# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on 11.05.17.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.
All rights reserved. See the LICENSE.TXT file for more details.
"""
import logging
import copy
import json
import matplotlib.pyplot as plt
import dill
import os
import numpy as np
import time
from time import gmtime, strftime
import warnings

# local imports
from properties import LabelProperties, IterationProperties, PlotProperties
from properties import instrument_start, instrument_close
from elasticity import load_isotropic_elasticity_matrix, load_TI_elasticity_matrix, mapping_old_indexes
from elasticity import load_isotropic_elasticity_matrix_toepliz
from mesh import CartesianMesh
from time_step_solution import attempt_time_step
from visualization import plot_footprint_analytical, plot_analytical_solution,\
                          plot_injection_source, get_elements
from symmetry import load_isotropic_elasticity_matrix_symmetric, symmetric_elasticity_matrix_from_full
from labels import TS_errorMessages, supported_projections, suitable_elements
from front_stability import audit_front_state, canonicalize_front_state, front_cfl_time_step


class Controller:
    """
    This class describes the controller which takes the given material, fluid, injection and loading properties and
    advances a given fracture according to the provided simulation properties.
    """

    errorMessages = TS_errorMessages

    def __init__(self, Fracture, Solid_prop, Fluid_prop, Injection_prop, Sim_prop, Load_prop=None, C=None):
        """ The constructor of the Controller class.

        Args:
           Fracture (Fracture):                     -- the fracture to be propagated.
           Solid_prop (MaterialProperties):         -- the MaterialProperties object giving the material properties.
           Fluid_prop (FluidProperties):            -- the FluidProperties object giving the fluid properties.
           Injection_prop (InjectionProperties):    -- the InjectionProperties object giving the injection.
                                                       properties.
           Sim_prop (SimulationProperties):         -- the SimulationProperties object giving the numerical
                                                       parameters to be used in the simulation.
           Load_prop (LoadingProperties):           -- the LoadingProperties object specifying how the material is
                                                       mechanically loaded.
           C (ndarray):                             -- the elasticity matrix.

        """
        log = logging.getLogger('PyFrac.controller')
        self.fracture = Fracture
        self.solid_prop = Solid_prop
        self.fluid_prop = Fluid_prop
        self.injection_prop = Injection_prop
        self.sim_prop = Sim_prop
        self.load_prop = Load_prop
        self.C = C
        self.fr_queue = [None, None, None, None, None]  # queue of fractures from the last five time steps
        # A fracture checkpoint is inseparable from the elasticity matrix
        # built on its mesh.  The legacy controller only queued the fracture;
        # after a remesh followed by a failed step it could therefore restore
        # an old fracture with a newer, differently-sized C matrix.  Keep only
        # a lightweight mesh signature here and rebuild C on rollback.  Five
        # deep copies of a dense elasticity matrix would be prohibitive on a
        # long run.
        self.C_queue = [None, None, None, None, None]
        self.c_was_provided = C is not None
        self.stepsFromChckPnt = 0
        self.tmStpPrefactor_copy = copy.copy(Sim_prop.tmStpPrefactor) # should be in simulation properties
        self.stagnant_TS = None     # time step if the front is stagnant. It is increased exponentialy to avoid uneccessary small steps.
        self.perfData = []
        self.lastSavedFile = 0
        self.lastSavedTime = np.NINF
        self.lastPlotTime = np.NINF
        self.TmStpCount = 0
        self.chkPntReattmpts = 0    # the number of re-attempts done from the checkpoint. Simulation is declared failed after 5 attempts.
        self.TmStpReductions = 0    # the number of times the time step has been reattempted because the fracture it was advancing too more than two cells in a row
        self.delta_w = None         # change in width between successive time steps. Used to limit time step.
        self.lstTmStp = None
        self.solveDetlaP_cp = self.sim_prop.solveDeltaP # copy of the flag indicating the solver to solve for pressure or delta p
        # The interactive application asks the user whether to jump to the
        # next positive injection after closure. Batch/assimilation runs must
        # make this decision deterministically; the adapter opts in through
        # ``autoJumpClosedFracture``.
        self.PstvInjJmp = (
            True if getattr(Sim_prop, "autoJumpClosedFracture", False) else None
        )
        self.fullyClosed = False    # should be related to the fracture state (thus in fracture class)
        self.setFigPos = True
        self.lastSuccessfulTS = Fracture.time
        self.maxTmStp = 0           # the maximum time step taken uptil now by the controller.
        self.frontCflLimitedSteps = 0
        self.frontStateRepairs = 0
        self.nonMonotonicStateRejects = 0
        self.lastTimeStep = float("nan")
        self.lastAcceptedDeltaTime = float("nan")
        self.lastTimeStepDiagnostics = {}
        self.lastAttemptStatus = None
        self.lastAttemptTimeStep = float("nan")
        self.lastAttemptFailureCause = None
        self.attemptStatusCounts = {}
        self.zeroInjectionJumps = []
        self.volumeProjectionCount = 0
        self.volumeProjectionLastFactor = float("nan")
        self.volumeProjectionMaxFactor = 1.0
        self.volumeProjectionMinFactor = 1.0
        self.progressFile = os.environ.get("PYFRAC_PROGRESS_FILE")
        self.nativeAttempt = int(os.environ.get("PYFRAC_NATIVE_ATTEMPT", "0"))


        # make a list of Nones with the size of the number of variables to plot during simulation
        self.Figures = [None for i in range(len(self.sim_prop.plotVar))]

        # Find the times where any parameter changes. These times will be added to the time series where the solution is
        # required to ensure the time is hit during time stepping and the change is applied at the exact time.
        param_change_at = np.array([], dtype=np.float64)
        if Injection_prop.injectionRate.shape[1] > 1:
           param_change_at = np.hstack((param_change_at, Injection_prop.injectionRate[0]))
        if isinstance(Sim_prop.fixedTmStp, np.ndarray):
           param_change_at = np.hstack((param_change_at, Sim_prop.fixedTmStp[0]))
        if isinstance(Sim_prop.tmStpPrefactor, np.ndarray):
           param_change_at = np.hstack((param_change_at, Sim_prop.tmStpPrefactor[0]))


        if len(param_change_at) > 0:
            if self.sim_prop.get_solTimeSeries() is not None:
                # add the times where any parameter changes to the required solution time series
                sol_time_srs = np.hstack((self.sim_prop.get_solTimeSeries(), param_change_at))
            else:
                sol_time_srs = param_change_at
            sol_time_srs = np.unique(sol_time_srs)
            if sol_time_srs[0] == 0:
                sol_time_srs = np.delete(sol_time_srs, 0)
        else:
           sol_time_srs = self.sim_prop.get_solTimeSeries()
        self.timeToHit = sol_time_srs

        if self.sim_prop.finalTime is None:
           if self.sim_prop.get_solTimeSeries() is None:
               ## Not necessarily an error
                raise ValueError("The final time to stop the simulation is not provided!")
           else:
               self.sim_prop.finalTime = np.max(self.sim_prop.get_solTimeSeries())
        else:
            if self.timeToHit is not None:
                greater_finalTime = np.where(self.timeToHit > self.sim_prop.finalTime)[0]
                self.timeToHit = np.delete(self.timeToHit, greater_finalTime)

        # Setting to volume control solver if viscosity is zero
        if self.fluid_prop.rheology == 'Newtonian' and self.fluid_prop.viscosity < 1e-15:
           print("Fluid viscosity is zero. Setting solver to volume control...")
           self.sim_prop.set_volumeControl(True)

        if self.injection_prop.sourceLocFunc is None:
            if not all(elem in self.fracture.EltChannel for elem in Injection_prop.sourceElem):
                message = 'INJECTION LOCATION ERROR: \n' \
                          'injection points are located outisde of the fracture footprints'
                raise SystemExit(message)

        # Setting whether sparse matrix is used to make fluid conductivity matrix
        if Sim_prop.solveSparse is None:
           if Fracture.mesh.NumberOfElts > 2500 or self.injection_prop.modelInjLine:
               Sim_prop.solveSparse = True
           else:
               Sim_prop.solveSparse = False

        # basic performance data
        self.remeshings = 0
        self.successfulTimeSteps = 0
        self.failedTimeSteps = 0

        # setting front advancing scheme to implicit if velocity is not available for the first time step.
        self.frontAdvancing = copy.copy(Sim_prop.frontAdvancing)
        if Sim_prop.frontAdvancing in ['explicit', 'predictor-corrector']:
            if np.nanmax(Fracture.v) <= 0 or np.isnan(Fracture.v).any():
                Sim_prop.frontAdvancing = 'implicit'

        if self.sim_prop.saveToDisk:
            self.logAddress = copy.copy(Sim_prop.get_outputFolder())
        else:
            self.logAddress = './'

        # setting up tip asymptote
        if self.fluid_prop.rheology in ["Herschel-Bulkley", "HBF"]:
            if self.sim_prop.get_tipAsymptote() not in ["HBF", "HBF_aprox", "HBF_num_quad"]:
                warnings.warn("Fluid rhelogy and tip asymptote does not match. Setting tip asymptote to \'HBF\'")
                self.sim_prop.set_tipAsymptote('HBF')
        if self.fluid_prop.rheology in ["power-law", "PLF"]:
            if self.sim_prop.get_tipAsymptote() not in ["PLF", "PLF_aprox", "PLF_num_quad", "PLF_M"]:
                warnings.warn("Fluid rhelogy and tip asymptote does not match. Setting tip asymptote to \'PLF\'")
                self.sim_prop.set_tipAsymptote('PLF')
        if self.fluid_prop.rheology == 'Newtonian':
            if self.sim_prop.get_tipAsymptote() not in ["K", "M", "Mt", "U", "U1", "MK", "MDR", "M_MDR"]:
                warnings.warn("Fluid rhelogy and tip asymptote does not match. Setting tip asymptote to \'U\'")
                self.sim_prop.set_tipAsymptote('U1')

        if self.fluid_prop.rheology != 'Newtonian':
            self.sim_prop.saveRegime = False

        # if you set the code to advance max 1 cell then remove the SimulProp.timeStepLimit
        if self.sim_prop.timeStepLimit is not None and self.sim_prop.limitAdancementTo2cells is True:
            if self.sim_prop.forceTmStpLmtANDLmtAdvTo2cells == False:
                warnings.warn("You have set sim_prop.limitAdancementTo2cells = True. This imply that sim_prop.timeStepLimit will be deactivated.")
                self.sim_prop.timeStepLimit = None
            else:
                warnings.warn(
                    "You have forced <limitAdancementTo2cells> to be True and set <timeStepLimit> - the first one might be uneffective onto the second one until the prefactor has been reduced to produce a time step < timeStepLimit")
#-----------------------------------------------------------------------------------------------------------------------

    def run(self):
        """
        This function runs the simulation according to the parameters given in the properties classes. See especially
        the documentation of the :py:class:`properties.SimulationProperties` class to get details of the parameters
        controlling the simulation run.
        """
        log = logging.getLogger('PyFrac.controller.run')
        log_only_to_logfile = logging.getLogger('PyFrac_LF.controller.run')
        self._write_progress(status=0, phase="controller_start")

        # output initial fracture
        if self.sim_prop.saveToDisk:
            # save properties
            if not os.path.exists(self.sim_prop.get_outputFolder()):
                os.makedirs(self.sim_prop.get_outputFolder())

            prop = (self.solid_prop, self.fluid_prop, self.injection_prop, self.sim_prop)
            with open(self.sim_prop.get_outputFolder() + "properties", 'wb') as output:
                dill.dump(prop, output, -1)

        if self.sim_prop.plotFigure or self.sim_prop.saveToDisk:
            # save or plot fracture
            self.output(self.fracture)
            self.lastSavedTime = self.fracture.time

        if self.sim_prop.log2file:
            self.sim_prop.set_logging_to_file(self.logAddress)

        # deactivate the block_toepliz_compression functions
        # DO THIS CHECK BEFORE COMPUTING C!
        if self.C is not None: # in the case C is provided
            self.sim_prop.useBlockToeplizCompression = False
        elif self.solid_prop.TI_elasticity: # in case of TI_elasticity
            self.sim_prop.useBlockToeplizCompression = False
        elif not self.solid_prop.TI_elasticity and self.sim_prop.symmetric:  # in case you save 1/4 of the elasticity due to domain symmetry
            self.sim_prop.useBlockToeplizCompression = False

        # load elasticity matrix
        if self.C is None:
            log.info("Making elasticity matrix...")
            if self.sim_prop.symmetric:
                if not self.sim_prop.get_volumeControl():
                    raise ValueError("Symmetric fracture is only supported for inviscid fluid yet!")

            if not self.solid_prop.TI_elasticity:
                if self.sim_prop.symmetric:
                    self.C = load_isotropic_elasticity_matrix_symmetric(self.fracture.mesh,
                                                                        self.solid_prop.Eprime)
                else:
                    if not self.sim_prop.useBlockToeplizCompression:
                        self.C = load_isotropic_elasticity_matrix(self.fracture.mesh,
                                                                  self.solid_prop.Eprime)
                    else:
                        self.C = load_isotropic_elasticity_matrix_toepliz(self.fracture.mesh,
                                                                          self.solid_prop.Eprime)
            else:
                C = load_TI_elasticity_matrix(self.fracture.mesh,
                                                   self.solid_prop,
                                                   self.sim_prop)
                # compressing the elasticity matrix for symmetric fracture
                if self.sim_prop.symmetric:
                    self.C = symmetric_elasticity_matrix_from_full(C, self.fracture.mesh)
                else:
                    self.C = C
            log.info('Done!')

        # # perform first time step with implicit front advancing due to non-availability of velocity
        # if not self.sim_prop.symmetric:
        #     if self.sim_prop.frontAdvancing == "predictor-corrector":
        #         self.sim_prop.frontAdvancing = "implicit"

        log.info("Starting time = " + repr(self.fracture.time))
        self._write_progress(status=0, phase="time_marching_ready")
        # starting time stepping loop
        # Upstream 1.1.1 used ``0.999 * finalTime`` here.  On long runs that
        # silently stops several model seconds early (4.435 s for a 4435 s
        # target) and can also create a zero-step restart near an assimilation
        # node.  Keep only a floating-point tolerance; get_time_step() already
        # clips the last accepted step to finalTime.
        final_time_tolerance = max(1e-8, 1e-10 * abs(self.sim_prop.finalTime))
        while self.fracture.time < self.sim_prop.finalTime - final_time_tolerance and self.TmStpCount < self.sim_prop.maxTimeSteps:

            # Optional diagnostic path for the legacy closed-front failure:
            # when the input rate is exactly zero, advance only the clock to
            # the next positive-injection event. This does not solve the
            # shut-in PDE; recording the interval makes the limitation
            # explicit and prevents it being mistaken for a full dynamic
            # validation.
            if getattr(self.sim_prop, "skipZeroInjectionIntervals", False):
                current_rate = np.asarray(
                    self.injection_prop.get_injection_rate(self.fracture.time, self.fracture),
                    dtype=float,
                ).reshape(-1)
                if (
                    current_rate.size
                    and np.all(np.isfinite(current_rate))
                    and np.max(np.abs(current_rate)) <= 1.0e-12
                ):
                    times = np.asarray(self.injection_prop.injectionRate[0, :], dtype=float)
                    rates = np.asarray(self.injection_prop.injectionRate[1, :], dtype=float)
                    candidates = times[
                        (times > self.fracture.time + final_time_tolerance)
                        & (rates > 1.0e-12)
                    ]
                    if candidates.size:
                        next_time = min(float(np.min(candidates)), float(self.sim_prop.finalTime))
                        if next_time > self.fracture.time + final_time_tolerance:
                            start_time = float(self.fracture.time)
                            self.fracture.time = next_time
                            self.zeroInjectionJumps.append({
                                "start_time_s": start_time,
                                "end_time_s": next_time,
                                "duration_s": next_time - start_time,
                            })
                            self.lastAcceptedDeltaTime = next_time - start_time
                            self._write_progress(status=14, phase="zero_injection_interval_skipped")
                            continue

            timeStep = self.get_time_step()
            # Apply the front-traversal cap before calling the nonlinear
            # solver.  The legacy implementation only detects a multi-cell
            # jump after reconstruction (status 17), which is too late for
            # the continuous-front bookkeeping to remain reliable.
            front_cfl = getattr(self.sim_prop, "frontCFL", 0.8)
            limited_step, was_limited = front_cfl_time_step(
                timeStep,
                getattr(self.fracture, "v", None),
                self.fracture.mesh.hx,
                self.fracture.mesh.hy,
                front_cfl,
            )
            if was_limited:
                self.frontCflLimitedSteps += 1
                log.debug(
                    "front CFL limited time step %.6g -> %.6g s at t=%.6g s",
                    timeStep,
                    limited_step,
                    self.fracture.time,
                )
                timeStep = limited_step
            self.lastTimeStep = float(timeStep)

            # A continuation window may contain a rate transition (for
            # example, a short ramp followed by a shut-in).  The legacy
            # controller restores ``solveDeltaP`` after every accepted step,
            # so setting it once when the Controller is constructed is not
            # sufficient.  When the adapter opts in, select the pressure
            # formulation at the actual current injection state for every
            # step: delta-pressure during positive injection, absolute
            # pressure during shut-in.  This prevents the delta-pressure
            # system from becoming singular after injection is stopped while
            # preserving the original behavior for all other callers.
            if getattr(self.sim_prop, "solveDeltaPByInjectionRate", False):
                current_q = self.injection_prop.get_injection_rate(self.fracture.time, self.fracture)
                # The first step after a closed-fracture jump must rebuild
                # absolute pressure from elasticity and the leak-off ledger.
                # Switching immediately to delta-pressure at the jumped time
                # uses a zero/closed pressure increment as its reference and
                # is a common source of an invalid EHL iterate.
                self.sim_prop.solveDeltaP = (
                    False if self.fullyClosed else bool(np.nanmax(current_q) > 1.0e-12)
                )

            if self.sim_prop.collectPerfData:
                tmStp_perf = IterationProperties(itr_type="time step")
            else:
                tmStp_perf = None

            # advancing time step
            status, Fr_n_pls1 = self.advance_time_step(self.fracture,
                                                         self.C,
                                                        timeStep,
                                                        tmStp_perf)

            if status == 1 and Fr_n_pls1 is not None:
                # A solver return code of 1 is not sufficient evidence that
                # the candidate state is usable.  Older PyFrac paths can
                # return a state whose time went backwards after a failed
                # reconstruction/remesh.  Accepting it would poison the
                # checkpoint queue and make the next loop silently replay an
                # earlier physical time while reporting success.
                current_time = float(self.fracture.time)
                next_time = float(getattr(Fr_n_pls1, "time", np.nan))
                time_tolerance = max(1.0e-10, 1.0e-10 * abs(current_time))
                if (not np.isfinite(next_time)) or next_time <= current_time + time_tolerance:
                    self.nonMonotonicStateRejects += 1
                    log.warning(
                        "front state rejected because time did not advance: %.12g -> %.12g",
                        current_time,
                        next_time,
                    )
                    status = 18
                else:
                    repairs = canonicalize_front_state(Fr_n_pls1)
                    self.frontStateRepairs += len(repairs)
                    if repairs:
                        log.warning("canonicalized front state after accepted step: %s", "; ".join(repairs))
                    structural_errors = audit_front_state(Fr_n_pls1)
                    if structural_errors:
                        # Never place a structurally inconsistent state in the
                        # five-state retry queue.  It will follow the normal
                        # controller rollback path instead.
                        log.warning("front state rejected after reconstruction: %s", "; ".join(structural_errors))
                        status = 18

            if self.sim_prop.collectPerfData:
                tmStp_perf.CpuTime_end = time.time()
                tmStp_perf.status = status == 1
                tmStp_perf.failure_cause = self.errorMessages[status]
                tmStp_perf.time = self.fracture.time
                tmStp_perf.NumbOfElts = len(self.fracture.EltCrack)
                self.perfData.append(tmStp_perf)

            if status == 1:
            # Successful time step
                log.info("Time step successful!")
                log.debug("Element in the crack: "+str(len(Fr_n_pls1.EltCrack)))
                log.debug("Nx: " + str(Fr_n_pls1.mesh.nx))
                log.debug("Ny: " + str(Fr_n_pls1.mesh.ny))
                log.debug("hx: " + str(Fr_n_pls1.mesh.hx))
                log.debug("hy: " + str(Fr_n_pls1.mesh.hy))

                # The legacy delta-pressure solve can produce an unphysical
                # negative absolute fluid pressure during a rapid rate ramp
                # down.  Once that value is carried into the next step, the
                # increment equation can diverge by many orders of magnitude.
                # The realtime continuation adapter opts into a conservative
                # zero-gauge-pressure floor.  This is a numerical/physical
                # admissibility guard, not an observation correction: it only
                # clips impossible absolute pressure and recomputes pNet from
                # the clipped pFluid and the current confining stress.
                pressure_floor = getattr(self.sim_prop, "pressureFloorPa", None)
                if pressure_floor is not None and hasattr(Fr_n_pls1, "pFluid"):
                    crack_indices = np.asarray(Fr_n_pls1.EltCrack, dtype=int)
                    if crack_indices.size:
                        p_floor = float(pressure_floor)
                        Fr_n_pls1.pFluid[crack_indices] = np.maximum(
                            np.asarray(Fr_n_pls1.pFluid[crack_indices], dtype=float),
                            p_floor,
                        )
                        Fr_n_pls1.pNet[crack_indices] = (
                            Fr_n_pls1.pFluid[crack_indices]
                            - self.solid_prop.SigmaO[crack_indices]
                        )
                self._apply_volume_balance_projection(Fr_n_pls1)
                self.delta_w = Fr_n_pls1.w - self.fracture.w
                self.lstTmStp = Fr_n_pls1.time - self.fracture.time
                self.lastAcceptedDeltaTime = float(self.lstTmStp)
                # output
                if self.sim_prop.plotFigure or self.sim_prop.saveToDisk:
                    if Fr_n_pls1.time > self.lastSavedTime:
                        self.output(Fr_n_pls1)

                # add the advanced fracture to the last five fractures list
                self.fracture = copy.deepcopy(Fr_n_pls1)
                queue_index = self.successfulTimeSteps % 5
                self.fr_queue[queue_index] = copy.deepcopy(Fr_n_pls1)
                self.C_queue[queue_index] = (
                    int(self.fracture.mesh.nx),
                    int(self.fracture.mesh.ny),
                    int(self.fracture.mesh.NumberOfElts),
                )

                if self.fracture.time > self.lastSuccessfulTS:
                    self.lastSuccessfulTS = self.fracture.time
                if self.maxTmStp < self.lstTmStp:
                    self.maxTmStp = self.lstTmStp
                # put check point reattempts to zero if the simulation has advanced past the time where it failed
                if Fr_n_pls1.time > self.lastSuccessfulTS + 2 * self.maxTmStp:
                    self.chkPntReattmpts = 0
                    # set the prefactor to the original value after four time steps (after the 5 time steps back jump)
                    self.sim_prop.tmStpPrefactor = self.tmStpPrefactor_copy
                self.successfulTimeSteps += 1
                # set to 0 the counter of time step reductions
                if self.TmStpReductions > 0:
                    self.TmStpReductions = 0
                    self.sim_prop.tmStpPrefactor = self.tmStpPrefactor_copy
                # resetting the parameters for closure
                if self.fullyClosed:
                    # set to solve for pressure if the fracture was fully closed in last time step and is open now
                    self.sim_prop.solveDeltaP = False
                else:
                    self.sim_prop.solveDeltaP = self.solveDetlaP_cp
                self.PstvInjJmp = (
                    True if getattr(self.sim_prop, "autoJumpClosedFracture", False)
                    else None
                )
                self.fullyClosed = False

                # Set front advancing back as set in simulation properties
                # originally if velocity becomes available.  A same-footprint
                # continuation can legitimately return an empty velocity array
                # in the legacy ILSA path; ``np.max(empty)`` used to abort the
                # whole native run here.  Treat that state as “velocity not
                # available” and let the next step use the implicit path.
                velocity = np.asarray(Fr_n_pls1.v)
                if velocity.size and (np.max(velocity) > 0 or not np.isnan(velocity).any()):
                    self.sim_prop.frontAdvancing = copy.copy(self.frontAdvancing)
                else:
                    self.sim_prop.frontAdvancing = 'implicit'

                if self.TmStpCount == self.sim_prop.maxTimeSteps:
                    log.warning("Max time steps reached!")

            elif status == 12 or status == 16:
                # re-meshing required
                if self.sim_prop.enableRemeshing:
                    # the following update is required because Fr_n_pls1.EltTip contains the intersection between the cells at the boundary of the mesh and
                    # the reconstructed front. For that reason in case of mesh extension
                    if hasattr(Fr_n_pls1, 'EltTipBefore'):
                        self.fracture.EltTipBefore = Fr_n_pls1.EltTipBefore
                    # we need to decide which remeshings are to be considered
                    compress = False
                    if status == 16:
                        # we reached cell number limit so we adapt by compressing the domain accordingly

                        # calculate the new number of cells
                        new_elems = [int((self.fracture.mesh.nx + np.round(self.sim_prop.meshReductionFactor, 0))
                                         / self.sim_prop.meshReductionFactor),
                                     int((self.fracture.mesh.ny + np.round(self.sim_prop.meshReductionFactor, 0))
                                         / self.sim_prop.meshReductionFactor)]
                        if new_elems[0] % 2 == 0:
                            new_elems[0] = new_elems[0] + 1
                        if new_elems[1] % 2 == 0:
                            new_elems[1] = new_elems[1] + 1

                        # Decide if we still can reduce the number of elements
                        if (2 * self.fracture.mesh.Lx / new_elems[0] > self.sim_prop.maxCellSize) or (2 *
                            self.fracture.mesh.Ly / new_elems[1] > self.fracture.mesh.hy / self.fracture.mesh.hx *
                            self.sim_prop.maxCellSize):
                            log.warning("Reduction of cells not possible as minimal cell size would be violated!")
                            self.sim_prop.meshReductionPossible = False
                        else:

                            log.info("Reducing cell number...")

                            # We need to make sure the injection point stays where it is. We also do this for two points
                            # on same x or y
                            if len(self.fracture.source) == 1:
                                index = self.fracture.source[0]
                                cent_point = self.fracture.mesh.CenterCoor[self.fracture.source[0]]

                                reduction_factor = self.sim_prop.meshReductionFactor
                            elif len(self.fracture.source) == 2:
                                index = self.fracture.source[0]
                                cent_point = self.fracture.mesh.CenterCoor[self.fracture.source[0]]

                                if self.fracture.mesh.CenterCoor[self.fracture.source[0]][0] == \
                                        self.fracture.mesh.CenterCoor[self.fracture.source[1]][0]:
                                    elems_inter = int(abs(self.fracture.mesh.CenterCoor[self.fracture.source[0]][1] -
                                        self.fracture.mesh.CenterCoor[self.fracture.source[1]][1]) / \
                                                  self.fracture.mesh.hy)
                                    new_inter = int(np.ceil(elems_inter/self.sim_prop.meshReductionFactor))
                                    reduction_factor = elems_inter / new_inter

                                elif self.fracture.mesh.CenterCoor[self.fracture.source[0]][1] == \
                                        self.fracture.mesh.CenterCoor[self.fracture.source[1]][1]:
                                    elems_inter = int(abs(self.fracture.mesh.CenterCoor[self.fracture.source[0]][0] -
                                        self.fracture.mesh.CenterCoor[self.fracture.source[1]][0]) / \
                                                  self.fracture.mesh.hx)
                                    new_inter = int(np.ceil(elems_inter / self.sim_prop.meshReductionFactor))
                                    reduction_factor = elems_inter / new_inter

                                else:
                                    reduction_factor = self.sim_prop.meshReductionFactor

                                log.info("The real reduction factor used is " + repr(reduction_factor))

                            else:
                                index = self.fracture.mesh.locate_element(0., 0.)[0]
                                cent_point = np.asarray([0., 0.])

                                reduction_factor = self.sim_prop.meshReductionFactor

                            row = int(index/self.fracture.mesh.nx)
                            column = index - self.fracture.mesh.nx * row

                            row_frac = (self.fracture.mesh.ny - (row + 1))/row
                            col_frac = column/(self.fracture.mesh.nx - (column + 1))

                            # calculate the new number of cells
                            new_elems = [int((self.fracture.mesh.nx + np.round(reduction_factor, 0))
                                             / reduction_factor),
                                         int((self.fracture.mesh.ny + np.round(reduction_factor, 0))
                                             / reduction_factor)]
                            if new_elems[0] % 2 == 0:
                                new_elems[0] = new_elems[0] + 1
                            if new_elems[1] % 2 == 0:
                                new_elems[1] = new_elems[1] + 1


                            # We calculate the new dimension of the meshed area
                            new_limits = [[cent_point[0] - round((new_elems[0] - 1)/(1 / col_frac + 1)) *
                                           self.fracture.mesh.hx * reduction_factor,
                                           cent_point[0] + (new_elems[0] - round((new_elems[0] - 1)/(1 / col_frac + 1))
                                                            - 1) * self.fracture.mesh.hx *
                                           reduction_factor],
                                          [cent_point[1] - round((new_elems[1] - 1) / (row_frac + 1)) *
                                           self.fracture.mesh.hy * reduction_factor,
                                           cent_point[1] + (new_elems[1] - round((new_elems[1] - 1) / (row_frac + 1))
                                                            - 1) * self.fracture.mesh.hy *
                                           reduction_factor]]

                            elems = new_elems
                            direction = 'reduce'
                            self.remesh(new_limits, elems, direction)

                            # set all other to zero
                            side_bools = [False, False, False, False]

                    elif status == 12:
                        if self.sim_prop.meshExtensionAllDir:
                            # we extend no matter how many boundaries we have hit
                            # ensure all directions to extend are true
                            self.sim_prop.set_mesh_extension_direction(['all'])

                        # ``Frontlist`` is an ordered boundary list, not a
                        # global mesh-index map.  After a remesh/extension its
                        # positional index cannot be used to infer which
                        # physical side was reached.  The old implementation
                        # did exactly that, causing horizontal-only extension
                        # to be skipped or the wrong side to be extended.
                        # Determine touched sides from actual tip-cell
                        # centers.  The order remains [bottom, top, left,
                        # right].
                        tip_cells = np.asarray(Fr_n_pls1.EltTip, dtype=int).reshape(-1)
                        tip_cells = tip_cells[
                            (tip_cells >= 0) &
                            (tip_cells < Fr_n_pls1.mesh.NumberOfElts)
                        ]
                        if tip_cells.size:
                            tip_centers = Fr_n_pls1.mesh.CenterCoor[tip_cells]
                            edge_x = max(float(Fr_n_pls1.mesh.hx), 1.0e-12) * 1.5
                            edge_y = max(float(Fr_n_pls1.mesh.hy), 1.0e-12) * 1.5
                            x_min = float(Fr_n_pls1.mesh.domainLimits[2])
                            x_max = float(Fr_n_pls1.mesh.domainLimits[3])
                            y_min = float(Fr_n_pls1.mesh.domainLimits[0])
                            y_max = float(Fr_n_pls1.mesh.domainLimits[1])
                            side_bools = [
                                bool(np.any(tip_centers[:, 1] <= y_min + edge_y)),
                                bool(np.any(tip_centers[:, 1] >= y_max - edge_y)),
                                bool(np.any(tip_centers[:, 0] <= x_min + edge_x)),
                                bool(np.any(tip_centers[:, 0] >= x_max - edge_x)),
                            ]
                        else:
                            side_bools = [False, False, False, False]

                        if not self.sim_prop.meshExtensionAllDir:
                            compress = \
                                not np.asarray(np.asarray(self.sim_prop.meshExtension) * np.asarray(side_bools)).any() \
                                or (len(np.asarray(side_bools)[np.asarray(side_bools) == True]) > 3)


                    # This is the classical remeshing where the sides of the elements are multiplied by a constant.
                    # The legacy fallback compresses the domain whenever an
                    # allowed extension is unavailable.  That keeps the
                    # front inside the grid but changes the physical scale
                    # and can create an artificial high-pressure state on a
                    # long run.  An opt-in fixed-cell regrid grows only the
                    # horizontal domain, preserving the initial small-grid
                    # resolution while avoiding an ever-growing elasticity
                    # matrix.  The remesh path remains subject to the normal
                    # front and mass audits.
                    if compress and getattr(self.sim_prop, "expandDomainOnBoundary", False):
                        expansion = max(
                            float(getattr(self.sim_prop, "domainExpansionFactor", 2.0)),
                            1.05,
                        )
                        x_min = float(self.fracture.mesh.domainLimits[2])
                        x_max = float(self.fracture.mesh.domainLimits[3])
                        y_min = float(self.fracture.mesh.domainLimits[0])
                        y_max = float(self.fracture.mesh.domainLimits[1])
                        x_center = 0.5 * (x_min + x_max)
                        x_half = 0.5 * (x_max - x_min) * expansion
                        new_limits = [
                            [x_center - x_half, x_center + x_half],
                            [y_min, y_max],
                        ]
                        elems = [self.fracture.mesh.nx, self.fracture.mesh.ny]
                        log.info(
                            "Regridding to expand horizontal domain by %.3g with fixed cell count",
                            expansion,
                        )
                        self.remesh(
                            new_limits,
                            elems,
                            direction=None,
                            rem_factor=expansion,
                        )
                        compress = False
                        side_bools = [False, False, False, False]

                    if compress:
                        log.info("Remeshing by compressing the domain...")

                        # We need to make sure the injection point stays where it is. We also do this for two points
                        # on same x or y
                        if len(self.fracture.source) == 1:
                            index = self.fracture.source[0]
                            cent_point = self.fracture.mesh.CenterCoor[self.fracture.source[0]]

                            compression_factor = self.sim_prop.remeshFactor
                        elif len(self.fracture.source) == 2:
                            index = self.fracture.source[0]
                            cent_point = self.fracture.mesh.CenterCoor[self.fracture.source[0]]

                            if self.fracture.mesh.CenterCoor[self.fracture.source[0]][0] == \
                                    self.fracture.mesh.CenterCoor[self.fracture.source[1]][0]:
                                elems_inter = int(abs(self.fracture.mesh.CenterCoor[self.fracture.source[0]][1] -
                                                      self.fracture.mesh.CenterCoor[self.fracture.source[1]][1]) / \
                                                  self.fracture.mesh.hy)
                                new_inter = int(np.ceil(elems_inter / self.sim_prop.remeshFactor))
                                compression_factor = elems_inter / new_inter

                            elif self.fracture.mesh.CenterCoor[self.fracture.source[0]][1] == \
                                    self.fracture.mesh.CenterCoor[self.fracture.source[1]][1]:
                                elems_inter = int(abs(self.fracture.mesh.CenterCoor[self.fracture.source[0]][0] -
                                                      self.fracture.mesh.CenterCoor[self.fracture.source[1]][0]) / \
                                                  self.fracture.mesh.hx)
                                new_inter = int(np.ceil(elems_inter / self.sim_prop.remeshFactor))
                                compression_factor = elems_inter / new_inter

                            else:
                                compression_factor = self.sim_prop.remeshFactor

                            log.info("The real reduction factor used is " + repr(compression_factor))

                        else:
                            index = self.fracture.mesh.locate_element(0., 0.)[0]
                            cent_point = np.asarray([0., 0.])

                            compression_factor = self.sim_prop.remeshFactor

                        row = int(index / self.fracture.mesh.nx)
                        column = index - self.fracture.mesh.nx * row

                        row_frac = (self.fracture.mesh.ny - (row + 1)) / row
                        col_frac = column / (self.fracture.mesh.nx - (column + 1))

                        # We calculate the new dimension of the meshed area
                        new_limits = [[cent_point[0] - round((self.fracture.mesh.nx - 1) / (1 / col_frac + 1)) *
                                       self.fracture.mesh.hx * compression_factor,
                                       cent_point[0] + (self.fracture.mesh.nx - round((self.fracture.mesh.nx - 1) /
                                                                                      (1 / col_frac + 1)) - 1) *
                                       self.fracture.mesh.hx * compression_factor],
                                      [cent_point[1] - round((self.fracture.mesh.ny - 1) / (row_frac + 1)) *
                                       self.fracture.mesh.hy * compression_factor,
                                       cent_point[1] + (self.fracture.mesh.ny - round((self.fracture.mesh.ny - 1) /
                                                                                       (row_frac + 1)) - 1) *
                                       self.fracture.mesh.hy * compression_factor]]

                        # # We calculate the new dimension of the meshed area
                        # new_dimensions = 2 * self.sim_prop.remeshFactor * np.asarray([self.fracture.mesh.Lx,
                        #                                                           self.fracture.mesh.Ly])
                        # new_limits = [[(self.fracture.mesh.domainLimits[2]+self.fracture.mesh.domainLimits[3]) / 2
                        #                - new_dimensions[0]/2, (self.fracture.mesh.domainLimits[2] +
                        #                                        self.fracture.mesh.domainLimits[3]) / 2
                        #                + new_dimensions[0]/2],
                        #               [(self.fracture.mesh.domainLimits[0]+self.fracture.mesh.domainLimits[1]) / 2
                        #                - new_dimensions[1]/2, (self.fracture.mesh.domainLimits[0] +
                        #                                        self.fracture.mesh.domainLimits[1]) / 2
                        #                + new_dimensions[1]/2]]

                        elems = [self.fracture.mesh.nx, self.fracture.mesh.ny]

                        if len(np.intersect1d(self.fracture.mesh.CenterElts, index)) == 0:
                            compression_factor = 10

                        self.remesh(new_limits, elems, rem_factor=compression_factor)

                        side_bools = [False, False, False, False]

                    else:
                        nx_init = self.fracture.mesh.nx
                        ny_init = self.fracture.mesh.ny
                        for side in range(4):
                            if np.asarray(np.asarray(self.sim_prop.meshExtension) * np.asarray(side_bools))[side]:
                                if side == 0:

                                    elems_add = int(ny_init * (self.sim_prop.meshExtensionFactor[side] - 1))
                                    if elems_add % 2 != 0:
                                        elems_add = elems_add + 1

                                    if not self.sim_prop.symmetric:
                                        log.info("Remeshing by extending towards negative y...")
                                        new_limits = [[self.fracture.mesh.domainLimits[2],
                                                       self.fracture.mesh.domainLimits[3]],
                                                      [self.fracture.mesh.domainLimits[0] -
                                                       elems_add * self.fracture.mesh.hy,
                                                       self.fracture.mesh.domainLimits[1]]]
                                    else:
                                        log.info("Remeshing by extending in vertical direction to keep symmetry...")
                                        new_limits = [[self.fracture.mesh.domainLimits[2],
                                                       self.fracture.mesh.domainLimits[3]],
                                                      [self.fracture.mesh.domainLimits[0] -
                                                       elems_add * self.fracture.mesh.hy/2,
                                                       self.fracture.mesh.domainLimits[1] +
                                                       elems_add * self.fracture.mesh.hy/2]]
                                        side_bools[1] = False

                                    # For a symmetric mesh both vertical sides
                                    # are added at once.  ``mapping_old_indexes``
                                    # needs the paired-direction token so that
                                    # old rows are centred in the new mesh;
                                    # using ``bottom`` here shifts every old
                                    # cell by the full added height.
                                    direction = 'vertical' if self.sim_prop.symmetric else 'bottom'

                                    elems = [self.fracture.mesh.nx, self.fracture.mesh.ny + elems_add]


                                if side == 1:

                                    elems_add = int(ny_init * (self.sim_prop.meshExtensionFactor[side] - 1))
                                    if elems_add % 2 != 0:
                                        elems_add = elems_add + 1

                                    if not self.sim_prop.symmetric:
                                        log.info("Remeshing by extending towards positive y...")
                                        new_limits = [[self.fracture.mesh.domainLimits[2],
                                                       self.fracture.mesh.domainLimits[3]],
                                                      [self.fracture.mesh.domainLimits[0],
                                                       self.fracture.mesh.domainLimits[1] +
                                                       elems_add * self.fracture.mesh.hy]]
                                    else:
                                        log.info("Remeshing by extending in vertical direction to keep symmetry...")
                                        new_limits = [[self.fracture.mesh.domainLimits[2],
                                                       self.fracture.mesh.domainLimits[3]],
                                                      [self.fracture.mesh.domainLimits[0] -
                                                       elems_add * self.fracture.mesh.hy/2,
                                                       self.fracture.mesh.domainLimits[1] +
                                                       elems_add * self.fracture.mesh.hy/2]]
                                        side_bools[0] = False

                                    direction = 'vertical' if self.sim_prop.symmetric else 'top'

                                    elems = [self.fracture.mesh.nx, self.fracture.mesh.ny + elems_add]

                                if side == 2:

                                    elems_add = int(nx_init * (self.sim_prop.meshExtensionFactor[side] - 1))
                                    if elems_add % 2 != 0:
                                        elems_add = elems_add + 1

                                    if not self.sim_prop.symmetric:
                                        log.info("Remeshing by extending towards negative x...")
                                        new_limits = [
                                            [self.fracture.mesh.domainLimits[2] - elems_add * self.fracture.mesh.hx,
                                             self.fracture.mesh.domainLimits[3]],
                                            [self.fracture.mesh.domainLimits[0],
                                             self.fracture.mesh.domainLimits[1]]]
                                    else:
                                        log.info("Remeshing by extending in horizontal direction to keep symmetry...")
                                        new_limits = [
                                            [self.fracture.mesh.domainLimits[2] - elems_add * self.fracture.mesh.hx/2,
                                             self.fracture.mesh.domainLimits[3] + elems_add * self.fracture.mesh.hx/2],
                                            [self.fracture.mesh.domainLimits[0],
                                             self.fracture.mesh.domainLimits[1]]]
                                        side_bools[3] = False

                                    # Symmetric horizontal extension grows the
                                    # domain on both x sides.  The old code
                                    # passed ``left``/``right`` even though
                                    # both sides had been added, so the index
                                    # map became row-dependent and moved crack
                                    # cells outside their physical locations.
                                    direction = 'horizontal' if self.sim_prop.symmetric else 'left'

                                    elems = [self.fracture.mesh.nx + elems_add, self.fracture.mesh.ny]

                                if side == 3:

                                    elems_add = int(nx_init * (self.sim_prop.meshExtensionFactor[side] - 1))
                                    if elems_add % 2 != 0:
                                        elems_add = elems_add + 1

                                    if not self.sim_prop.symmetric:
                                        log.info("Remeshing by extending towards positive x...")
                                        new_limits = [[self.fracture.mesh.domainLimits[2],
                                                       self.fracture.mesh.domainLimits[
                                                           3] + elems_add * self.fracture.mesh.hx],
                                                      [self.fracture.mesh.domainLimits[0],
                                                       self.fracture.mesh.domainLimits[1]]]
                                    else:
                                        log.info("Remeshing by extending in horizontal direction to keep symmetry...")
                                        new_limits = [
                                            [self.fracture.mesh.domainLimits[2] - elems_add * self.fracture.mesh.hx/2,
                                             self.fracture.mesh.domainLimits[3] + elems_add * self.fracture.mesh.hx/2],
                                            [self.fracture.mesh.domainLimits[0],
                                             self.fracture.mesh.domainLimits[1]]]
                                        side_bools[2] = False

                                    direction = 'horizontal' if self.sim_prop.symmetric else 'right'

                                    elems = [self.fracture.mesh.nx + elems_add, self.fracture.mesh.ny]

                                self.remesh(new_limits, elems, direction=direction)
                                side_bools[side] = False

                    if np.asarray(side_bools).any():
                        log.info("Remeshing by compressing the domain...")

                        # We need to make sure the injection point stays where it is. We also do this for two points
                        # on same x or y
                        if len(self.fracture.source) == 1:
                            index = self.fracture.source[0]
                            cent_point = self.fracture.mesh.CenterCoor[self.fracture.source[0]]

                            compression_factor = self.sim_prop.remeshFactor
                        elif len(self.fracture.source) == 2:
                            index = self.fracture.source[0]
                            cent_point = self.fracture.mesh.CenterCoor[self.fracture.source[0]]

                            if self.fracture.mesh.CenterCoor[self.fracture.source[0]][0] == \
                                    self.fracture.mesh.CenterCoor[self.fracture.source[1]][0]:
                                elems_inter = int(abs(self.fracture.mesh.CenterCoor[self.fracture.source[0]][1] -
                                                      self.fracture.mesh.CenterCoor[self.fracture.source[1]][1]) / \
                                                  self.fracture.mesh.hy)
                                new_inter = int(np.ceil(elems_inter / self.sim_prop.remeshFactor))
                                compression_factor = elems_inter / new_inter

                            elif self.fracture.mesh.CenterCoor[self.fracture.source[0]][1] == \
                                    self.fracture.mesh.CenterCoor[self.fracture.source[1]][1]:
                                elems_inter = int(abs(self.fracture.mesh.CenterCoor[self.fracture.source[0]][0] -
                                                      self.fracture.mesh.CenterCoor[self.fracture.source[1]][0]) / \
                                                  self.fracture.mesh.hx)
                                new_inter = int(np.ceil(elems_inter / self.sim_prop.remeshFactor))
                                compression_factor = elems_inter / new_inter

                            else:
                                compression_factor = self.sim_prop.remeshFactor

                            log.info("The real reduction factor used is " + repr(compression_factor))

                        else:
                            index = self.fracture.mesh.locate_element(0., 0.)
                            cent_point = np.asarray([0., 0.])

                            compression_factor = self.sim_prop.remeshFactor

                        row = int(index / self.fracture.mesh.nx)
                        column = index - self.fracture.mesh.nx * row

                        row_frac = (self.fracture.mesh.ny - (row + 1)) / row
                        col_frac = column / (self.fracture.mesh.nx - (column + 1))

                        # We calculate the new dimension of the meshed area
                        new_limits = [[cent_point[0] - round((self.fracture.mesh.nx - 1) / (1 / col_frac + 1)) *
                                       self.fracture.mesh.hx * compression_factor,
                                       cent_point[0] + (self.fracture.mesh.nx - round((self.fracture.mesh.nx - 1) /
                                                                                      (1 / col_frac + 1)) - 1) *
                                       self.fracture.mesh.hx * compression_factor],
                                      [cent_point[1] - round((self.fracture.mesh.ny - 1) / (row_frac + 1)) *
                                       self.fracture.mesh.hy * compression_factor,
                                       cent_point[1] + (self.fracture.mesh.ny - round((self.fracture.mesh.ny - 1) /
                                                                                      (row_frac + 1)) - 1) *
                                       self.fracture.mesh.hy * compression_factor]]

                        # # We calculate the new dimension of the meshed area
                        # new_dimensions = 2 * self.sim_prop.remeshFactor * np.asarray([self.fracture.mesh.Lx,
                        #                                                           self.fracture.mesh.Ly])
                        # new_limits = [[(self.fracture.mesh.domainLimits[2]+self.fracture.mesh.domainLimits[3]) / 2
                        #                - new_dimensions[0]/2, (self.fracture.mesh.domainLimits[2] +
                        #                                        self.fracture.mesh.domainLimits[3]) / 2
                        #                + new_dimensions[0]/2],
                        #               [(self.fracture.mesh.domainLimits[0]+self.fracture.mesh.domainLimits[1]) / 2
                        #                - new_dimensions[1]/2, (self.fracture.mesh.domainLimits[0] +
                        #                                        self.fracture.mesh.domainLimits[1]) / 2
                        #                + new_dimensions[1]/2]]

                        elems = [self.fracture.mesh.nx, self.fracture.mesh.ny]

                        if len(np.intersect1d(self.fracture.mesh.CenterElts, index)) == 0:
                            compression_factor = 10

                        self.remesh(new_limits, elems, rem_factor=compression_factor)

                    log_only_to_logfile.info("\nRemeshed at " + repr(self.fracture.time))

                else:
                    log.info("Reached end of the domain. Exiting...")
                    break

            elif status == 14:
                # fracture fully closed
                self.output(Fr_n_pls1)
                if self.PstvInjJmp is None:
                    inp = input("Fracture is fully closed.\n\nDo you want to jump to"
                            " the time of next positive injection? [y/n]")
                    t0 = time.time()
                    while inp not in ['y', 'Y', 'n', 'N'] and time.time() - t0 < 600:
                        inp = input("Press y or n")

                    if inp == 'y' or inp == 'Y' or time.time() - t0 >= 600:
                        self.PstvInjJmp = True
                    else:
                        self.PstvInjJmp = False

                if self.PstvInjJmp:
                    self.sim_prop.solveDeltaP = False
                    # index of current time in the time series (first row) of the injection rate array
                    time_larger = np.where(Fr_n_pls1.time <= self.injection_prop.injectionRate[0, :])[0]
                    pos_inj = np.where(self.injection_prop.injectionRate[1, :] > 0)[0]
                    Qact = self.injection_prop.get_injection_rate(self.fracture.time, self.fracture)
                    after_time = np.intersect1d(time_larger, pos_inj)
                    if len(after_time) == 0 and max(Qact) == 0.:
                        log.warning("Positive injection not found!")
                        break
                    elif len(after_time) == 0:
                        jump_to = self.fracture.time + self.fracture.time * 0.1
                    else:
                        jump_to = min(self.injection_prop.injectionRate[0, np.intersect1d(time_larger, pos_inj)])
                    Fr_n_pls1.time = jump_to
                elif inp == 'n' or inp == 'N':
                    self.sim_prop.solveDeltaP = True
                self.fullyClosed = True
                self.fracture = copy.deepcopy(Fr_n_pls1)
            elif status == 17:
                # time step too big: you advanced more than one cell
                log.info("The fracture is advancing more than two cells in a row at time "+ repr(self.fracture.time))

                if self.TmStpReductions == self.sim_prop.maxReattemptsFracAdvMore2Cells:
                    log.warning("We can not reduce the time step more than that")
                    if self.sim_prop.collectPerfData:
                        if self.sim_prop.saveToDisk:
                            file_address = self.sim_prop.get_outputFolder() + "perf_data.dat"
                        else:
                            file_address = "./perf_data.dat"
                        with open(file_address, 'wb') as perf_output:
                            dill.dump(self.perfData, perf_output, -1)

                    log.info("\n\n---Simulation failed---")

                    raise SystemExit("Simulation failed.")
                else:
                    log.info("- limiting the time step - ")
                    # decrease time step pre-factor before taking the next fracture in the queue having last
                    # five time steps
                    if isinstance(self.sim_prop.tmStpPrefactor, np.ndarray):
                        indxCurTime = max(np.where(self.fracture.time >= self.sim_prop.tmStpPrefactor[0, :])[0])
                        self.sim_prop.tmStpPrefactor[1, indxCurTime] *= 0.5**self.TmStpReductions
                    else:
                        self.sim_prop.tmStpPrefactor *= 0.5**self.TmStpReductions
                    self.TmStpReductions += 1
            else:
                # time step failed
                log.warning("\n" + self.errorMessages[status])
                log.warning("\nTime step failed at = " + repr(self.fracture.time))
                # check if the queue with last 5 time steps is not empty, or max check points jumps done
                if self.fr_queue[self.successfulTimeSteps % 5] is None or \
                   self.chkPntReattmpts == 4:
                    if self.sim_prop.collectPerfData:
                        if self.sim_prop.saveToDisk:
                            file_address = self.sim_prop.get_outputFolder() + "perf_data.dat"
                        else:
                            file_address = "./perf_data.dat"
                        with open(file_address, 'wb') as perf_output:
                            dill.dump(self.perfData, perf_output, -1)

                    log.info("\n\n---Simulation failed---")

                    raise SystemExit("Simulation failed.")
                else:
                    # decrease time step pre-factor before taking the next fracture in the queue having last
                    # five time steps
                    if isinstance(self.sim_prop.tmStpPrefactor, np.ndarray):
                        indxCurTime = max(np.where(self.fracture.time >= self.sim_prop.tmStpPrefactor[0, :])[0])
                        self.sim_prop.tmStpPrefactor[1, indxCurTime] *= 0.8
                        current_PreFctr = self.sim_prop.tmStpPrefactor[1, indxCurTime]
                    else:
                        self.sim_prop.tmStpPrefactor *= 0.8
                        current_PreFctr = self.sim_prop.tmStpPrefactor

                    self.chkPntReattmpts += 1

                    # We need to re-update first the properties (in case meshing occurred in between!)
                    rollback_index = (self.successfulTimeSteps + self.chkPntReattmpts) % 5
                    rollback_fracture = self.fr_queue[rollback_index]
                    self.solid_prop.remesh(rollback_fracture.mesh)
                    self.injection_prop.remesh(rollback_fracture.mesh
                                               ,self.fracture.mesh)

                    self.fracture = copy.deepcopy(rollback_fracture)
                    if not self.c_was_provided:
                        # Rebuild the matrix on the restored mesh instead of
                        # restoring a potentially huge matrix snapshot.  This
                        # also repairs the exact mismatch that used to occur
                        # when a remesh happened between two checkpoints.
                        self.C = self._build_elasticity_matrix(self.fracture.mesh)
                    else:
                        signature = self.C_queue[rollback_index]
                        if signature is not None and np.asarray(self.C).ndim >= 2:
                            expected = int(signature[2])
                            if int(np.asarray(self.C).shape[0]) != expected:
                                raise RuntimeError(
                                    "provided elasticity matrix cannot be paired with "
                                    f"rollback mesh ({np.asarray(self.C).shape[0]} != {expected})"
                                )
                    log.warning("Time step have failed despite of reattempts with slightly smaller/bigger time steps...\n"
                                  "Going " + repr(5 - self.chkPntReattmpts) + " time steps back and re-attempting with the"
                                    " time step pre-factor of " + repr(current_PreFctr))

                    self.failedTimeSteps += 1

            self._write_progress(status=status)
            self.TmStpCount += 1

        print("\n")
        log.info("Final time = " + repr(self.fracture.time))
        log.info("-----Simulation finished------")
        log.info("number of time steps = " + repr(self.successfulTimeSteps))
        log.info("failed time steps = " + repr(self.failedTimeSteps))
        log.info("number of remeshings = " + repr(self.remeshings))
        log.info("front CFL limited steps = " + repr(self.frontCflLimitedSteps))
        log.info("front state repairs = " + repr(self.frontStateRepairs))
        log.info("non-monotonic state rejects = " + repr(self.nonMonotonicStateRejects))

        # Headless/native workers explicitly disable plotting.  Calling
        # ``plt.show`` unconditionally here can still enter a backend/event
        # loop after the physical time march has reached ``finalTime``;
        # isolated inversion workers then appear to finish in the heartbeat
        # but never return their result before the parent timeout.  Only run
        # plotting cleanup when a caller actually requested a figure.
        if self.sim_prop.plotFigure:
            plt.show(block=False)
            plt.close('all')

        if self.sim_prop.collectPerfData:
            file_address = self.sim_prop.get_outputFolder() + "perf_data.dat"
            os.makedirs(os.path.dirname(file_address), exist_ok=True)
            with open(file_address, 'wb') as output:
                dill.dump(self.perfData, output, -1)
        return True

    # ------------------------------------------------------------------------------------------------------------------

    def _apply_volume_balance_projection(self, fracture):
        """Project an accepted state onto the global fluid-volume ledger.

        The legacy viscous EHL system may lose global volume after a domain
        regrid even when each local nonlinear solve reports success.  This
        optional experiment applies a transparent global projection only
        after a successful step: target opening volume is injected volume
        minus cumulative leak-off, then the opening is scaled and pressure is
        recomputed from the elasticity matrix.  It is deliberately disabled
        by default and is recorded separately from an unmodified native run.
        """
        if not getattr(self.sim_prop, "enableVolumeBalanceProjection", False):
            return False
        if self.C is None:
            return False
        crack = np.asarray(getattr(fracture, "EltCrack", []), dtype=int).reshape(-1)
        if crack.size == 0:
            return False
        area = float(getattr(fracture.mesh, "EltArea", np.nan))
        if not np.isfinite(area) or area <= 0.0:
            return False
        try:
            injected = float(np.asarray(getattr(fracture, "injectedVol", np.nan)).reshape(-1)[0])
            leakoff = float(np.nansum(np.asarray(getattr(fracture, "LkOffTotal", 0.0), dtype=float)))
            current_volume = float(np.nansum(np.asarray(fracture.w, dtype=float)[crack]) * area)
        except (TypeError, ValueError):
            return False
        target_volume = injected - leakoff
        if not np.isfinite((injected, leakoff, current_volume, target_volume)).all():
            return False
        if target_volume <= 0.0 or current_volume <= 0.0:
            return False
        factor = target_volume / current_volume
        if not np.isfinite(factor) or factor <= 0.0:
            return False

        width = np.asarray(fracture.w, dtype=float).copy()
        width[crack] *= factor
        if not np.isfinite(width[crack]).all() or np.any(width[crack] < 0.0):
            return False
        matrix = np.asarray(self.C)
        if matrix.ndim != 2 or matrix.shape[0] <= int(np.max(crack)):
            return False
        try:
            net = matrix[np.ix_(crack, crack)].dot(width[crack])
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return False
        sigma = np.asarray(self.solid_prop.SigmaO, dtype=float)[crack]
        pressure = net + sigma
        if not np.isfinite(pressure).all():
            return False
        fracture.w = width
        fracture.pFluid = np.zeros_like(np.asarray(fracture.pFluid, dtype=float))
        fracture.pFluid[crack] = pressure
        fracture.pNet = np.zeros_like(np.asarray(fracture.pNet, dtype=float))
        fracture.pNet[crack] = net
        if hasattr(fracture, "wHist"):
            fracture.wHist = np.maximum(np.asarray(fracture.wHist, dtype=float), width)
        fracture.FractureVolume = float(target_volume)
        fracture.efficiency = float(target_volume / max(injected, 1.0e-12))
        self.volumeProjectionCount += 1
        self.volumeProjectionLastFactor = float(factor)
        self.volumeProjectionMaxFactor = max(float(self.volumeProjectionMaxFactor), float(factor))
        self.volumeProjectionMinFactor = min(float(self.volumeProjectionMinFactor), float(factor))
        return True

    def _build_elasticity_matrix(self, mesh):
        """Build an elasticity matrix whose indexing matches ``mesh``.

        This is used only when a failed step rolls back across a remesh.  The
        old controller restored the fracture object but left ``self.C`` on the
        newer mesh, so the next extension attempted to concatenate matrices
        with incompatible row counts.  Rebuilding on the restored mesh is
        slower than a pointer swap, but it is bounded to rollback events and
        avoids retaining several dense matrices in memory.
        """

        if self.solid_prop.TI_elasticity:
            matrix = load_TI_elasticity_matrix(mesh, self.solid_prop, self.sim_prop)
            if self.sim_prop.symmetric:
                return symmetric_elasticity_matrix_from_full(matrix, mesh)
            return matrix

        if self.sim_prop.symmetric:
            return load_isotropic_elasticity_matrix_symmetric(mesh, self.solid_prop.Eprime)
        if self.sim_prop.useBlockToeplizCompression:
            return load_isotropic_elasticity_matrix_toepliz(mesh, self.solid_prop.Eprime)
        return load_isotropic_elasticity_matrix(mesh, self.solid_prop.Eprime)

    def _write_progress(self, status=None, phase=None):
        """Atomically publish the last controller state for long runs.

        The parent process may terminate a slow candidate without being able
        to receive the Python object in memory.  This small heartbeat is
        deliberately metadata-only; it is not treated as a valid PyFrac
        result and it never replaces the acceptance check on final time.
        """
        if not self.progressFile:
            return
        try:
            path = os.path.abspath(self.progressFile)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = {
                "time_s": float(getattr(self.fracture, "time", float("nan"))),
                "final_time_s": float(getattr(self.sim_prop, "finalTime", float("nan"))),
                "status": None if status is None else int(status),
                "phase": phase or "time_marching",
                "native_attempt": int(self.nativeAttempt),
                "successful_time_steps": int(self.successfulTimeSteps),
                "failed_time_steps": int(self.failedTimeSteps),
                "controller_attempts": int(self.TmStpCount),
                "max_time_steps": int(getattr(self.sim_prop, "maxTimeSteps", -1)),
                "zero_injection_jumps": list(self.zeroInjectionJumps),
                "front_cfl_limited_steps": int(self.frontCflLimitedSteps),
                "front_state_repairs": int(self.frontStateRepairs),
                "non_monotonic_state_rejects": int(self.nonMonotonicStateRejects),
                "last_time_step_s": float(self.lastTimeStep),
                "last_accepted_delta_time_s": float(self.lastAcceptedDeltaTime),
                "last_attempt_status": self.lastAttemptStatus,
                "last_attempt_time_step_s": float(self.lastAttemptTimeStep),
                "last_attempt_failure_cause": self.lastAttemptFailureCause,
                "attempt_status_counts": {
                    str(key): int(value)
                    for key, value in self.attemptStatusCounts.items()
                },
                "time_step_diagnostics": self.lastTimeStepDiagnostics,
                "mesh_nx": int(getattr(self.fracture.mesh, "nx", 0)),
                "mesh_ny": int(getattr(self.fracture.mesh, "ny", 0)),
                "crack_cells": int(len(np.asarray(getattr(self.fracture, "EltCrack", [])))),
                "tip_cells": int(len(np.asarray(getattr(self.fracture, "EltTip", [])))),
                "volume_projection_count": int(self.volumeProjectionCount),
                "volume_projection_last_factor": float(self.volumeProjectionLastFactor),
                "volume_projection_max_factor": float(self.volumeProjectionMaxFactor),
                "volume_projection_min_factor": float(self.volumeProjectionMinFactor),
            }
            # Publish the conserved-volume ledger with the heartbeat.  This
            # is diagnostic metadata only, but it lets a long run identify
            # whether a mass-balance jump is introduced by the EHL step or by
            # the subsequent remesh/state transfer.  Do not repair values in
            # this reporting path: the acceptance gate must see the raw state.
            injected = float(np.asarray(getattr(self.fracture, "injectedVol", np.nan)).reshape(-1)[0])
            fracture_volume = float(getattr(self.fracture, "FractureVolume", np.nan))
            leakoff = float(np.nansum(np.asarray(getattr(self.fracture, "LkOffTotal", np.nan), dtype=float)))
            payload["volume_ledger"] = {
                "injected_volume_m3": injected,
                "fracture_volume_m3": fracture_volume,
                "leakoff_volume_m3": leakoff,
                "mass_balance_residual_m3": injected - fracture_volume - leakoff,
                "mesh_half_length_m": float(getattr(self.fracture.mesh, "Lx", np.nan)),
                "mesh_half_height_m": float(getattr(self.fracture.mesh, "Ly", np.nan)),
            }
            # Keep a compact, real field snapshot for the desktop playback.
            # This is diagnostic output only and never feeds values back into
            # the solver. Sampling bounds file size on long native runs.
            crack = np.asarray(getattr(self.fracture, "EltCrack", []), dtype=int)
            centers = np.asarray(getattr(self.fracture.mesh, "CenterCoor", []), dtype=float)
            if crack.size and centers.ndim == 2 and centers.shape[1] >= 2:
                valid = crack[(crack >= 0) & (crack < centers.shape[0])]
                if valid.size:
                    coordinates = centers[valid, :2]
                    payload["half_length_m"] = float(np.nanmax(np.abs(coordinates[:, 0])))
                    payload["fracture_height_m"] = float(np.nanmax(coordinates[:, 1]) - np.nanmin(coordinates[:, 1]))
                    widths = np.asarray(getattr(self.fracture, "w", []), dtype=float)
                    pressures = np.asarray(getattr(self.fracture, "pNet", []), dtype=float)
                    if widths.size > int(valid.max()):
                        payload["max_aperture_mm"] = float(np.nanmax(widths[valid]) * 1.0e3)
                    if pressures.size > int(valid.max()):
                        field_pressure = pressures[valid] / 1.0e6
                        payload["mean_net_pressure_mpa"] = float(np.nanmean(field_pressure))
                        payload["max_net_pressure_mpa"] = float(np.nanmax(field_pressure))
                    stride = max(1, int(np.ceil(valid.size / 180.0)))
                    sampled = valid[::stride]
                    payload["pressure_field"] = [
                        {
                            "x_m": float(centers[index, 0]),
                            "y_m": float(centers[index, 1]),
                            "net_pressure_mpa": float(pressures[index] / 1.0e6) if pressures.size > index else 0.0,
                            "width_mm": float(widths[index] * 1.0e3) if widths.size > index else 0.0,
                        }
                        for index in sampled
                    ]
                    tip = np.asarray(getattr(self.fracture, "EltTip", []), dtype=int)
                    tip = tip[(tip >= 0) & (tip < centers.shape[0])]
                    payload["front_geometry"] = centers[tip, :2].tolist() if tip.size else []
            temporary = path + ".tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(temporary, path)
            history_path = os.environ.get("PYFRAC_PROGRESS_HISTORY_FILE")
            if history_path:
                history_path = os.path.abspath(history_path)
                os.makedirs(os.path.dirname(history_path), exist_ok=True)
                with open(history_path, "a", encoding="utf-8") as history_handle:
                    history_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            # Progress reporting must never change the solver result.
            logging.getLogger("PyFrac.controller").debug(
                "could not write progress heartbeat: %s", exc
            )


#-----------------------------------------------------------------------------------------------------------------------

    def advance_time_step(self, Frac, C, timeStep, perfNode=None):
        """
        This function advances the fracture by the given time step. In case of failure, reattempts are made with smaller
        time steps.

        Arguments:
            Frac (Fracture object):         -- fracture object from the last time step
            C (ndarray-float):              -- the elasticity matrix
            timeStep (float):               -- time step to be attempted
            perfNode (IterationProperties)  -- An IterationProperties instance to store performance data

        Return:
            - exitstatus (int)        -- see documentation for possible values.
            - Fr (Fracture)           -- fracture after advancing time step.
        """
        log = logging.getLogger('PyFrac.controller.advance_time_step')
        # ``maxReattempts`` is the number of *extra* retries in the original
        # code, but the surrounding controller always expects at least one
        # actual attempt.  ``range(0, 0)`` used to leave ``status`` and ``Fr``
        # uninitialised when a caller deliberately disabled retries, causing
        # an unrelated ``UnboundLocalError`` before the real solver failure
        # could be reported.  Keep zero as “no retry” while still performing
        # the initial attempt.
        attempt_count = max(int(getattr(self.sim_prop, "maxReattempts", 0)), 0) + 1
        for i in range(attempt_count):
            # Always retry with a smaller step.  The upstream controller
            # switched to *larger* steps in the second half of this loop,
            # which is counterproductive for an EHL non-convergence: after a
            # failed small-step retry it could immediately re-enter the same
            # unstable regime.  Monotone reduction makes rollback reproducible
            # and gives the nonlinear solve a genuine chance to recover.
            tmStp_to_attempt = timeStep * self.sim_prop.reAttemptFactor ** i

            # check for final time
            if Frac.time + tmStp_to_attempt > 1.01 * self.sim_prop.finalTime:
                log.info(repr(Frac.time + tmStp_to_attempt))
                return status, Fr
            print('\n')
            log.info('Evaluating solution at time = ' + repr(Frac.time+tmStp_to_attempt) + " ...")
            log.debug("Attempting time step of " + repr(tmStp_to_attempt) + " sec...")

            perfNode_TmStpAtmpt = instrument_start('time step attempt', perfNode)

            self.attmptedTimeStep = tmStp_to_attempt
            status, Fr = attempt_time_step(Frac,
                                            C,
                                            self.solid_prop,
                                            self.fluid_prop,
                                            self.sim_prop,
                                            self.injection_prop,
                                            tmStp_to_attempt,
                                            perfNode_TmStpAtmpt)

            # Publish the inner status immediately.  A long native run may
            # be stopped by the outer wall-clock budget before Controller.run
            # returns, so the final result alone is not enough to diagnose
            # where the solver stopped.
            self.lastAttemptStatus = int(status)
            self.lastAttemptTimeStep = float(tmStp_to_attempt)
            self.lastAttemptFailureCause = (
                None if status == 1 else self.errorMessages[status]
            )
            self.attemptStatusCounts[status] = self.attemptStatusCounts.get(status, 0) + 1
            self._write_progress(status=status, phase="time_step_attempt")

            if perfNode_TmStpAtmpt is not None:
                instrument_close(perfNode, perfNode_TmStpAtmpt,
                                 None, len(Frac.EltCrack), status == 1,
                                 self.errorMessages[status], Frac.time)
                perfNode.attempts_data.append(perfNode_TmStpAtmpt)

            if status in [1, 12, 14, 16, 17]:
                break
            else:
                log.warning(self.errorMessages[status])
                log.warning("Time step failed...")


        return status, Fr

#-----------------------------------------------------------------------------------------------------------------------

    def output(self, Fr_advanced):
        """
        This function plot the fracture footprint and/or save file to disk according to the parameters set in the
        simulation properties. See documentation of SimulationProperties class to get the details of parameters which
        determines when and how the output is made.

        Arguments:
            Fr_advanced (Fracture object):       -- fracture after time step is advanced.

        """
        log = logging.getLogger('Pyfrac.output')
        in_req_TSrs = False
        # current time in the time series given at which the solution is to be evaluated
        if self.sim_prop.get_solTimeSeries() is not None and  self.sim_prop.plotATsolTimeSeries :
            if Fr_advanced.time in self.sim_prop.get_solTimeSeries():
                in_req_TSrs = True

        # if the time is final time
        if Fr_advanced.time >= self.sim_prop.finalTime:
            in_req_TSrs = True

        if self.sim_prop.saveToDisk:

            save_TP_exceeded = False
            save_TS_exceeded = False

            # check if save time period is exceeded since last save
            if self.sim_prop.saveTimePeriod is not None:
                if Fr_advanced.time >= self.lastSavedTime + self.sim_prop.saveTimePeriod:
                    save_TP_exceeded = True

            # check if the number of time steps since last save exceeded
            if self.sim_prop.saveTSJump is not None:
                if self.successfulTimeSteps % self.sim_prop.saveTSJump == 0:
                    save_TS_exceeded = True

            if save_TP_exceeded or in_req_TSrs or save_TS_exceeded:

                # save fracture to disk
                log.info("Saving solution at " + repr(Fr_advanced.time) + "...")
                Fr_advanced.SaveFracture(self.sim_prop.get_outputFolder() +
                                         self.sim_prop.get_simulation_name() +
                                         '_file_' + repr(self.lastSavedFile))
                self.lastSavedFile += 1
                log.info("Done! ")

                self.lastSavedTime = Fr_advanced.time

        # plot fracture variables
        if self.sim_prop.plotFigure:

            plot_TP_exceeded = False
            plot_TS_exceeded = False

            # check if plot time period is exceeded since last plot
            if self.sim_prop.plotTimePeriod is not None:
                if Fr_advanced.time >= self.lastPlotTime + self.sim_prop.plotTimePeriod:
                    plot_TP_exceeded = True

            # check if the number of time steps since last plot exceeded
            if self.sim_prop.plotTSJump is not None:
                if self.successfulTimeSteps % self.sim_prop.plotTSJump == 0:
                    plot_TS_exceeded = True

            if plot_TP_exceeded or in_req_TSrs or plot_TS_exceeded:

                for index, plt_var in enumerate(self.sim_prop.plotVar):
                    log.info("Plotting solution at " + repr(Fr_advanced.time) + "...")
                    plot_prop = PlotProperties()

                    if self.Figures[index]:
                        axes = self.Figures[index].get_axes()   # save axes from last figure
                        plt.figure(self.Figures[index].number)
                        plt.clf()                              # clear figure
                        self.Figures[index].add_axes(axes[0])   # add axis to the figure

                    if plt_var == 'footprint':
                        # footprint is plotted if variable to plot is not given
                        plot_prop.lineColor = 'b'
                        if self.sim_prop.plotAnalytical:
                            self.Figures[index] = plot_footprint_analytical(self.sim_prop.analyticalSol,
                                                                       self.solid_prop,
                                                                       self.injection_prop,
                                                                       self.fluid_prop,
                                                                       [Fr_advanced.time],
                                                                       fig=self.Figures[index],
                                                                       h=self.sim_prop.height,
                                                                       samp_cell=None,
                                                                       plot_prop=plot_prop,
                                                                       gamma=self.sim_prop.aspectRatio,
                                                                       inj_point=self.injection_prop.sourceCoordinates)

                        self.Figures[index] = Fr_advanced.plot_fracture(variable='mesh',
                                                                       mat_properties=self.solid_prop,
                                                                       projection='2D',
                                                                       backGround_param=self.sim_prop.bckColor,
                                                                       fig=self.Figures[index],
                                                                       plot_prop=plot_prop)

                        plot_prop.lineColor = 'k'
                        self.Figures[index] = Fr_advanced.plot_fracture(variable='footprint',
                                                                       projection='2D',
                                                                       fig=self.Figures[index],
                                                                       plot_prop=plot_prop)

                    elif plt_var in ('fluid velocity as vector field','fvvf','fluid flux as vector field','ffvf'):
                        if self.fluid_prop.viscosity == 0. :
                            raise SystemExit('ERROR: if the fluid viscosity is equal to 0 does not make sense to ask a plot of the fluid velocity or fluid flux')
                        elif self.sim_prop._SimulationProperties__tipAsymptote == 'K':
                            raise SystemExit('ERROR: if tipAsymptote == K, does not make sense to ask a plot of the fluid velocity or fluid flux')
                        self.Figures[index] = Fr_advanced.plot_fracture(variable='mesh',
                                                                       mat_properties=self.solid_prop,
                                                                       projection='2D',
                                                                       backGround_param=self.sim_prop.bckColor,
                                                                       fig=self.Figures[index],
                                                                       plot_prop=plot_prop)

                        plot_prop.lineColor = 'k'
                        self.Figures[index] = Fr_advanced.plot_fracture(variable='footprint',
                                                                       projection='2D',
                                                                       fig=self.Figures[index],
                                                                       plot_prop=plot_prop)

                        self.Figures[index] = Fr_advanced.plot_fracture(variable=plt_var,
                                                                       projection='2D_vectorfield',
                                                                       mat_properties=self.solid_prop,
                                                                       fig=self.Figures[index])
                    else:
                        if self.sim_prop.plotAnalytical:
                            proj = supported_projections[plt_var][0]
                            self.Figures[index] = plot_analytical_solution(regime=self.sim_prop.analyticalSol,
                                                                      variable=plt_var,
                                                                      mat_prop=self.solid_prop,
                                                                      inj_prop=self.injection_prop,
                                                                      fluid_prop=self.fluid_prop,
                                                                      projection=proj,
                                                                      time_srs=[Fr_advanced.time],
                                                                      h=self.sim_prop.height,
                                                                      gamma=self.sim_prop.aspectRatio)

                        fig_labels = LabelProperties(plt_var, 'whole mesh', '2D')
                        fig_labels.figLabel = ''
                        self.Figures[index] = Fr_advanced.plot_fracture(variable='footprint',
                                                                       projection='2D',
                                                                       fig=self.Figures[index],
                                                                       labels=fig_labels)

                        self.Figures[index] = Fr_advanced.plot_fracture(variable=plt_var,
                                                                       projection='2D_clrmap',
                                                                       mat_properties=self.solid_prop,
                                                                       fig=self.Figures[index],
                                                                       elements=get_elements(suitable_elements[plt_var], Fr_advanced))
                        # plotting source elements
                        self.Figures[index] = plot_injection_source(Fr_advanced,
                                              fig=self.Figures[index])

                    # plotting closed cells
                    if len(Fr_advanced.closed) > 0:
                        plot_prop.lineColor = 'orangered'
                        self.Figures[index] = Fr_advanced.mesh.identify_elements(Fr_advanced.closed,
                                                                                fig=self.Figures[index],
                                                                                plot_prop=plot_prop,
                                                                                plot_mesh=False,
                                                                                print_number=False)
                    plt.ion()
                    plt.pause(0.4)
                    
                # set figure position
                if self.setFigPos:
                    for i in range(len(self.sim_prop.plotVar)):
                        plt.figure(i + 1)
                        mngr = plt.get_current_fig_manager()
                        x_offset = 650 * i
                        y_ofset = 50
                        if i >= 3:
                            x_offset = (i - 3) * 650
                            y_ofset = 500
                        try:
                            mngr.window.setGeometry(x_offset, y_ofset, 640, 545)
                        except AttributeError:
                            pass
                    self.setFigPos = False

                # plot the figure
                log.info("Done! ")
                if self.sim_prop.blockFigure:
                    print("click on the window to continue...")
                    plt.waitforbuttonpress()

                self.lastPlotTime = Fr_advanced.time


#------------------------------------------------------------------------------------------------------------------

    def get_time_step(self):
        """
        This function calculates the appropriate time step. It takes minimum of the time steps evaluated according to
        the following:

            - time step evaluated with the current front velocity to limit the increase in length compared to a cell \
                length
            - time step evaluated with the current front velocity to limit the increase in length compared to the \
                current fracture length
            - time step evaluated with the injection rate in the coming time step
            - time step evaluated to limit the change in total volume of the fracture
        In addition, the limit on the time step and the times at which the solution is required are also taken in
        account to get the appropriate time step.

        Returns:
            - time_step (float)   -- the appropriate time step.

        """
        log = logging.getLogger('PyFrac.get_time_step')
        time_step_given = False
        TS_cell_length = np.inf
        TS_fracture_length = np.inf
        TS_inj_cell = np.inf
        TS_delta_vol = np.inf
        if self.sim_prop.fixedTmStp is not None:
            # fixed time step
            if isinstance(self.sim_prop.fixedTmStp, float) or isinstance(self.sim_prop.fixedTmStp, int):
                time_step = self.sim_prop.fixedTmStp
                time_step_given = True
            elif isinstance(self.sim_prop.fixedTmStp, np.ndarray) and self.sim_prop.fixedTmStp.shape[0] == 2:
                # fixed time step list is given
                times_past = np.where(self.fracture.time >= self.sim_prop.fixedTmStp[0, :])[0]
                if len(times_past) > 0:
                    indxCurTime = max(times_past)
                    if self.sim_prop.fixedTmStp[1, indxCurTime] is not None:
                        # time step is not given as None.
                        time_step = self.sim_prop.fixedTmStp[1, indxCurTime]  # current time step
                        time_step_given = True
                    else:
                        time_step_given = False
                else:
                    # time step is given as None. In this case time step will be evaluated with current state
                    time_step_given = False
            else:
                raise ValueError("Fixed time step can be a float or an ndarray with two rows giving the time and"
                                 " corresponding time steps.")

        if not time_step_given:
            delta_x = min(self.fracture.mesh.hx, self.fracture.mesh.hy)
            velocity = np.asarray(self.fracture.v, dtype=float).reshape(-1)
            if np.any(~np.isfinite(velocity)):
                log.warning("non-finite front velocities are present; ignoring them for time-step estimation")
            non_zero_v = np.where(np.isfinite(velocity) & (velocity > 0.0))[0]
            # time step is calculated with the current propagation velocity
            if len(non_zero_v) > 0:
                if len(self.injection_prop.sourceElem) < 4:
                    # if point source
                    tipVrtxCoord = self.fracture.mesh.VertexCoor[self.fracture.mesh.Connectivity[self.fracture.EltTip,
                                                                                             self.fracture.ZeroVertex]]
                    # the distance of tip from the injection point in each of the tip cell
                    dist_Inj_pnt = ((tipVrtxCoord[:, 0] - self.injection_prop.sourceCoordinates[0]) ** 2 +
                                    (tipVrtxCoord[:, 1] - self.injection_prop.sourceCoordinates[1]) ** 2) ** 0.5 \
                                   + self.fracture.l

                    # The legacy continuous-front update can expose a new
                    # velocity entry for one internal iteration before the
                    # corresponding tip-distance metadata is rebuilt.  Keep
                    # the time-step estimate conservative, but do not index
                    # a shorter distance array with the longer velocity list.
                    valid_non_zero_v = non_zero_v[non_zero_v < dist_Inj_pnt.size]
                    if valid_non_zero_v.size > 0:
                        length_fraction = max(
                            float(getattr(self.sim_prop, "fractureLengthFraction", 0.2)),
                            0.0,
                        )
                        TS_fracture_length = min(
                            abs(
                                length_fraction
                                * dist_Inj_pnt[valid_non_zero_v]
                                / velocity[valid_non_zero_v]
                            )
                        ) if length_fraction > 0.0 else np.inf
                    else:
                        TS_fracture_length = np.inf
                else:
                    TS_fracture_length = np.inf

                # the time step evaluated by restricting the fraction of the cell that would be traversed in the time
                # step. e.g., if the pre-factor is 0.5, the tip in the cell with the largest velocity will progress half
                # of the cell width in either x or y direction depending on which is smaller.
                cell_traversal_fraction = max(
                    float(getattr(self.sim_prop, "cellTraversalFraction", 1.0)),
                    1.0e-6,
                )
                TS_cell_length = (
                    cell_traversal_fraction
                    * delta_x
                    / np.max(velocity[non_zero_v])
                )

            else:
                TS_cell_length = np.inf
                TS_fracture_length = np.inf

            # index of current time in the time series (first row) of the injection rate array
            indx_cur_time = max(np.where(self.fracture.time >= self.injection_prop.injectionRate[0, :])[0])
            current_rate = self.injection_prop.injectionRate[1, indx_cur_time]  # current injection rate
            if current_rate < 0:
                vel_injection = current_rate / (2 * (self.fracture.mesh.hx + self.fracture.mesh.hy) *
                                    self.fracture.w[self.fracture.mesh.CenterElts])
                TS_inj_cell = 10 * delta_x / abs(vel_injection[0])
            elif current_rate > 0:
                # for positive injection, use the increase in total fracture volume criteria
                injection_fraction = max(
                    float(getattr(self.sim_prop, "injectionVolumeStepFraction", 0.10)),
                    1.0e-6,
                )
                TS_inj_cell = injection_fraction * sum(self.fracture.w) * self.fracture.mesh.EltArea / current_rate
            else:
                TS_inj_cell = np.inf

            TS_delta_vol = np.inf
            if self.delta_w is not None:
                delta_vol = sum(self.delta_w) / sum(self.fracture.w)
                if abs(delta_vol) <= 1.0e-12:
                    TS_delta_vol = np.inf
                elif delta_vol < 0:
                    volume_change_fraction = max(
                        float(getattr(self.sim_prop, "negativeVolumeChangeStepFraction", 0.05)),
                        1.0e-6,
                    )
                    TS_delta_vol = self.lstTmStp / abs(delta_vol) * volume_change_fraction
                else:
                    volume_change_fraction = max(
                        float(getattr(self.sim_prop, "positiveVolumeChangeStepFraction", 0.12)),
                        1.0e-6,
                    )
                    TS_delta_vol = self.lstTmStp / abs(delta_vol) * volume_change_fraction

            # getting pre-factor for current time
            current_prefactor = self.sim_prop.get_time_step_prefactor(self.fracture.time)
            time_step = current_prefactor * min(TS_cell_length,
                                              TS_fracture_length,
                                              TS_inj_cell,
                                              TS_delta_vol)

            # limit time step to be max 2 * last time step
            if (self.lstTmStp != None and not np.isinf(time_step)) and time_step > 2 * self.lstTmStp:
                time_step = 2 * self.lstTmStp

            # limit the time step to be at max 15% of the actual time
            time_fraction = max(
                float(getattr(self.sim_prop, "timeStepTimeFraction", 0.15)),
                1.0e-6,
            )
            if time_step > time_fraction * self.fracture.time:
                time_step = time_fraction * self.fracture.time

        # in case of fracture not propagating
        if time_step <= 0 or np.isinf(time_step):
            if self.stagnant_TS is not None:
                time_step = self.stagnant_TS
                self.stagnant_TS = time_step * 1.2
            else:
                TS_obtained = False
                log.warning("The fracture front is stagnant and there is no injection. In these conditions, "
                            "there is no criterion to calculate time step size.")
                while not TS_obtained:
                    try:
                        inp = input("Enter the time step size(seconds) you would like to try:")
                        time_step = float(inp)
                        TS_obtained = True
                    except ValueError:
                        pass

        # to get the solution at the times given in time series, any change in parameters or final time
        next_in_TS = self.sim_prop.finalTime

        if self.timeToHit is not None:
            larger_in_TS = np.where(self.timeToHit > self.fracture.time)[0]
            if len(larger_in_TS) > 0:
                next_in_TS = np.min(self.timeToHit[larger_in_TS])

        if next_in_TS < self.fracture.time:
            raise SystemExit('The minimum time required in the given time series or the end time'
                             ' is less than initial time.')

        # check if time step would step over the next time in required time series
        if self.fracture.time + time_step > next_in_TS:
            time_step = next_in_TS - self.fracture.time
        # check if the current time is very close the next time to hit. If yes, set it to the next time to avoid
        # very small time step in the next time step advance.
        elif next_in_TS - self.fracture.time < 1.05 * time_step:
            time_step = next_in_TS - self.fracture.time

        # Do not let a native PDE step straddle a positive/zero injection
        # regime transition.  The legacy controller can otherwise present a
        # single nonlinear solve with both an open and a shut-in source,
        # which is the recurrent trigger for front reconstruction failures on
        # long runs.  Only regime changes are clipped; ordinary measured rate
        # changes remain eligible for the existing larger time steps.
        if getattr(self.sim_prop, "clipToInjectionRegimeEvents", False):
            rate_times = np.asarray(self.injection_prop.injectionRate[0, :], dtype=float)
            rate_values = np.asarray(self.injection_prop.injectionRate[1, :], dtype=float)
            finite = np.isfinite(rate_times) & np.isfinite(rate_values)
            rate_times = rate_times[finite]
            rate_values = rate_values[finite]
            if rate_times.size > 1:
                positive = rate_values > 1.0e-12
                transition_indices = np.flatnonzero(positive[1:] != positive[:-1]) + 1
                event_tolerance = max(
                    1.0e-8,
                    1.0e-10 * max(abs(float(self.sim_prop.finalTime)), 1.0),
                )
                future = transition_indices[
                    rate_times[transition_indices] > self.fracture.time + event_tolerance
                ]
                if future.size:
                    next_regime_event = float(rate_times[future[0]])
                    if self.fracture.time + time_step > next_regime_event:
                        time_step = next_regime_event - self.fracture.time

        # checking if the time step is above the limit
        if self.sim_prop.timeStepLimit is not None and time_step > self.sim_prop.timeStepLimit:
            log.warning("Evaluated/given time step is more than the time step limit! Limiting time step...")
            time_step = self.sim_prop.timeStepLimit

        self.lastTimeStepDiagnostics = {
            "selected_s": float(time_step),
            "cell_length_s": float(TS_cell_length),
            "fracture_length_s": float(TS_fracture_length),
            "injection_volume_s": float(TS_inj_cell),
            "volume_change_s": float(TS_delta_vol),
            "injection_volume_fraction": float(
                getattr(self.sim_prop, "injectionVolumeStepFraction", 0.10)
            ),
            "positive_volume_change_fraction": float(
                getattr(self.sim_prop, "positiveVolumeChangeStepFraction", 0.12)
            ),
            "negative_volume_change_fraction": float(
                getattr(self.sim_prop, "negativeVolumeChangeStepFraction", 0.05)
            ),
            "cell_traversal_fraction": float(
                getattr(self.sim_prop, "cellTraversalFraction", 1.0)
            ),
            "time_step_limit_s": (
                None if self.sim_prop.timeStepLimit is None
                else float(self.sim_prop.timeStepLimit)
            ),
            "time_step_given": bool(time_step_given),
        }

        return time_step

# ------------------------------------------------------------------------------------------------------------------

    def remesh(self, new_limits, elems, direction=None, rem_factor=10):
        log = logging.getLogger('PyFrac.remesh')
        # Generating the new mesh (with new limits but same number of elements)
        coarse_mesh = CartesianMesh(new_limits[0],
                                    new_limits[1],
                                    elems[0],
                                    elems[1],
                                    symmetric=self.sim_prop.symmetric)

        # Finalizing the transfer of information from the fine to the coarse mesh
        self.solid_prop.remesh(coarse_mesh)
        self.injection_prop.remesh(coarse_mesh, self.fracture.mesh)

        # We adapt the elasticity matrix
        if not self.sim_prop.useBlockToeplizCompression:
            if direction is None:
                # The project-level fixed-cell path expands only the
                # horizontal domain.  The legacy ``C *= 1/rem_factor`` shortcut
                # is valid only for an isotropic scaling of every coordinate;
                # applying it here changes the pressure/width relation and
                # can produce a large nonphysical pressure error.  Rebuild C
                # on the new mesh whenever the domain is expanded anisotropically.
                fixed_domain_regrid = bool(
                    getattr(self.sim_prop, "expandDomainOnBoundary", False)
                )
                if rem_factor == self.sim_prop.remeshFactor and not fixed_domain_regrid:
                    self.C *= 1 / self.sim_prop.remeshFactor
                else:
                    if not self.sim_prop.symmetric:
                        self.C = load_isotropic_elasticity_matrix(coarse_mesh, self.solid_prop.Eprime)
                    else:
                        self.C = load_isotropic_elasticity_matrix_symmetric(coarse_mesh, self.solid_prop.Eprime)
                #rem_factor = self.sim_prop.remeshFactor
                #self.C *= 1 / rem_factor
            elif direction == 'reduce':
                #rem_factor = 10
                if not self.sim_prop.symmetric:
                    self.C = load_isotropic_elasticity_matrix(coarse_mesh, self.solid_prop.Eprime)
                else:
                    self.C = load_isotropic_elasticity_matrix_symmetric(coarse_mesh, self.solid_prop.Eprime)
            else:
                #rem_factor = 10
                log.info("Extending the elasticity matrix...")
                self.extend_isotropic_elasticity_matrix(coarse_mesh, direction=direction)
        else:
            # if direction is None:
            #     rem_factor = self.sim_prop.remeshFactor
            # else:
            #     rem_factor = 10
            self.C.reload(coarse_mesh)

        self.fracture = self.fracture.remesh(rem_factor,
                                             self.C,
                                             coarse_mesh,
                                             self.solid_prop,
                                             self.fluid_prop,
                                             self.injection_prop,
                                             self.sim_prop,
                                             direction)

        self.fracture.mesh = coarse_mesh

        # update the saved properties
        if self.sim_prop.saveToDisk:
            if os.path.exists(self.sim_prop.get_outputFolder() + "properties"):
                os.remove(self.sim_prop.get_outputFolder() + "properties")
            prop = (self.solid_prop, self.fluid_prop, self.injection_prop, self.sim_prop)
            with open(self.sim_prop.get_outputFolder() + "properties", 'wb') as output:
                dill.dump(prop, output, -1)
        self.remeshings += 1

        log.info("Done!")

# -----------------------------------------------------------------------------------------------------------------------

    def extend_isotropic_elasticity_matrix(self, new_mesh, direction=None):
        """
        In the case of extension of the mesh we don't need to recalculate the entire elasticity matrix. All we need to do is
        to map all the elements to their new index and calculate what lasts

        Arguments:
            new_mesh (object CartesianMesh):    -- a mesh object describing the domain.
        """

        a = new_mesh.hx / 2.
        b = new_mesh.hy / 2.
        Ne = new_mesh.NumberOfElts
        Ne_old = self.fracture.mesh.NumberOfElts

        new_indexes = np.array(mapping_old_indexes(new_mesh, self.fracture.mesh, direction))

        if len(self.C) != Ne and not self.sim_prop.symmetric:
            self.C = np.vstack((np.hstack((self.C, np.full((Ne_old, Ne - Ne_old), 0.))),
               np.full((Ne - Ne_old, Ne), 0.)))

            self.C[np.ix_(new_indexes, new_indexes)] = self.C[np.ix_(np.arange(Ne_old), np.arange(Ne_old))]

            add_el = np.setdiff1d(np.arange(Ne), new_indexes)

            for i in add_el:
                x = new_mesh.CenterCoor[i, 0] - new_mesh.CenterCoor[:, 0]
                y = new_mesh.CenterCoor[i, 1] - new_mesh.CenterCoor[:, 1]

                self.C[i] = (self.solid_prop.Eprime / (8. * np.pi)) * (
                        np.sqrt(np.square(a - x) + np.square(b - y)) / ((a - x) * (b - y)) + np.sqrt(
                    np.square(a + x) + np.square(b - y)
                ) / ((a + x) * (b - y)) + np.sqrt(np.square(a - x) + np.square(b + y)) / ((a - x) * (b + y)) + np.sqrt(
                    np.square(a + x) + np.square(b + y)) / ((a + x) * (b + y)))

            self.C[np.ix_(new_indexes, add_el)] = np.transpose(self.C[np.ix_(add_el, new_indexes)])

        elif not self.sim_prop.symmetric:
            self.C = load_isotropic_elasticity_matrix(new_mesh, self.solid_prop.Eprime)
        else:
            self.C = load_isotropic_elasticity_matrix_symmetric(new_mesh, self.solid_prop.Eprime)
