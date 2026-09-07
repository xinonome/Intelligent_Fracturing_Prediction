"""Adapter around the vendored PyFrac source.

PyFrac is an external GPLv3 simulator.  This module loads its documented
classes and converts their output to the project's common fracture-state
schema.  The vendored 1.1.1 controller contains one audited final-time loop
compatibility patch recorded in ``third_party/PyFrac/IMPLEMENTATION_LOG.md``.

Two modes are exposed:

``snapshot``
    Build a PyFrac Cartesian mesh and initialize a PKN fracture at the
    requested time.  This is quick and useful for model-space comparison and
    teacher-data prototyping, but it is *not* a time-marching PyFrac run.

``native``
    Start from the explicit native initial condition at ``initial_time_s``
    (the formal runner uses 1 s) and let ``Controller.run`` advance the
    fracture. It is the high-fidelity reference path and can be slow or
    numerically sensitive on old PyFrac versions. The adapter checks that the
    final time actually reaches the requested target; failures are returned
    with diagnostic metadata instead of being silently relabelled as PyFrac
    output. Older callers may still request a compatibility warm start by
    setting ``native_start_time_s`` above ``initial_time_s``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import gc
import os
import sys
import time
import traceback
import builtins
import traceback
from typing import Any

import numpy as np

from .pyfrac_config import PyFracConfig
from .pyfrac_robustness import CheckpointManager, MeshDecision, choose_mesh
from .fracture_length_models import LengthForwardModel


def target_time_reached(final_time_s: float, target_time_s: float) -> bool:
    """Use a numerical tolerance, never a percentage-of-duration shortcut."""

    tolerance_s = max(1.0e-6, 1.0e-6 * max(abs(float(target_time_s)), 1.0))
    return float(final_time_s) >= float(target_time_s) - tolerance_s


def _release_native_controller(controller: Any | None) -> None:
    """Release large transient arrays retained by the legacy controller."""

    if controller is None:
        return
    # Each controller owns a dense elasticity matrix and a five-state
    # fracture queue. Continuation creates a controller per window, so make
    # their lifetime explicit instead of waiting for cyclic GC.
    for name, value in (
        ("C", None),
        ("fr_queue", [None, None, None, None, None]),
        ("perfData", []),
        ("Figures", []),
        ("fracture", None),
        ("solid_prop", None),
        ("fluid_prop", None),
        ("injection_prop", None),
        ("sim_prop", None),
    ):
        try:
            setattr(controller, name, value)
        except Exception:
            pass
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass
    gc.collect()


def native_state_health_errors(fracture: Any, previous_fracture: Any | None = None) -> list[str]:
    """Return reasons why a continuation state must not become a checkpoint.

    The legacy solver can stop at its per-controller step budget after its
    last accepted internal state has already developed an inconsistent front
    (for example a channel cell with no arrival time).  Treating that object
    as successful partial progress makes every later retry fail at the exact
    same model time.  This gate is intentionally conservative for the long
    inversion experiment.
    """

    errors: list[str] = []
    if not hasattr(fracture, "EltCrack"):
        return errors
    crack = np.asarray(fracture.EltCrack, dtype=int)
    if crack.size == 0:
        errors.append("empty fracture footprint")
        return errors

    def _indexed_finite(array_name: str, index_name: str) -> None:
        if not hasattr(fracture, array_name) or not hasattr(fracture, index_name):
            return
        values = np.asarray(getattr(fracture, array_name))
        indices = np.asarray(getattr(fracture, index_name), dtype=int)
        if indices.size and (indices.min() < 0 or indices.max() >= values.size):
            errors.append(f"{index_name} outside {array_name}")
        elif indices.size and not np.isfinite(values[indices]).all():
            errors.append(f"non-finite {array_name} on {index_name}")

    _indexed_finite("Tarrival", "EltChannel")
    _indexed_finite("TarrvlZrVrtx", "EltTip")
    _indexed_finite("TarrvlZrVrtx", "EltRibbon")
    tip_count = len(np.asarray(getattr(fracture, "EltTip", [])))
    for name in ("l", "alpha", "v", "FillF", "ZeroVertex", "Ffront"):
        if hasattr(fracture, name) and len(np.asarray(getattr(fracture, name))) != tip_count:
            errors.append(
                f"tip metadata size mismatch: {name}={len(np.asarray(getattr(fracture, name)))} EltTip={tip_count}"
            )
    for name in ("w", "pNet", "pFluid"):
        if hasattr(fracture, name):
            values = np.asarray(getattr(fracture, name))
            if crack.max(initial=-1) >= values.size or not np.isfinite(values[crack]).all():
                errors.append(f"non-finite {name} on EltCrack")
    if hasattr(fracture, "pNet"):
        net_pressure = np.asarray(fracture.pNet, dtype=float)[crack]
        # The legacy solver can produce a few MPa of cell-level variation near
        # the front during remeshing.  A pressure magnitude above 1000 MPa is
        # a numerical blow-up for this engineering case and must be rolled
        # back instead of being recorded as a successful continuation.
        if net_pressure.size and np.nanmax(np.abs(net_pressure)) > 1.0e9:
            errors.append("extreme net pressure on EltCrack")
    if hasattr(fracture, "w") and np.any(np.asarray(fracture.w)[crack] < -1.0e-12):
        errors.append("negative fracture width")
    if hasattr(fracture, "v"):
        velocity = np.asarray(fracture.v, dtype=float)
        if velocity.size and (not np.isfinite(velocity).all() or np.nanmin(velocity) < -1.0e-10):
            errors.append("negative or non-finite front velocity")
    if previous_fracture is not None and hasattr(previous_fracture, "EltCrack"):
        previous_count = len(np.asarray(previous_fracture.EltCrack))
        if previous_count > 0 and crack.size < 0.8 * previous_count:
            errors.append(f"fracture footprint collapsed from {previous_count} to {crack.size} cells")
    return errors


def repair_native_front_metadata(fracture: Any) -> list[str]:
    """Repair only legacy front bookkeeping; never alter pressure/width/volume."""

    repairs: list[str] = []
    current_time = float(getattr(fracture, "time", 0.0))
    for array_name, index_name in (
        ("Tarrival", "EltChannel"),
        ("TarrvlZrVrtx", "EltTip"),
        ("TarrvlZrVrtx", "EltRibbon"),
    ):
        if not hasattr(fracture, array_name) or not hasattr(fracture, index_name):
            continue
        values = np.asarray(getattr(fracture, array_name))
        indices = np.asarray(getattr(fracture, index_name), dtype=int)
        if not indices.size:
            continue
        missing = indices[~np.isfinite(values[indices])]
        if missing.size:
            values[missing] = current_time
            repairs.append(f"filled {missing.size} {array_name} values on {index_name}")
    tip_count = len(np.asarray(getattr(fracture, "EltTip", [])))
    for name in ("l", "alpha", "v", "FillF", "ZeroVertex", "Ffront"):
        if not hasattr(fracture, name):
            continue
        original = np.asarray(getattr(fracture, name))
        values = np.asarray(original, dtype=float)
        if values.ndim > 0 and values.shape[0] == tip_count:
            continue
        if values.ndim == 0:
            continue
        old_count = values.shape[0]
        if name == "Ffront":
            target_shape = (tip_count, *values.shape[1:])
            resized = np.zeros(target_shape, dtype=float)
            copied = min(old_count, tip_count)
            if copied:
                resized[:copied] = values[:copied]
                if copied < tip_count:
                    resized[copied:] = np.nanmedian(values[:copied], axis=0)
            setattr(fracture, name, resized)
            repairs.append(f"resized {name} metadata from {old_count} to {tip_count}")
            continue
        finite = values[np.isfinite(values)]
        if name == "v":
            finite = np.abs(finite)
            default = float(np.nanmedian(finite)) if finite.size else 1.0e-9
            default = max(default, 1.0e-9)
        elif name == "l":
            finite = np.abs(finite)
            default = float(np.nanmedian(finite)) if finite.size else 1.0e-6
            default = max(default, 1.0e-6)
        elif name == "FillF":
            default = float(np.clip(np.nanmedian(finite), 0.0, 1.0)) if finite.size else 0.5
        elif name == "ZeroVertex":
            default = int(np.rint(np.nanmedian(finite))) if finite.size else 0
        else:
            default = float(np.nanmedian(finite)) if finite.size else 0.0
        dtype = original.dtype if name == "ZeroVertex" else float
        resized = np.full(tip_count, default, dtype=dtype)
        copied = min(old_count, tip_count)
        if copied:
            resized[:copied] = values[:copied]
        setattr(fracture, name, resized)
        repairs.append(f"resized {name} metadata from {old_count} to {tip_count}")
    if hasattr(fracture, "v"):
        velocity = np.asarray(fracture.v, dtype=float)
        invalid = ~np.isfinite(velocity) | (velocity < 0.0)
        if invalid.any():
            finite_magnitude = np.abs(np.nan_to_num(velocity[invalid], nan=0.0, posinf=0.0, neginf=0.0))
            velocity[invalid] = np.maximum(finite_magnitude, 1.0e-9)
            fracture.v = velocity
            repairs.append(f"projected {int(invalid.sum())} front velocities to non-negative values")
    return repairs


@dataclass(frozen=True)
class PyFracRunResult:
    """Normalized result from one PyFrac evaluation."""

    half_length_m: float
    max_aperture_mm: float
    area_m2: float
    volume_m3: float
    net_pressure_mpa: float
    bottomhole_pressure_mpa: float
    front_geometry: list[list[float]]
    runtime_seconds: float
    model_name: str
    engine_mode: str
    success: bool
    error: str | None = None
    final_time_s: float = float("nan")
    successful_time_steps: int = 0
    failed_time_steps: int = 0
    target_reached: bool = False
    partial_progress: bool = False
    injected_volume_m3: float = float("nan")
    fracture_volume_m3: float = float("nan")
    leakoff_volume_m3: float = float("nan")
    mass_balance_residual_m3: float = float("nan")
    mass_balance_relative_error: float = float("nan")
    efficiency: float = float("nan")
    time_step_limit_s: float = float("nan")
    fracture_height_m: float = float("nan")
    maximum_width_m: float = float("nan")
    mesh_level: int = 0
    mesh_nx: int = 0
    mesh_ny: int = 0
    mesh_dx_m: float = float("nan")
    mesh_dy_m: float = float("nan")
    mesh_cells_across_front: float = float("nan")
    adaptive_mesh_status: str = "not_evaluated"
    checkpoint_id: str | None = None
    rollback_applied: bool = False
    retry_count: int = 0
    convergence_status: str = "not_evaluated"
    time_step_settings: dict[str, Any] | None = None
    front_metadata_repair_count: int = 0
    # A completed clock is not sufficient for a valid dynamic result.  The
    # legacy controller can terminate with an occupied crack footprint but no
    # active front after the footprint has been pressed against the domain
    # boundary.  Keep this structural flag explicit so callers cannot promote
    # that state to a native acceptance merely because conservation happened
    # to close numerically.
    boundary_limited: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PyFracAdapter:
    """Load PyFrac and run one planar fracture case."""

    def __init__(self, config: PyFracConfig | None = None, project_root: str | Path | None = None) -> None:
        self.config = config or PyFracConfig()
        self.project_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
        self.pyfrac_root = self.config.resolved_root(self.project_root)
        self.src_root = self.pyfrac_root / "src"
        self._modules: dict[str, Any] | None = None

    def verify_installation(self) -> dict[str, Any]:
        return {
            "root": str(self.pyfrac_root),
            "src": str(self.src_root),
            "source_exists": self.src_root.is_dir(),
            "license_exists": (self.pyfrac_root / "LICENSE.TXT").is_file(),
            "gpl_exists": (self.pyfrac_root / "GPL.txt").is_file(),
            "commit": _git_commit(self.pyfrac_root),
        }

    def run(
        self,
        injection_rate_m3_s: float,
        time_s: float,
        mode: str = "snapshot",
        height_m: float | None = None,
        viscosity_pa_s: float | None = None,
        e_prime_pa: float | None = None,
        leakoff_coefficient_m_sqrt_s: float | None = None,
        min_horizontal_stress_pa: float | None = None,
        fracture_toughness_pa_sqrt_m: float | None = None,
        confining_stress_func: Any | None = None,
        injection_rate_history: np.ndarray | None = None,
    ) -> PyFracRunResult:
        started = time.perf_counter()
        mode = mode.lower().strip()
        if mode not in {"snapshot", "native"}:
            raise ValueError("PyFrac mode must be 'snapshot' or 'native'")
        if not self.src_root.is_dir():
            return self._failed(started, mode, "PyFrac source directory does not exist")

        try:
            modules = self._load_modules()
            result = self._run_case(
                modules,
                injection_rate_m3_s=(
                    np.asarray(injection_rate_history, dtype=float)
                    if injection_rate_history is not None
                    else max(float(injection_rate_m3_s), 1.0e-9)
                ),
                time_s=max(float(time_s), self.config.initial_time_s),
                mode=mode,
                height_m=float(height_m or self.config.height_m),
                viscosity_pa_s=float(viscosity_pa_s or self.config.viscosity_pa_s),
                e_prime_pa=float(e_prime_pa or self.config.e_prime_pa),
                leakoff_coefficient_m_sqrt_s=float(
                    leakoff_coefficient_m_sqrt_s or self.config.leakoff_coefficient_m_sqrt_s
                ),
                min_horizontal_stress_pa=float(min_horizontal_stress_pa or self.config.min_horizontal_stress_pa),
                fracture_toughness_pa_sqrt_m=float(
                    fracture_toughness_pa_sqrt_m or self.config.fracture_toughness_pa_sqrt_m
                ),
                confining_stress_func=confining_stress_func,
            )
            return _replace_runtime(result, time.perf_counter() - started)
        except Exception as exc:  # PyFrac is legacy code; preserve diagnostics for reproducibility.
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            return self._failed(started, mode, error)

    def _load_modules(self) -> dict[str, Any]:
        if self._modules is not None:
            return self._modules
        source = str(self.src_root)
        if source not in sys.path:
            sys.path.insert(0, source)

        # PyFrac 1.1.1 predates NumPy 1.24. These aliases are process-local
        # compatibility shims; the vendored GPL source remains unchanged.
        for name, value in (("int", int), ("float", float), ("bool", bool)):
            if name not in np.__dict__:
                setattr(np, name, value)

        # PyFrac 1.1.1 calls ``Polygon(vertices, True)``.  Matplotlib 3.8+
        # made ``closed`` keyword-only.  Keep this compatibility shim local
        # to the process; the vendored GPL source is not modified.
        import matplotlib.patches as mpatches

        polygon = mpatches.Polygon
        if not getattr(polygon, "_pyfrac_legacy_compat", False):
            def _pyfrac_polygon(xy, closed=True, *args, **kwargs):
                return polygon(xy, closed=closed, *args, **kwargs)

            _pyfrac_polygon._pyfrac_legacy_compat = True
            mpatches.Polygon = _pyfrac_polygon

        import level_set

        level_set.Eikonal_Res = _scalar_eikonal_residual
        from controller import Controller
        from fracture import Fracture
        from fracture_initialization import Geometry, InitializationParameters
        from mesh import CartesianMesh
        from properties import FluidProperties, InjectionProperties, MaterialProperties, SimulationProperties

        self._modules = {
            "Controller": Controller,
            "Fracture": Fracture,
            "Geometry": Geometry,
            "InitializationParameters": InitializationParameters,
            "CartesianMesh": CartesianMesh,
            "FluidProperties": FluidProperties,
            "InjectionProperties": InjectionProperties,
            "MaterialProperties": MaterialProperties,
            "SimulationProperties": SimulationProperties,
        }
        return self._modules

    def _configure_mesh_extension(self, simulation: Any) -> None:
        """Apply the explicitly requested legacy PyFrac mesh extension policy."""

        directions = [False, False, False, False]
        if self.config.mesh_extension_enabled:
            if self.config.mesh_extension_all_directions:
                directions = [True, True, True, True]
            else:
                configured = tuple(self.config.mesh_extension_directions)
                if len(configured) != 4:
                    raise ValueError("mesh_extension_directions must contain four booleans")
                directions = [bool(value) for value in configured]
        simulation.meshExtension = directions
        simulation.meshExtensionAllDir = bool(
            self.config.mesh_extension_enabled and self.config.mesh_extension_all_directions
        )
        if self.config.mesh_extension_enabled:
            simulation.set_mesh_extension_factor(float(self.config.mesh_extension_factor))

    def _configure_solver_limits(self, simulation: Any) -> None:
        """Bound legacy inner iterations so outer rollback remains effective."""

        simulation.maxSolverItrs = max(int(self.config.max_solver_iterations), 1)
        simulation.toleranceEHL = max(float(self.config.ehl_tolerance), 1.0e-12)
        simulation.relaxation_factor = min(max(float(self.config.ehl_relaxation), 0.0), 1.0)
        simulation.maxFrontItrs = max(int(self.config.max_front_iterations), 1)
        simulation.maxReattempts = max(int(self.config.max_pyfrac_reattempts), 0)
        simulation.remeshFactor = max(float(self.config.remesh_factor), 1.01)
        simulation.frontCFL = max(float(self.config.front_cfl), 0.0)
        simulation.fractureLengthFraction = max(float(self.config.front_length_fraction), 0.0)
        simulation.injectionVolumeStepFraction = max(
            float(self.config.injection_volume_step_fraction), 1.0e-6
        )
        simulation.positiveVolumeChangeStepFraction = max(
            float(self.config.positive_volume_change_step_fraction), 1.0e-6
        )
        simulation.negativeVolumeChangeStepFraction = max(
            float(self.config.negative_volume_change_step_fraction), 1.0e-6
        )
        simulation.timeStepTimeFraction = max(
            float(self.config.time_step_time_fraction), 1.0e-6
        )
        simulation.cellTraversalFraction = max(
            float(self.config.cell_traversal_fraction), 1.0e-6
        )
        simulation.skipZeroInjectionIntervals = bool(
            self.config.skip_zero_injection_intervals
        )
        simulation.clipToInjectionRegimeEvents = bool(
            self.config.clip_to_injection_regime_events
        )
        simulation.expandDomainOnBoundary = bool(
            self.config.expand_domain_on_boundary
        )
        simulation.domainExpansionFactor = max(
            float(self.config.domain_expansion_factor), 1.05
        )

    def _run_case(self, modules: dict[str, Any], **kwargs: Any) -> PyFracRunResult:
        height = kwargs["height_m"]
        q = kwargs["injection_rate_m3_s"]
        q_scale = float(q[1, 0]) if isinstance(q, np.ndarray) else float(q)
        t = kwargs["time_s"]
        eprime = kwargs["e_prime_pa"]
        viscosity = kwargs["viscosity_pa_s"]
        leakoff = kwargs["leakoff_coefficient_m_sqrt_s"]
        stress = kwargs["min_horizontal_stress_pa"]
        toughness = kwargs["fracture_toughness_pa_sqrt_m"]
        confining_stress_func = kwargs.get("confining_stress_func")
        mode = kwargs["mode"]
        native_start = max(float(self.config.native_start_time_s), float(self.config.initial_time_s))

        # Domain sizing has two different jobs.  Snapshot mode needs a domain
        # large enough for the requested final-time analytical fracture.  A
        # native run, however, must first resolve the explicit t=1 s initial
        # fracture; sizing its initial mesh from a 4,435 s front makes the
        # first fracture sub-cell-sized and causes the legacy front
        # reconstruction to fail before time marching begins.  Native growth
        # is handled later by PyFrac's mesh extension/remeshing policy.
        pkn_length = 0.68 * ((q_scale**3 * eprime) / (viscosity * height**4)) ** 0.2 * t**0.8
        initial_pkn_length = 0.68 * ((q_scale**3 * eprime) / (viscosity * height**4)) ** 0.2 * native_start**0.8
        sizing_length = initial_pkn_length if mode == "native" and t > native_start else pkn_length
        half_length = max(self.config.mesh_half_length_m, 1.8 * sizing_length)
        # Keep several cells across the fixed PKN height.  A fixed 25 m
        # half-height would make a small smoke-test fracture sub-cell-sized.
        half_height = max(height * 1.5, min(self.config.mesh_half_height_m, height * 4.0))
        # Native mode starts from the native initial state. Resolve that state
        # first; sizing only from the final-time front can still leave the
        # initial fracture sub-cell-sized on a coarse mesh.
        resolution_length = sizing_length
        if self.config.adaptive_mesh_enabled:
            mesh_decision = choose_mesh(
                estimated_half_length_m=resolution_length,
                height_m=height,
                base_half_length_m=half_length,
                base_half_height_m=half_height,
                base_nx=self.config.mesh_nx,
                base_ny=self.config.mesh_ny,
                min_front_cells=self.config.min_front_cells,
                boundary_margin_cells=self.config.boundary_margin_cells,
                refinement_factor=self.config.mesh_refinement_factor,
                max_levels=self.config.max_mesh_levels,
                max_nx=self.config.max_mesh_nx,
                max_ny=self.config.max_mesh_ny,
                minimum_half_length_m=self.config.mesh_half_length_m,
            )
        else:
            mesh_decision = MeshDecision(
                level=0,
                half_length_m=float(half_length),
                half_height_m=float(half_height),
                nx=max(int(self.config.mesh_nx), 31),
                # A height-contained PKN run can be benchmarked with fewer
                # vertical cells, but never fewer than 9: below this the
                # initial footprint is no longer resolved reliably.
                # The height-contained PKN formulation is effectively
                # one-dimensional across the fixed fracture height. Five
                # transverse cells are the minimum controlled throughput
                # setting; the default remains the finer 9/31-cell grids.
                ny=max(int(self.config.mesh_ny), 5),
                dx_m=2.0 * float(half_length) / max(int(self.config.mesh_nx) - 1, 1),
                dy_m=2.0 * float(half_height) / max(int(self.config.mesh_ny) - 1, 1),
                estimated_half_length_m=float(resolution_length),
                cells_across_front=float("nan"),
                boundary_margin_m=float(half_length - pkn_length),
                reason="adaptive_mesh_disabled",
            )
        mesh = modules["CartesianMesh"](
            mesh_decision.half_length_m,
            mesh_decision.half_height_m,
            mesh_decision.nx,
            mesh_decision.ny,
        )
        solid_kwargs = {"Carters_coef": leakoff}
        if confining_stress_func is None:
            solid_kwargs["confining_stress"] = stress
        else:
            solid_kwargs["confining_stress_func"] = confining_stress_func
        solid = modules["MaterialProperties"](mesh, eprime, toughness, **solid_kwargs)
        fluid = modules["FluidProperties"](viscosity=viscosity)
        injection = modules["InjectionProperties"](q, mesh)
        sim = modules["SimulationProperties"]()
        sim.finalTime = t
        sim.maxTimeSteps = int(self.config.max_time_steps)
        sim.timeStepLimit = float(self.config.dynamic_step_limit_s) if self.config.dynamic_step_limit_s > 0 else None
        sim.elastohydrSolver = str(self.config.elastohydr_solver)
        sim.solveDeltaP = bool(self.config.solve_delta_p)
        sim.enableVolumeBalanceProjection = bool(
            self.config.enable_volume_balance_projection
        )
        sim.plotFigure = False
        sim.saveToDisk = False
        sim.log2file = False
        # Adaptive initial sizing and runtime domain remeshing are separate
        # concerns.  A deliberately coarse diagnostic run may disable the
        # former while still needing the latter to follow a growing front.
        sim.enableRemeshing = bool(self.config.enable_pyfrac_remeshing)
        # This adapter uses height-contained PKN geometry. Growing the mesh
        # vertically is both unphysical here and dangerous because legacy
        # PyFrac rebuilds a dense N^2 elasticity matrix after extension.
        self._configure_mesh_extension(sim)
        sim.enableGPU = False
        sim.blockFigure = False
        sim.verbositylevel = "error"
        sim.frontAdvancing = str(self.config.front_advancing)
        sim.projMethod = str(self.config.projection_method)
        self._configure_solver_limits(sim)

        geometry = modules["Geometry"]("height contained", fracture_height=height)
        # In native mode the first Fracture object is an explicit initial
        # condition.  Constructing it at the requested final time before the
        # native branch runs defeats the t=1 s mesh sizing and can immediately
        # raise ``fracture is larger than domain`` for a long target such as
        # 4,435 s.  Snapshot mode still initializes directly at its requested
        # time.
        initialization_time = native_start if mode == "native" and t > native_start else t
        init = modules["InitializationParameters"](geometry, regime="PKN", time=initialization_time)
        fracture = modules["Fracture"](mesh, init, solid, fluid, injection, sim)

        engine_mode = "pyfrac_pkn_grid_snapshot"
        final = fracture
        final_time_s = float(t)
        successful_time_steps = 0
        failed_time_steps = 0
        target_reached = True
        native_retry_count = 0
        native_rollback_applied = False
        native_checkpoint_id = None
        zero_injection_jumps: list[dict[str, float]] = []
        # Keep the effective step limit defined for early native requests too.
        # When t == native_start, the native time-marching branch is skipped,
        # but the normalized result still reports the chosen step setting.
        step_limit = float(self.config.dynamic_step_limit_s)
        if mode == "native" and t > self.config.initial_time_s:
            # Native mode starts from a resolved PKN state and time-marches it.
            # The old solver may fail for particular meshes/versions; the
            # caller receives an explicit failure rather than fake output.
            if t <= native_start:
                raise ValueError(
                    f"native target time {t:g}s is not later than the resolved warm-start "
                    f"time {native_start:g}s; use snapshot for early-time output"
                )
            init_geometry = modules["Geometry"]("height contained", fracture_height=height)
            init_param = modules["InitializationParameters"](
                init_geometry, regime="PKN", time=native_start
            )
            initial = modules["Fracture"](mesh, init_param, solid, fluid, injection, sim)
            checkpoint_manager = CheckpointManager()
            checkpoint = checkpoint_manager.save(
                time_s=native_start,
                fracture=initial,
                last_parameters={},
                successful_steps=0,
                failed_steps=0,
                active=True,
            )
            native_checkpoint_id = checkpoint.checkpoint_id
            for attempt in range(max(int(self.config.max_retries), 0) + 1):
                controller = None
                try:
                    trial_solid = modules["MaterialProperties"](mesh, eprime, toughness, **solid_kwargs)
                    trial_fluid = modules["FluidProperties"](viscosity=viscosity)
                    trial_injection = modules["InjectionProperties"](q, mesh)
                    trial_sim = modules["SimulationProperties"]()
                    trial_sim.finalTime = t
                    trial_sim.maxTimeSteps = int(self.config.max_time_steps)
                    trial_sim.timeStepLimit = step_limit if step_limit > 0.0 else None
                    trial_sim.elastohydrSolver = str(self.config.elastohydr_solver)
                    trial_sim.solveDeltaP = bool(self.config.solve_delta_p)
                    trial_sim.enableVolumeBalanceProjection = bool(
                        self.config.enable_volume_balance_projection
                    )
                    trial_sim.plotFigure = False
                    trial_sim.saveToDisk = False
                    trial_sim.log2file = False
                    trial_sim.enableRemeshing = bool(self.config.enable_pyfrac_remeshing)
                    self._configure_mesh_extension(trial_sim)
                    trial_sim.enableGPU = False
                    trial_sim.blockFigure = False
                    trial_sim.verbositylevel = "error"
                    # PyFrac's original controller asks an interactive
                    # question after closure. Native candidates run in a
                    # worker without stdin, so deterministically jump to
                    # the next positive-injection time instead of blocking.
                    trial_sim.autoJumpClosedFracture = True
                    # Near a closed-fracture re-opening, the legacy EHL
                    # matrix can be numerically rank-deficient.  Permit the
                    # solver's finite-checked least-squares fallback for
                    # that case; it is still rejected if the returned state
                    # is non-finite or fails the outer acceptance checks.
                    trial_sim.allowLeastSquaresLinearFallback = True
                    trial_sim.frontAdvancing = str(self.config.front_advancing)
                    trial_sim.projMethod = str(self.config.projection_method)
                    self._configure_solver_limits(trial_sim)
                    current = checkpoint_manager.restore(checkpoint, reason="native_attempt_start", retry_count=attempt)
                    current.pFluid[:] = 0.0
                    current.pFluid[current.EltCrack] = current.pNet[current.EltCrack] + trial_solid.SigmaO[current.EltCrack]
                    previous_attempt = os.environ.get("PYFRAC_NATIVE_ATTEMPT")
                    os.environ["PYFRAC_NATIVE_ATTEMPT"] = str(attempt)
                    original_input = builtins.input
                    try:
                        controller = modules["Controller"](current, trial_solid, trial_fluid, trial_injection, trial_sim)
                        builtins.input = lambda _prompt="": (_ for _ in ()).throw(
                            RuntimeError("PyFrac requested an interactive time step; non-interactive run aborted")
                        )
                        controller.run()
                    finally:
                        builtins.input = original_input
                        if previous_attempt is None:
                            os.environ.pop("PYFRAC_NATIVE_ATTEMPT", None)
                        else:
                            os.environ["PYFRAC_NATIVE_ATTEMPT"] = previous_attempt
                    final = controller.fracture
                    final_time_s = float(final.time)
                    successful_time_steps = int(controller.successfulTimeSteps)
                    failed_time_steps = int(controller.failedTimeSteps)
                    zero_injection_jumps = list(getattr(controller, "zeroInjectionJumps", []))
                    target_reached = target_time_reached(final_time_s, float(t))
                    if not target_reached:
                        raise RuntimeError(
                            "PyFrac native solver stopped before target time: "
                            f"final_time={final_time_s:g}s, target_time={float(t):g}s, "
                            f"successful_steps={successful_time_steps}, failed_steps={failed_time_steps}"
                        )
                    native_retry_count = attempt
                    native_rollback_applied = attempt > 0
                    break
                except (Exception, SystemExit) as exc:
                    native_rollback_applied = True
                    native_retry_count = attempt
                    if attempt >= int(self.config.max_retries):
                        raise RuntimeError(
                            f"native PyFrac failed after rollback/retries: {type(exc).__name__}: {exc}"
                        ) from exc
                    step_limit = max(
                        float(self.config.min_dynamic_step_s),
                        step_limit * float(self.config.retry_time_step_factor),
                    ) if step_limit > 0.0 else float(self.config.min_dynamic_step_s)
                finally:
                    _release_native_controller(controller)
            engine_mode = (
                "pyfrac_native_dynamic_with_shutin_skip"
                if zero_injection_jumps
                else "pyfrac_native_dynamic"
            )
            if self.config.enable_volume_balance_projection:
                engine_mode += "_volume_projection"

        mesh = getattr(final, "mesh", mesh)
        crack = np.asarray(final.EltCrack, dtype=int)
        if crack.size == 0:
            raise RuntimeError("PyFrac returned an empty fracture footprint")
        coords = np.asarray(mesh.CenterCoor[crack], dtype=float)
        tip_cells = np.asarray(final.EltTip, dtype=int).reshape(-1)
        tip_cells = tip_cells[(tip_cells >= 0) & (tip_cells < mesh.NumberOfElts)]
        front = np.asarray(mesh.CenterCoor[tip_cells], dtype=float)
        boundary_limited = tip_cells.size == 0
        length = float(np.max(np.abs(coords[:, 0])))
        aperture = float(np.nanmax(np.asarray(final.w)[crack])) * 1000.0
        area = float(crack.size * mesh.EltArea)
        volume = float(getattr(final, "FractureVolume", np.nansum(final.w[crack]) * mesh.EltArea))
        injected = _scalar_volume(getattr(final, "injectedVol", np.nan))
        if not np.isfinite(injected):
            if isinstance(q, np.ndarray) and q.ndim == 2 and q.shape[0] >= 2:
                injected = float(np.trapz(q[1], q[0]))
            else:
                injected = float(q) * t
        leakoff_volume = _scalar_volume(getattr(final, "LkOffTotal", 0.0))
        residual = float(injected - volume - leakoff_volume)
        relative_error = abs(residual) / max(abs(injected), 1.0e-12)
        net = float(np.nanmean(np.asarray(final.pNet)[crack])) / 1.0e6
        return PyFracRunResult(
            half_length_m=length,
            max_aperture_mm=aperture,
            area_m2=area,
            volume_m3=volume,
            net_pressure_mpa=net,
            bottomhole_pressure_mpa=net + stress / 1.0e6,
            front_geometry=front.tolist(),
            runtime_seconds=0.0,
            model_name="PyFrac",
            engine_mode=engine_mode,
            success=True,
            final_time_s=final_time_s,
            successful_time_steps=successful_time_steps,
            failed_time_steps=failed_time_steps,
            target_reached=target_reached,
            injected_volume_m3=injected,
            fracture_volume_m3=volume,
            leakoff_volume_m3=leakoff_volume,
            mass_balance_residual_m3=residual,
            mass_balance_relative_error=relative_error,
            efficiency=float(volume / injected) if injected > 0 else float("nan"),
            time_step_limit_s=float(step_limit if mode == "native" else self.config.dynamic_step_limit_s),
            fracture_height_m=float(height),
            maximum_width_m=float(aperture / 1000.0),
            mesh_level=mesh_decision.level,
            mesh_nx=mesh_decision.nx,
            mesh_ny=mesh_decision.ny,
            mesh_dx_m=mesh_decision.dx_m,
            mesh_dy_m=mesh_decision.dy_m,
            mesh_cells_across_front=mesh_decision.cells_across_front,
            adaptive_mesh_status=mesh_decision.reason,
            checkpoint_id=native_checkpoint_id,
            rollback_applied=native_rollback_applied,
            retry_count=native_retry_count,
            time_step_settings={
                "max_time_steps": int(self.config.max_time_steps),
                "time_step_limit_s": float(step_limit if mode == "native" else self.config.dynamic_step_limit_s),
                "native_mode": mode == "native",
                "max_retries": int(self.config.max_retries),
                "skip_zero_injection_intervals": bool(self.config.skip_zero_injection_intervals),
                "clip_to_injection_regime_events": bool(
                    self.config.clip_to_injection_regime_events
                ),
                "expand_domain_on_boundary": bool(
                    self.config.expand_domain_on_boundary
                ),
                "domain_expansion_factor": float(
                    self.config.domain_expansion_factor
                ),
                "zero_injection_jumps": zero_injection_jumps,
            },
            boundary_limited=boundary_limited,
        )

    def _failed(self, started: float, mode: str, error: str) -> PyFracRunResult:
        return PyFracRunResult(
            half_length_m=float("nan"),
            max_aperture_mm=float("nan"),
            area_m2=float("nan"),
            volume_m3=float("nan"),
            net_pressure_mpa=float("nan"),
            bottomhole_pressure_mpa=float("nan"),
            front_geometry=[],
            runtime_seconds=time.perf_counter() - started,
            model_name="PyFrac",
            engine_mode=f"pyfrac_{mode}",
            success=False,
            error=error,
        )


class PyFracNativeSession:
    """A continuously advancing native PyFrac state.

    ``PyFracAdapter.run(mode="native")`` is convenient for isolated target
    times.  An EnKF inversion needs a different lifecycle: the fracture state
    at one assimilation time must be the initial state of the next window.
    This session keeps that state and rebuilds the native Controller whenever
    EnKF changes the physical parameters or the injection schedule.

    This is intentionally a single planar fracture session.  It is the
    correct foundation for a continuous single-cluster PyFrac-EnKF experiment;
    it is not labelled as a native six-cluster horizontal-well solver.
    """

    def __init__(
        self,
        adapter: PyFracAdapter,
        injection_rate_history: np.ndarray,
        native_initial_time_s: float,
        height_m: float,
        viscosity_pa_s: float,
        e_prime_pa: float,
        leakoff_coefficient_m_sqrt_s: float,
        min_horizontal_stress_pa: float,
        fracture_toughness_pa_sqrt_m: float,
        max_time_steps: int | None = None,
        domain_time_s: float | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval_s: float = 60.0,
        initial_fracture: Any | None = None,
        initial_parameters: dict[str, float] | None = None,
        initial_successful_steps: int = 0,
        initial_failed_steps: int = 0,
        initial_step_limit_s: float | None = None,
        initial_consecutive_failures: int = 0,
        initial_success_streak: int = 0,
        initial_front_metadata_repair_count: int = 0,
    ) -> None:
        self.adapter = adapter
        self.config = adapter.config
        self.modules = adapter._load_modules()
        self.height_m = float(height_m)
        self.max_time_steps = int(max_time_steps or self.config.max_time_steps)
        self.total_successful_steps = int(initial_successful_steps)
        self.total_failed_steps = int(initial_failed_steps)
        # A failed attempt is rolled back to the last valid state. The member
        # is disabled only after all configured retries have failed.
        self.active = True
        self.last_parameters: dict[str, float] = {}
        self.checkpoints = CheckpointManager()
        self.checkpoint_dir = Path(checkpoint_dir).resolve() if checkpoint_dir else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval_s = max(float(checkpoint_interval_s), 0.0)
        self._last_disk_checkpoint_time_s = float("-inf")
        self.disk_checkpoint_paths: list[str] = []
        self.last_checkpoint_id: str | None = None
        self.last_rollback_applied = False
        self.last_retry_count = 0
        # Keep the reduced continuation step across outer retries.  A long
        # synchronized run may need several calls to recover from one stiff
        # front-reconstruction region; resetting to the original limit on
        # every call only repeats the same failure.
        self._continuation_step_limit_s = max(
            float(
                initial_step_limit_s
                if initial_step_limit_s is not None
                else self.config.initial_step_limit_s
            ),
            max(float(self.config.min_dynamic_step_s), 1.0e-6),
        )
        self.consecutive_failures = max(int(initial_consecutive_failures), 0)
        self._continuation_success_streak = max(int(initial_success_streak), 0)
        self.front_metadata_repair_count = max(int(initial_front_metadata_repair_count), 0)
        self.last_front_metadata_repairs: list[str] = []
        self.mesh_decision: MeshDecision | None = None
        schedule = _validate_injection_history(injection_rate_history)
        default_parameters = {
            "height_m": float(height_m),
            "viscosity_pa_s": float(viscosity_pa_s),
            "e_prime_pa": float(e_prime_pa),
            "leakoff_coefficient_m_sqrt_s": float(leakoff_coefficient_m_sqrt_s),
            "min_horizontal_stress_pa": float(min_horizontal_stress_pa),
            "fracture_toughness_pa_sqrt_m": float(fracture_toughness_pa_sqrt_m),
        }
        if initial_fracture is None:
            native_initial_time = max(float(native_initial_time_s), float(self.config.initial_time_s))
            self.mesh = self._build_mesh(
                schedule,
                float(e_prime_pa),
                native_initial_time,
            )
            self.fracture = self._initialize_fracture(
                schedule,
                native_initial_time,
                float(height_m),
                float(viscosity_pa_s),
                float(e_prime_pa),
                float(leakoff_coefficient_m_sqrt_s),
                float(min_horizontal_stress_pa),
                float(fracture_toughness_pa_sqrt_m),
            )
            checkpoint_kind = "initial_native"
        else:
            # A durable checkpoint already contains the mesh and all mutable
            # Fracture arrays. Do not analytically reinitialize it on resume.
            self.fracture = initial_fracture
            self.mesh = getattr(initial_fracture, "mesh", None)
            if self.mesh is None:
                raise ValueError("initial_fracture checkpoint does not contain a PyFrac mesh")
            checkpoint_kind = "resume"
        self.last_parameters = dict(initial_parameters or default_parameters)
        self._accepted_state_history: list[Any] = []
        if self.config.checkpoint_enabled:
            initial = self.checkpoints.save(
                time_s=float(self.fracture.time),
                fracture=self.fracture,
                last_parameters=self.last_parameters,
                successful_steps=self.total_successful_steps,
                failed_steps=self.total_failed_steps,
                active=self.active,
            )
            self._accepted_state_history.append(initial)
            if self.checkpoint_dir is not None:
                self._persist_checkpoint(initial, kind=checkpoint_kind, force=True)

    def advance_to(
        self,
        target_time_s: float,
        injection_rate_history: np.ndarray,
        *,
        height_m: float,
        viscosity_pa_s: float,
        e_prime_pa: float,
        leakoff_coefficient_m_sqrt_s: float,
        min_horizontal_stress_pa: float,
        fracture_toughness_pa_sqrt_m: float,
        allow_partial: bool = False,
    ) -> PyFracRunResult:
        """Advance the retained fracture state to one later target time."""

        started = time.perf_counter()
        target = float(target_time_s)
        if not self.active:
            return self._failed_session(
                started,
                "member disabled after a previous native PyFrac window failure",
            )
        if target <= float(self.fracture.time) + 1.0e-8:
            return self._summarize(
                started,
                final_time=float(self.fracture.time),
                successful_steps=0,
                failed_steps=0,
                target_reached=True,
            )
        schedule = _validate_injection_history(injection_rate_history)
        if float(schedule[0, -1]) < target - 1.0e-8:
            raise ValueError("injection history must cover the native target time")

        checkpoint = None
        if self.config.checkpoint_enabled:
            checkpoint = self.checkpoints.save(
                time_s=float(self.fracture.time),
                fracture=self.fracture,
                last_parameters=self.last_parameters,
                successful_steps=self.total_successful_steps,
                failed_steps=self.total_failed_steps,
                active=self.active,
            )
            self.last_checkpoint_id = checkpoint.checkpoint_id
            if self.checkpoint_dir is not None:
                self._persist_checkpoint(checkpoint, kind="rollback", force=True)
        self.last_rollback_applied = False
        self.last_retry_count = 0
        requested_parameters = {
            "height_m": float(height_m),
            "viscosity_pa_s": float(viscosity_pa_s),
            "e_prime_pa": float(e_prime_pa),
            "leakoff_coefficient_m_sqrt_s": float(leakoff_coefficient_m_sqrt_s),
            "min_horizontal_stress_pa": float(min_horizontal_stress_pa),
            "fracture_toughness_pa_sqrt_m": float(fracture_toughness_pa_sqrt_m),
        }
        retry_limit = max(int(self.config.max_retries), 0)
        step_limit = float(self._continuation_step_limit_s)
        last_error: Exception | None = None
        last_traceback = ""
        last_steps = 0
        last_failed_steps = 0
        for attempt in range(retry_limit + 1):
            controller = None
            try:
                modules = self.modules
                solid = self._material(
                    requested_parameters["e_prime_pa"],
                    requested_parameters["leakoff_coefficient_m_sqrt_s"],
                    requested_parameters["min_horizontal_stress_pa"],
                    requested_parameters["fracture_toughness_pa_sqrt_m"],
                )
                fluid = modules["FluidProperties"](viscosity=requested_parameters["viscosity_pa_s"])
                injection = modules["InjectionProperties"](schedule, self.mesh)
                sim = self._simulation(target, step_limit)

                # The legacy delta-pressure formulation is well behaved for
                # continued positive injection, but it can become singular
                # when a continuation window ends in a shut-in.  In that
                # case the solver must recover the absolute pressure field
                # from elasticity and the current leak-off state.  Switch
                # back automatically when the target window ends with
                # positive injection (e.g. at re-start after a shut-in).
                target_rate = float(schedule[1, -1])
                if np.isfinite(target_rate):
                    sim.solveDeltaP = bool(target_rate > 0.0)
                # Let Controller.run refresh the formulation at every native
                # time step.  A continuation window can span a ramp, shut-in,
                # and re-start; choosing from only the window-end rate is not
                # physically safe for the legacy delta-pressure solver.
                sim.solveDeltaPByInjectionRate = True
                # See the native path above: continuation and assimilation
                # windows must not enter PyFrac's interactive closure prompt.
                sim.autoJumpClosedFracture = True
                sim.allowLeastSquaresLinearFallback = True
                # Gauge pressure cannot be negative.  The floor is applied by
                # Controller after each accepted native step, before the state
                # is retained for the next continuation window.
                sim.pressureFloorPa = 0.0

                # Rebuild pressure for the updated material before creating a
                # controller. This is the physical-state handoff from EnKF to
                # the next native PyFrac window.
                self.fracture.pFluid[:] = 0.0
                self.fracture.pFluid[self.fracture.EltCrack] = (
                    self.fracture.pNet[self.fracture.EltCrack] + solid.SigmaO[self.fracture.EltCrack]
                )
                controller = modules["Controller"](self.fracture, solid, fluid, injection, sim)
                original_input = builtins.input
                builtins.input = lambda _prompt="": (_ for _ in ()).throw(
                    RuntimeError("PyFrac requested an interactive time step; non-interactive run aborted")
                )
                try:
                    controller.run()
                finally:
                    builtins.input = original_input
                candidate_fracture = controller.fracture
                health_errors = native_state_health_errors(
                    candidate_fracture,
                    checkpoint.fracture if checkpoint is not None else self.fracture,
                )
                repairable = {
                    "non-finite Tarrival on EltChannel",
                    "non-finite TarrvlZrVrtx on EltTip",
                    "non-finite TarrvlZrVrtx on EltRibbon",
                    "negative or non-finite front velocity",
                }
                self.last_front_metadata_repairs = []
                if health_errors and all(
                    error in repairable or error.startswith("tip metadata size mismatch:")
                    for error in health_errors
                ):
                    repairs = repair_native_front_metadata(candidate_fracture)
                    remaining = native_state_health_errors(
                        candidate_fracture,
                        checkpoint.fracture if checkpoint is not None else self.fracture,
                    )
                    if repairs and not remaining:
                        self.last_front_metadata_repairs = repairs
                        self.front_metadata_repair_count += 1
                        health_errors = []
                if health_errors:
                    raise RuntimeError("invalid PyFrac continuation state: " + "; ".join(health_errors))
                self.fracture = candidate_fracture
                self.total_successful_steps += int(controller.successfulTimeSteps)
                self.total_failed_steps += int(controller.failedTimeSteps)
                last_steps = int(controller.successfulTimeSteps)
                last_failed_steps = int(controller.failedTimeSteps)
                reached = target_time_reached(float(self.fracture.time), target)
                if not reached:
                    # The real-time experiment deliberately keeps a valid
                    # partial advance when PyFrac reaches its internal step
                    # budget before the requested source point.  It is a
                    # computed state, not a fabricated target-time result.
                    if allow_partial and float(self.fracture.time) > float(checkpoint.time_s if checkpoint else 0.0) + 1.0e-8 and last_steps > 0:
                        self.last_parameters = requested_parameters
                        self.active = True
                        self._record_successful_continuation(step_limit)
                        self.last_retry_count = attempt
                        result = self._summarize(
                            started,
                            final_time=float(self.fracture.time),
                            successful_steps=last_steps,
                            failed_steps=last_failed_steps,
                            target_reached=False,
                            partial_progress=True,
                        )
                        self._persist_accepted_state(force=False)
                        return result
                    raise RuntimeError(
                        "PyFrac continuous session stopped before target time: "
                        f"final_time={float(self.fracture.time):g}s, target_time={target:g}s, "
                        f"successful_steps={last_steps}, failed_steps={last_failed_steps}"
                    )
                self.last_parameters = requested_parameters
                self.active = True
                self._record_successful_continuation(step_limit)
                self.last_retry_count = attempt
                result = self._summarize(
                    started,
                    final_time=float(self.fracture.time),
                    successful_steps=last_steps,
                    failed_steps=last_failed_steps,
                    target_reached=True,
                )
                self._persist_accepted_state(force=False)
                return result
            except (Exception, SystemExit) as exc:
                last_error = exc if isinstance(exc, Exception) else RuntimeError(str(exc))
                last_traceback = traceback.format_exc(limit=12)
                if checkpoint is not None:
                    self.fracture = self.checkpoints.restore(
                        checkpoint,
                        reason=f"{type(exc).__name__}: {exc}",
                        retry_count=attempt + 1,
                    )
                    self.last_rollback_applied = True
                    self.total_successful_steps = checkpoint.successful_steps
                    self.total_failed_steps = checkpoint.failed_steps + 1
                    self.last_parameters = dict(checkpoint.last_parameters)
                    self.active = bool(checkpoint.active)
                else:
                    self.total_failed_steps += 1
                self.last_retry_count = attempt
                if attempt < retry_limit:
                    reduced = step_limit * float(self.config.retry_time_step_factor)
                    floor = max(float(self.config.min_dynamic_step_s), 1.0e-6)
                    step_limit = max(floor, reduced) if step_limit > 0.0 else floor
                    continue
                # Keep a member recoverable at its last valid checkpoint.
                # The synchronized outer runner can retry the same target
                # with this smaller persistent step limit.
                floor = max(float(self.config.min_dynamic_step_s), 1.0e-6)
                reduced = step_limit * float(self.config.retry_time_step_factor)
                self._continuation_step_limit_s = max(floor, reduced) if step_limit > 0.0 else floor
                self.consecutive_failures += 1
                self._continuation_success_streak = 0
                self.active = bool(checkpoint.active) if checkpoint is not None else False
                break
            finally:
                _release_native_controller(controller)

        return PyFracRunResult(
            half_length_m=float("nan"),
            max_aperture_mm=float("nan"),
            area_m2=float("nan"),
            volume_m3=float("nan"),
            net_pressure_mpa=float("nan"),
            bottomhole_pressure_mpa=float("nan"),
            front_geometry=[],
            runtime_seconds=time.perf_counter() - started,
            model_name="PyFrac",
            engine_mode="pyfrac_native_dynamic_continuation",
            success=False,
            error=(
                f"{type(last_error).__name__}: {last_error}\n"
                f"{last_traceback}"
                if last_error
                else "native continuation failed"
            ),
            final_time_s=float(self.fracture.time),
            successful_time_steps=last_steps,
            failed_time_steps=max(last_failed_steps, 1),
            target_reached=False,
            partial_progress=False,
            mesh_level=self.mesh_decision.level if self.mesh_decision else 0,
            mesh_nx=self.mesh_decision.nx if self.mesh_decision else 0,
            mesh_ny=self.mesh_decision.ny if self.mesh_decision else 0,
            mesh_dx_m=self.mesh_decision.dx_m if self.mesh_decision else float("nan"),
            mesh_dy_m=self.mesh_decision.dy_m if self.mesh_decision else float("nan"),
            mesh_cells_across_front=self.mesh_decision.cells_across_front if self.mesh_decision else float("nan"),
            adaptive_mesh_status=self.mesh_decision.reason if self.mesh_decision else "not_evaluated",
            checkpoint_id=self.last_checkpoint_id,
            rollback_applied=self.last_rollback_applied,
            retry_count=self.last_retry_count,
            time_step_limit_s=step_limit,
            time_step_settings={
                "initial_limit_s": float(self.config.dynamic_step_limit_s),
                "last_attempt_limit_s": float(step_limit),
                "next_attempt_limit_s": float(self._continuation_step_limit_s),
                "min_dynamic_step_s": float(self.config.min_dynamic_step_s),
                "max_retries": retry_limit,
            },
        )

    def _record_successful_continuation(self, step_limit: float) -> None:
        """Keep a safe step, but recover one level after three good windows."""

        self.consecutive_failures = 0
        self._continuation_success_streak += 1
        next_limit = float(step_limit)
        initial = max(float(self.config.dynamic_step_limit_s), float(self.config.min_dynamic_step_s))
        if self._continuation_success_streak >= 3 and next_limit < initial:
            factor = float(self.config.retry_time_step_factor)
            growth = 1.0 / factor if 0.0 < factor < 1.0 else 2.0
            next_limit = min(initial, next_limit * growth)
            self._continuation_success_streak = 0
        self._continuation_step_limit_s = next_limit

    def _persist_accepted_state(self, *, force: bool) -> None:
        """Persist an accepted native state at a controlled interval.

        The in-memory checkpoint remains the rollback mechanism.  These dill
        files are durable restart artifacts for a long experiment and are
        accompanied by a small manifest entry; they are never treated as
        snapshot-mode evidence.
        """

        if not self.config.checkpoint_enabled:
            return
        current_time = float(self.fracture.time)
        checkpoint = self.checkpoints.save(
            time_s=current_time,
            fracture=self.fracture,
            last_parameters=self.last_parameters,
            successful_steps=self.total_successful_steps,
            failed_steps=self.total_failed_steps,
            active=self.active,
        )
        self._accepted_state_history.append(checkpoint)
        # Two prior accepted generations are enough to escape a numerically
        # marginal state while keeping the memory footprint bounded.
        self._accepted_state_history = self._accepted_state_history[-3:]
        if self.checkpoint_dir is None:
            return
        if not force and current_time - self._last_disk_checkpoint_time_s < self.checkpoint_interval_s:
            return
        self._persist_checkpoint(checkpoint, kind="accepted", force=True)

    def rollback_to_previous_accepted(self, *, reason: str) -> bool:
        """Backtrack one accepted generation after repeated continuation failure."""

        current_time = float(self.fracture.time)
        candidates = [
            checkpoint
            for checkpoint in self._accepted_state_history
            if float(checkpoint.time_s) < current_time - 1.0e-8
        ]
        if not candidates:
            return False
        checkpoint = candidates[-1]
        self.fracture = self.checkpoints.restore(checkpoint, reason=reason, retry_count=0)
        self.mesh = self.fracture.mesh
        self.total_successful_steps = int(checkpoint.successful_steps)
        self.total_failed_steps = int(checkpoint.failed_steps) + 1
        self.last_parameters = dict(checkpoint.last_parameters)
        self.active = True
        self.last_rollback_applied = True
        floor = max(float(self.config.min_dynamic_step_s), 1.0e-6)
        reduced = float(self._continuation_step_limit_s) * float(self.config.retry_time_step_factor)
        self._continuation_step_limit_s = max(floor, reduced)
        self.consecutive_failures = 0
        self._continuation_success_streak = 0
        self._accepted_state_history = [
            item for item in self._accepted_state_history if float(item.time_s) <= float(checkpoint.time_s) + 1.0e-8
        ]
        if self.checkpoint_dir is not None:
            restored = self.checkpoints.save(
                time_s=float(self.fracture.time),
                fracture=self.fracture,
                last_parameters=self.last_parameters,
                successful_steps=self.total_successful_steps,
                failed_steps=self.total_failed_steps,
                active=self.active,
            )
            self._persist_checkpoint(restored, kind="backtrack", force=True)
        return True

    def _persist_checkpoint(self, checkpoint: Any, *, kind: str, force: bool) -> None:
        if self.checkpoint_dir is None or not self.config.checkpoint_enabled:
            return
        if not force and float(checkpoint.time_s) - self._last_disk_checkpoint_time_s < self.checkpoint_interval_s:
            return
        try:
            import dill

            safe_time = f"{float(checkpoint.time_s):.3f}".replace(".", "_")
            path = self.checkpoint_dir / f"{kind}_{checkpoint.checkpoint_id}_t{safe_time}s.dill"
            payload = {
                "checkpoint_id": checkpoint.checkpoint_id,
                "kind": kind,
                "time_s": float(checkpoint.time_s),
                "last_parameters": dict(checkpoint.last_parameters),
                "successful_steps": int(checkpoint.successful_steps),
                "failed_steps": int(checkpoint.failed_steps),
                "active": bool(checkpoint.active),
                "fracture": checkpoint.fracture,
                "engine_mode": "pyfrac_native_dynamic_continuation",
                "continuation_step_limit_s": float(self._continuation_step_limit_s),
                "consecutive_failures": int(self.consecutive_failures),
                "continuation_success_streak": int(self._continuation_success_streak),
                "front_metadata_repair_count": int(self.front_metadata_repair_count),
            }
            with path.open("wb") as handle:
                dill.dump(payload, handle, -1)
            self.disk_checkpoint_paths.append(str(path))
            # Rollback archives are written on every attempt, but they must
            # not reset the cadence for durable accepted-state checkpoints.
            if kind in {"initial", "resume", "accepted"}:
                self._last_disk_checkpoint_time_s = float(checkpoint.time_s)
            self.checkpoints.events.append(
                {
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "time_s": float(checkpoint.time_s),
                    "event": "disk_saved",
                    "kind": kind,
                    "path": str(path),
                }
            )
        except Exception as exc:
            # A failed archival write must be visible in the audit but must
            # not corrupt the in-memory rollback path.
            self.checkpoints.events.append(
                {
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "time_s": float(checkpoint.time_s),
                    "event": "disk_save_failed",
                    "kind": kind,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    def _failed_session(self, started: float, error: str) -> PyFracRunResult:
        """Return a deterministic failure for a member disabled after error."""

        return PyFracRunResult(
            half_length_m=float("nan"),
            max_aperture_mm=float("nan"),
            area_m2=float("nan"),
            volume_m3=float("nan"),
            net_pressure_mpa=float("nan"),
            bottomhole_pressure_mpa=float("nan"),
            front_geometry=[],
            runtime_seconds=time.perf_counter() - started,
            model_name="PyFrac",
            engine_mode="pyfrac_native_dynamic_continuation",
            success=False,
            error=error,
            final_time_s=float(self.fracture.time),
            successful_time_steps=0,
            failed_time_steps=0,
            target_reached=False,
            partial_progress=False,
        )

    def _build_mesh(self, schedule: np.ndarray, e_prime_pa: float, time_s: float):
        # Reserve the domain for the requested continuation horizon, but
        # resolve the native initial front when choosing cell size. Using the
        # later high-rate front for both decisions can make the t=1 s initial
        # state sub-cell-sized and fail before the first native step.
        active = schedule[1, schedule[0] <= float(time_s) + 1.0e-8]
        q_scale = max(float(np.nanmax(active if active.size else schedule[1])), 1.0e-9)
        pkn_length = 0.68 * ((q_scale**3 * e_prime_pa) / (self.config.viscosity_pa_s * self.height_m**4)) ** 0.2 * time_s**0.8
        initial_time = max(float(self.config.native_start_time_s), float(self.config.initial_time_s))
        initial_active = schedule[1, schedule[0] <= initial_time + 1.0e-8]
        initial_q_scale = max(float(np.nanmax(initial_active if initial_active.size else schedule[1])), 1.0e-9)
        resolution_length = 0.68 * ((initial_q_scale**3 * e_prime_pa) / (self.config.viscosity_pa_s * self.height_m**4)) ** 0.2 * initial_time**0.8
        half_length = max(self.config.mesh_half_length_m, 1.8 * pkn_length)
        half_height = max(self.height_m * 1.5, min(self.config.mesh_half_height_m, self.height_m * 4.0))
        if self.config.adaptive_mesh_enabled:
            decision = choose_mesh(
                estimated_half_length_m=resolution_length,
                height_m=self.height_m,
                base_half_length_m=half_length,
                base_half_height_m=half_height,
                base_nx=self.config.mesh_nx,
                base_ny=self.config.mesh_ny,
                min_front_cells=self.config.min_front_cells,
                boundary_margin_cells=self.config.boundary_margin_cells,
                refinement_factor=self.config.mesh_refinement_factor,
                max_levels=self.config.max_mesh_levels,
                max_nx=self.config.max_mesh_nx,
                max_ny=self.config.max_mesh_ny,
                minimum_half_length_m=self.config.mesh_half_length_m,
            )
        else:
            decision = MeshDecision(
                level=0,
                half_length_m=float(half_length),
                half_height_m=float(half_height),
                nx=max(int(self.config.mesh_nx), 31),
                ny=max(int(self.config.mesh_ny), 5),
                dx_m=2.0 * float(half_length) / max(max(int(self.config.mesh_nx), 31) - 1, 1),
                dy_m=2.0 * float(half_height) / max(max(int(self.config.mesh_ny), 5) - 1, 1),
                estimated_half_length_m=float(pkn_length),
                cells_across_front=float("nan"),
                boundary_margin_m=float(half_length - pkn_length),
                reason="adaptive_mesh_disabled",
            )
        self.mesh_decision = decision
        return self.modules["CartesianMesh"](
            decision.half_length_m,
            decision.half_height_m,
            decision.nx,
            decision.ny,
        )

    def _material(self, e_prime_pa: float, leakoff: float, stress_pa: float, toughness: float):
        return self.modules["MaterialProperties"](
            self.mesh,
            e_prime_pa,
            toughness,
            Carters_coef=leakoff,
            confining_stress=stress_pa,
        )

    def _initialize_fracture(
        self,
        schedule: np.ndarray,
        native_initial_time: float,
        height_m: float,
        viscosity: float,
        e_prime_pa: float,
        leakoff: float,
        stress_pa: float,
        toughness: float,
    ):
        modules = self.modules
        solid = self._material(e_prime_pa, leakoff, stress_pa, toughness)
        fluid = modules["FluidProperties"](viscosity=viscosity)
        injection = modules["InjectionProperties"](schedule, self.mesh)
        sim = self._simulation(native_initial_time)
        geometry = modules["Geometry"]("height contained", fracture_height=height_m)
        init = modules["InitializationParameters"](geometry, regime="PKN", time=native_initial_time)
        fracture = modules["Fracture"](self.mesh, init, solid, fluid, injection, sim)

        # A PKN analytical initial condition has a valid footprint but the
        # legacy Fracture constructor leaves ``v`` as NaN when no explicit tip
        # velocity is supplied. Controller then performs one same-footprint
        # step, stores a zero velocity, and the next predictor-corrector step
        # can evaluate a zero tip volume and stop at t ~= 1 s. For a native
        # continuation the first front speed is available from the same PKN
        # similarity scale used to build the initial condition:
        # dL/dt ~= 0.8 L/t. This only initializes front bookkeeping; it does
        # not modify pressure, width, volume, or any EnKF parameter.
        if (
            str(getattr(sim, "projMethod", "")) == "LS_continousfront"
            and np.asarray(getattr(fracture, "EltTip", []), dtype=int).size
        ):
            crack = np.asarray(fracture.EltCrack, dtype=int)
            half_length = (
                float(np.max(np.abs(self.mesh.CenterCoor[crack, 0])))
                if crack.size
                else 0.0
            )
            initial_front_speed = max(
                0.8 * half_length / max(float(native_initial_time), 1.0e-6),
                1.0e-6,
            )
            fracture.v = np.full(
                int(np.asarray(fracture.EltTip, dtype=int).size),
                initial_front_speed,
                dtype=np.float64,
            )
        return fracture

    def _simulation(self, final_time: float, time_step_limit_s: float | None = None):
        sim = self.modules["SimulationProperties"]()
        sim.finalTime = float(final_time)
        sim.maxTimeSteps = int(self.max_time_steps)
        limit = self.config.dynamic_step_limit_s if time_step_limit_s is None else time_step_limit_s
        sim.timeStepLimit = float(limit) if float(limit) > 0 else None
        sim.elastohydrSolver = str(self.config.elastohydr_solver)
        sim.solveDeltaP = bool(self.config.solve_delta_p)
        sim.enableVolumeBalanceProjection = bool(
            self.config.enable_volume_balance_projection
        )
        sim.plotFigure = False
        sim.saveToDisk = False
        sim.log2file = False
        # Initial adaptive sizing and runtime PyFrac remeshing are separate
        # controls.  A continuation session may intentionally use the fixed
        # initial grid while still allowing the native solver to extend or
        # compress the domain as the front grows.
        sim.enableRemeshing = bool(self.config.enable_pyfrac_remeshing)
        adapter = getattr(self, "adapter", None)
        if adapter is not None:
            adapter._configure_mesh_extension(sim)
        else:
            # Lightweight unit tests construct the session without its
            # adapter; retain the safe legacy default for that path.
            sim.meshExtension = [False, False, False, False]
            sim.meshExtensionAllDir = False
        sim.enableGPU = False
        sim.blockFigure = False
        sim.verbositylevel = "error"
        sim.autoJumpClosedFracture = True
        sim.allowLeastSquaresLinearFallback = True
        sim.frontAdvancing = str(self.config.front_advancing)
        sim.projMethod = str(self.config.projection_method)
        adapter = getattr(self, "adapter", None)
        if adapter is not None:
            adapter._configure_solver_limits(sim)
        else:
            sim.maxSolverItrs = 140
            sim.maxFrontItrs = 25
            sim.maxReattempts = 8
        return sim

    def _summarize(
        self,
        started: float,
        *,
        final_time: float,
        successful_steps: int,
        failed_steps: int,
        target_reached: bool,
        partial_progress: bool = False,
    ) -> PyFracRunResult:
        final = self.fracture
        self.mesh = getattr(final, "mesh", self.mesh)
        crack = np.asarray(final.EltCrack, dtype=int)
        if crack.size == 0:
            raise RuntimeError("PyFrac returned an empty fracture footprint")
        coords = np.asarray(self.mesh.CenterCoor[crack], dtype=float)
        length = float(np.max(np.abs(coords[:, 0])))
        aperture = float(np.nanmax(np.asarray(final.w)[crack])) * 1000.0
        area = float(crack.size * self.mesh.EltArea)
        volume = _scalar_volume(getattr(final, "FractureVolume", np.nansum(final.w[crack]) * self.mesh.EltArea))
        injected = _scalar_volume(getattr(final, "injectedVol", np.nan))
        leakoff = _scalar_volume(getattr(final, "LkOffTotal", 0.0))
        residual = float(injected - volume - leakoff)
        relative = abs(residual) / max(abs(injected), 1.0e-12)
        net = float(np.nanmean(np.asarray(final.pNet)[crack])) / 1.0e6
        tip_cells = np.asarray(final.EltTip, dtype=int).reshape(-1)
        tip_cells = tip_cells[(tip_cells >= 0) & (tip_cells < self.mesh.NumberOfElts)]
        front = np.asarray(self.mesh.CenterCoor[tip_cells], dtype=float)
        stress_pa = float(self.last_parameters.get("min_horizontal_stress_pa", self.config.min_horizontal_stress_pa))
        return PyFracRunResult(
            half_length_m=length,
            max_aperture_mm=aperture,
            area_m2=area,
            volume_m3=volume,
            net_pressure_mpa=net,
            bottomhole_pressure_mpa=net + stress_pa / 1.0e6,
            front_geometry=front.tolist(),
            runtime_seconds=time.perf_counter() - started,
            model_name="PyFrac",
            engine_mode="pyfrac_native_dynamic_continuation",
            success=True,
            final_time_s=final_time,
            successful_time_steps=successful_steps,
            failed_time_steps=failed_steps,
            target_reached=target_reached,
            partial_progress=partial_progress,
            injected_volume_m3=injected,
            fracture_volume_m3=volume,
            leakoff_volume_m3=leakoff,
            mass_balance_residual_m3=residual,
            mass_balance_relative_error=relative,
            efficiency=float((injected - leakoff) / max(injected, 1.0e-12)),
            time_step_limit_s=float(self.config.dynamic_step_limit_s),
            fracture_height_m=float(self.height_m),
            maximum_width_m=float(aperture / 1000.0),
            mesh_level=self.mesh_decision.level if self.mesh_decision else 0,
            mesh_nx=self.mesh_decision.nx if self.mesh_decision else 0,
            mesh_ny=self.mesh_decision.ny if self.mesh_decision else 0,
            mesh_dx_m=self.mesh_decision.dx_m if self.mesh_decision else float("nan"),
            mesh_dy_m=self.mesh_decision.dy_m if self.mesh_decision else float("nan"),
            mesh_cells_across_front=self.mesh_decision.cells_across_front if self.mesh_decision else float("nan"),
            adaptive_mesh_status=self.mesh_decision.reason if self.mesh_decision else "not_evaluated",
            checkpoint_id=self.last_checkpoint_id,
            rollback_applied=self.last_rollback_applied,
            retry_count=self.last_retry_count,
            time_step_settings={
                "time_step_limit_s": float(self._continuation_step_limit_s),
                "max_time_steps": int(self.max_time_steps),
                "pyfrac_remeshing_enabled": bool(self.config.enable_pyfrac_remeshing),
                "last_front_metadata_repairs": list(self.last_front_metadata_repairs),
            },
            front_metadata_repair_count=int(self.front_metadata_repair_count),
            boundary_limited=tip_cells.size == 0,
        )


def _validate_injection_history(value: np.ndarray) -> np.ndarray:
    history = np.asarray(value, dtype=float)
    if history.ndim != 2 or history.shape[0] != 2 or history.shape[1] == 0:
        raise ValueError("injection_rate_history must be a 2 x N array")
    if not np.isclose(history[0, 0], 0.0):
        raise ValueError("injection_rate_history must start at t=0")
    if np.any(~np.isfinite(history)) or np.any(np.diff(history[0]) <= 0.0):
        raise ValueError("injection_rate_history times must be finite and strictly increasing")
    history = history.copy()
    # A zero rate is a physical shut-in and must remain zero.  Only the
    # initial rate used to construct a warm analytical state needs to be
    # positive; callers create that initial value in make_schedule().
    history[1] = np.maximum(history[1], 0.0)
    if history.shape[1] > 1 and history[1, 0] <= 1.0e-12:
        positive = np.flatnonzero(history[1] > 1.0e-12)
        history[1, 0] = history[1, positive[0]] if positive.size else 1.0e-9
    return history


class PyFracLengthForwardModel(LengthForwardModel):
    """LengthForwardModel-compatible wrapper for offline PyFrac studies.

    The wrapper is intentionally not used by the online EnKF default.  It is
    available through ``build_length_forward_model('pyfrac')`` for offline
    comparison and uses the explicit ``snapshot`` mode by default.
    """

    model_name = "pyfrac_reference"

    def __init__(self, config: PyFracConfig | None = None, mode: str = "snapshot") -> None:
        self.config = config or PyFracConfig()
        self.mode = mode
        self.adapter = PyFracAdapter(self.config)

    def simulate_lengths(
        self,
        factor_state: np.ndarray,
        cluster_x: np.ndarray,
        q_base: np.ndarray | float,
        viscosity_pa_s: float,
        e_prime_pa: float,
        height_m: float,
        t_seconds: float,
    ):
        import pandas as pd
        from .fracture_length_models import LengthForwardResult

        x = np.asarray(cluster_x, dtype=float)
        n_clusters = len(x)
        factors = np.clip(np.resize(np.asarray(factor_state, dtype=float), n_clusters), 0.65, 1.35)
        base = np.maximum(np.resize(np.asarray(q_base, dtype=float), n_clusters), 1.0e-9)
        capacity = base * factors
        q_cluster = capacity * base.sum() / max(float(capacity.sum()), 1.0e-12)
        rows: list[dict[str, float]] = []
        for index, (cluster_x_value, q_value, factor) in enumerate(zip(x, q_cluster, factors), start=1):
            result = self.adapter.run(
                injection_rate_m3_s=float(q_value),
                time_s=float(t_seconds),
                mode=self.mode,
                height_m=height_m,
                viscosity_pa_s=viscosity_pa_s,
                e_prime_pa=e_prime_pa,
            )
            if not result.success:
                raise RuntimeError(f"PyFrac cluster {index} failed: {result.error}")
            rows.append(
                {
                    "cluster_id": index,
                    "x_center_m": float(cluster_x_value),
                    "Q_cluster_m3s": float(q_value),
                    "cluster_factor": float(factor),
                    "half_length_m": result.half_length_m,
                    "max_aperture_mm": result.max_aperture_mm,
                    "area_m2": result.area_m2,
                    "volume_m3": result.volume_m3,
                    "net_pressure_mpa": result.net_pressure_mpa,
                    "bottomhole_pressure_mpa": result.bottomhole_pressure_mpa,
                    "pyfrac_runtime_seconds": result.runtime_seconds,
                }
            )
        return LengthForwardResult(pd.DataFrame(rows), self.model_name)


def _replace_runtime(result: PyFracRunResult, runtime: float) -> PyFracRunResult:
    payload = result.to_dict()
    payload["runtime_seconds"] = float(runtime)
    return PyFracRunResult(**payload)


def _git_commit(path: Path) -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _scalar_volume(value: Any) -> float:
    try:
        array = np.asarray(value, dtype=float)
        if array.size == 0:
            return 0.0
        return float(np.nansum(array))
    except (TypeError, ValueError):
        return float("nan")


def _scalar_eikonal_residual(Tij: Any, *args: Any) -> float:
    """NumPy-2-compatible form of PyFrac's old scalar Eikonal residual."""

    scalar = lambda value: float(np.asarray(value).reshape(-1)[0])
    value = scalar(Tij)
    left, right, bottom, top, fij, dx, dy = [scalar(value) for value in args]
    return (
        max((value - left) / dx, 0.0) ** 2
        + min((right - value) / dx, 0.0) ** 2
        + max((value - bottom) / dy, 0.0) ** 2
        + min((top - value) / dy, 0.0) ** 2
        - fij**2
    )
