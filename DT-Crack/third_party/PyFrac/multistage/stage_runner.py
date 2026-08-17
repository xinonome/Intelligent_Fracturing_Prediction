from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .diagnostics import barrier_contacts, handover_diagnostic
from .global_mapping import local_to_global
from .property_mapping import build_stage_material, local_stress_function
from .pyfrac_adapter import StageSimulationInput, run_pyfrac_stage
from .schemas import ProjectConfig, StageDefinition, StageResult
from .stage_definition import stage_boundary_tvd


@dataclass
class ProjectContext:
    config: ProjectConfig
    logs: pd.DataFrame
    trajectory: pd.DataFrame
    output_dir: Path


def run_stage(stage: StageDefinition, context: ProjectContext, overrides: dict[str, Any] | None = None) -> StageResult:
    overrides = overrides or {}
    config = context.config
    material = build_stage_material(context.logs, context.trajectory, stage, config)
    fluid = config.fluid
    if "viscosity_pa_s" in overrides:
        fluid = type(fluid)(float(overrides["viscosity_pa_s"]), fluid.compressibility_pa_inv, fluid.density_kg_m3)
    if "toughness_pa_sqrt_m" in overrides:
        material = type(material)(**{**material.__dict__, "toughness_pa_sqrt_m": float(overrides["toughness_pa_sqrt_m"])})
    simulation = config.simulation
    if "save_every_s" in overrides:
        simulation = type(simulation)(simulation.final_time_s, float(overrides["save_every_s"]), simulation.front_advancing, simulation.mode, simulation.fracture_height_m)
    stage_dir = context.output_dir / stage.stage_id
    stress_function = local_stress_function(context.logs, stage, context.trajectory)
    sim_input = StageSimulationInput(stage.stage_id, config.mesh, material, fluid, config.injection, simulation, config.runtime, stress_function)
    result = run_pyfrac_stage(sim_input, stage_dir)
    top_tvd, bottom_tvd, center_tvd = stage_boundary_tvd(context.trajectory, stage)
    stage_top_v, stage_bottom_v = center_tvd - min(top_tvd, bottom_tvd), center_tvd - max(top_tvd, bottom_tvd)
    tolerance = float(config.diagnostics["barrier_detection"]["tolerance_m"])
    contacts = barrier_contacts(result.metrics, result.snapshots, stage_top_v, stage_bottom_v, tolerance)
    evidence, handover = handover_diagnostic(
        result.metrics,
        contacts,
        int(config.diagnostics["handover"]["min_consecutive_snapshots"]),
        float(config.diagnostics["handover"]["vertical_growth_epsilon_m_s"]),
        float(config.diagnostics["handover"]["lateral_growth_min_m_s"]),
    )
    contacts.to_csv(stage_dir / "barrier_contacts.csv", index=False)
    evidence.to_csv(stage_dir / "handover_evidence.csv", index=False)
    last = result.snapshots[-1]
    local = pd.DataFrame(last.front_coordinates_local, columns=["u_m", "v_m"])
    local.insert(0, "time_s", last.time_s)
    local.insert(0, "stage_id", stage.stage_id)
    global_front = local_to_global(local, context.trajectory, stage, config.mapping.get("fracture_azimuth_deg"))
    local.to_csv(stage_dir / "final_front_local.csv", index=False)
    global_front.to_csv(stage_dir / "final_front_global.csv", index=False)
    result.metadata.update({
        "stage_material": material.__dict__,
        "stage_top_tvd_m": top_tvd,
        "stage_bottom_tvd_m": bottom_tvd,
        "stage_center_tvd_m": center_tvd,
        "stage_top_v_m": stage_top_v,
        "stage_bottom_v_m": stage_bottom_v,
        "handover_time_s": handover,
        "mass_balance": _mass_balance_summary(result.snapshots),
    })
    if not result.metadata["mass_balance"].get("pass", False):
        raise RuntimeError(f"PyFrac stage {stage.stage_id} failed mass-balance acceptance: {result.metadata['mass_balance']}")
    if not last.target_reached or last.failed_time_steps:
        raise RuntimeError(
            f"PyFrac stage {stage.stage_id} did not reach target: target_reached={last.target_reached}, failed_steps={last.failed_time_steps}"
        )
    return result


def _mass_balance_summary(snapshots) -> dict[str, Any]:
    if not snapshots:
        return {"status": "NOT_COMPUTED", "reason": "no snapshots"}
    final = snapshots[-1]
    value = float(final.mass_balance_relative_error)
    return {
        "status": "COMPUTED" if value == value else "NOT_COMPUTED",
        "injected_volume_m3": float(final.injected_volume_m3),
        "fracture_volume_m3": float(final.fracture_volume_m3),
        "leakoff_volume_m3": float(final.leakoff_volume_m3),
        "residual_m3": float(final.mass_balance_residual_m3),
        "relative_error": value,
        "pass": bool(value == value and value <= 0.10),
    }
