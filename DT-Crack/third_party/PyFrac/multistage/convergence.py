from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import load_config, load_inputs
from .schemas import MeshConfig, SimulationConfig
from .stage_runner import ProjectContext, run_stage


def run_convergence(config_path: str | Path, output_dir: str | Path | None = None) -> Path:
    config = load_config(config_path)
    logs, trajectory = load_inputs(config)
    root = Path(output_dir) if output_dir else config.output_root / "convergence"
    root.mkdir(parents=True, exist_ok=True)
    stage = config.stages[0]
    cases = {
        "spatial_coarse": {"mesh": (61, 31), "time_step_limit_s": None},
        "spatial_medium": {"mesh": (81, 41), "time_step_limit_s": None},
        "spatial_fine": {"mesh": (101, 51), "time_step_limit_s": None},
        "temporal_coarse": {"mesh": (81, 41), "time_step_limit_s": 30.0},
        "temporal_medium": {"mesh": (81, 41), "time_step_limit_s": 15.0},
        "temporal_fine": {"mesh": (81, 41), "time_step_limit_s": 7.5},
    }
    rows = []
    for name, case in cases.items():
        nx, ny = case["mesh"]
        # Compare the same resolved final-time state. Early-time PKN
        # initialization can be sub-cell-sized on coarse meshes, which is a
        # known legacy initializer constraint rather than a convergence datum.
        simulation = SimulationConfig(config.simulation.final_time_s, config.simulation.final_time_s, config.simulation.front_advancing, config.simulation.mode, config.simulation.fracture_height_m)
        runtime = dict(config.runtime)
        if case["time_step_limit_s"] is not None:
            runtime["time_step_limit_s"] = case["time_step_limit_s"]
        case_config = replace(config, mesh=MeshConfig(config.mesh.x_half_extent_m, config.mesh.y_half_extent_m, nx, ny), simulation=simulation, runtime=runtime)
        result = run_stage(stage, ProjectContext(case_config, logs, trajectory, root / name))
        final = result.metrics.iloc[-1]
        rows.append({
            "case": name, "family": "temporal" if name.startswith("temporal") else "spatial", "nx": nx, "ny": ny,
            "time_step_limit_s": case["time_step_limit_s"],
            "final_half_length_m": final.half_length_m,
            "final_full_height_m": final.full_height_m,
            "maximum_width_m": final.max_width_m,
            "fracture_volume_m3": final.fracture_volume_m3,
            "handover_time_s": result.metadata.get("handover_time_s"),
            "runtime_s": result.metadata.get("elapsed_s"),
            "mass_balance_relative_error": result.metadata.get("mass_balance", {}).get("relative_error"),
            "target_reached": bool(result.snapshots[-1].target_reached) if result.snapshots else False,
            "failed_time_steps": int(result.snapshots[-1].failed_time_steps) if result.snapshots else -1,
            "injected_volume_m3": result.metadata.get("mass_balance", {}).get("injected_volume_m3"),
            "leakoff_volume_m3": result.metadata.get("mass_balance", {}).get("leakoff_volume_m3"),
            "mass_balance_residual_m3": result.metadata.get("mass_balance", {}).get("residual_m3"),
        })
    summary = pd.DataFrame(rows)
    fields = ["final_half_length_m", "final_full_height_m", "maximum_width_m", "fracture_volume_m3"]
    spatial_medium = summary.loc[summary.case == "spatial_medium"].iloc[0]
    spatial_fine = summary.loc[summary.case == "spatial_fine"].iloc[0]
    temporal_medium = summary.loc[summary.case == "temporal_medium"].iloc[0]
    temporal_fine = summary.loc[summary.case == "temporal_fine"].iloc[0]
    spatial_relative = {field: abs(float(spatial_medium[field]) - float(spatial_fine[field])) / max(abs(float(spatial_fine[field])), 1e-12) for field in fields}
    temporal_relative = {field: abs(float(temporal_medium[field]) - float(temporal_fine[field])) / max(abs(float(temporal_fine[field])), 1e-12) for field in fields}
    mass_balance = {
        str(row["case"]): float(row["mass_balance_relative_error"])
        for _, row in summary.iterrows()
        if pd.notna(row["mass_balance_relative_error"])
    }
    acceptance = {
        "criterion": "spatial and temporal medium_vs_fine relative difference <= 5%; mass balance <= 10%",
        "spatial_relative_differences": spatial_relative,
        "temporal_relative_differences": temporal_relative,
        "mass_balance_relative_errors": mass_balance,
        "spatial_pass": all(value <= 0.05 for value in spatial_relative.values()),
        "temporal_pass": all(value <= 0.05 for value in temporal_relative.values()),
        "mass_balance_pass": all((not pd.notna(value)) or value <= 0.10 for value in mass_balance.values()),
        "solver_pass": bool(summary["target_reached"].all() and (summary["failed_time_steps"] == 0).all() and np.isfinite(summary[fields].to_numpy(dtype=float)).all()),
    }
    acceptance["pass"] = bool(acceptance["spatial_pass"] and acceptance["temporal_pass"] and acceptance["mass_balance_pass"] and acceptance["solver_pass"])
    summary.to_csv(root / "summary.csv", index=False)
    (root / "convergence_summary.json").write_text(json.dumps({"cases": cases, "spatial": spatial_relative, "temporal": temporal_relative}, indent=2), encoding="utf-8")
    summary[["case", "family", "injected_volume_m3", "fracture_volume_m3", "leakoff_volume_m3", "mass_balance_residual_m3", "mass_balance_relative_error"]].to_csv(root / "mass_balance.csv", index=False)
    (root / "acceptance.json").write_text(json.dumps(acceptance, indent=2, allow_nan=True), encoding="utf-8")
    return root / "summary.csv"
