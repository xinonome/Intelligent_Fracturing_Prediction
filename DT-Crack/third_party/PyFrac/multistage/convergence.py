from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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
        "coarse": (61, 31),
        "medium": (81, 41),
        "fine": (101, 51),
    }
    rows = []
    for name, (nx, ny) in cases.items():
        # Compare the same resolved final-time state. Early-time PKN
        # initialization can be sub-cell-sized on coarse meshes, which is a
        # known legacy initializer constraint rather than a convergence datum.
        simulation = SimulationConfig(config.simulation.final_time_s, config.simulation.final_time_s, config.simulation.front_advancing, config.simulation.mode, config.simulation.fracture_height_m)
        case_config = replace(config, mesh=MeshConfig(config.mesh.x_half_extent_m, config.mesh.y_half_extent_m, nx, ny), simulation=simulation)
        result = run_stage(stage, ProjectContext(case_config, logs, trajectory, root / name))
        final = result.metrics.iloc[-1]
        rows.append({
            "case": name, "nx": nx, "ny": ny,
            "final_half_length_m": final.half_length_m,
            "final_full_height_m": final.full_height_m,
            "maximum_width_m": final.max_width_m,
            "fracture_volume_m3": final.fracture_volume_m3,
            "handover_time_s": result.metadata.get("handover_time_s"),
            "runtime_s": result.metadata.get("elapsed_s"),
        })
    summary = pd.DataFrame(rows)
    medium = summary.loc[summary.case == "medium"].iloc[0]
    fine = summary.loc[summary.case == "fine"].iloc[0]
    fields = ["final_half_length_m", "final_full_height_m", "maximum_width_m", "fracture_volume_m3"]
    relative = {field: abs(float(medium[field]) - float(fine[field])) / max(abs(float(fine[field])), 1e-12) for field in fields}
    summary.to_csv(root / "summary.csv", index=False)
    (root / "acceptance.json").write_text(__import__("json").dumps({"criterion": "medium_vs_fine_relative_difference <= 5%", "relative_differences": relative, "pass": all(value <= 0.05 for value in relative.values())}, indent=2), encoding="utf-8")
    return root / "summary.csv"
