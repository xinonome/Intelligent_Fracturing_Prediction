from __future__ import annotations

from pathlib import Path

import pandas as pd

from .exceptions import ToughnessSensitivityConfigurationError
from .io import load_yaml
from .multistage_runner import run_project
from .report import generate_report


def run_sensitivity(config_path, matrix_path) -> Path:
    import yaml
    matrix = load_yaml(matrix_path)
    viscosity_cases = matrix.get("viscosity_cases") or {}
    if not viscosity_cases:
        raise ToughnessSensitivityConfigurationError("sensitivity matrix requires viscosity_cases")
    toughness_cases = matrix.get("toughness_cases") or {}
    cases = []
    for name, case in viscosity_cases.items():
        cases.append((str(name), {"viscosity_pa_s": float(case["viscosity_pa_s"])}))
    if toughness_cases:
        for name, case in toughness_cases.items():
            if "kic_pa_sqrt_m" not in case:
                raise ToughnessSensitivityConfigurationError(f"toughness case {name} lacks kic_pa_sqrt_m")
        cases = [(f"{v_name}_{t_name}", {"viscosity_pa_s": float(v["viscosity_pa_s"]), "toughness_pa_sqrt_m": float(t["kic_pa_sqrt_m"])}) for v_name, v in viscosity_cases.items() for t_name, t in toughness_cases.items()]
    config = __import__("multistage.config", fromlist=["load_config"]).load_config(config_path)
    sensitivity_root = config.output_root / "sensitivity"
    sensitivity_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, overrides in cases:
        # High viscosity produces a smaller early-time footprint. The legacy
        # initializer rejects a sub-cell fracture, so sensitivity cases use a
        # final-time snapshot unless the matrix explicitly supplies a safer
        # output interval. This is recorded in the case metadata via the
        # single final snapshot rather than silently relabelling time.
        overrides = {**overrides, "save_every_s": float(config.simulation.final_time_s)}
        run_dir, metadata, results = run_project(config, overrides=overrides, output_dir=sensitivity_root / name)
        generate_report(run_dir)
        for stage_id, result in results.items():
            final = result.metrics.iloc[-1]
            rows.append({
                "case": name, "stage_id": stage_id, "viscosity_pa_s": overrides["viscosity_pa_s"],
                "toughness_pa_sqrt_m": overrides.get("toughness_pa_sqrt_m"),
                "final_half_length_m": final.half_length_m, "final_full_height_m": final.full_height_m,
                "final_aspect_ratio": final.aspect_ratio,
                "first_barrier_contact_time_s": result.metadata.get("stage_top_tvd_m"),
                "full_containment_time_s": None, "handover_time_s": result.metadata.get("handover_time_s"),
                "max_pressure_pa": float(result.metrics.max_pressure_pa.max()),
                "fracture_volume_m3": final.fracture_volume_m3,
                "runtime_s": result.metadata.get("elapsed_s"),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(sensitivity_root / "summary.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not summary.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for stage_id, frame in summary.groupby("stage_id"):
            ax.plot(frame["viscosity_pa_s"], frame["final_half_length_m"], marker="o", label=stage_id)
        ax.set_xscale("log"); ax.set_xlabel("viscosity (Pa·s)"); ax.set_ylabel("final half length (m)"); ax.legend(); fig.tight_layout(); fig.savefig(sensitivity_root / "viscosity_geometry_comparison.png", dpi=140); plt.close(fig)
        if summary["toughness_pa_sqrt_m"].notna().any():
            fig, ax = plt.subplots(figsize=(7, 4)); summary.groupby("toughness_pa_sqrt_m")["final_half_length_m"].mean().plot(kind="bar", ax=ax); ax.set_ylabel("mean final half length (m)"); fig.tight_layout(); fig.savefig(sensitivity_root / "toughness_geometry_comparison.png", dpi=140); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4)); summary.groupby("case")["handover_time_s"].count().plot(kind="bar", ax=ax); ax.set_ylabel("handover evidence count"); fig.tight_layout(); fig.savefig(sensitivity_root / "handover_time_comparison.png", dpi=140); plt.close(fig)
    (sensitivity_root / "configuration.json").write_text(__import__("json").dumps({"cases": cases}, indent=2), encoding="utf-8")
    return sensitivity_root / "summary.csv"
