"""Compare matched constrained and unbounded EnKF control runs.

This script is intentionally separate from the APP runtime.  It reads two
completed ``validate_direct_observations.py`` outputs produced with the same
data, seed, replay points and filter settings, then reports whether removing
parameter bounds improves pressure fit at the cost of parameter instability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PARAMETER_CLASSES = {
    "pressure_fracture": [
        "eprime_gpa",
        "leakoff_m_sqrt_s",
        "viscosity_pa_s",
        "min_stress_mpa",
        "fracture_toughness_pa_sqrt_m",
    ],
    "cluster_intake": [
        "intake_capacity_factor_c1",
        "intake_capacity_factor_c2",
        "intake_capacity_factor_c3",
        "intake_capacity_factor_c4",
        "intake_capacity_factor_c5",
        "intake_capacity_factor_c6",
    ],
    "interaction_allocation": [
        "stress_shadow_scale",
        "boundary_relief_scale",
        "allocation_exponent",
    ],
}


def read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_rows(summaries: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_names = [
        "all_steps_bhp_mae_mpa",
        "all_steps_bhp_rmse_mpa",
        "validation_bhp_mae_mpa",
        "validation_bhp_relative_error_mean",
        "validation_prior_bhp_relative_error_mean",
        "validation_liquid_tvd_mean",
        "validation_sand_tvd_mean",
        "all_steps_compute_p95_ms",
        "validation_pass",
    ]
    for mode, summary in summaries.items():
        metrics = summary.get("metrics", {})
        for metric in metric_names:
            rows.append(
                {
                    "row_type": "run_metric",
                    "mode": mode,
                    "parameter_class": "",
                    "parameter": metric,
                    "value": metrics.get(metric),
                    "start": None,
                    "end": None,
                    "range": None,
                    "std": None,
                    "max_abs_step_change": None,
                    "p95_abs_step_change": None,
                }
            )
        trajectory = metrics.get("parameter_trajectory_summary", {})
        for class_name, names in PARAMETER_CLASSES.items():
            for name in names:
                item = trajectory.get(class_name, {}).get(name)
                if not item:
                    continue
                rows.append(
                    {
                        "row_type": "parameter_trajectory",
                        "mode": mode,
                        "parameter_class": class_name,
                        "parameter": name,
                        "value": None,
                        "start": item.get("start"),
                        "end": item.get("end"),
                        "range": item.get("range"),
                        "std": item.get("std"),
                        "max_abs_step_change": item.get("max_abs_step_change"),
                        "p95_abs_step_change": item.get("p95_abs_step_change"),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No comparable metrics were found in the supplied summaries")
    return frame


def make_delta_table(frame: pd.DataFrame) -> pd.DataFrame:
    constrained = frame[frame["mode"].eq("constrained")].set_index(
        ["row_type", "parameter_class", "parameter"]
    )
    unbounded = frame[frame["mode"].eq("unbounded_control")].set_index(
        ["row_type", "parameter_class", "parameter"]
    )
    common = constrained.index.intersection(unbounded.index)
    rows = []
    for key in common:
        row = {
            "row_type": key[0],
            "parameter_class": key[1],
            "parameter": key[2],
        }
        for column in ["value", "start", "end", "range", "std", "max_abs_step_change", "p95_abs_step_change"]:
            left = constrained.loc[key, column]
            right = unbounded.loc[key, column]
            row[f"constrained_{column}"] = left
            row[f"unbounded_{column}"] = right
            if pd.notna(left) and pd.notna(right):
                row[f"unbounded_minus_constrained_{column}"] = float(right) - float(left)
            else:
                row[f"unbounded_minus_constrained_{column}"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def plot_pressure_histories(summaries: dict[str, dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    colors = {"constrained": "#2f80ed", "unbounded_control": "#d95d39"}
    for mode, summary in summaries.items():
        history_path = Path(summary["outputs"]["history"])
        history = pd.read_csv(history_path)
        axes[0].plot(
            history["time_s"],
            history["posterior_bottomhole_pressure_mpa"],
            color=colors.get(mode),
            label=f"{mode} posterior",
        )
    first_history = pd.read_csv(Path(next(iter(summaries.values()))["outputs"]["history"]))
    axes[0].plot(
        first_history["time_s"],
        first_history["observed_bottomhole_pressure_mpa"],
        color="black",
        linewidth=1.2,
        label="observed",
    )
    axes[0].set_ylabel("Bottom-hole pressure / MPa")
    axes[0].set_title("Matched EnKF pressure fit")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for mode, summary in summaries.items():
        trajectory_path = Path(summary["outputs"]["parameter_trajectory"])
        trajectory = pd.read_csv(trajectory_path)
        for class_name, names in PARAMETER_CLASSES.items():
            class_relative_ranges = []
            for name in names:
                column = f"posterior_{name}"
                if column in trajectory:
                    values = trajectory[column].to_numpy(dtype=float)
                    class_relative_ranges.append(
                        float(values.max() - values.min())
                        / max(abs(float(values.mean())), 1.0e-12)
                    )
            if class_relative_ranges:
                axes[1].plot(
                    [class_name],
                    [sum(class_relative_ranges) / len(class_relative_ranges)],
                    marker="o",
                    color=colors.get(mode),
                    label=mode if class_name == list(PARAMETER_CLASSES)[0] else None,
                )
    axes[1].set_ylabel("Mean relative range")
    axes[1].set_title("Mean full-process posterior parameter volatility by class")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constrained-summary", required=True, type=Path)
    parser.add_argument("--unbounded-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        "constrained": read_summary(args.constrained_summary.resolve()),
        "unbounded_control": read_summary(args.unbounded_summary.resolve()),
    }
    frame = make_rows(summaries)
    delta = make_delta_table(frame)
    frame.to_csv(args.output_dir / "parameter_bound_mode_comparison.csv", index=False, encoding="utf-8-sig")
    delta.to_csv(args.output_dir / "parameter_bound_mode_delta.csv", index=False, encoding="utf-8-sig")
    plot_pressure_histories(summaries, args.output_dir / "parameter_bound_mode_comparison.png")
    result = {
        "experiment": "matched_enkf_parameter_bound_mode_control",
        "constrained_summary": str(args.constrained_summary.resolve()),
        "unbounded_summary": str(args.unbounded_summary.resolve()),
        "comparison_csv": str((args.output_dir / "parameter_bound_mode_comparison.csv").resolve()),
        "delta_csv": str((args.output_dir / "parameter_bound_mode_delta.csv").resolve()),
        "figure": str((args.output_dir / "parameter_bound_mode_comparison.png").resolve()),
        "modes": {
            mode: {
                "parameter_bound_mode": summary.get("parameter_bound_mode"),
                "pressure_fit_priority": summary.get("pressure_fit_priority"),
                "metrics": summary.get("metrics", {}),
            }
            for mode, summary in summaries.items()
        },
    }
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
