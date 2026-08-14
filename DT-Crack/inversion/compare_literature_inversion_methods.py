"""Compare a literature-style full-time-step PKN inversion with online EnKF.

The full-time-step baseline is a reproducible adaptation of DAS-PKN inversion
work: fit one physical parameter vector to all calibration timestamps, then
run the forward model on held-out timestamps without assimilating them.  It is
not claimed to be a byte-for-byte reproduction of a paper's conjugate-gradient
or raw DAS signal operator because this project currently has cluster liquid,
sand and pressure observations rather than raw DAS strain-rate matrices.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


DT_ROOT = Path(__file__).resolve().parents[1]
if str(DT_ROOT) not in sys.path:
    sys.path.insert(0, str(DT_ROOT))

from data_fusion import controls_for_step, load_frac_monitor_text, load_stage_pressure_schedule, pressure_for_step
from data_fusion.pressure_schedule_adapter import PressureModelConfig
from inversion import PhysicalEnKFConfig, physical_values, pkn_with_carter_leakoff, state_record
from validate_direct_observations import (
    batch_calibrate_state,
    build_parser as build_validation_parser,
    normalize_positive,
    observed_cluster_shares,
    predicted_observation,
    total_variation,
)


def configure_font() -> None:
    for path in [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")]:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare online parameter-EnKF with a full-time-step PKN least-squares inversion baseline."
    )
    parser.add_argument("--frac-monitor-text", required=True)
    parser.add_argument("--construction-pressure-xls", required=True)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--calibration-ratio", type=float, default=0.70)
    parser.add_argument("--ensemble-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lsq-max-nfev", type=int, default=80)
    parser.add_argument("--run-dir", default=str(DT_ROOT.parent / "outputs" / "dt" / "literature_method_comparison"))
    return parser


def configure_enhanced_validation_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the same enhanced configuration used by the formal EnKF run."""

    argv = [
        "--frac-monitor-text", str(args.frac_monitor_text),
        "--construction-pressure-xls", str(args.construction_pressure_xls),
        "--max-steps", str(args.max_steps),
        "--calibration-ratio", str(args.calibration_ratio),
        "--ensemble-size", str(args.ensemble_size),
        "--seed", str(args.seed),
        "--physics-profile", "enhanced",
        "--validation-mode", "online",
        "--batch-max-nfev", str(args.lsq_max_nfev),
        "--run-dir", str(Path(args.run_dir) / "enkf_internal"),
    ]
    return build_validation_parser().parse_args(argv)


def select_context(args: argparse.Namespace) -> dict:
    monitor = load_frac_monitor_text(args.frac_monitor_text, default_step_seconds=1.0)
    controls = monitor.controls
    n_clusters = int(controls["cluster_id"].nunique())
    available_steps = np.asarray(sorted(controls["step"].unique()), dtype=int)
    count = min(max(args.max_steps, 2), len(available_steps))
    indices = np.linspace(0, len(available_steps) - 1, count).round().astype(int)
    source_steps = np.unique(available_steps[indices])
    calibration_count = max(1, min(len(source_steps) - 1, int(round(len(source_steps) * args.calibration_ratio))))
    validation_args = configure_enhanced_validation_args(args)
    pressure_cfg = PressureModelConfig(min_horizontal_stress_mpa=validation_args.base_min_stress_mpa)
    pressure, pressure_meta = load_stage_pressure_schedule(
        args.construction_pressure_xls,
        pressure_cfg,
        validation_args.measured_depth_m,
        validation_args.vertical_depth_m,
    )
    cfg = PhysicalEnKFConfig(
        base_eprime_pa=validation_args.base_eprime_pa,
        base_leakoff_m_sqrt_s=validation_args.base_leakoff_m_sqrt_s,
        base_viscosity_pa_s=validation_args.base_viscosity_pa_s,
        base_min_stress_mpa=validation_args.base_min_stress_mpa,
        height_m=validation_args.height_m,
        pressure_proxy_scale=validation_args.pressure_proxy_scale,
        max_leakoff_fraction=0.85,
        hydraulic_coupling_mode="coupled",
        conductance_exponent=validation_args.conductance_exponent,
        stress_shadow_feedback=0.75,
        pressure_leakoff_exponent=0.12,
        pressure_leakoff_reference_mpa=15.0,
    )
    prior_mean = np.r_[
        0.0,
        0.0,
        0.0,
        validation_args.base_min_stress_mpa,
        0.0,
    ]
    validation_args.batch_calibrate = True
    return {
        "monitor": monitor,
        "controls": controls,
        "n_clusters": n_clusters,
        "source_steps": source_steps,
        "calibration_count": calibration_count,
        "pressure": pressure,
        "pressure_meta": pressure_meta,
        "cfg": cfg,
        "prior_mean": prior_mean,
        "validation_args": validation_args,
    }


def evaluate_state(state: np.ndarray, source_step: int, context: dict) -> dict:
    controls = context["controls"]
    pressure = context["pressure"]
    cfg: PhysicalEnKFConfig = context["cfg"]
    n_clusters = int(context["n_clusters"])
    validation_args = context["validation_args"]
    step_controls = controls_for_step(controls, int(source_step))
    t_seconds = max(float(source_step), 1.0)
    cumulative_volume = float(step_controls["cumulative_liquid_volume_m3"].sum())
    total_rate = max(cumulative_volume / t_seconds, 1.0e-8)
    current_total_rate = max(float(step_controls["flow_rate_m3_min"].sum()) / 60.0, 1.0e-8)
    q_base = np.full(n_clusters, total_rate / n_clusters, dtype=float)
    q_current = np.full(n_clusters, current_total_rate / n_clusters, dtype=float)
    observed_liquid, observed_sand = observed_cluster_shares(step_controls)
    observed_bhp = float(pressure_for_step(pressure, int(source_step))["bottomhole_pressure_mpa"])
    forward = pkn_with_carter_leakoff(state, q_base, t_seconds, cfg, q_current)
    predicted, _ = predicted_observation(forward, n_clusters)
    predicted_liquid = predicted[: n_clusters - 1]
    predicted_sand = predicted[n_clusters - 1 : 2 * (n_clusters - 1)]
    predicted_bhp = float(predicted[-1])
    row = {
        "source_step": int(source_step),
        "time_s": t_seconds,
        "observed_bottomhole_pressure_mpa": observed_bhp,
        "predicted_bottomhole_pressure_mpa": predicted_bhp,
        "liquid_tvd": total_variation(
            np.r_[predicted_liquid, 1.0 - float(predicted_liquid.sum())], observed_liquid
        ),
        "sand_tvd": total_variation(
            np.r_[predicted_sand, 1.0 - float(predicted_sand.sum())], observed_sand
        ),
        "bhp_relative_error": abs(predicted_bhp - observed_bhp) / max(abs(observed_bhp), 1.0),
        "half_length_max_m": float(np.max(forward["half_length_m"])),
        "leakoff_fraction": float(forward["leakoff_fraction"]),
        "rate_conservation_error": float(forward["rate_conservation_error"]),
        "eprime_gpa": float(physical_values(state, cfg, n_clusters)["eprime_pa"]) / 1.0e9,
        "min_stress_mpa": float(physical_values(state, cfg, n_clusters)["min_horizontal_stress_mpa"]),
        "step_compute_ms": 0.0,
    }
    return row


def run_full_time_step_lsq(context: dict) -> tuple[pd.DataFrame, dict]:
    args = context["validation_args"]
    source_steps = context["source_steps"]
    calibration_count = context["calibration_count"]
    start = time.perf_counter()
    center, fit_info = batch_calibrate_state(
        args,
        source_steps,
        calibration_count,
        context["controls"],
        context["pressure"],
        context["cfg"],
        context["n_clusters"],
        context["prior_mean"],
    )
    fit_seconds = time.perf_counter() - start
    rows: list[dict] = []
    for index, source_step in enumerate(source_steps):
        step_start = time.perf_counter()
        row = evaluate_state(center, int(source_step), context)
        row["sequence_index"] = index
        row["phase"] = "calibration" if index < calibration_count else "validation"
        row["step_compute_ms"] = (time.perf_counter() - step_start) * 1000.0
        rows.append(row)
    history = pd.DataFrame(rows)
    validation = history[history["phase"].eq("validation")]
    metrics = {
        "calibration_steps": int((history["phase"] == "calibration").sum()),
        "validation_steps": int(len(validation)),
        "validation_liquid_tvd_mean": float(validation["liquid_tvd"].mean()),
        "validation_sand_tvd_mean": float(validation["sand_tvd"].mean()),
        "validation_bhp_relative_error_mean": float(validation["bhp_relative_error"].mean()),
        "validation_all_observations_within_15_percent_rate": float(
            ((validation["liquid_tvd"] <= 0.15) & (validation["sand_tvd"] <= 0.15) & (validation["bhp_relative_error"] <= 0.15)).mean()
        ),
        "validation_pass": bool(
            validation["liquid_tvd"].mean() <= 0.15
            and validation["sand_tvd"].mean() <= 0.15
            and validation["bhp_relative_error"].mean() <= 0.15
        ),
        "forward_step_p50_ms": float(history["step_compute_ms"].quantile(0.50)),
        "forward_step_p95_ms": float(history["step_compute_ms"].quantile(0.95)),
        "fit_runtime_seconds": float(fit_seconds),
        "fit_runtime_amortized_ms_per_timestep": float(fit_seconds / max(len(source_steps), 1) * 1000.0),
        "max_rate_conservation_error": float(history["rate_conservation_error"].max()),
        "mean_leakoff_fraction": float(history["leakoff_fraction"].mean()),
    }
    return history, {"metrics": metrics, "fit_info": fit_info, "center": center.tolist()}


def summarize_enkf(history: pd.DataFrame, result: dict) -> dict:
    metrics = result["metrics"]
    return {
        "method": "online_parameter_enkf",
        "method_label": "当前方法：参数空间 EnKF 在线更新",
        "validation_liquid_tvd_mean": metrics["validation_liquid_tvd_mean"],
        "validation_sand_tvd_mean": metrics["validation_sand_tvd_mean"],
        "validation_bhp_relative_error_mean": metrics["validation_bhp_relative_error_mean"],
        "validation_all_observations_within_15_percent_rate": metrics["validation_all_observations_within_15_percent_rate"],
        "validation_pass": metrics["validation_pass"],
        "runtime_p50_ms": metrics["all_steps_compute_p50_ms"],
        "runtime_p95_ms": metrics["all_steps_compute_p95_ms"],
        "fit_runtime_seconds": float(result["batch_calibration"].get("runtime_seconds", 0.0)),
        "notes": "EnKF每个新观测到达后更新PKN参数，再重新运行PKN。",
    }


def summarize_lsq(history: pd.DataFrame, result: dict) -> dict:
    metrics = result["metrics"]
    return {
        "method": "full_time_step_lsq",
        "method_label": "文献型基线：全时间步 PKN 最小二乘拟合",
        "validation_liquid_tvd_mean": metrics["validation_liquid_tvd_mean"],
        "validation_sand_tvd_mean": metrics["validation_sand_tvd_mean"],
        "validation_bhp_relative_error_mean": metrics["validation_bhp_relative_error_mean"],
        "validation_all_observations_within_15_percent_rate": metrics["validation_all_observations_within_15_percent_rate"],
        "validation_pass": metrics["validation_pass"],
        "runtime_p50_ms": metrics["forward_step_p50_ms"],
        "runtime_p95_ms": metrics["forward_step_p95_ms"],
        "fit_runtime_seconds": metrics["fit_runtime_seconds"],
        "notes": "只用前70%拟合一组静态参数，后30%不吸收新观测；对应全时间步历史匹配基线。",
    }


def plot_comparison(frame: pd.DataFrame, path: Path) -> None:
    labels = frame["method_label"].tolist()
    x = np.arange(len(frame))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    metric_specs = [
        ("validation_liquid_tvd_mean", "液量TVD / %", "#2f6db0"),
        ("validation_sand_tvd_mean", "砂量TVD / %", "#ef8b2c"),
        ("validation_bhp_relative_error_mean", "压力相对误差 / %", "#3a9d5d"),
    ]
    for ax, (column, title, color) in zip(axes, metric_specs):
        values = frame[column].to_numpy(dtype=float) * 100.0
        bars = ax.bar(x, values, color=color, width=0.58)
        ax.axhline(15.0, color="#c62828", ls="--", lw=1.3, label="15%目标")
        ax.set_title(title)
        ax.set_xticks(x, ["参数EnKF", "全时间步LSQ"])
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value:.2f}", ha="center", fontsize=10)
    fig.suptitle("文献型全时间步PKN反演与当前参数EnKF对比", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_runtime(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = ["参数EnKF", "全时间步LSQ"]
    values = frame["runtime_p95_ms"].to_numpy(dtype=float)
    bars = ax.bar(labels, values, color=["#1f77b4", "#7f8c8d"], width=0.55)
    ax.axhline(15000.0, color="#c62828", ls="--", label="15秒指标")
    ax.set_ylabel("单步正演/更新耗时 P95 / ms")
    ax.set_title("计算耗时对比（全时间步拟合的总拟合耗时另列）")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.03, f"{value:.1f}", ha="center")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    configure_font()
    args = build_parser().parse_args()
    output = Path(args.run_dir).resolve() / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    context = select_context(args)

    validation_args = context["validation_args"]
    enkf_history, _, enkf_result = __import__("validate_direct_observations").run(validation_args)
    lsq_history, lsq_result = run_full_time_step_lsq(context)
    comparison = pd.DataFrame([summarize_enkf(enkf_history, enkf_result), summarize_lsq(lsq_history, lsq_result)])
    comparison.to_csv(output / "method_comparison.csv", index=False, encoding="utf-8-sig")
    lsq_history.to_csv(output / "full_time_step_lsq_history.csv", index=False, encoding="utf-8-sig")
    enkf_history.to_csv(output / "online_enkf_history.csv", index=False, encoding="utf-8-sig")
    plot_comparison(comparison, output / "method_accuracy_comparison.png")
    plot_runtime(comparison, output / "method_runtime_comparison.png")
    summary = {
        "demo": "literature_inversion_method_comparison",
        "data": {
            "fiber_monitor": str(Path(args.frac_monitor_text).resolve()),
            "construction_pressure": str(Path(args.construction_pressure_xls).resolve()),
            "time_steps": int(len(context["source_steps"])),
            "calibration_steps": int(context["calibration_count"]),
            "validation_steps": int(len(context["source_steps"]) - context["calibration_count"]),
        },
        "methods": comparison.to_dict(orient="records"),
        "literature_alignment": {
            "das_pkn_full_time_step": "Adapted: PKN forward model + multi-time-step least-squares parameter inversion.",
            "dts_enkf": "Adapted by current online parameter EnKF; current data has no DTS temperature channel.",
            "vae_kan_enkf": "Not claimed as reproduced: requires raw fracture-network images, latent training data and a KAN surrogate.",
        },
        "limitations": [
            "The available fiber file contains cluster liquid/sand/balance fields, not raw DAS strain-rate matrices or DTS temperature profiles.",
            "The comparison uses the same reduced observation operator for fairness; it is not a direct reproduction of raw-signal papers.",
            "The 15% values are observation-space errors, not independent fracture-length truth errors.",
        ],
        "outputs": {
            "comparison_csv": str(output / "method_comparison.csv"),
            "accuracy_figure": str(output / "method_accuracy_comparison.png"),
            "runtime_figure": str(output / "method_runtime_comparison.png"),
            "full_time_step_history": str(output / "full_time_step_lsq_history.csv"),
            "online_enkf_history": str(output / "online_enkf_history.csv"),
        },
        "config": asdict(context["cfg"]),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_root": str(output), "methods": summary["methods"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
