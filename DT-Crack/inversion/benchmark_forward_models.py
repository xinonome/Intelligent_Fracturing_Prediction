from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
DATA_ROOT = Path(os.environ.get("FRACTURING_DATA_ROOT", str(PROJECT_ROOT / "data")))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from forward_models import build_length_forward_model  # noqa: E402
from inversion.playback import (  # noqa: E402
    build_parser as build_demo_parser,
    calc_eprime,
    enkf_update,
    pkn_proxy_net_pressure_mpa,
    run_demo,
)


def configure_matplotlib_chinese_font() -> None:
    for font_path in [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf"), Path(r"C:\Windows\Fonts\simsun.ttc")]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            return


configure_matplotlib_chinese_font()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark PKN/BEM/data-driven forward models under the same EnKF loop.")
    parser.add_argument("--models", nargs="+", default=["pkn4", "bem_reduced", "physics_hybrid", "data_surrogate"])
    parser.add_argument("--frac-monitor-text", default=str(DATA_ROOT / "frac_monitor.txt"))
    parser.add_argument("--well-trajectory-csv", default=str(DATA_ROOT / "well_trajectory.csv"))
    parser.add_argument("--construction-pressure-xls", default=None)
    parser.add_argument("--max-playback-steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--ensemble-size", type=int, default=80)
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--default-step-seconds", type=float, default=1.0)
    parser.add_argument("--visual-aperture-gain", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "outputs" / "dt" / "forward_model_benchmark"))
    parser.add_argument("--render-best-demo", action="store_true", help="Render the best model with the full HTML playback script after benchmark.")
    parser.add_argument("--open", action="store_true", help="Open rendered best-demo HTML if --render-best-demo is set.")
    return parser


def make_demo_args(args: argparse.Namespace, model_name: str) -> argparse.Namespace:
    demo_args = build_demo_parser().parse_args([])
    demo_args.forward_model = model_name
    demo_args.frac_monitor_text = args.frac_monitor_text
    demo_args.well_trajectory_csv = args.well_trajectory_csv
    demo_args.construction_pressure_xls = args.construction_pressure_xls
    demo_args.use_real_monitor_timeline = True
    demo_args.max_playback_steps = args.max_playback_steps
    demo_args.default_step_seconds = args.default_step_seconds
    demo_args.n_clusters = args.n_clusters
    demo_args.ensemble_size = args.ensemble_size
    demo_args.visual_aperture_gain = args.visual_aperture_gain
    demo_args.seed = args.seed
    demo_args.open = False
    return demo_args


def percentile_ms(values_seconds: list[float], percentile: float) -> float:
    if not values_seconds:
        return 0.0
    return float(np.percentile(np.asarray(values_seconds, dtype=float) * 1000.0, percentile))


def benchmark_single_forward(model_name: str, table: pd.DataFrame, demo_args: argparse.Namespace, repeats: int) -> dict[str, float]:
    model = build_length_forward_model(model_name)
    final_time = float(table["time_s"].max())
    final = table[table["time_s"] == final_time].sort_values("cluster_id")
    cluster_x = final["x_center_m"].to_numpy(dtype=float)
    factor = final.get("posterior_growth_factor", final["cluster_factor"]).to_numpy(dtype=float)
    q_base = final["Q_cluster_m3s"].to_numpy(dtype=float) / np.maximum(factor, 1.0e-9)
    e_prime = calc_eprime(demo_args.young_modulus_pa, demo_args.poisson_ratio)

    durations = []
    for _ in range(max(int(repeats), 1)):
        start = time.perf_counter()
        model.simulate_lengths(
            factor_state=factor,
            cluster_x=cluster_x,
            q_base=q_base,
            viscosity_pa_s=demo_args.viscosity_pa_s,
            e_prime_pa=e_prime,
            height_m=demo_args.height_m,
            t_seconds=final_time,
        )
        durations.append(time.perf_counter() - start)
    return {
        "single_forward_p50_ms": percentile_ms(durations, 50),
        "single_forward_p95_ms": percentile_ms(durations, 95),
        "single_forward_max_ms": percentile_ms(durations, 100),
    }


def benchmark_enkf_step(model_name: str, table: pd.DataFrame, demo_args: argparse.Namespace, repeats: int) -> dict[str, float]:
    model = build_length_forward_model(model_name)
    rng = np.random.default_rng(demo_args.seed + 909)
    final_time = float(table["time_s"].max())
    final = table[table["time_s"] == final_time].sort_values("cluster_id")
    cluster_x = final["x_center_m"].to_numpy(dtype=float)
    factor = final.get("posterior_growth_factor", final["cluster_factor"]).to_numpy(dtype=float)
    q_base = final["Q_cluster_m3s"].to_numpy(dtype=float) / np.maximum(factor, 1.0e-9)
    observed_lengths = final["observed_half_length_m"].to_numpy(dtype=float)
    e_prime = calc_eprime(demo_args.young_modulus_pa, demo_args.poisson_ratio)
    pressure_available = "observed_net_pressure_mpa" in final.columns and final["observed_net_pressure_mpa"].notna().any()
    observed_pressure = float(final["observed_net_pressure_mpa"].dropna().iloc[0]) if pressure_available else None

    durations = []
    for _ in range(max(int(repeats), 1)):
        ensemble = np.clip(
            factor.reshape(1, -1) + rng.normal(0.0, 0.055, size=(demo_args.ensemble_size, len(factor))),
            0.65,
            1.35,
        )
        start = time.perf_counter()
        pred_tables = [
            model.simulate_lengths(
                factor_state=member,
                cluster_x=cluster_x,
                q_base=q_base,
                viscosity_pa_s=demo_args.viscosity_pa_s,
                e_prime_pa=e_prime,
                height_m=demo_args.height_m,
                t_seconds=final_time,
            ).table
            for member in ensemble
        ]
        predicted_lengths = np.asarray([frame["half_length_m"].to_numpy(dtype=float) for frame in pred_tables])
        if observed_pressure is not None and demo_args.assimilate_net_pressure:
            predicted_pressure = np.asarray(
                [
                    pkn_proxy_net_pressure_mpa(frame, e_prime, demo_args.height_m, demo_args.net_pressure_proxy_scale)
                    for frame in pred_tables
                ],
                dtype=float,
            ).reshape(-1, 1)
            predicted_obs = np.hstack([predicted_lengths, predicted_pressure])
            observed_obs = np.r_[observed_lengths, observed_pressure]
            obs_noise = np.r_[np.full(len(factor), demo_args.length_noise_m), demo_args.net_pressure_noise_mpa]
        else:
            predicted_obs = predicted_lengths
            observed_obs = observed_lengths
            obs_noise = np.full(len(factor), demo_args.length_noise_m)
        updated, _ = enkf_update(ensemble, predicted_obs, observed_obs, obs_noise, rng)
        posterior_factor = updated.mean(axis=0)
        model.simulate_lengths(
            factor_state=posterior_factor,
            cluster_x=cluster_x,
            q_base=q_base,
            viscosity_pa_s=demo_args.viscosity_pa_s,
            e_prime_pa=e_prime,
            height_m=demo_args.height_m,
            t_seconds=final_time,
        )
        durations.append(time.perf_counter() - start)

    return {
        "enkf_step_p50_ms": percentile_ms(durations, 50),
        "enkf_step_p95_ms": percentile_ms(durations, 95),
        "enkf_step_max_ms": percentile_ms(durations, 100),
    }


def run_one_model(model_name: str, args: argparse.Namespace) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    demo_args = make_demo_args(args, model_name)
    start = time.perf_counter()
    result = run_demo(demo_args)
    timeline_total_seconds = time.perf_counter() - start
    table = result["table"]
    history = result["history"]
    single = benchmark_single_forward(model_name, table, demo_args, args.repeats)
    step = benchmark_enkf_step(model_name, table, demo_args, args.repeats)
    metrics = result["metrics"]
    instantiated_model = build_length_forward_model(model_name)
    row = {
        "model": model_name,
        "model_name": instantiated_model.model_name,
        **single,
        **step,
        "timeline_total_seconds": float(timeline_total_seconds),
        "time_steps": int(history["step"].nunique() if "step" in history else len(history)),
        "within_15_percent_rate": float(metrics["within_15_percent_rate"]),
        "final_prior_error": float(metrics["final_prior_error"]),
        "final_posterior_error": float(metrics["final_posterior_error"]),
        "final_within_15_percent": bool(metrics["final_within_15_percent"]),
    }
    if hasattr(instantiated_model, "n_panels"):
        row["bem_panels_per_fracture"] = int(instantiated_model.n_panels)
        row["bem_max_iterations"] = int(instantiated_model.max_iterations)
    if hasattr(instantiated_model, "validation_metrics"):
        validation = instantiated_model.validation_metrics
        row["surrogate_validation_scenarios"] = int(validation["validation_scenarios"])
        row["surrogate_half_length_mape"] = float(validation["half_length_mape"])
        row["surrogate_half_length_p95_relative_error"] = float(validation["half_length_p95_relative_error"])
        row["surrogate_aperture_mape"] = float(validation["aperture_mape"])
    row["single_forward_pass_15s"] = bool(row["single_forward_p95_ms"] < 15000.0)
    row["enkf_step_pass_15s"] = bool(row["enkf_step_p95_ms"] < 15000.0)
    row["pass_15s"] = bool(row["single_forward_pass_15s"] and row["enkf_step_pass_15s"])
    return row, table, history


def plot_runtime(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(frame))
    width = 0.35
    ax.bar(x - width / 2, frame["single_forward_p95_ms"], width, label="单次正演 P95", color="#4C78A8")
    ax.bar(x + width / 2, frame["enkf_step_p95_ms"], width, label="EnKF单步 P95", color="#F58518")
    ax.axhline(15000.0, color="#D62728", linestyle="--", linewidth=1.5, label="15秒指标")
    ax.set_xticks(x, frame["model"])
    ax.set_ylabel("耗时 ms")
    ax.set_title("正演模型计算时间对比")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_tradeoff(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for _, row in frame.iterrows():
        ax.scatter(row["enkf_step_p95_ms"], row["final_posterior_error"] * 100.0, s=130)
        ax.text(row["enkf_step_p95_ms"], row["final_posterior_error"] * 100.0, f" {row['model']}", va="center")
    ax.axhline(15.0, color="#D62728", linestyle="--", linewidth=1.4, label="15%误差目标")
    ax.axvline(15000.0, color="#9467BD", linestyle="--", linewidth=1.4, label="15秒计算目标")
    ax.set_xlabel("EnKF单步 P95 耗时 ms")
    ax.set_ylabel("最终后验误差 %")
    ax.set_title("精度-计算时间折中")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pkn_validation(frame: pd.DataFrame, path: Path) -> None:
    pkn = frame[frame["model"] == "pkn4"]
    if pkn.empty:
        return
    row = pkn.iloc[0]
    labels = ["单次正演P95", "EnKF单步P95"]
    values = [row["single_forward_p95_ms"], row["enkf_step_p95_ms"]]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    colors = ["#54A24B" if value < 15000.0 else "#E45756" for value in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(15000.0, color="#D62728", linestyle="--", linewidth=1.4, label="15秒指标")
    ax.set_ylabel("耗时 ms")
    ax.set_title("PKN 15秒计算指标验证")
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.2f} ms", ha="center", va="bottom")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_best_demo(best_model: str, args: argparse.Namespace, run_root: Path) -> str:
    script = ROOT / "inversion" / "playback.py"
    cmd = [
        sys.executable,
        str(script),
        "--forward-model",
        best_model,
        "--frac-monitor-text",
        args.frac_monitor_text,
        "--well-trajectory-csv",
        args.well_trajectory_csv,
        "--use-real-monitor-timeline",
        "--max-playback-steps",
        str(args.max_playback_steps),
        "--default-step-seconds",
        str(args.default_step_seconds),
        "--n-clusters",
        str(args.n_clusters),
        "--ensemble-size",
        str(args.ensemble_size),
        "--visual-aperture-gain",
        str(args.visual_aperture_gain),
        "--run-dir",
        str(run_root / "best_model_demo"),
    ]
    if args.construction_pressure_xls:
        cmd.extend(["--construction-pressure-xls", args.construction_pressure_xls])
    if args.open:
        cmd.append("--open")
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return completed.stdout


def main() -> None:
    args = build_arg_parser().parse_args()
    run_root = Path(args.run_dir).resolve() / time.strftime("%Y%m%d_%H%M%S")
    figures = run_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    rows = []
    model_tables: dict[str, pd.DataFrame] = {}
    model_histories: dict[str, pd.DataFrame] = {}
    for model_name in args.models:
        row, table, history = run_one_model(model_name, args)
        rows.append(row)
        model_tables[model_name] = table
        model_histories[model_name] = history
        table.to_csv(run_root / f"{model_name}_summary_all_time_steps.csv", index=False, encoding="utf-8-sig")
        history.to_csv(run_root / f"{model_name}_enkf_history.csv", index=False, encoding="utf-8-sig")
        print(json.dumps(row, ensure_ascii=False))

    summary_frame = pd.DataFrame(rows)
    summary_frame.to_csv(run_root / "benchmark_summary.csv", index=False, encoding="utf-8-sig")
    plot_runtime(summary_frame, figures / "model_runtime_comparison.png")
    plot_tradeoff(summary_frame, figures / "accuracy_runtime_tradeoff.png")
    plot_pkn_validation(summary_frame, figures / "pkn_15s_validation.png")

    best = summary_frame.sort_values(["pass_15s", "within_15_percent_rate", "final_posterior_error"], ascending=[False, False, True]).iloc[0]
    rendered_stdout = None
    if args.render_best_demo:
        rendered_stdout = render_best_demo(str(best["model"]), args, run_root)

    payload = {
        "demo": "forward_model_15s_benchmark",
        "run_root": str(run_root),
        "models": args.models,
        "criteria": {
            "single_forward_p95_ms": "< 15000",
            "enkf_step_p95_ms": "< 15000",
            "error_target": "posterior observation error <= 15%",
            "render_time_excluded": True,
        },
        "best_model": str(best["model"]),
        "pkn_pass_15s": bool(summary_frame.loc[summary_frame["model"] == "pkn4", "pass_15s"].iloc[0]) if "pkn4" in set(summary_frame["model"]) else None,
        "summary": summary_frame.to_dict(orient="records"),
        "outputs": {
            "benchmark_summary_csv": str(run_root / "benchmark_summary.csv"),
            "benchmark_summary_json": str(run_root / "benchmark_summary.json"),
        },
        "figures": {
            "model_runtime_comparison": str(figures / "model_runtime_comparison.png"),
            "accuracy_runtime_tradeoff": str(figures / "accuracy_runtime_tradeoff.png"),
            "pkn_15s_validation": str(figures / "pkn_15s_validation.png"),
        },
        "render_best_demo_stdout": rendered_stdout,
        "args": vars(args),
    }
    (run_root / "benchmark_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
