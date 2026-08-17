"""Benchmark PKN, offline PyFrac and the online residual surrogate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DT_ROOT = ROOT / "DT-Crack"
if str(DT_ROOT) not in sys.path:
    sys.path.insert(0, str(DT_ROOT))

from forward_models.fracture_length_models import calc_pkn
from forward_models.pyfrac_adapter import PyFracAdapter
from forward_models.pyfrac_config import PyFracConfig
from forward_models.pyfrac_surrogate import (
    FEATURE_COLUMNS,
    PyFracResidualSurrogate,
    generate_teacher_dataset,
)
from forward_models.pyfrac_surrogate import _group_values, _prepare_feature_frame, _strict_group_split


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_csv = output_dir / "teacher_samples.csv"
    if args.teacher_csv:
        teacher_csv = Path(args.teacher_csv).resolve()
    elif not teacher_csv.exists():
        generate_teacher_dataset(
            teacher_csv,
            samples=args.samples,
            seed=args.seed,
            pyfrac_mode=args.pyfrac_mode,
            steps_per_scenario=args.steps_per_scenario,
            n_clusters=args.n_clusters,
        )
    teacher = pd.read_csv(teacher_csv)
    surrogate = PyFracResidualSurrogate.train(teacher)
    non_regression = _evaluate_pkn_non_regression(surrogate, teacher)
    surrogate.metrics["non_regression_gate"] = bool(non_regression["pass"])
    surrogate.metrics["non_regression"] = non_regression
    surrogate_path = output_dir / "pyfrac_residual_surrogate.joblib"
    surrogate.save(surrogate_path)

    adapter = PyFracAdapter(PyFracConfig())
    benchmark_rows: list[dict[str, object]] = []
    for index, item in teacher.head(args.max_points).iterrows():
        pkn_started = time.perf_counter()
        pkn_w, pkn_l = calc_pkn(
            np.asarray([item["q_m3_s"]]),
            item["viscosity_pa_s"],
            item["e_prime_gpa"] * 1.0e9,
            item["height_m"],
            item["time_s"],
        )
        pkn_runtime = time.perf_counter() - pkn_started
        pyfrac_started = time.perf_counter()
        pyfrac = adapter.run(
            injection_rate_m3_s=float(item["q_m3_s"]),
            time_s=float(item["time_s"]),
            mode=args.pyfrac_mode,
            height_m=float(item["height_m"]),
            viscosity_pa_s=float(item["viscosity_pa_s"]),
            e_prime_pa=float(item["e_prime_gpa"] * 1.0e9),
            leakoff_coefficient_m_sqrt_s=float(item["leakoff_m_sqrt_s"]),
            min_horizontal_stress_pa=float(item["min_stress_mpa"] * 1.0e6),
            fracture_toughness_pa_sqrt_m=float(item["fracture_toughness_pa_sqrt_m"]),
        )
        pyfrac_wall_runtime = time.perf_counter() - pyfrac_started
        features = pd.DataFrame([item[FEATURE_COLUMNS].to_dict()])
        surrogate_started = time.perf_counter()
        delta = surrogate.predict_delta(features).iloc[0]
        surrogate_runtime = time.perf_counter() - surrogate_started
        benchmark_rows.append(
            {
                "sample": int(index),
                "pkn_runtime_ms": pkn_runtime * 1000.0,
                "pyfrac_runtime_ms": pyfrac_wall_runtime * 1000.0,
                "pyfrac_internal_runtime_ms": pyfrac.runtime_seconds * 1000.0,
                "surrogate_runtime_ms": surrogate_runtime * 1000.0,
                "pyfrac_success": pyfrac.success,
                "pkn_half_length_m": float(pkn_l[0]),
                "pyfrac_half_length_m": pyfrac.half_length_m,
                "surrogate_half_length_m": float(pkn_l[0] + delta["delta_length_m"]),
                "pyfrac_mode": args.pyfrac_mode,
            }
        )
    frame = pd.DataFrame(benchmark_rows)
    frame.to_csv(output_dir / "benchmark_samples.csv", index=False, encoding="utf-8-sig")
    successful = frame[frame["pyfrac_success"]].copy()
    runtime_p95 = _quantile(frame["surrogate_runtime_ms"], 0.95)
    quality_gate = surrogate.online_readiness(min_test_r2=args.min_test_r2)
    online_ready = bool(
        quality_gate["ready"]
        and runtime_p95 is not None
        and runtime_p95 < 15000.0
    )
    summary = {
        "teacher_csv": str(teacher_csv),
        "surrogate_path": str(surrogate_path),
        "pyfrac_mode": args.pyfrac_mode,
        "teacher_generation": {
            "scenario_groups": args.samples,
            "steps_per_scenario": args.steps_per_scenario,
            "clusters_per_scenario": args.n_clusters,
            "teacher_rows": int(len(teacher)),
            "expected_rows_if_all_snapshots_succeed": int(
                args.samples * args.steps_per_scenario * args.n_clusters
            ),
            "successful_row_rate": float(
                len(teacher)
                / max(args.samples * args.steps_per_scenario * args.n_clusters, 1)
            ),
            "scenario_split": "strict_group_holdout_70_15_15",
            "scenario_seed": args.seed,
        },
        "sample_count": int(len(frame)),
        "pyfrac_success_count": int(len(successful)),
        "runtime_ms": {
            "pkn_p50": _quantile(frame["pkn_runtime_ms"], 0.50),
            "pkn_p95": _quantile(frame["pkn_runtime_ms"], 0.95),
            "pyfrac_p50": _quantile(successful["pyfrac_runtime_ms"], 0.50),
            "pyfrac_p95": _quantile(successful["pyfrac_runtime_ms"], 0.95),
            "surrogate_p50": _quantile(frame["surrogate_runtime_ms"], 0.50),
            "surrogate_p95": _quantile(frame["surrogate_runtime_ms"], 0.95),
        },
        "surrogate_validation": surrogate.metrics,
        "online_15s_scope": "surrogate plus EnKF; PyFrac is offline teacher only",
        "online_enkf_approval": online_ready,
        "online_chain": {
            "default": "PKN + EnKF",
            "surrogate_role": "candidate residual correction; explicit path required",
            "pyfrac_role": "offline teacher only",
        },
        "online_quality_gate": quality_gate,
        "online_gate_targets": ["delta_length_m", "delta_pressure_mpa"],
        "pkn_non_regression": non_regression,
        "online_gate_reason": (
            "approved for online EnKF candidate integration"
            if online_ready
            else "not approved: expand teacher scenarios and improve held-out residual accuracy before online EnKF"
        ),
        "limitations": [
            "当前教师样本是单簇参数场景，不能替代真实裂缝几何标定。",
            "PyFrac native 动态模式的失败会保留在样本中，不会自动标成成功。",
            "本 benchmark 未把 Plotly/PyVista 渲染时间计入模型时间。",
        ],
    }
    (output_dir / "benchmark_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_plot(output_dir, frame)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def write_plot(output_dir: Path, frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    values = [
        frame["pkn_runtime_ms"].median(),
        frame["surrogate_runtime_ms"].median(),
        frame["pyfrac_runtime_ms"].dropna().median(),
    ]
    ax.bar(["PKN", "代理模型", "PyFrac"], values, color=["#2563eb", "#16a34a", "#dc2626"])
    ax.set_ylabel("单次计算时间 (ms)")
    ax.set_title("PKN / PyFrac / PyFrac残差代理模型耗时对比")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_comparison.png", dpi=180)
    plt.close(fig)


def _quantile(values: pd.Series, q: float) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.quantile(q)) if not values.empty else None


def _evaluate_pkn_non_regression(surrogate: PyFracResidualSurrogate, teacher: pd.DataFrame) -> dict[str, object]:
    frame = _prepare_feature_frame(teacher).dropna(subset=FEATURE_COLUMNS + ["delta_length_m", "delta_pressure_mpa"]).reset_index(drop=True)
    groups = _group_values(frame)
    _, _, test_mask, split = _strict_group_split(groups, random_state=20260810)
    actual = frame[["delta_length_m", "delta_pressure_mpa"]].to_numpy(dtype=float)[test_mask]
    predicted = surrogate.predict_delta(frame.loc[test_mask]).loc[:, ["delta_length_m", "delta_pressure_mpa"]].to_numpy(dtype=float)
    pkn_mae = np.mean(np.abs(actual), axis=0)
    surrogate_mae = np.mean(np.abs(actual - predicted), axis=0)
    names = ["delta_length_m", "delta_pressure_mpa"]
    values = {name: {"pkn_mae": float(pkn_mae[i]), "surrogate_mae": float(surrogate_mae[i]), "non_worse": bool(surrogate_mae[i] <= pkn_mae[i] + 1.0e-9)} for i, name in enumerate(names)}
    return {"pass": bool(all(item["non_worse"] for item in values.values())), "test_groups": split["test"], "targets": values, "criterion": "surrogate residual MAE <= PKN zero-residual baseline"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-csv")
    parser.add_argument("--output-dir", default="outputs/dt/pyfrac_benchmark")
    parser.add_argument("--samples", type=int, default=96, help="number of PyFrac scenario groups")
    parser.add_argument("--steps-per-scenario", type=int, default=4)
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--min-test-r2", type=float, default=0.80)
    parser.add_argument("--max-points", type=int, default=24)
    parser.add_argument("--ensemble-size", type=int, default=40)
    parser.add_argument("--pyfrac-mode", choices=["snapshot", "native"], default="snapshot")
    return parser.parse_args()


if __name__ == "__main__":
    main()
