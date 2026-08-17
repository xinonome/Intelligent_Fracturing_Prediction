"""Run a reproducible Stage-08 PKN versus PyFrac single-cluster comparison.

Examples
--------
PowerShell::

    python DT-Crack\\forward_models\\run_pyfrac_single_cluster.py `
      --fiber-monitor Data\\3Dfrac\\光纤本井监测08.txt `
      --pressure-xls Data\\3Dfrac\\JY84-Z1-stage08-f1.xls `
      --trajectory-csv Data\\3Dfrac\\JY84-Z1HF-1011.csv `
      --stage 8 --cluster 1 --model pyfrac --max-points 12 `
      --output-dir outputs\\dt\\pyfrac_single_cluster

Use ``--model pkn`` to generate only the PKN baseline. Use ``--model pyfrac``
to generate both the PKN baseline and the requested PyFrac reference.

The default ``snapshot`` mode invokes PyFrac's Cartesian mesh and PKN
initialization independently at each selected time.  It is a quick, explicit
model-space reference.  ``--pyfrac-mode native`` attempts PyFrac time
marching and records failures instead of silently falling back.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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

from data_fusion.frac_monitor_text_adapter import load_frac_monitor_text
from data_fusion.pressure_schedule_adapter import PressureModelConfig, load_stage_pressure_schedule
from data_fusion.well_trajectory_adapter import load_well_trajectory
from forward_models.fracture_length_models import calc_pkn
from forward_models.pyfrac_adapter import PyFracAdapter
from forward_models.pyfrac_config import PyFracConfig


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    fiber = load_frac_monitor_text(args.fiber_monitor, default_step_seconds=args.default_step_seconds)
    controls = fiber.controls.copy()
    if controls.empty:
        raise SystemExit("Fiber monitor text produced no cluster controls")
    if args.stage is not None and "stage" in fiber.stage_info:
        stage_text = str(args.stage).lstrip("0") or "0"
        mask = fiber.stage_info["stage"].astype(str).str.lstrip("0").replace("", "0") == stage_text
        if mask.any():
            selected_steps = set(fiber.stage_info.loc[mask, "step"].astype(int))
            controls = controls[controls["step"].isin(selected_steps)].copy()
    if controls.empty:
        raise SystemExit(f"No controls found for stage={args.stage}")

    cluster_ids = sorted(int(v) for v in controls["cluster_id"].dropna().unique())
    if args.cluster not in cluster_ids:
        raise SystemExit(f"Cluster {args.cluster} is not available; found {cluster_ids}")

    pressure = None
    pressure_meta: dict[str, object] | None = None
    if args.pressure_xls:
        pressure, pressure_meta = load_stage_pressure_schedule(
            args.pressure_xls,
            config=PressureModelConfig(step_seconds=args.default_step_seconds),
        )
    trajectory = load_well_trajectory(args.trajectory_csv) if args.trajectory_csv else pd.DataFrame()
    timeline = build_timeline(controls, args.cluster, pressure, args.default_step_seconds)
    if timeline.empty:
        raise SystemExit("Could not build a usable comparison timeline")
    if args.min_time_s > 0:
        timeline = timeline[timeline["time_s"] >= args.min_time_s].copy()
        if timeline.empty:
            raise SystemExit(f"No timeline points at or after min-time-s={args.min_time_s}")
    selected = timeline.iloc[_select_indices(len(timeline), args.max_points)].reset_index(drop=True)

    pyfrac_config = PyFracConfig(
        pyfrac_root=str(args.pyfrac_root) if args.pyfrac_root else "",
        height_m=args.height_m,
        viscosity_pa_s=args.viscosity_pa_s,
        min_horizontal_stress_pa=args.min_stress_mpa * 1.0e6,
        mesh_nx=args.mesh_nx,
        mesh_ny=args.mesh_ny,
        native_start_time_s=args.native_start_time_s,
        max_time_steps=args.max_time_steps,
    )
    adapter = PyFracAdapter(pyfrac_config, project_root=ROOT)
    pyfrac_rows: list[dict[str, object]] = []
    pkn_rows: list[dict[str, object]] = []
    run_pyfrac = args.model == "pyfrac"
    for _, item in selected.iterrows():
        time_s = max(float(item["time_s"]), 1.0)
        q_cluster = max(float(item["cluster_flow_m3_s"]), 1.0e-8)
        w_pkn, length_pkn = calc_pkn(
            np.asarray([q_cluster]),
            args.viscosity_pa_s,
            pyfrac_config.e_prime_pa,
            args.height_m,
            time_s,
        )
        pkn_rows.append(
            {
                "step": int(item["step"]),
                "time_s": time_s,
                "cluster_id": args.cluster,
                "q_total_m3_s": float(item["q_total_m3_s"]),
                "q_cluster_m3_s": q_cluster,
                "liquid_allocation": float(item["allocation_weight"]),
                "pkn_half_length_m": float(length_pkn[0]),
                "pkn_max_aperture_mm": float(2.0 * w_pkn[0] * 1000.0),
                "bottomhole_pressure_mpa": float(item["bottomhole_pressure_mpa"]),
                "net_pressure_mpa": float(item["net_pressure_mpa"]),
            }
        )
        if run_pyfrac:
            result = adapter.run(
                injection_rate_m3_s=q_cluster,
                time_s=time_s,
                mode=args.pyfrac_mode,
                height_m=args.height_m,
                viscosity_pa_s=args.viscosity_pa_s,
                e_prime_pa=pyfrac_config.e_prime_pa,
                min_horizontal_stress_pa=args.min_stress_mpa * 1.0e6,
            )
        else:
            result = None
        pyfrac_rows.append(
            {
                "step": int(item["step"]),
                "time_s": time_s,
                "cluster_id": args.cluster,
                "q_cluster_m3_s": q_cluster,
                "pyfrac_half_length_m": result.half_length_m if result else float("nan"),
                "pyfrac_max_aperture_mm": result.max_aperture_mm if result else float("nan"),
                "pyfrac_area_m2": result.area_m2 if result else float("nan"),
                "pyfrac_volume_m3": result.volume_m3 if result else float("nan"),
                "pyfrac_net_pressure_mpa": result.net_pressure_mpa if result else float("nan"),
                "pyfrac_bottomhole_pressure_mpa": result.bottomhole_pressure_mpa if result else float("nan"),
                "pyfrac_runtime_seconds": result.runtime_seconds if result else float("nan"),
                "pyfrac_final_time_s": result.final_time_s if result else float("nan"),
                "pyfrac_successful_time_steps": result.successful_time_steps if result else 0,
                "pyfrac_failed_time_steps": result.failed_time_steps if result else 0,
                "pyfrac_target_reached": result.target_reached if result else False,
                "pyfrac_engine_mode": result.engine_mode if result else "not_requested",
                "pyfrac_success": bool(result.success) if result else False,
                "pyfrac_error": (result.error or "") if result else "model=pkn; PyFrac not requested",
                "front_geometry_json": json.dumps(result.front_geometry, ensure_ascii=False) if result else "[]",
            }
        )

    pkn_frame = pd.DataFrame(pkn_rows)
    pyfrac_frame = pd.DataFrame(pyfrac_rows)
    comparison = pkn_frame.merge(pyfrac_frame, on=["step", "time_s", "cluster_id", "q_cluster_m3_s"], how="left")
    comparison["length_relative_error"] = _relative_error(
        comparison["pyfrac_half_length_m"], comparison["pkn_half_length_m"]
    )
    comparison["aperture_relative_error"] = _relative_error(
        comparison["pyfrac_max_aperture_mm"], comparison["pkn_max_aperture_mm"]
    )
    comparison["pressure_relative_error"] = _relative_error(
        comparison["pyfrac_net_pressure_mpa"], comparison["net_pressure_mpa"]
    )

    pkn_frame.to_csv(output_dir / "pkn_predictions.csv", index=False, encoding="utf-8-sig")
    pyfrac_frame.to_csv(output_dir / "pyfrac_predictions.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output_dir / "single_cluster_comparison.csv", index=False, encoding="utf-8-sig")
    write_figures(output_dir, comparison)

    success = comparison["pyfrac_success"].astype(bool)
    metrics = {
        "cluster_id": args.cluster,
        "stage": args.stage,
        "model": args.model,
        "pyfrac_mode": args.pyfrac_mode if run_pyfrac else None,
        "timeline_points": int(len(comparison)),
        "pyfrac_success_points": int(success.sum()),
        "pyfrac_success_rate": float(success.mean()),
        "length_mean_relative_error": _safe_mean(comparison.loc[success, "length_relative_error"]),
        "length_p95_relative_error": _safe_percentile(comparison.loc[success, "length_relative_error"], 95),
        "aperture_mean_relative_error": _safe_mean(comparison.loc[success, "aperture_relative_error"]),
        "pressure_mean_relative_error": _safe_mean(comparison.loc[success, "pressure_relative_error"]),
        "pkn_vs_pyfrac_is_model_difference": run_pyfrac,
        "not_real_fracture_length_accuracy": True,
    }
    (output_dir / "single_cluster_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "demo": "stage08_single_cluster_pkn_pyfrac_comparison" if run_pyfrac else "stage08_single_cluster_pkn_baseline",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": time.perf_counter() - started,
        "inputs": {
            "fiber_monitor": str(Path(args.fiber_monitor).resolve()),
            "pressure_xls": str(Path(args.pressure_xls).resolve()) if args.pressure_xls else None,
            "trajectory_csv": str(Path(args.trajectory_csv).resolve()) if args.trajectory_csv else None,
            "fiber_meta": fiber.meta,
            "pressure_meta": pressure_meta,
            "trajectory_rows": int(len(trajectory)),
        },
        "config": asdict(pyfrac_config),
        "pyfrac_installation": adapter.verify_installation(),
        "metrics": metrics,
        "outputs": {
            "pkn_predictions": str(output_dir / "pkn_predictions.csv"),
            "pyfrac_predictions": str(output_dir / "pyfrac_predictions.csv"),
            "comparison": str(output_dir / "single_cluster_comparison.csv"),
            "metrics": str(output_dir / "single_cluster_metrics.json"),
        },
        "limitations": [
            "当前光纤文本没有独立裂缝半长真值，PKN/PyFrac误差是模型间差异，不是现场几何误差。",
            "PyFrac snapshot 模式是网格化PKN初始化，不等同于完整时间推进。",
            "砂量作为外部观测约束，未作为PyFrac内部支撑剂输运变量。",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_timeline(
    controls: pd.DataFrame,
    cluster_id: int,
    pressure: pd.DataFrame | None,
    default_step_seconds: float,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    grouped = controls.groupby("step", sort=True)
    all_steps = sorted(int(v) for v in controls["step"].unique())
    t0 = pd.to_datetime(controls["time"], errors="coerce").min()
    for step in all_steps:
        step_df = grouped.get_group(step).sort_values("cluster_id")
        selected = step_df[step_df["cluster_id"] == cluster_id]
        if selected.empty:
            continue
        row = selected.iloc[0]
        cumulative = step_df["cumulative_liquid_volume_m3"].astype(float).fillna(0.0).to_numpy()
        increments = step_df["liquid_volume_m3"].astype(float).fillna(0.0).clip(lower=0.0).to_numpy()
        if increments.sum() <= 1.0e-12:
            previous_step = controls[controls["step"] < step]
            if not previous_step.empty:
                previous = previous_step.sort_values("step").groupby("cluster_id").tail(1)
                previous_map = previous.set_index("cluster_id")["cumulative_liquid_volume_m3"].to_dict()
                increments = np.asarray(
                    [max(float(c) - float(previous_map.get(int(cid), 0.0)), 0.0) for c, cid in zip(cumulative, step_df["cluster_id"])]
                )
        allocation = float(increments[list(step_df["cluster_id"]).index(cluster_id)] / max(increments.sum(), 1.0e-12))
        if increments.sum() <= 1.0e-12:
            allocation = float(row.get("allocation_weight", 1.0 / max(len(step_df), 1)))
        time_value = pd.to_datetime(row.get("time"), errors="coerce")
        time_s = float((time_value - t0).total_seconds()) if pd.notna(time_value) and pd.notna(t0) else step * default_step_seconds
        fiber_q = float(increments.sum()) / max(default_step_seconds, 1.0e-9)
        if pressure is not None and not pressure.empty:
            progress = step / max(float(all_steps[-1]), 1.0)
            pressure_progress = np.linspace(0.0, 1.0, len(pressure))
            q_total = float(np.interp(progress, pressure_progress, pressure["flow_rate_m3_s"].to_numpy(dtype=float)))
            bottomhole = float(np.interp(progress, pressure_progress, pressure["bottomhole_pressure_mpa"].to_numpy(dtype=float)))
            net = float(np.interp(progress, pressure_progress, pressure["net_pressure_mpa"].to_numpy(dtype=float)))
        else:
            q_total, bottomhole, net = fiber_q, float("nan"), float("nan")
        if q_total <= 1.0e-9:
            q_total = fiber_q
        rows.append(
            {
                "step": step,
                "time_s": max(time_s, 0.0),
                "q_total_m3_s": q_total,
                "cluster_flow_m3_s": q_total * allocation,
                "allocation_weight": allocation,
                "bottomhole_pressure_mpa": bottomhole,
                "net_pressure_mpa": net,
            }
        )
    return pd.DataFrame(rows)


def write_figures(output_dir: Path, comparison: pd.DataFrame) -> None:
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180})
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(comparison["time_s"], comparison["pkn_half_length_m"], "o-", label="PKN", linewidth=2)
    axes[0].plot(comparison["time_s"], comparison["pyfrac_half_length_m"], "o-", label="PyFrac", linewidth=2)
    failed = comparison[~comparison["pyfrac_success"].astype(bool)]
    if not failed.empty:
        axes[0].plot(
            failed["time_s"],
            failed["pkn_half_length_m"],
            "x",
            color="crimson",
            markersize=9,
            label="PyFrac未完成点",
        )
    axes[0].set_ylabel("半缝长 (m)")
    axes[0].set_title("第8段单簇 PKN / PyFrac 模型空间对照")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(comparison["time_s"], comparison["pkn_max_aperture_mm"], "o-", label="PKN", linewidth=2)
    axes[1].plot(comparison["time_s"], comparison["pyfrac_max_aperture_mm"], "o-", label="PyFrac", linewidth=2)
    if not failed.empty:
        axes[1].plot(
            failed["time_s"],
            failed["pkn_max_aperture_mm"],
            "x",
            color="crimson",
            markersize=9,
            label="PyFrac未完成点",
        )
    axes[1].set_xlabel("相对时间 (s)")
    axes[1].set_ylabel("最大开度 (mm)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "single_cluster_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    final = comparison.iloc[-1]
    try:
        points = np.asarray(json.loads(final["front_geometry_json"]), dtype=float)
        if points.ndim == 2 and points.shape[1] >= 2:
            ax.plot(points[:, 0], points[:, 1], "o-", label="PyFrac front")
    except Exception:
        pass
    ax.set_title("PyFrac 单簇最终前缘（投影）")
    ax.set_xlabel("裂缝长度方向 x (m)")
    ax.set_ylabel("高度方向 y (m)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fracture_front_comparison.png")
    plt.close(fig)


def _relative_error(reference: pd.Series, estimate: pd.Series) -> pd.Series:
    ref = pd.to_numeric(reference, errors="coerce").abs()
    est = pd.to_numeric(estimate, errors="coerce").abs()
    return (ref - est).abs() / ref.clip(lower=1.0e-9)


def _safe_mean(values: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _safe_percentile(values: pd.Series, percentile: float) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(np.percentile(values, percentile)) if not values.empty else None


def _select_indices(length: int, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, max_points).round().astype(int))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fiber-monitor", required=True)
    parser.add_argument("--pressure-xls")
    parser.add_argument("--trajectory-csv")
    parser.add_argument("--stage", type=int, default=8)
    parser.add_argument("--cluster", type=int, default=1)
    parser.add_argument("--model", default="pyfrac", choices=["pkn", "pyfrac"])
    parser.add_argument("--pyfrac-mode", choices=["snapshot", "native"], default="snapshot")
    parser.add_argument("--pyfrac-root")
    parser.add_argument("--output-dir", default="outputs/dt/pyfrac_single_cluster")
    parser.add_argument("--max-points", type=int, default=12)
    parser.add_argument(
        "--min-time-s",
        type=float,
        default=0.0,
        help="Only evaluate timeline points at or after this relative time; useful for native warm-start runs.",
    )
    parser.add_argument("--default-step-seconds", type=float, default=10.0)
    parser.add_argument("--height-m", type=float, default=30.0)
    parser.add_argument("--viscosity-pa-s", type=float, default=0.1)
    parser.add_argument("--min-stress-mpa", type=float, default=60.0)
    parser.add_argument("--mesh-nx", type=int, default=61)
    parser.add_argument("--mesh-ny", type=int, default=41)
    parser.add_argument("--native-start-time-s", type=float, default=120.0)
    parser.add_argument(
        "--max-time-steps",
        type=int,
        default=30,
        help="Maximum native PyFrac time steps; increase for long offline reference runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
