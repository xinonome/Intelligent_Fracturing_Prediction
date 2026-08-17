"""Run a six-cluster PyFrac snapshot with external stress-shadow coupling.

This is an offline feasibility path, not a claim that PyFrac natively solves
the complete horizontal-well six-cluster problem.  Each cluster receives its
measured fiber liquid allocation, runs one planar PyFrac snapshot, and then a
small external stress-shadow matrix produces the coupled display length.
"""

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

from data_fusion.frac_monitor_text_adapter import load_frac_monitor_text  # noqa: E402
from data_fusion.pressure_schedule_adapter import PressureModelConfig, load_stage_pressure_schedule  # noqa: E402
from data_fusion.well_trajectory_adapter import load_well_trajectory  # noqa: E402
from forward_models.pyfrac_adapter import PyFracAdapter  # noqa: E402
from forward_models.pyfrac_config import PyFracConfig  # noqa: E402


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    monitor = load_frac_monitor_text(args.fiber_monitor, default_step_seconds=args.default_step_seconds)
    controls = monitor.controls.copy()
    if args.stage is not None and "stage" in monitor.stage_info:
        stage_text = str(args.stage).lstrip("0") or "0"
        mask = monitor.stage_info["stage"].astype(str).str.lstrip("0").replace("", "0") == stage_text
        if mask.any():
            steps = set(monitor.stage_info.loc[mask, "step"].astype(int))
            controls = controls[controls["step"].isin(steps)].copy()
    if controls.empty:
        raise SystemExit("No fiber controls found for the requested stage")

    pressure = None
    pressure_meta = None
    if args.pressure_xls:
        pressure, pressure_meta = load_stage_pressure_schedule(
            args.pressure_xls,
            config=PressureModelConfig(step_seconds=args.default_step_seconds),
        )
    trajectory = load_well_trajectory(args.trajectory_csv) if args.trajectory_csv else pd.DataFrame()
    timeline = build_timeline(controls, pressure, args.default_step_seconds)
    selected = timeline.iloc[select_indices(len(timeline), args.max_points)].reset_index(drop=True)
    cluster_ids = sorted(int(v) for v in controls["cluster_id"].dropna().unique())
    adapter = PyFracAdapter(
        PyFracConfig(
            height_m=args.height_m,
            viscosity_pa_s=args.viscosity_pa_s,
            min_horizontal_stress_pa=args.min_stress_mpa * 1.0e6,
        ),
        project_root=ROOT,
    )

    rows: list[dict[str, object]] = []
    for _, item in selected.iterrows():
        allocation = np.asarray([item[f"allocation_c{cluster}"] for cluster in cluster_ids], dtype=float)
        q_total = float(item["q_total_m3_s"])
        q_clusters = q_total * allocation
        raw_lengths: list[float] = []
        raw_results = []
        for cluster, q_cluster in zip(cluster_ids, q_clusters):
            result = adapter.run(
                injection_rate_m3_s=float(q_cluster),
                time_s=max(float(item["time_s"]), 1.0),
                mode=args.pyfrac_mode,
                height_m=args.height_m,
                viscosity_pa_s=args.viscosity_pa_s,
                e_prime_pa=adapter.config.e_prime_pa,
                min_horizontal_stress_pa=args.min_stress_mpa * 1.0e6,
            )
            raw_results.append(result)
            raw_lengths.append(result.half_length_m if result.success else np.nan)

        raw = np.nan_to_num(np.asarray(raw_lengths, dtype=float), nan=0.0)
        distance = np.abs(np.arange(len(cluster_ids))[:, None] - np.arange(len(cluster_ids))[None, :])
        shadow_kernel = np.exp(-distance / max(args.shadow_decay_clusters, 1e-6))
        np.fill_diagonal(shadow_kernel, 0.0)
        shadow_kernel /= np.maximum(shadow_kernel.sum(axis=1, keepdims=True), 1.0e-12)
        neighbor_length = shadow_kernel @ raw
        scale = neighbor_length / max(float(np.mean(raw)), 1.0e-9) if raw.any() else np.zeros_like(raw)
        coupled = raw * np.clip(1.0 - args.stress_shadow_strength * np.tanh(scale), 0.75, 1.0)
        for index, (cluster, q_cluster, allocation_weight, result) in enumerate(
            zip(cluster_ids, q_clusters, allocation, raw_results)
        ):
            rows.append(
                {
                    "step": int(item["step"]),
                    "time_s": float(item["time_s"]),
                    "cluster_id": cluster,
                    "q_total_m3_s": q_total,
                    "q_cluster_m3_s": float(q_cluster),
                    "fiber_liquid_allocation": float(allocation_weight),
                    "pyfrac_half_length_m": float(raw[index]) if result.success else np.nan,
                    "coupled_half_length_m": float(coupled[index]) if result.success else np.nan,
                    "pyfrac_max_aperture_mm": result.max_aperture_mm,
                    "pyfrac_net_pressure_mpa": result.net_pressure_mpa,
                    "pyfrac_success": bool(result.success),
                    "pyfrac_error": result.error or "",
                    "pyfrac_runtime_seconds": result.runtime_seconds,
                    "stress_shadow_factor": float(coupled[index] / raw[index]) if result.success and raw[index] > 0 else np.nan,
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "multicluster_predictions.csv", index=False, encoding="utf-8-sig")
    write_figure(frame, output / "multicluster_length_comparison.png")
    summary = {
        "demo": "stage08_six_cluster_pyfrac_external_coupling",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": time.perf_counter() - started,
        "stage": args.stage,
        "cluster_ids": cluster_ids,
        "timeline_points": int(len(selected)),
        "rows": int(len(frame)),
        "pyfrac_success_rate": float(frame["pyfrac_success"].mean()) if not frame.empty else 0.0,
        "total_rate_conservation_error": float(
            np.max(np.abs(frame.groupby("step")["q_cluster_m3_s"].sum() - frame.groupby("step")["q_total_m3_s"].first()))
        ) if not frame.empty else None,
        "inputs": {
            "fiber_monitor": str(Path(args.fiber_monitor).resolve()),
            "pressure_xls": str(Path(args.pressure_xls).resolve()) if args.pressure_xls else None,
            "trajectory_csv": str(Path(args.trajectory_csv).resolve()) if args.trajectory_csv else None,
            "pressure_meta": pressure_meta,
            "trajectory_rows": int(len(trajectory)),
        },
        "coupling": {
            "method": "six independent planar PyFrac snapshots plus external stress-shadow matrix",
            "stress_shadow_strength": args.stress_shadow_strength,
            "shadow_decay_clusters": args.shadow_decay_clusters,
            "fiber_liquid_allocation_is_boundary_condition": True,
        },
        "pyfrac_installation": adapter.verify_installation(),
        "outputs": {
            "predictions": str(output / "multicluster_predictions.csv"),
            "figure": str(output / "multicluster_length_comparison.png"),
        },
        "limitations": [
            "该路径不是PyFrac原生六簇水平井求解，而是单簇平面模型加外部耦合验证。",
            "砂量没有作为PyFrac内部支撑剂输运变量，只保留在上游观测链路。",
            "没有真实裂缝几何真值时，缝长仍是模型参考量，不是现场精度标签。",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_timeline(controls: pd.DataFrame, pressure: pd.DataFrame | None, step_seconds: float) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    grouped = controls.groupby("step", sort=True)
    all_steps = sorted(int(v) for v in controls["step"].unique())
    t0 = pd.to_datetime(controls["time"], errors="coerce").min()
    previous_cumulative: dict[int, float] = {}
    for step in all_steps:
        current = grouped.get_group(step).sort_values("cluster_id")
        ids = current["cluster_id"].astype(int).to_numpy()
        cumulative = current["cumulative_liquid_volume_m3"].astype(float).fillna(0.0).to_numpy()
        increment = current["liquid_volume_m3"].astype(float).fillna(0.0).clip(lower=0.0).to_numpy()
        if float(increment.sum()) <= 1.0e-12:
            increment = np.asarray([max(c - previous_cumulative.get(int(cid), 0.0), 0.0) for c, cid in zip(cumulative, ids)])
        previous_cumulative.update({int(cid): float(value) for cid, value in zip(ids, cumulative)})
        if float(increment.sum()) <= 1.0e-12:
            increment = np.ones(len(ids), dtype=float)
        allocation = increment / max(float(increment.sum()), 1.0e-12)
        time_value = pd.to_datetime(current.iloc[0].get("time"), errors="coerce")
        time_s = float((time_value - t0).total_seconds()) if pd.notna(time_value) and pd.notna(t0) else step * step_seconds
        if pressure is not None and not pressure.empty:
            progress = step / max(float(all_steps[-1]), 1.0)
            axis = np.linspace(0.0, 1.0, len(pressure))
            q_total = float(np.interp(progress, axis, pressure["flow_rate_m3_s"].to_numpy(dtype=float)))
        else:
            q_total = float(increment.sum()) / max(step_seconds, 1.0e-9)
        row: dict[str, float] = {"step": float(step), "time_s": max(time_s, 0.0), "q_total_m3_s": max(q_total, 1.0e-9)}
        for cluster, weight in zip(ids, allocation):
            row[f"allocation_c{int(cluster)}"] = float(weight)
        rows.append(row)
    return pd.DataFrame(rows)


def write_figure(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for cluster, group in frame.groupby("cluster_id"):
        group = group.sort_values("time_s")
        ax.plot(group["time_s"], group["pyfrac_half_length_m"], "--", alpha=0.55, label=f"C{cluster} PyFrac")
        ax.plot(group["time_s"], group["coupled_half_length_m"], linewidth=2, label=f"C{cluster} coupled")
    ax.set_xlabel("相对时间 (s)")
    ax.set_ylabel("半缝长 (m)")
    ax.set_title("第8段六簇：真实光纤液量分配 + PyFrac单簇快照 + 外部应力阴影")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def select_indices(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, maximum).round().astype(int))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fiber-monitor", required=True)
    parser.add_argument("--pressure-xls")
    parser.add_argument("--trajectory-csv")
    parser.add_argument("--stage", type=int, default=8)
    parser.add_argument("--pyfrac-mode", choices=["snapshot", "native"], default="snapshot")
    parser.add_argument("--output-dir", default="outputs/dt/pyfrac_multicluster")
    parser.add_argument("--max-points", type=int, default=6)
    parser.add_argument("--default-step-seconds", type=float, default=10.0)
    parser.add_argument("--height-m", type=float, default=30.0)
    parser.add_argument("--viscosity-pa-s", type=float, default=0.1)
    parser.add_argument("--min-stress-mpa", type=float, default=60.0)
    parser.add_argument("--stress-shadow-strength", type=float, default=0.08)
    parser.add_argument("--shadow-decay-clusters", type=float, default=1.25)
    return parser.parse_args()


if __name__ == "__main__":
    main()
