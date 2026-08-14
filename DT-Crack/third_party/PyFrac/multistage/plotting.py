from __future__ import annotations

from pathlib import Path

import numpy as np


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_stage(result, stage_dir: Path) -> None:
    plt = _plt()
    metrics = result.metrics
    last = result.snapshots[-1]
    front = np.asarray(last.front_coordinates_local, dtype=float)
    if front.size:
        fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(front[:, 0], front[:, 1], s=5); ax.set_xlabel("u (m)"); ax.set_ylabel("v (m)"); ax.set_title("Final fracture footprint"); fig.tight_layout(); fig.savefig(stage_dir / "footprint_final.png", dpi=140); plt.close(fig)
    for column, title, filename in (("max_width_m", "Width", "width_final.png"), ("max_pressure_pa", "Pressure", "pressure_final.png")):
        fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(metrics["time_s"], metrics[column]); ax.set_xlabel("time (s)"); ax.set_ylabel(column); ax.set_title(title); fig.tight_layout(); fig.savefig(stage_dir / filename, dpi=140); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True); axes[0].plot(metrics.time_s, metrics.half_length_m, label="half length"); axes[0].plot(metrics.time_s, metrics.full_height_m, label="full height"); axes[0].legend(); axes[1].plot(metrics.time_s, metrics.aspect_ratio); axes[1].set_ylabel("aspect ratio"); axes[1].set_xlabel("time (s)"); fig.tight_layout(); fig.savefig(stage_dir / "geometry_vs_time.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(metrics.time_s, metrics.aspect_ratio); ax.set_xlabel("time (s)"); ax.set_ylabel("aspect ratio"); fig.tight_layout(); fig.savefig(stage_dir / "aspect_ratio_vs_time.png", dpi=140); plt.close(fig)
    evidence_path = stage_dir / "handover_evidence.csv"
    if evidence_path.is_file():
        evidence = __import__("pandas").read_csv(evidence_path)
        fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True); axes[0].plot(evidence.time_s, evidence.dL_dt_m_s, label="dL/dt"); axes[0].plot(evidence.time_s, evidence.dH_dt_m_s, label="dH/dt"); axes[0].legend(); axes[1].step(evidence.time_s, evidence.handover_candidate.astype(int)); axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("candidate"); fig.tight_layout(); fig.savefig(stage_dir / "handover_diagnostic.png", dpi=140); plt.close(fig)


def plot_combined(results: dict, trajectory, output_dir: Path) -> None:
    plt = _plt()
    combined_dir = output_dir / "combined"
    combined_dir.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection="3d")
    ax.plot(trajectory["X_m"], trajectory["Y_m"], -trajectory["TVD_m"], color="black", alpha=0.35, label="well trajectory")
    for stage_id, result in results.items():
        front_path = output_dir / stage_id / "final_front_global.csv"
        if front_path.is_file():
            frame = __import__("pandas").read_csv(front_path)
            ax.scatter(frame.X_m, frame.Y_m, frame.Z_m, s=4, label=stage_id)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m, up)"); ax.legend(); fig.tight_layout(); fig.savefig(combined_dir / "well_x_3d_fractures.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5));
    for stage_id, result in results.items(): ax.plot(result.metrics.time_s, result.metrics.half_length_m, label=stage_id)
    ax.set_xlabel("time (s)"); ax.set_ylabel("half length (m)"); ax.legend(); fig.tight_layout(); fig.savefig(combined_dir / "stage_geometry_comparison.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(trajectory["X_m"], trajectory["TVD_m"], color="black", alpha=.5, label="well trajectory")
    for stage_id in results:
        frame = __import__("pandas").read_csv(output_dir / stage_id / "final_front_global.csv")
        if not frame.empty: ax.scatter(frame.X_m, frame.TVD_m, s=4, label=stage_id)
    ax.set_xlabel("X (m)"); ax.set_ylabel("TVD (m, down)"); ax.invert_yaxis(); ax.legend(); fig.tight_layout(); fig.savefig(combined_dir / "well_x_tvd_stage_projection.png", dpi=150); plt.close(fig)
