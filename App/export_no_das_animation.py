"""Export a portable GIF for a single-stage no-DAS fracture demo.

The animation is based on the selected pressure-only cache and shows the
stage-level PKN result distributed across the configured display cluster
count.  The source data and the selected well/stage are read from the cache,
so the same exporter can produce one GIF per registered well section.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#2563eb", "#0f766e", "#d97706", "#dc2626", "#7c3aed", "#0891b2"]


def _scenario_payload(cache_path: Path, scenario_id: str) -> dict:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios", {})
    if isinstance(scenarios, dict) and isinstance(scenarios.get(scenario_id), dict):
        return scenarios[scenario_id]
    return payload


def _value(series, index: int, default: float = 0.0) -> float:
    if series is None or len(series) == 0:
        return default
    try:
        return float(series[min(max(index, 0), len(series) - 1)])
    except (TypeError, ValueError):
        return default


def _last_value(series, default: float = 0.0) -> float:
    if series is None or len(series) == 0:
        return default
    return _value(series, len(series) - 1, default)


def export_gif(
    cache_path: Path,
    output_path: Path,
    *,
    scenario_id: str = "no_das_pressure_only",
    frame_count: int = 72,
    fps: int = 8,
) -> Path:
    """Render the animation one frame at a time to keep memory bounded."""

    payload = _scenario_payload(cache_path, scenario_id)
    meta = payload.get("meta", {}) or {}
    timeline = np.asarray(payload.get("timeline_s", []), dtype=float)
    positions = sorted(payload.get("cluster_positions", []) or [], key=lambda row: int(row.get("cluster_id", 0)))
    clusters = payload.get("clusters", {}) or {}
    if meta.get("observation_mode") != "pressure_only" or timeline.size < 2 or not positions:
        raise ValueError("invalid no-DAS cache for animation export")
    indices = np.unique(np.linspace(0, timeline.size - 1, max(2, int(frame_count)), dtype=int))
    md = np.asarray([float(row.get("measured_depth_m", i)) for i, row in enumerate(positions)], dtype=float)
    x = (md - md.min()) / max(float(md.max() - md.min()), 1.0) * 1600.0
    max_length = max([_last_value((clusters.get(str(i), {}) or {}).get("posterior_half_length_m", []), 1.0) for i in range(len(positions))] or [1.0])
    stage = str(meta.get("stage", "unknown")); well = str(meta.get("well", "单井段"))
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(11, 6.5), dpi=110)
    writer = PillowWriter(fps=max(1, int(fps)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(output_path), dpi=110):
        for index in indices:
            fig.clear()
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor("#101820"); fig.patch.set_facecolor("#101820")
            ax.set_xlim(-80, 1680); ax.set_ylim(-max_length * 1.18, max_length * 1.18); ax.set_zlim(-28, 28)
            ax.set_box_aspect((2.5, 1.45, 0.8)); ax.view_init(elev=23, azim=-58); ax.grid(True, color="#405361", alpha=0.55)
            ax.tick_params(colors="#B7C7D3", labelsize=8)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.label.set_color("#E8F0F5"); axis.pane.set_facecolor((0.06, 0.09, 0.12, 1.0))
            ax.set_xlabel("假设井段轴向 / m", color="#E8F0F5", labelpad=8); ax.set_ylabel("裂缝横向半长 / m", color="#E8F0F5", labelpad=8); ax.set_zlabel("裂缝高度 / m", color="#E8F0F5", labelpad=6)
            ax.plot(np.linspace(-80, 1680, 80), np.zeros(80), np.zeros(80), color="#8292A6", linewidth=3.2)
            ax.scatter(x, np.zeros_like(x), np.zeros_like(x), color="#F2A93B", s=22, depthshade=False)
            for cluster_index, x_value in enumerate(x):
                record = clusters.get(str(cluster_index), {}) or {}
                length = max(_value(record.get("posterior_half_length_m", []), int(index), 0.0), 0.0)
                height = 22.0
                verts = [[(x_value, -length, -height / 2), (x_value, length, -height / 2), (x_value, length, height / 2), (x_value, -length, height / 2)]]
                ax.add_collection3d(Poly3DCollection(verts, facecolors=COLORS[cluster_index % len(COLORS)], alpha=0.68, edgecolors="#E8F0F5", linewidths=0.35))
                ax.text(x_value, 0, height / 2 + 2, f"C{cluster_index + 1}", color="#F2A93B", fontsize=8, ha="center")
            writer.grab_frame()
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a no-DAS stage-level fracture evolution GIF")
    parser.add_argument("--cache", default=str(ROOT / "outputs" / "app" / "dt_realtime_cache.json"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "app" / "no_das_stage08_fracture_evolution.gif"))
    parser.add_argument("--scenario", default="no_das_pressure_only")
    parser.add_argument("--frame-count", type=int, default=72)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    result = export_gif(Path(args.cache), Path(args.output), scenario_id=args.scenario, frame_count=args.frame_count, fps=args.fps)
    print(json.dumps({"output": str(result), "size_bytes": result.stat().st_size}, ensure_ascii=False, indent=2))
