"""Build an offline Plotly 3D well/fracture-cluster playback page.

The page is generated from the synchronized APP cache. It uses the real well
trajectory coordinates and the posterior per-cluster PKN half-length. Fracture
height is a fixed PKN input (30 m); it is not an inverted field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[1]


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1.0e-9 else fallback.astype(float)


def _cluster_mesh(center: np.ndarray, tangent: np.ndarray, half_length: float, height_m: float = 30.0):
    """Return a thin PKN fracture surface attached to a cluster location."""

    vertical = np.array([0.0, 0.0, 1.0], dtype=float)
    tangent = _unit(tangent, np.array([1.0, 0.0, 0.0], dtype=float))
    width_direction = _unit(np.cross(tangent, vertical), np.array([0.0, 1.0, 0.0], dtype=float))
    height_direction = _unit(np.cross(width_direction, tangent), vertical)
    length = max(float(half_length), 1.0)
    half_height = max(float(height_m) / 2.0, 1.0)
    corners = np.array(
        [
            center - width_direction * length - height_direction * half_height,
            center + width_direction * length - height_direction * half_height,
            center + width_direction * length + height_direction * half_height,
            center - width_direction * length + height_direction * half_height,
        ]
    )
    return {
        "x": corners[:, 0].tolist(),
        "y": corners[:, 1].tolist(),
        "z": corners[:, 2].tolist(),
        "i": [0, 0],
        "j": [1, 2],
        "k": [2, 3],
    }


def _trajectory_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    points = np.array(
        [[float(row.get("east_m", 0.0)), float(row.get("north_m", 0.0)), float(row.get("vertical_depth_m", 0.0))] for row in records],
        dtype=float,
    )
    if len(points) < 2:
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=float)
    tangents = np.empty_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    for index in range(1, len(points) - 1):
        tangents[index] = points[index + 1] - points[index - 1]
    return points, tangents


def build_html(cache_path: Path, output_path: Path, frame_count: int = 120, scenario_id: str | None = None) -> Path:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if scenario_id and isinstance(payload.get("scenarios"), dict) and scenario_id in payload["scenarios"]:
        payload = payload["scenarios"][scenario_id]
    timeline = np.asarray(payload.get("timeline_s", []), dtype=int)
    trajectory = payload.get("trajectory", [])
    positions = payload.get("cluster_positions", [])
    clusters = payload.get("clusters", {})
    if timeline.size == 0 or not trajectory:
        raise ValueError("DT cache lacks timeline or trajectory")

    well_points, well_tangents = _trajectory_arrays(trajectory)
    cluster_records = sorted(positions, key=lambda row: int(row.get("cluster_id", 0)))
    cluster_centers = np.array(
        [[float(row.get("east_m", 0.0)), float(row.get("north_m", 0.0)), float(row.get("vertical_depth_m", 0.0))] for row in cluster_records],
        dtype=float,
    )
    if not cluster_records:
        cluster_centers = np.empty((0, 3), dtype=float)
    cluster_tangents = []
    for row in cluster_records:
        md = float(row.get("measured_depth_m", 0.0))
        index = int(np.argmin([abs(float(item.get("measured_depth_m", 0.0)) - md) for item in trajectory]))
        cluster_tangents.append(well_tangents[index])
    cluster_tangents = np.asarray(cluster_tangents, dtype=float)

    frame_count = max(2, min(int(frame_count), int(timeline.size)))
    frame_indices = np.unique(np.linspace(0, timeline.size - 1, frame_count, dtype=int))
    max_lengths = []
    max_prior_lengths = []
    for cluster_record in cluster_records:
        cluster_id = str(int(cluster_record.get("cluster_id", 0)))
        record = clusters.get(cluster_id, {})
        series = record.get("posterior_half_length_m", [])
        prior_series = record.get("prior_half_length_m", [])
        max_lengths.append(max([float(value) for value in series] or [1.0]))
        max_prior_lengths.append(max([float(value) for value in prior_series] or [1.0]))
    max_length = max(max_lengths or [1.0])

    # Compute one global scene box from the entire trajectory and the largest
    # fracture extent across all frames.  Keeping these ranges fixed prevents
    # the 3D view from zooming or changing scale as playback advances.
    extent_points = [well_points]
    if cluster_centers.size:
        extent_points.append(cluster_centers)
    for center, tangent, cluster_max in zip(cluster_centers, cluster_tangents, max_lengths):
        mesh = _cluster_mesh(center, tangent, cluster_max)
        extent_points.append(np.column_stack([mesh["x"], mesh["y"], mesh["z"]]))
    for center, tangent, cluster_max in zip(cluster_centers, cluster_tangents, max_prior_lengths):
        mesh = _cluster_mesh(center, tangent, cluster_max)
        extent_points.append(np.column_stack([mesh["x"], mesh["y"], mesh["z"]]))
    all_extent = np.vstack(extent_points)
    scene_min = all_extent.min(axis=0)
    scene_max = all_extent.max(axis=0)
    scene_span = np.maximum(scene_max - scene_min, 1.0)
    scene_padding = np.maximum(scene_span * 0.05, 1.0)
    scene_ranges = {
        "x": [float(scene_min[0] - scene_padding[0]), float(scene_max[0] + scene_padding[0])],
        "y": [float(scene_min[1] - scene_padding[1]), float(scene_max[1] + scene_padding[1])],
        "z": [float(scene_max[2] + scene_padding[2]), float(scene_min[2] - scene_padding[2])],
    }

    def length_at(cluster_id: int, index: int, key: str = "posterior_half_length_m") -> float:
        series = clusters.get(str(cluster_id), {}).get(key, [])
        if not series:
            return 1.0
        return float(series[min(max(index, 0), len(series) - 1)])

    initial_index = int(frame_indices[0])
    initial_prior_meshes = [_cluster_mesh(center, tangent, length_at(cid, initial_index, "prior_half_length_m")) for center, tangent, cid in zip(cluster_centers, cluster_tangents, range(len(cluster_records)))]
    initial_meshes = [_cluster_mesh(center, tangent, length_at(cid, initial_index)) for center, tangent, cid in zip(cluster_centers, cluster_tangents, range(len(cluster_records)))]
    colors = ["#2563eb", "#0f766e", "#d97706", "#dc2626", "#7c3aed", "#0891b2"]
    data = [
        go.Scatter3d(
            x=well_points[:, 0], y=well_points[:, 1], z=well_points[:, 2],
            mode="lines", line={"color": "#334155", "width": 7}, name="",
            hovertemplate="东=%{x:.1f} m<br>北向展示坐标=%{y:.1f} m<br>垂深=%{z:.1f} m<extra></extra>",
        ),
        go.Scatter3d(
            x=cluster_centers[:, 0], y=cluster_centers[:, 1], z=cluster_centers[:, 2],
            mode="markers", marker={"size": 5, "color": "#ef4444"},
            name="", hovertemplate="簇位置<br>北向展示坐标=%{y:.1f} m<extra></extra>",
        ),
    ]
    for index, mesh in enumerate(initial_meshes):
        prior_mesh = initial_prior_meshes[index]
        data.append(
            go.Mesh3d(
                **prior_mesh, color="#F2A93B", opacity=0.22,
                name=f"簇 C{index + 1} PKN先验", hovertemplate=f"C{index + 1}<br>PKN先验半缝长=%{{customdata:.2f}} m<extra></extra>",
                customdata=[length_at(index, initial_index, "prior_half_length_m")] * 4,
                showscale=False,
            )
        )
        data.append(
            go.Mesh3d(
                **mesh, color="#20C7C2", opacity=0.68,
                name=f"簇 C{index + 1} 裂缝", hovertemplate=f"C{index + 1}<br>PKN后验半缝长=%{{customdata:.2f}} m<extra></extra>",
                customdata=[length_at(index, initial_index)] * 4,
                showscale=False,
            )
        )

    frames = []
    for index in frame_indices:
        posterior_lengths = [length_at(cluster_index, int(index)) for cluster_index in range(len(cluster_centers))]
        active_cluster = int(np.argmax(posterior_lengths)) if posterior_lengths else 0
        frame_data = [
            {
                "type": "scatter3d",
                "x": cluster_centers[:, 0].tolist(), "y": cluster_centers[:, 1].tolist(), "z": cluster_centers[:, 2].tolist(),
                "text": [f"C{cluster_index + 1}" for cluster_index in range(len(cluster_centers))],
                "marker": {"size": [10 if cluster_index == active_cluster else 5 for cluster_index in range(len(cluster_centers))], "color": ["#F2A93B" if cluster_index == active_cluster else "#E05252" for cluster_index in range(len(cluster_centers))]},
            }
        ]
        for cluster_index, (center, tangent) in enumerate(zip(cluster_centers, cluster_tangents)):
            prior_mesh = _cluster_mesh(center, tangent, length_at(cluster_index, int(index), "prior_half_length_m"))
            prior_mesh.update({"type": "mesh3d", "color": "#F2A93B", "opacity": 0.22, "customdata": [length_at(cluster_index, int(index), "prior_half_length_m")] * 4})
            frame_data.append(prior_mesh)
            mesh = _cluster_mesh(center, tangent, length_at(cluster_index, int(index)))
            mesh.update({"type": "mesh3d", "color": "#20C7C2", "opacity": 0.68, "customdata": [length_at(cluster_index, int(index))] * 4})
            frame_data.append(mesh)
        frames.append(go.Frame(name=str(int(timeline[index])), data=frame_data, traces=list(range(1, len(data)))))

    fig = go.Figure(data=data, frames=frames)
    fig.update_layout(
        # Let the embedded QtWebEngine view determine the height.  The APP
        # allocates the 3D panel across the full height of the three charts;
        # a fixed Plotly height would leave the lower part of that panel blank.
        template="plotly_dark", autosize=True, height=None, showlegend=False,
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        scene={
            "xaxis_title": "东向 East (m)", "yaxis_title": "北向 North (m)", "zaxis_title": "垂深 TVD (m)",
            "aspectmode": "data",
            "xaxis": {"range": scene_ranges["x"], "gridcolor": "#344B5A", "zerolinecolor": "#344B5A"},
            "yaxis": {"range": scene_ranges["y"], "gridcolor": "#344B5A", "zerolinecolor": "#344B5A"},
            "zaxis": {"range": scene_ranges["z"], "gridcolor": "#344B5A", "zerolinecolor": "#344B5A"},
            "bgcolor": "#1B2A36",
        },
        paper_bgcolor="#1B2A36", plot_bgcolor="#1B2A36", font={"color": "#E8F0F5"},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_html = fig.to_html(
        include_plotlyjs=True,
        full_html=False,
        div_id="dt3d",
        config={"responsive": True, "displaylogo": False},
        default_width="100%",
        default_height="100%",
    )
    frame_times_json = json.dumps([int(timeline[index]) for index in frame_indices])
    html = f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>六簇裂缝三维状态回放</title>
<style>
html,body{{width:100%;height:100%;margin:0;overflow:hidden;font-family:'Microsoft YaHei',sans-serif;background:#1B2A36;color:#E8F0F5;}}
body>div{{width:100%;height:100%;display:flex;flex-direction:column;min-height:0;}}
#dt3d,#dt3d.plotly-graph-div{{width:100%!important;height:100%!important;min-height:0;flex:1 1 auto;}}
</style>
</head><body>{plot_html}
<script>
(() => {{
  const graph = document.getElementById('dt3d');
  const resize = () => {{
    if (window.Plotly && graph) Plotly.Plots.resize(graph);
  }};
  window.addEventListener('resize', resize);
  if (window.ResizeObserver && graph) new ResizeObserver(resize).observe(graph.parentElement || graph);
  window.setTimeout(resize, 0);
  window.setTimeout(resize, 400);
}})();
window.setTimeIndex = function(time) {{
  if (typeof Plotly === 'undefined') return;
  const times = {frame_times_json};
  let nearest = times[0];
  for (const candidate of times) {{
    if (Math.abs(candidate - time) < Math.abs(nearest - time)) nearest = candidate;
  }}
  Plotly.animate('dt3d', [String(nearest)], {{mode:'immediate', frame:{{duration:0, redraw:true}}, transition:{{duration:0}}}});
}};
window.setCamera = function(camera) {{
  if (typeof Plotly === 'undefined' || !camera) return;
  Plotly.relayout('dt3d', {{'scene.camera': camera}});
}};
window.getCamera = function() {{
  const graph = document.getElementById('dt3d');
  return graph && graph.layout && graph.layout.scene ? graph.layout.scene.camera : null;
}};
</script></body></html>"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build offline 3D well trajectory and fracture-cluster playback")
    parser.add_argument("--cache", default=str(ROOT / "outputs" / "app" / "dt_realtime_cache.json"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "app" / "dt_realtime_3d.html"))
    parser.add_argument("--frame-count", type=int, default=120)
    parser.add_argument("--scenario", choices=["das_cluster_observation", "no_das_pressure_only"], default="das_cluster_observation")
    args = parser.parse_args()
    result = build_html(Path(args.cache), Path(args.output), args.frame_count, args.scenario)
    print(json.dumps({"html": str(result), "frame_count": args.frame_count}, ensure_ascii=False, indent=2))
