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


def _cluster_mesh(
    center: np.ndarray,
    tangent: np.ndarray,
    half_length: float,
    height_m: float = 30.0,
    *,
    anchor_bottom: bool = False,
):
    """Return a thin PKN fracture surface attached to a cluster location."""

    vertical = np.array([0.0, 0.0, 1.0], dtype=float)
    tangent = _unit(tangent, np.array([1.0, 0.0, 0.0], dtype=float))
    width_direction = _unit(np.cross(tangent, vertical), np.array([0.0, 1.0, 0.0], dtype=float))
    height_direction = _unit(np.cross(width_direction, tangent), vertical)
    length = max(float(half_length), 1.0)
    half_height = max(float(height_m) / 2.0, 1.0)
    if anchor_bottom:
        # The APP reverses the Z axis so TVD increases visually downward.  In
        # that coordinate system the bottom reference line is the high-Z side
        # and the fracture surface must be shifted toward lower Z.
        center = center - height_direction * half_height
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
    cluster_records = sorted(positions, key=lambda row: int(row.get("cluster_id", 0)))
    pressure_only = payload.get("meta", {}).get("observation_mode") == "pressure_only"
    if timeline.size == 0 or (not trajectory and not pressure_only):
        raise ValueError("DT cache lacks timeline or trajectory")

    if pressure_only:
        # No-DAS is a stage-level model demonstration.  Do not reuse the
        # actual well path: draw a straight assumed stage and place the
        # estimated clusters on that line.
        if not cluster_records:
            cluster_keys = sorted(
                (str(key) for key in clusters.keys()),
                key=lambda value: int(value) if value.lstrip("-").isdigit() else value,
            ) if isinstance(clusters, dict) else []
            if cluster_keys:
                cluster_records = [
                    {
                        "cluster_id": int(key) if key.lstrip("-").isdigit() else index,
                    }
                    for index, key in enumerate(cluster_keys)
                ]
            else:
                fallback_count = int(payload.get("meta", {}).get("cluster_count", 6) or 6)
                cluster_records = [{"cluster_id": index} for index in range(max(fallback_count, 1))]
        count = len(cluster_records)
        span = max(1000.0, 240.0 * max(count - 1, 1))
        line_x = np.linspace(-span / 2.0 - 180.0, span / 2.0 + 180.0, 80)
        display_height = 30.0
        # Z is reversed in the Plotly scene; the high-Z side is therefore the
        # visual bottom of the communication view.
        base_z = display_height
        well_points = np.column_stack([line_x, np.zeros_like(line_x), np.full_like(line_x, base_z)])
        cluster_centers = np.column_stack([
            np.linspace(-span / 2.0, span / 2.0, count),
            np.zeros(count),
            np.full(count, base_z),
        ]) if count else np.empty((0, 3), dtype=float)
        cluster_tangents = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=float), (count, 1))
    else:
        well_points, well_tangents = _trajectory_arrays(trajectory)
        cluster_centers = None
    cluster_centers = np.array(
        [[float(row.get("east_m", 0.0)), float(row.get("north_m", 0.0)), float(row.get("vertical_depth_m", 0.0))] for row in cluster_records],
        dtype=float,
    ) if cluster_centers is None else cluster_centers
    if not cluster_records:
        cluster_centers = np.empty((0, 3), dtype=float)
        cluster_tangents = np.empty((0, 3), dtype=float)
    elif not pressure_only:
        cluster_tangents = []
        for row in cluster_records:
            md = float(row.get("measured_depth_m", 0.0))
            index = int(np.argmin([abs(float(item.get("measured_depth_m", 0.0)) - md) for item in trajectory]))
            cluster_tangents.append(well_tangents[index])
        cluster_tangents = np.asarray(cluster_tangents, dtype=float)

    # Some historical DAS cache versions contain six independent state series
    # but no geometry file.  Keep those series visible by placing display-only
    # markers along the terminal trajectory interval.  These positions are
    # never used by EnKF or treated as field geometry; a supplied geometry file
    # replaces them on the next cache build.
    if not pressure_only and not cluster_records and isinstance(clusters, dict):
        cluster_keys = sorted(
            (str(key) for key in clusters.keys()),
            key=lambda value: int(value) if value.lstrip("-").isdigit() else value,
        )
        if len(cluster_keys) > 1 and len(trajectory) >= 2:
            trajectory_md = np.asarray(
                [float(row.get("measured_depth_m", index)) for index, row in enumerate(trajectory)],
                dtype=float,
            )
            trajectory_xyz = np.asarray(
                [
                    [
                        float(row.get("east_m", 0.0)),
                        float(row.get("north_m", 0.0)),
                        float(row.get("vertical_depth_m", 0.0)),
                    ]
                    for row in trajectory
                ],
                dtype=float,
            )
            order = np.argsort(trajectory_md)
            trajectory_md = trajectory_md[order]
            trajectory_xyz = trajectory_xyz[order]
            start_md = max(float(trajectory_md[0]), float(trajectory_md[-1]) - 1600.0)
            target_md = np.linspace(start_md, float(trajectory_md[-1]), len(cluster_keys))
            cluster_records = [
                {
                    "cluster_id": int(key) if key.lstrip("-").isdigit() else index,
                    "measured_depth_m": float(md),
                    "east_m": float(np.interp(md, trajectory_md, trajectory_xyz[:, 0])),
                    "north_m": float(np.interp(md, trajectory_md, trajectory_xyz[:, 1])),
                    "vertical_depth_m": float(np.interp(md, trajectory_md, trajectory_xyz[:, 2])),
                }
                for index, (key, md) in enumerate(zip(cluster_keys, target_md))
            ]

    # Rebuild the display geometry after the fallback above.  The original
    # cache has no cluster_positions in this case, so the first pass created
    # an empty array before the display-only records were available.
    if cluster_records and not pressure_only:
        cluster_centers = np.array(
            [
                [
                    float(row.get("east_m", 0.0)),
                    float(row.get("north_m", 0.0)),
                    float(row.get("vertical_depth_m", 0.0)),
                ]
                for row in cluster_records
            ],
            dtype=float,
        )
        cluster_tangents = []
        for row in cluster_records:
            md = float(row.get("measured_depth_m", 0.0))
            index = int(np.argmin([abs(float(item.get("measured_depth_m", 0.0)) - md) for item in trajectory]))
            cluster_tangents.append(well_tangents[index])
        cluster_tangents = np.asarray(cluster_tangents, dtype=float)

    stage_only = not bool(cluster_records)
    arrays = payload.get("arrays", {}) or {}

    def _series_numbers(values) -> list[float]:
        result = []
        for value in values or []:
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = 0.0
            result.append(number if np.isfinite(number) else 0.0)
        return result

    cluster_ids = [str(int(row.get("cluster_id", index))) for index, row in enumerate(cluster_records)]

    def _stage_length_series(key: str, preferred_array: str | None = None) -> list[float]:
        preferred = _series_numbers(arrays.get(preferred_array, [])) if preferred_array else []
        if preferred:
            return preferred
        count = int(timeline.size)
        result = [0.0] * count
        source_cluster_ids = cluster_ids or sorted(
            (str(key) for key in clusters.keys()),
            key=lambda value: int(value) if str(value).lstrip("-").isdigit() else str(value),
        )
        for cluster_id in source_cluster_ids:
            values = _series_numbers((clusters.get(cluster_id, {}) or {}).get(key, []))
            for index in range(min(count, len(values))):
                result[index] += max(values[index], 0.0)
        return result

    if stage_only:
        if pressure_only:
            stage_center = np.array([0.0, 0.0, 0.0], dtype=float)
            stage_tangent = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            stage_index = len(well_points) // 2
            stage_center = well_points[stage_index]
            stage_tangent = well_tangents[stage_index]
        stage_prior_lengths = _stage_length_series("prior_half_length_m")
        stage_posterior_lengths = _stage_length_series(
            "posterior_half_length_m", "pkn_stage_half_length_m"
        )

    frame_count = max(2, min(int(frame_count), int(timeline.size)))
    frame_indices = np.unique(np.linspace(0, timeline.size - 1, frame_count, dtype=int))
    max_lengths = []
    max_prior_lengths = []
    for cluster_id in cluster_ids:
        record = clusters.get(cluster_id, {}) or {}
        series = _series_numbers(record.get("posterior_half_length_m", []))
        prior_series = _series_numbers(record.get("prior_half_length_m", []))
        max_lengths.append(max(series or [1.0]))
        max_prior_lengths.append(max(prior_series or [1.0]))
    if stage_only:
        max_lengths = [max(stage_posterior_lengths or [1.0])]
        max_prior_lengths = [max(stage_prior_lengths or [1.0])]
    max_length = max(max_lengths or [1.0])

    # Compute one global scene box from the entire trajectory and the largest
    # fracture extent across all frames.  Keeping these ranges fixed prevents
    # the 3D view from zooming or changing scale as playback advances.
    extent_points = [well_points]
    if cluster_centers.size:
        extent_points.append(cluster_centers)
    if stage_only:
        for stage_max in (max_lengths[0], max_prior_lengths[0]):
            mesh = _cluster_mesh(stage_center, stage_tangent, stage_max, anchor_bottom=pressure_only)
            extent_points.append(np.column_stack([mesh["x"], mesh["y"], mesh["z"]]))
    else:
        for center, tangent, cluster_max in zip(cluster_centers, cluster_tangents, max_lengths):
            mesh = _cluster_mesh(center, tangent, cluster_max, anchor_bottom=pressure_only)
            extent_points.append(np.column_stack([mesh["x"], mesh["y"], mesh["z"]]))
        for center, tangent, cluster_max in zip(cluster_centers, cluster_tangents, max_prior_lengths):
            mesh = _cluster_mesh(center, tangent, cluster_max, anchor_bottom=pressure_only)
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

    def length_at(cluster_index: int, index: int, key: str = "posterior_half_length_m") -> float:
        if cluster_index < 0 or cluster_index >= len(cluster_ids):
            return 1.0
        series = (clusters.get(cluster_ids[cluster_index], {}) or {}).get(key, [])
        if not series:
            return 1.0
        value = float(series[min(max(index, 0), len(series) - 1)])
        return value if np.isfinite(value) else 1.0

    def stage_length_at(index: int, prior: bool = False) -> float:
        series = stage_prior_lengths if prior else stage_posterior_lengths
        if not series:
            return 1.0
        value = float(series[min(max(index, 0), len(series) - 1)])
        return value if np.isfinite(value) else 1.0

    initial_index = int(frame_indices[0])
    colors = ["#2563eb", "#0f766e", "#d97706", "#dc2626", "#7c3aed", "#0891b2"]
    well_hover = (
        "直线假设井段轴向=%{x:.1f} m<br>模型展示坐标<extra></extra>"
        if pressure_only
        else "东=%{x:.1f} m<br>北向展示坐标=%{y:.1f} m<br>垂深=%{z:.1f} m<extra></extra>"
    )
    cluster_hover = (
        "簇位置<br>轴向=%{x:.1f} m<extra></extra>"
        if pressure_only
        else "簇位置<br>北向展示坐标=%{y:.1f} m<extra></extra>"
    )
    data = [
        go.Scatter3d(
            x=well_points[:, 0], y=well_points[:, 1], z=well_points[:, 2],
            mode="lines", line={"color": "#334155", "width": 7}, name="",
            hovertemplate=well_hover,
        ),
    ]
    if stage_only:
        data.append(
            go.Scatter3d(
                x=[stage_center[0]], y=[stage_center[1]], z=[stage_center[2]], mode="markers",
                marker={"size": 7, "color": "#ef4444"}, name="阶段模型位置",
                hovertemplate="阶段级模型位置<extra></extra>",
            )
        )
        data.append(
            go.Mesh3d(
                **_cluster_mesh(stage_center, stage_tangent, stage_length_at(initial_index, True), anchor_bottom=pressure_only),
                color="#F2A93B", opacity=0.22, name="阶段 PKN 先验",
                hovertemplate="阶段 PKN 先验半缝长=%{customdata:.2f} m<extra></extra>",
                customdata=[stage_length_at(initial_index, True)] * 4, showscale=False,
            )
        )
        data.append(
            go.Mesh3d(
                **_cluster_mesh(stage_center, stage_tangent, stage_length_at(initial_index), anchor_bottom=pressure_only),
                color="#20C7C2", opacity=0.68, name="阶段级裂缝",
                hovertemplate="阶段级后验半缝长=%{customdata:.2f} m<extra></extra>",
                customdata=[stage_length_at(initial_index)] * 4, showscale=False,
            )
        )
    else:
        data.append(
            go.Scatter3d(
                x=cluster_centers[:, 0], y=cluster_centers[:, 1], z=cluster_centers[:, 2],
                mode="markers", marker={"size": 5, "color": "#ef4444"},
                name="", hovertemplate=cluster_hover,
            )
        )
        for index in range(len(cluster_records)):
            prior_length = length_at(index, initial_index, "prior_half_length_m")
            posterior_length = length_at(index, initial_index)
            data.append(
                go.Mesh3d(
                    **_cluster_mesh(cluster_centers[index], cluster_tangents[index], prior_length, anchor_bottom=pressure_only),
                    color="#F2A93B", opacity=0.22,
                    name=f"簇 C{index + 1} PKN先验", hovertemplate=f"C{index + 1}<br>PKN先验半缝长=%{{customdata:.2f}} m<extra></extra>",
                    customdata=[prior_length] * 4, showscale=False,
                )
            )
            data.append(
                go.Mesh3d(
                    **_cluster_mesh(cluster_centers[index], cluster_tangents[index], posterior_length, anchor_bottom=pressure_only),
                    color="#20C7C2", opacity=0.68,
                    name=f"簇 C{index + 1} 裂缝", hovertemplate=f"C{index + 1}<br>PKN后验半缝长=%{{customdata:.2f}} m<extra></extra>",
                    customdata=[posterior_length] * 4, showscale=False,
                )
            )

    frames = []
    for index in frame_indices:
        if stage_only:
            frame_data = [
                {
                    "type": "scatter3d", "x": [stage_center[0]], "y": [stage_center[1]], "z": [stage_center[2]],
                    "text": ["阶段级模型"], "marker": {"size": 7, "color": "#ef4444"},
                },
                {
                    **_cluster_mesh(stage_center, stage_tangent, stage_length_at(int(index), True), anchor_bottom=pressure_only),
                    "type": "mesh3d", "color": "#F2A93B", "opacity": 0.22,
                    "customdata": [stage_length_at(int(index), True)] * 4,
                },
                {
                    **_cluster_mesh(stage_center, stage_tangent, stage_length_at(int(index)), anchor_bottom=pressure_only),
                    "type": "mesh3d", "color": "#20C7C2", "opacity": 0.68,
                    "customdata": [stage_length_at(int(index))] * 4,
                },
            ]
        else:
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
                prior_length = length_at(cluster_index, int(index), "prior_half_length_m")
                prior_mesh = _cluster_mesh(center, tangent, prior_length, anchor_bottom=pressure_only)
                prior_mesh.update({"type": "mesh3d", "color": "#F2A93B", "opacity": 0.22, "customdata": [prior_length] * 4})
                frame_data.append(prior_mesh)
                posterior_length = length_at(cluster_index, int(index))
                mesh = _cluster_mesh(center, tangent, posterior_length, anchor_bottom=pressure_only)
                mesh.update({"type": "mesh3d", "color": "#20C7C2", "opacity": 0.68, "customdata": [posterior_length] * 4})
                frame_data.append(mesh)
        frames.append(go.Frame(name=str(int(timeline[index])), data=frame_data, traces=list(range(1, len(data)))))

    fig = go.Figure(data=data, frames=frames)
    # The no-DAS view is an assumed straight stage used for communication and
    # cluster-pattern demonstration.  Its axial span is much larger than the
    # fixed PKN fracture height, so ``aspectmode=data`` would compress the
    # fractures into a thin strip.  Use a visual aspect ratio for this view;
    # this changes only rendering, never the estimated coordinates or lengths.
    if pressure_only:
        scene_aspectmode = "manual"
        scene_aspectratio = {"x": 2.2, "y": 1.45, "z": 1.8}
        scene_camera = {
            "eye": {"x": 1.45, "y": 1.55, "z": 1.30},
            "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
    else:
        scene_aspectmode = "data"
        scene_aspectratio = None
        scene_camera = None

    fig.update_layout(
        # Let the embedded QtWebEngine view determine the height.  The APP
        # allocates the 3D panel across the full height of the three charts;
        # a fixed Plotly height would leave the lower part of that panel blank.
        template="plotly_dark", autosize=True, height=None, showlegend=False,
        uirevision="dt-camera-v1",
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        scene={
            "xaxis_title": "假设井段轴向 (m)" if pressure_only else "东向 East (m)",
            "yaxis_title": "裂缝横向 (m)" if pressure_only else "北向 North (m)",
            "zaxis_title": "裂缝高度 (m)" if pressure_only else "垂深 TVD (m)",
            "aspectmode": scene_aspectmode,
            **({"aspectratio": scene_aspectratio} if scene_aspectratio else {}),
            **({"camera": scene_camera} if scene_camera else {}),
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
        config={"responsive": True, "displaylogo": False, "scrollZoom": True, "doubleClick": "reset"},
        default_width="100%",
        default_height="100%",
    )
    frame_times_json = json.dumps([int(timeline[index]) for index in frame_indices])
    html = f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>{'无 DAS 直线假设井段 · PKN 估计裂缝扩展' if pressure_only else '六簇裂缝三维状态回放'}</title>
<style>
html,body{{width:100%;height:100%;margin:0;overflow:hidden;font-family:'Microsoft YaHei',sans-serif;background:#1B2A36;color:#E8F0F5;}}
body>div{{width:100%;height:100%;display:flex;flex-direction:column;min-height:0;}}
#dt3d,#dt3d.plotly-graph-div{{width:100%!important;height:100%!important;min-height:0;flex:1 1 auto;}}
</style>
</head><body>{plot_html}
<script>
(() => {{
  const graph = document.getElementById('dt3d');
  window.__dtGraph = graph;
  window.__dtCurrentCamera = window.__dtCurrentCamera || null;
  window.__dtInteractionEnabled = true;
  const rememberCamera = () => {{
    const current = graph && graph.layout && graph.layout.scene ? graph.layout.scene.camera : null;
    if (current) window.__dtCurrentCamera = JSON.parse(JSON.stringify(current));
  }};
  if (graph && graph.on) graph.on('plotly_relayout', (event) => {{
    if (window.__dtInteractionEnabled && event && (event['scene.camera'] || Object.keys(event).some((key) => key.indexOf('scene.camera.') === 0))) rememberCamera();
  }});
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
  const graph = window.__dtGraph || document.getElementById('dt3d');
  if (!graph) return;
  const times = {frame_times_json};
  let nearest = times[0];
  for (const candidate of times) {{
    if (Math.abs(candidate - time) < Math.abs(nearest - time)) nearest = candidate;
  }}
  const current = graph.layout && graph.layout.scene ? graph.layout.scene.camera : null;
  if (current) window.__dtCurrentCamera = JSON.parse(JSON.stringify(current));
  window.__dtAnimationToken = (window.__dtAnimationToken || 0) + 1;
  const token = window.__dtAnimationToken;
  Promise.resolve(Plotly.animate('dt3d', [String(nearest)], {{mode:'immediate', frame:{{duration:0, redraw:true}}, transition:{{duration:0}}}}))
    .then(() => {{
      if (token !== window.__dtAnimationToken || !window.__dtCurrentCamera) return;
      return Plotly.relayout('dt3d', {{'scene.camera': window.__dtCurrentCamera}});
    }});
}};
window.setCamera = function(camera) {{
  if (typeof Plotly === 'undefined' || !camera) return;
  window.__dtCurrentCamera = JSON.parse(JSON.stringify(camera));
  Plotly.relayout('dt3d', {{'scene.camera': camera}});
}};
window.setInteractionEnabled = function(enabled) {{
  const graph = window.__dtGraph || document.getElementById('dt3d');
  window.__dtInteractionEnabled = !!enabled;
  if (graph) graph.style.pointerEvents = window.__dtInteractionEnabled ? 'auto' : 'none';
}};
window.getCamera = function() {{
  const graph = window.__dtGraph || document.getElementById('dt3d');
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
