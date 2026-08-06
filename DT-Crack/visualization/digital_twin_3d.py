from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
try:
    import pyvista as pv
except ImportError:
    pv = None

if pv is not None:
    pv.global_theme.smooth_shading = False
    pv.global_theme.depth_peeling["enabled"] = False
    pv.global_theme.font.family = "arial"

ROOT = Path(__file__).resolve().parents[2]
DT_ROOT = ROOT / "DT-Crack"
if str(DT_ROOT) not in sys.path:
    sys.path.insert(0, str(DT_ROOT))
from inversion.playback import build_parser as build_closed_loop_parser
from inversion.playback import run_demo


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive PyVista demo for real 3Dfrac PKN + EnKF closed loop.")
    parser.add_argument("--frac-monitor-text", default=str(ROOT / "Data" / "3Dfrac" / "光纤本井监测08.txt"))
    parser.add_argument("--well-trajectory-csv", default=str(ROOT / "Data" / "3Dfrac" / "JY84-Z1HF-1011.csv"))
    parser.add_argument("--construction-pressure-xls", default=str(ROOT / "Data" / "3Dfrac" / "JY84-Z1-stage08-f1.xls"))
    parser.add_argument("--forward-model", choices=["pkn4", "bem_reduced", "physics_hybrid", "data_surrogate"], default="pkn4")
    parser.add_argument("--max-playback-steps", type=int, default=14)
    parser.add_argument("--visual-aperture-gain", type=float, default=120.0)
    parser.add_argument("--length-scale", type=float, default=0.08, help="Display scale for fracture half-length in 3D view.")
    parser.add_argument("--height-scale", type=float, default=1.0)
    parser.add_argument("--backend", choices=["qt", "native", "plotly-html"], default="qt", help="Interactive backend. Use plotly-html if VTK windows crash.")
    parser.add_argument("--rich-labels", action="store_true", help="Enable 3D text labels. Disabled by default for stable VTK interaction.")
    parser.add_argument("--screenshot-only", action="store_true", help="Render one screenshot without opening an interactive window.")
    parser.add_argument("--screenshot", default=str(ROOT / "outputs" / "dt" / "pyvista_3dfrac_demo_preview.png"))
    parser.add_argument("--html", default=str(ROOT / "outputs" / "dt" / "plotly_3dfrac_demo.html"))
    parser.add_argument("--agent-decisions-json", default=None, help="Optional HMI-KE agent_decisions.json to fuse risk cards into the Plotly page.")
    parser.add_argument("--open", action="store_true", help="Open generated browser demo when using --backend plotly-html.")
    parser.add_argument("--window-size", type=int, nargs=2, default=[1500, 900])
    return parser.parse_args()


def make_closed_loop_args(args: argparse.Namespace) -> argparse.Namespace:
    base = build_closed_loop_parser().parse_args([])
    base.forward_model = args.forward_model
    base.frac_monitor_text = args.frac_monitor_text
    base.well_trajectory_csv = args.well_trajectory_csv
    base.construction_pressure_xls = args.construction_pressure_xls
    base.use_real_monitor_timeline = True
    base.max_playback_steps = args.max_playback_steps
    base.default_step_seconds = 1.0
    base.n_clusters = 6
    base.well_len_m = 120.0
    base.visual_aperture_gain = args.visual_aperture_gain
    base.run_dir = str(ROOT / "outputs" / "dt" / "visualization_data")
    base.open = False
    return base


def fracture_surface(
    x_center: float,
    half_length_m: float,
    height_m: float,
    aperture_mm: float,
    length_scale: float,
    height_scale: float,
    aperture_gain: float,
    side: float,
    ny: int = 64,
    nz: int = 36,
) -> pv.StructuredGrid:
    y_extent = max(half_length_m * length_scale, 1e-3)
    z_extent = max(height_m * height_scale, 1e-3)
    y = np.linspace(-y_extent, y_extent, ny)
    z = np.linspace(-z_extent / 2.0, z_extent / 2.0, nz)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    length_decay = np.clip(1.0 - np.abs(yy) / max(y_extent, 1e-9), 0.0, None) ** 0.25
    height_decay = np.sqrt(np.clip(1.0 - (2.0 * zz / z_extent) ** 2, 0.0, None))
    aperture_field = aperture_mm * length_decay * height_decay
    x_offset = side * (aperture_field / 1000.0) * aperture_gain
    xx = np.full_like(yy, x_center) + x_offset
    grid = pv.StructuredGrid(xx, yy, zz)
    grid["aperture_mm"] = aperture_field.ravel(order="F")
    return grid


def add_line(plotter: pv.Plotter, points: np.ndarray, color: str, width: int = 4, name: str | None = None) -> None:
    if len(points) < 2:
        return
    line = pv.lines_from_points(points)
    tube = line.tube(radius=max(width * 0.015, 0.02))
    plotter.add_mesh(tube, color=color, name=name, smooth_shading=False)


def build_scene(
    plotter: pv.Plotter,
    table: pd.DataFrame,
    history: pd.DataFrame,
    state: SimpleNamespace,
    args: argparse.Namespace,
) -> None:
    plotter.clear()
    time_s = state.time_steps[state.step_index]
    frame = table[table["time_s"] == time_s].sort_values("cluster_id")
    h = history.iloc[state.step_index]
    max_x = float(table["x_center_m"].max())
    height = 30.0

    # Wellbore and real trajectory projection.
    well_points = np.array([[0.0, 0.0, 0.0], [max_x + 18.0, 0.0, 0.0]])
    add_line(plotter, well_points, "#111827", width=7, name="wellbore")
    if {"north_m", "east_m"}.issubset(frame.columns):
        north = frame["north_m"].to_numpy(dtype=float)
        north = north - np.nanmean(north)
        denom = max(float(np.nanmax(np.abs(north))), 1e-9)
        y = north / denom * 9.0
        traj_points = np.c_[frame["x_center_m"].to_numpy(dtype=float), y, np.full(len(y), -18.0)]
        add_line(plotter, traj_points, "#64748b", width=3, name="trajectory")

    aperture_max = max(float(table["max_aperture_mm"].max()), 1e-6)
    for _, row in frame.iterrows():
        x = float(row["x_center_m"])
        length = float(row["half_length_m"])
        prior = float(row["prior_half_length_m"]) * args.length_scale
        observed = float(row["observed_half_length_m"]) * args.length_scale
        aperture = float(row["max_aperture_mm"])
        cid = int(row["cluster_id"])

        for side in [-1.0, 1.0]:
            surf = fracture_surface(
                x,
                length,
                height,
                aperture,
                args.length_scale,
                args.height_scale,
                args.visual_aperture_gain,
                side,
            )
            plotter.add_mesh(
                surf,
                scalars="aperture_mm",
                cmap="turbo",
                clim=[0, aperture_max],
                opacity=0.88,
                smooth_shading=False,
                show_scalar_bar=False,
                name=f"frac_{cid}_{side}",
            )

        add_line(plotter, np.array([[x, -prior, 13.0], [x, prior, 13.0]]), "#f97316", width=3, name=f"prior_{cid}")
        add_line(plotter, np.array([[x, -observed, 17.0], [x, observed, 17.0]]), "#111827", width=2, name=f"obs_{cid}")
        add_line(plotter, np.array([[x, 0.0, -height / 2], [x, 0.0, height / 2]]), "#ef4444", width=3, name=f"cluster_{cid}")
        if args.rich_labels:
            md_label = ""
            if "measured_depth_m" in row and pd.notna(row["measured_depth_m"]):
                md_label = f"\nMD {float(row['measured_depth_m']):.0f}m"
            plotter.add_point_labels(
                np.array([[x, -7.5, 19.0]]),
                [f"C{cid}{md_label}"],
                font_size=12,
                text_color="#dc2626",
                shape_opacity=0.0,
                name=f"label_{cid}",
            )

    plotter.add_text(
        (
            "3Dfrac stage 08: PKN forward + DAS observation + EnKF length update\n"
            f"t={time_s:.0f}s | prior error={float(h['prior_error'])*100:.2f}% | "
            f"posterior error={float(h['posterior_error'])*100:.2f}% | "
            f"15% target={'OK' if bool(h['within_15_percent']) else 'NO'}\n"
            "Left mouse: rotate | Wheel: zoom | Right mouse: pan | orange=PKN prior, black=DAS obs, surface=EnKF posterior"
        ),
        position="upper_left",
        font_size=11,
        color="#111827",
        name="title",
    )
    plotter.add_scalar_bar("PKN aperture mm", n_labels=5, vertical=True, position_x=0.88, position_y=0.20)
    plotter.add_axes()
    plotter.show_grid(color="#cbd5e1")
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.18)
    plotter.render()


def write_plotly_html(table: pd.DataFrame, history: pd.DataFrame, args: argparse.Namespace, path: str | Path) -> Path:
    import plotly.graph_objects as go

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    time_steps = history["time_s"].to_numpy(dtype=float)
    aperture_max = max(float(table["max_aperture_mm"].max()), 1e-6)
    max_x = float(table["x_center_m"].max())
    height = 30.0

    def make_mesh_trace(row: pd.Series, side: float, visible: bool) -> go.Surface:
        x = float(row["x_center_m"])
        length = float(row["half_length_m"])
        aperture = float(row["max_aperture_mm"])
        y_extent = max(length * args.length_scale, 1e-3)
        z_extent = max(height * args.height_scale, 1e-3)
        y = np.linspace(-y_extent, y_extent, 34)
        z = np.linspace(-z_extent / 2.0, z_extent / 2.0, 20)
        yy, zz = np.meshgrid(y, z, indexing="ij")
        length_decay = np.clip(1.0 - np.abs(yy) / max(y_extent, 1e-9), 0.0, None) ** 0.25
        height_decay = np.sqrt(np.clip(1.0 - (2.0 * zz / z_extent) ** 2, 0.0, None))
        aperture_field = aperture * length_decay * height_decay
        xx = np.full_like(yy, x) + side * (aperture_field / 1000.0) * args.visual_aperture_gain
        return go.Surface(
            x=xx,
            y=yy,
            z=zz,
            surfacecolor=aperture_field,
            cmin=0,
            cmax=aperture_max,
            colorscale="Turbo",
            opacity=0.86,
            showscale=False,
            visible=visible,
            hovertemplate="aperture=%{surfacecolor:.3f} mm<extra></extra>",
        )

    traces: list[go.BaseTraceType] = []
    step_trace_indices: list[list[int]] = []
    for step_idx, time_s in enumerate(time_steps):
        frame = table[table["time_s"] == time_s].sort_values("cluster_id")
        visible = step_idx == len(time_steps) - 1
        current_indices: list[int] = []

        # Wellbore.
        current_indices.append(len(traces))
        traces.append(
            go.Scatter3d(
                x=[0, max_x + 18],
                y=[0, 0],
                z=[0, 0],
                mode="lines",
                line=dict(color="#111827", width=8),
                name="wellbore",
                visible=visible,
            )
        )

        if {"north_m", "east_m"}.issubset(frame.columns):
            north = frame["north_m"].to_numpy(dtype=float)
            north = north - np.nanmean(north)
            denom = max(float(np.nanmax(np.abs(north))), 1e-9)
            y = north / denom * 9.0
            current_indices.append(len(traces))
            traces.append(
                go.Scatter3d(
                    x=frame["x_center_m"],
                    y=y,
                    z=np.full(len(y), -18.0),
                    mode="lines",
                    line=dict(color="#64748b", width=5, dash="dot"),
                    name="trajectory",
                    visible=visible,
                )
            )

        for _, row in frame.iterrows():
            cid = int(row["cluster_id"])
            x = float(row["x_center_m"])
            prior = float(row["prior_half_length_m"]) * args.length_scale
            observed = float(row["observed_half_length_m"]) * args.length_scale
            obs_factor = float(row["fiber_observation_factor"]) if "fiber_observation_factor" in row else 1.0

            for side in [-1.0, 1.0]:
                current_indices.append(len(traces))
                traces.append(make_mesh_trace(row, side, visible))

            for name, y_value, z_value, color in [
                ("PKN prior", prior, 13.0, "#f97316"),
                ("DAS obs", observed, 17.0, "#111827"),
            ]:
                current_indices.append(len(traces))
                traces.append(
                    go.Scatter3d(
                        x=[x, x],
                        y=[-y_value, y_value],
                        z=[z_value, z_value],
                    mode="lines",
                    line=dict(color=color, width=5),
                    name=f"C{cid} {name}",
                    visible=visible,
                    hovertemplate=(
                        f"C{cid} {name}<br>"
                        f"fiber obs factor={obs_factor:.3f}<br>"
                        f"cum liquid score={float(row.get('cum_liquid_score', 1.0)):.3f}<br>"
                        f"cum sand score={float(row.get('cum_sand_score', 1.0)):.3f}<extra></extra>"
                    ),
                )
            )

            current_indices.append(len(traces))
            traces.append(
                go.Scatter3d(
                    x=[x],
                    y=[0],
                    z=[height / 2 + 3],
                    mode="text",
                    text=[f"C{cid}"],
                    textfont=dict(color="#dc2626", size=12),
                    name=f"C{cid}",
                    visible=visible,
                )
            )

        step_trace_indices.append(current_indices)

    buttons = []
    total = len(traces)
    for step_idx, time_s in enumerate(time_steps):
        visibility = [False] * total
        for idx in step_trace_indices[step_idx]:
            visibility[idx] = True
        h = history.iloc[step_idx]
        title = (
            f"3Dfrac stage 08 | t={time_s:.0f}s | "
            f"prior error={float(h['prior_error'])*100:.2f}% | "
            f"EnKF posterior error={float(h['posterior_error'])*100:.2f}%"
        )
        buttons.append(
            dict(
                method="update",
                label=f"{time_s:.0f}s",
                args=[{"visible": visibility}, {"title": title}],
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=buttons[-1]["args"][1]["title"] if buttons else "3Dfrac EnKF demo",
        width=1500,
        height=900,
        scene=dict(
            xaxis_title="well direction X (m)",
            yaxis_title="fracture half-length display Y (scaled)",
            zaxis_title="height Z (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=-1.6, z=1.1)),
        ),
        updatemenus=[
            dict(
                type="dropdown",
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
            )
        ],
        annotations=[
            dict(
                text="Browser WebGL fallback: drag to rotate, wheel to zoom. Surface=EnKF posterior, orange=PKN prior, black=DAS equivalent observation.",
                x=0.5,
                y=0.02,
                xref="paper",
                yref="paper",
                showarrow=False,
            )
        ],
    )
    # Embed Plotly so the acceptance demo remains usable on an isolated field laptop.
    fig.write_html(path, include_plotlyjs=True, auto_open=False)
    return path


def load_agent_decisions(path: str | Path | None) -> list[dict]:
    if not path:
        return []
    decision_path = Path(path)
    if not decision_path.exists():
        raise FileNotFoundError(f"Agent decisions JSON not found: {decision_path}")
    data = json.loads(decision_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("decisions", []) if isinstance(data.get("decisions"), list) else []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def default_agent_decision(time_s: float) -> dict:
    return {
        "time_s": float(time_s),
        "risk_level": "unknown",
        "uncertainty": "unknown",
        "main_risk": "未接入智能体输出",
        "recommendation": "当前页面未提供 HMI-KE agent_decisions.json，仅展示数字孪生正反演结果。",
        "action_candidates": [],
        "forbidden_actions": [],
        "requires_confirmation": False,
        "approval_level": "not_available",
        "confirmation_flow": {"status": "not_available", "reason": "未接入智能体输出"},
        "evidence": {},
    }


def write_plotly_html_v2(
    table: pd.DataFrame,
    history: pd.DataFrame,
    args: argparse.Namespace,
    path: str | Path,
    agent_decisions: list[dict] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    time_steps = history["time_s"].to_numpy(dtype=float)
    aperture_max = max(float(table["max_aperture_mm"].max()), 1e-6)
    max_x = float(table["x_center_m"].max())
    height = 30.0
    decision_by_time = {
        round(float(decision.get("time_s", -1)), 6): decision
        for decision in (agent_decisions or [])
        if "time_s" in decision
    }

    def make_mesh_trace(row: pd.Series, side: float) -> dict:
        x = float(row["x_center_m"])
        length = float(row["half_length_m"])
        aperture = float(row["max_aperture_mm"])
        y_extent = max(length * args.length_scale, 1e-3)
        z_extent = max(height * args.height_scale, 1e-3)
        y = np.linspace(-y_extent, y_extent, 28)
        z = np.linspace(-z_extent / 2.0, z_extent / 2.0, 16)
        yy, zz = np.meshgrid(y, z, indexing="ij")
        length_decay = np.clip(1.0 - np.abs(yy) / max(y_extent, 1e-9), 0.0, None) ** 0.25
        height_decay = np.sqrt(np.clip(1.0 - (2.0 * zz / z_extent) ** 2, 0.0, None))
        aperture_field = aperture * length_decay * height_decay
        xx = np.full_like(yy, x) + side * (aperture_field / 1000.0) * args.visual_aperture_gain
        return {
            "type": "surface",
            "x": xx.tolist(),
            "y": yy.tolist(),
            "z": zz.tolist(),
            "surfacecolor": aperture_field.tolist(),
            "cmin": 0,
            "cmax": aperture_max,
            "colorscale": "Turbo",
            "opacity": 0.86,
            "showscale": False,
            "hovertemplate": "EnKF posterior aperture=%{surfacecolor:.3f} mm<extra></extra>",
        }

    frames: list[dict] = []
    for step_idx, time_s in enumerate(time_steps):
        frame = table[table["time_s"] == time_s].sort_values("cluster_id")
        h = history.iloc[step_idx]
        traces_3d: list[dict] = [
            {
                "type": "scatter3d",
                "x": [0, max_x + 18],
                "y": [0, 0],
                "z": [0, 0],
                "mode": "lines",
                "line": {"color": "#111827", "width": 8},
                "name": "水平井筒",
                "hovertemplate": "horizontal wellbore<extra></extra>",
            }
        ]

        if {"north_m", "east_m"}.issubset(frame.columns):
            north = frame["north_m"].to_numpy(dtype=float)
            north = north - np.nanmean(north)
            denom = max(float(np.nanmax(np.abs(north))), 1e-9)
            y = north / denom * 9.0
            traces_3d.append(
                {
                    "type": "scatter3d",
                    "x": frame["x_center_m"].to_numpy(dtype=float).tolist(),
                    "y": y.tolist(),
                    "z": np.full(len(y), -18.0).tolist(),
                    "mode": "lines",
                    "line": {"color": "#64748b", "width": 5, "dash": "dot"},
                    "name": "井轨迹投影",
                    "hovertemplate": "well trajectory projection<extra></extra>",
                }
            )

        for _, row in frame.iterrows():
            cid = int(row["cluster_id"])
            x = float(row["x_center_m"])
            prior = float(row["prior_half_length_m"]) * args.length_scale
            observed = float(row["observed_half_length_m"]) * args.length_scale
            obs_factor = float(row["fiber_observation_factor"]) if "fiber_observation_factor" in row else 1.0

            for side in [-1.0, 1.0]:
                traces_3d.append(make_mesh_trace(row, side))

            for name, y_value, z_value, color in [
                ("PKN 先验缝长", prior, 13.0, "#f97316"),
                ("光纤等效观测缝长", observed, 17.0, "#111827"),
            ]:
                traces_3d.append(
                    {
                        "type": "scatter3d",
                        "x": [x, x],
                        "y": [-y_value, y_value],
                        "z": [z_value, z_value],
                        "mode": "lines",
                        "line": {"color": color, "width": 5},
                        "name": f"C{cid} {name}",
                        "hovertemplate": (
                            f"C{cid} {name}<br>"
                            f"fiber obs factor={obs_factor:.3f}<br>"
                            f"cum liquid score={float(row.get('cum_liquid_score', 1.0)):.3f}<br>"
                            f"cum sand score={float(row.get('cum_sand_score', 1.0)):.3f}<extra></extra>"
                        ),
                    }
                )

            traces_3d.append(
                {
                    "type": "scatter3d",
                    "x": [x],
                    "y": [0],
                    "z": [height / 2 + 3],
                    "mode": "text",
                    "text": [f"C{cid}"],
                    "textfont": {"color": "#dc2626", "size": 12},
                    "name": f"C{cid}",
                    "hovertemplate": f"C{cid}<extra></extra>",
                }
            )

        frames.append(
            {
                "time_s": float(time_s),
                "prior_error": float(h["prior_error"]),
                "posterior_error": float(h["posterior_error"]),
                "within_15_percent": bool(h["within_15_percent"]),
                "traces3d": traces_3d,
                "clusters": frame["cluster_id"].astype(int).tolist(),
                "prior_lengths": frame["prior_half_length_m"].astype(float).tolist(),
                "observed_lengths": frame["observed_half_length_m"].astype(float).tolist(),
                "posterior_lengths": frame["half_length_m"].astype(float).tolist(),
                "decision": decision_by_time.get(round(float(time_s), 6), default_agent_decision(float(time_s))),
            }
        )

    payload_json = json.dumps({"frames": frames, "aperture_max": aperture_max, "max_x": max_x}, ensure_ascii=False)
    try:
        from plotly.offline import get_plotlyjs

        plotly_script = f"<script>{get_plotlyjs()}</script>"
    except Exception as exc:
        raise RuntimeError("无法生成离线3D页面：Plotly本地资源不可用。") from exc
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>3Dfrac PKN + EnKF 连续播放演示</title>
  {plotly_script}
  <style>
    :root {{ --bg:#eef2f7; --panel:#fff; --ink:#0f172a; --muted:#64748b; --line:#dbe3ef; --green:#16a34a; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; }}
    header {{ padding:18px 28px 14px; background:linear-gradient(135deg,#0f172a,#164e63); color:white; box-shadow:0 8px 24px rgba(15,23,42,.18); }}
    header h1 {{ margin:0 0 6px; font-size:24px; }}
    header p {{ margin:0; color:#cbd5e1; font-size:14px; }}
    .controls {{ margin:16px 20px 12px; padding:14px 16px; border-radius:14px; background:var(--panel); display:grid; grid-template-columns:auto auto auto 1fr auto auto; gap:12px; align-items:center; box-shadow:0 8px 24px rgba(15,23,42,.08); }}
    button {{ border:0; border-radius:10px; background:#0f172a; color:white; padding:10px 16px; font-weight:700; cursor:pointer; }}
    button.secondary {{ background:#334155; }}
    input[type=range] {{ width:100%; }}
    select {{ padding:9px 12px; border:1px solid var(--line); border-radius:10px; background:white; }}
    .time-label {{ font-weight:800; min-width:160px; }}
    .dashboard {{ margin:0 20px 20px; display:grid; grid-template-columns:minmax(640px,1.42fr) minmax(460px,.9fr); gap:16px; align-items:start; }}
    .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:18px; overflow:hidden; box-shadow:0 8px 24px rgba(15,23,42,.08); }}
    .panel-title {{ padding:12px 16px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:12px; align-items:center; }}
    .panel-title h2 {{ margin:0; font-size:17px; }}
    .panel-title span {{ color:var(--muted); font-size:13px; }}
    #plot3d {{ height:720px; }}
    #plot2d {{ height:520px; }}
    .cards {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; padding:0 14px 14px; }}
    .card {{ border:1px solid var(--line); border-radius:14px; padding:12px; background:#f8fafc; }}
    .card .label {{ color:var(--muted); font-size:12px; font-weight:700; }}
    .card .value {{ margin-top:6px; font-size:22px; font-weight:900; }}
    .agent-card {{ margin:0 14px 14px; border:1px solid var(--line); border-radius:16px; background:linear-gradient(180deg,#ffffff,#f8fafc); padding:14px; }}
    .agent-head {{ display:flex; justify-content:space-between; gap:12px; align-items:center; margin-bottom:10px; }}
    .agent-head h3 {{ margin:0; font-size:16px; }}
    .pill {{ display:inline-flex; align-items:center; border-radius:999px; padding:5px 10px; font-size:12px; font-weight:900; color:white; background:#64748b; }}
    .pill.low {{ background:#16a34a; }} .pill.medium {{ background:#f59e0b; }} .pill.high {{ background:#dc2626; }} .pill.unknown {{ background:#64748b; }}
    .agent-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
    .agent-field {{ border:1px solid var(--line); border-radius:12px; padding:10px; background:white; }}
    .agent-field.full {{ grid-column:1 / -1; }}
    .agent-label {{ font-size:12px; font-weight:800; color:var(--muted); margin-bottom:5px; }}
    .agent-text {{ font-size:14px; line-height:1.55; white-space:pre-wrap; }}
    .agent-list {{ margin:0; padding-left:18px; font-size:13px; line-height:1.55; }}
    .forbidden {{ color:#b91c1c; }}
    .ok {{ color:var(--green); }} .bad {{ color:#dc2626; }}
    @media (max-width:1100px) {{ .dashboard {{ grid-template-columns:1fr; }} .controls {{ grid-template-columns:auto auto auto 1fr; }} #plot3d {{ height:620px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>3Dfrac 第 8 段：PKN 正演 + 光纤观测 + EnKF 反演校正连续演示</h1>
    <p>左侧展示 3D 裂缝扩展形态，右侧只展示每簇缝长。播放过程体现“PKN正演预测 -> 光纤观测约束 -> EnKF更新PKN参数 -> 参数更新后再正演”。</p>
  </header>
  <div class="controls">
    <button id="playBtn">播放</button>
    <button id="prevBtn" class="secondary">上一帧</button>
    <button id="nextBtn" class="secondary">下一帧</button>
    <input id="frameSlider" type="range" min="0" max="{max(len(frames)-1,0)}" value="0" />
    <select id="speedSelect"><option value="1200">慢速</option><option value="750" selected>正常</option><option value="350">快速</option></select>
    <div id="timeLabel" class="time-label">t=0s</div>
  </div>
  <main class="dashboard">
    <section class="panel"><div class="panel-title"><h2>3D 裂缝形态</h2><span>表面=EnKF更新参数后PKN再正演，橙线=PKN先验，黑线=光纤等效观测</span></div><div id="plot3d"></div></section>
    <section class="panel"><div class="panel-title"><h2>2D 缝长对比</h2><span>按簇展示半长，不展示缝高/缝宽</span></div><div id="plot2d"></div><div class="cards">
      <div class="card"><div class="label">PKN 先验误差</div><div id="priorError" class="value">--</div></div>
      <div class="card"><div class="label">参数更新后PKN误差</div><div id="postError" class="value">--</div></div>
      <div class="card"><div class="label">15% 目标</div><div id="targetStatus" class="value">--</div></div>
      <div class="card"><div class="label">当前帧</div><div id="stepStatus" class="value">--</div></div>
    </div>
    <div class="agent-card">
      <div class="agent-head"><h3>知识嵌入智能体风险建议</h3><span id="riskPill" class="pill unknown">--</span></div>
      <div class="agent-grid">
        <div class="agent-field"><div class="agent-label">不确定性</div><div id="uncertaintyText" class="agent-text">--</div></div>
        <div class="agent-field"><div class="agent-label">审批级别</div><div id="approvalText" class="agent-text">--</div></div>
        <div class="agent-field full"><div class="agent-label">主要风险</div><div id="mainRiskText" class="agent-text">--</div></div>
        <div class="agent-field full"><div class="agent-label">推荐动作</div><div id="recommendationText" class="agent-text">--</div></div>
        <div class="agent-field"><div class="agent-label">候选动作</div><ul id="actionList" class="agent-list"></ul></div>
        <div class="agent-field"><div class="agent-label forbidden">禁止/需避免动作</div><ul id="forbiddenList" class="agent-list forbidden"></ul></div>
        <div class="agent-field full"><div class="agent-label">确认流程与证据</div><div id="evidenceText" class="agent-text">--</div></div>
      </div>
    </div></section>
  </main>
  <script>
    const DEMO = {payload_json};
    let index = 0, timer = null, currentCamera = {{eye:{{x:1.5,y:-1.6,z:1.1}}}};
    const playBtn = document.getElementById("playBtn"), slider = document.getElementById("frameSlider"), speedSelect = document.getElementById("speedSelect");
    function pct(v) {{ return v === null || v === undefined || Number.isNaN(v) ? "--" : (v*100).toFixed(2)+"%"; }}
    function layout3d(frame) {{ return {{ margin:{{l:0,r:0,t:38,b:0}}, title:{{text:`t=${{frame.time_s.toFixed(0)}}s | PKN error=${{pct(frame.prior_error)}} | EnKF error=${{pct(frame.posterior_error)}}`,font:{{size:15}}}}, showlegend:false, scene:{{ xaxis:{{title:"井筒方向 X (m)"}}, yaxis:{{title:"缝长方向 Y (缩放显示)"}}, zaxis:{{title:"高度 Z (m)"}}, aspectmode:"data", camera:currentCamera }} }}; }}
    function render2d(frame) {{
      const labels = frame.clusters.map(c => "C"+c);
      const traces = [
        {{type:"bar", x:labels, y:frame.prior_lengths, name:"PKN 先验缝长", marker:{{color:"#f97316"}}, hovertemplate:"%{{x}}<br>PKN=%{{y:.2f}} m<extra></extra>"}},
        {{type:"bar", x:labels, y:frame.observed_lengths, name:"光纤等效观测", marker:{{color:"#111827"}}, hovertemplate:"%{{x}}<br>Obs=%{{y:.2f}} m<extra></extra>"}},
        {{type:"bar", x:labels, y:frame.posterior_lengths, name:"参数更新后PKN缝长", marker:{{color:"#16a34a"}}, hovertemplate:"%{{x}}<br>Updated PKN=%{{y:.2f}} m<extra></extra>"}}
      ];
      const layout = {{ margin:{{l:58,r:18,t:34,b:58}}, title:{{text:"每簇裂缝半长对比",font:{{size:15}}}}, barmode:"group", yaxis:{{title:"裂缝半长 L (m)",rangemode:"tozero"}}, xaxis:{{title:"簇号"}}, legend:{{orientation:"h",x:0,y:-.22}} }};
      Plotly.react("plot2d", traces, layout, {{responsive:true,displaylogo:false}});
    }}
    function updateCards(frame) {{
      document.getElementById("priorError").textContent = pct(frame.prior_error);
      document.getElementById("postError").textContent = pct(frame.posterior_error);
      const target = document.getElementById("targetStatus");
      target.textContent = frame.within_15_percent ? "达标" : "未达标";
      target.className = "value " + (frame.within_15_percent ? "ok" : "bad");
      document.getElementById("stepStatus").textContent = `${{index+1}} / ${{DEMO.frames.length}}`;
      document.getElementById("timeLabel").textContent = `t=${{frame.time_s.toFixed(0)}}s`;
    }}
    function riskText(risk) {{
      const map = {{low:"低风险", medium:"中风险", high:"高风险", unknown:"未接入"}};
      return map[risk] || risk || "--";
    }}
    function listText(items, formatter) {{
      if (!Array.isArray(items) || !items.length) return "<li>无</li>";
      return items.map(item => `<li>${{formatter(item)}}</li>`).join("");
    }}
    function renderAgent(decision) {{
      decision = decision || {{}};
      const risk = decision.risk_level || "unknown";
      const pill = document.getElementById("riskPill");
      pill.className = "pill " + risk;
      pill.textContent = riskText(risk);
      document.getElementById("uncertaintyText").textContent = decision.uncertainty || "--";
      document.getElementById("approvalText").textContent = `${{decision.approval_level || "--"}} / 人工确认=${{decision.requires_confirmation ? "是" : "否"}}`;
      document.getElementById("mainRiskText").textContent = decision.main_risk || "--";
      document.getElementById("recommendationText").textContent = decision.recommendation || "--";
      document.getElementById("actionList").innerHTML = listText(decision.action_candidates, item => {{
        if (typeof item === "string") return item;
        const clusters = Array.isArray(item.target_clusters) && item.target_clusters.length ? `，目标簇=${{item.target_clusters.join(",")}}` : "";
        return `${{item.action || "--"}}（${{item.type || "action"}}${{clusters}}）`;
      }});
      document.getElementById("forbiddenList").innerHTML = listText(decision.forbidden_actions, item => String(item));
      const ev = decision.evidence || {{}};
      const confirm = decision.confirmation_flow || {{}};
      document.getElementById("evidenceText").textContent =
        `确认状态：${{confirm.status || "--"}}；原因：${{confirm.reason || "--"}}\n` +
        `EnKF后验误差：${{pct(ev.posterior_error)}}；15%目标：${{ev.within_15_percent ? "达标" : "未达标/未知"}}；` +
        `分簇不均衡：${{ev.fiber_balance_imbalance ?? "--"}}；观测因子跨度：${{ev.fiber_factor_spread ?? "--"}}；` +
        `高响应簇：${{Array.isArray(ev.dominant_clusters) ? ev.dominant_clusters.join(",") : "--"}}`;
    }}
    function renderFrame(nextIndex) {{
      if (!DEMO.frames.length) return;
      index = (nextIndex + DEMO.frames.length) % DEMO.frames.length;
      const frame = DEMO.frames[index];
      slider.value = index;
      Plotly.react("plot3d", frame.traces3d, layout3d(frame), {{responsive:true,displaylogo:false}});
      render2d(frame); updateCards(frame); renderAgent(frame.decision);
    }}
    function startPlayback() {{ if (timer) return; playBtn.textContent="暂停"; timer=setInterval(()=>renderFrame(index+1), Number(speedSelect.value)); }}
    function stopPlayback() {{ if (!timer) return; clearInterval(timer); timer=null; playBtn.textContent="播放"; }}
    playBtn.addEventListener("click", () => timer ? stopPlayback() : startPlayback());
    document.getElementById("prevBtn").addEventListener("click", () => {{ stopPlayback(); renderFrame(index-1); }});
    document.getElementById("nextBtn").addEventListener("click", () => {{ stopPlayback(); renderFrame(index+1); }});
    slider.addEventListener("input", e => {{ stopPlayback(); renderFrame(Number(e.target.value)); }});
    speedSelect.addEventListener("change", () => {{ if (timer) {{ stopPlayback(); startPlayback(); }} }});
    document.getElementById("plot3d").on("plotly_relayout", e => {{ if (e["scene.camera"]) currentCamera = e["scene.camera"]; }});
    renderFrame(0);
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path


def main() -> None:
    args = build_args()
    closed_loop_args = make_closed_loop_args(args)
    result = run_demo(closed_loop_args)
    table = result["table"]
    history = result["history"]
    state = SimpleNamespace(step_index=len(history) - 1, time_steps=history["time_s"].to_numpy(dtype=float))

    if args.backend == "plotly-html":
        agent_decisions = load_agent_decisions(args.agent_decisions_json)
        html_path = write_plotly_html_v2(table, history, args, args.html, agent_decisions=agent_decisions)
        if args.open:
            webbrowser.open(html_path.resolve().as_uri())
        print(json.dumps({"backend": "plotly-html", "html": str(html_path)}, ensure_ascii=False, indent=2))
        return

    if pv is None:
        raise RuntimeError("PyVista backend requested but pyvista is not installed. Use --backend plotly-html.")

    if args.screenshot_only:
        pv.start_xvfb(wait=0.1) if hasattr(pv, "start_xvfb") else None
    if args.backend == "qt" and not args.screenshot_only:
        try:
            from pyvistaqt import BackgroundPlotter

            plotter = BackgroundPlotter(window_size=tuple(args.window_size), title="3Dfrac PKN + EnKF PyVista Qt Demo")
        except Exception as exc:
            print(f"Qt backend unavailable, fallback to native PyVista. Reason: {exc}", file=sys.stderr)
            plotter = pv.Plotter(window_size=tuple(args.window_size), off_screen=False)
    else:
        plotter = pv.Plotter(window_size=tuple(args.window_size), off_screen=args.screenshot_only)
    plotter.ren_win.SetMultiSamples(0)
    plotter.enable_trackball_style()
    build_scene(plotter, table, history, state, args)

    def slider_callback(value: float) -> None:
        state.step_index = int(round(value))
        build_scene(plotter, table, history, state, args)

    plotter.add_slider_widget(
        slider_callback,
        rng=[0, len(state.time_steps) - 1],
        value=state.step_index,
        title="time step",
        pointa=(0.18, 0.08),
        pointb=(0.82, 0.08),
        style="modern",
    )

    if args.screenshot_only:
        screenshot_path = Path(args.screenshot)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(screenshot_path))
        plotter.close()
        print(json.dumps({"screenshot": str(screenshot_path), "time_steps": len(state.time_steps)}, ensure_ascii=False, indent=2))
        return

    if args.backend == "qt":
        plotter.app.exec_()
    else:
        plotter.show(title="3Dfrac PKN + EnKF PyVista Interactive Demo", interactive=True)


if __name__ == "__main__":
    main()
