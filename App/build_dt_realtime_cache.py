"""Build the lightweight, synchronized DT playback cache used by the APP.

The Qt environment intentionally does not need numpy/pandas.  This script is
run with the algorithm Python environment and converts the real stage-08
sources into one relative-second timeline for the presentation layer.
"""

from __future__ import annotations

import csv
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "DT-Crack"))

from data_fusion.frac_monitor_text_adapter import load_frac_monitor_text
from data_fusion.pressure_schedule_adapter import load_pressure_model_config, load_stage_pressure_schedule
from data_fusion.well_trajectory_adapter import load_well_trajectory
from data_fusion.scenario import load_cluster_geometry
from data_fusion.observation_quality import validate_cluster_controls
from inversion.pressure_only_enkf import PressureOnlyConfig, run_pressure_only_correction


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _resolve_dt_run(run_path: str | Path) -> tuple[Path, Path, Path]:
    path = Path(run_path)
    if path.is_file():
        path = path.parent
    history = path / "direct_observation_history.csv"
    cluster = path / "cluster_share_history.csv"
    summary = path / "summary.json"
    if not history.exists() or not cluster.exists():
        raise FileNotFoundError(f"DT run is incomplete: {path}")
    return history, cluster, summary


def _registered_dt_run() -> tuple[Path, Path, Path] | None:
    registry_path = ROOT / "App" / "config" / "demo_registry.json"
    if not registry_path.exists():
        return None
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        tables = registry.get("modules", {}).get("dt", {}).get("tables", [])
        for value in tables:
            if str(value).endswith("direct_observation_history.csv"):
                candidate = ROOT / str(value)
                if candidate.exists():
                    return _resolve_dt_run(candidate)
    except (OSError, json.JSONDecodeError, FileNotFoundError):
        return None
    return None


def _latest_dt_run(preferred_run: str | Path | None = None) -> tuple[Path, Path, Path]:
    if preferred_run is not None:
        return _resolve_dt_run(preferred_run)
    registered = _registered_dt_run()
    if registered is not None:
        return registered
    candidates = []
    for path in (ROOT / "outputs" / "dt").glob("**/direct_observation_history.csv"):
        try:
            table = _read_csv(path)
        except Exception:
            continue
        if len(table) >= 20 and "time_s" in table.columns:
            candidates.append((path.stat().st_mtime, path))
    if not candidates:
        raise FileNotFoundError("No DT direct_observation_history.csv with at least 20 steps was found")
    candidates.sort(reverse=True)
    history = candidates[0][1]
    cluster = history.with_name("cluster_share_history.csv")
    summary = history.with_name("summary.json")
    if not cluster.exists():
        raise FileNotFoundError(f"Missing cluster history beside {history}")
    return history, cluster, summary


def _interp(values: pd.Series | np.ndarray, source_t: np.ndarray, target_t: np.ndarray) -> list[float]:
    arr = pd.to_numeric(values, errors="coerce").interpolate().ffill().bfill().to_numpy(dtype=float)
    return np.interp(target_t, source_t, arr).astype(float).tolist()


def _build_deep_display_geometry(
    trajectory: pd.DataFrame,
    n_clusters: int,
    cluster_geometry: pd.DataFrame | None = None,
    *,
    vertical_start_m: float = 3000.0,
    north_start_m: float = 77.0,
    north_end_m: float = 1650.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Use measured/configured cluster positions; never fabricate equal splits."""

    if trajectory.empty or n_clusters <= 0 or cluster_geometry is None or cluster_geometry.empty:
        return trajectory.copy(), pd.DataFrame(), {"mode": "empty"}
    result = trajectory.copy().reset_index(drop=True)
    result["source_north_m"] = result["north_m"]
    result["display_north_m"] = result["north_m"]
    result["geometry_provenance"] = "trajectory_source"
    ordered = result.sort_values("measured_depth_m")
    md = ordered["measured_depth_m"].to_numpy(dtype=float)
    positions_frame = cluster_geometry.sort_values("cluster_id").copy()
    positions = []
    for _, item in positions_frame.iterrows():
        cluster_id = int(item["cluster_id"])
        md_value = float(item["md_m"])
        def interp(column: str, optional: str) -> float:
            if optional in item and pd.notna(item[optional]):
                return float(item[optional])
            return float(np.interp(md_value, md, ordered[column].to_numpy(dtype=float)))
        row = {
            "cluster_id": cluster_id,
            "measured_depth_m": md_value,
            "vertical_depth_m": interp("vertical_depth_m", "tvd_m"),
            "north_m": interp("north_m", "north_m"),
            "east_m": interp("east_m", "east_m"),
            "source_north_m": interp("north_m", "north_m"),
            "display_north_m": interp("north_m", "north_m"),
            "geometry_provenance": "configured" if all(pd.notna(item.get(value)) for value in ("tvd_m", "east_m", "north_m")) else "trajectory_interpolated",
        }
        positions.append(row)

    metadata = {
        "mode": "configured_cluster_geometry",
        "cluster_placement": "configured_md_with_optional_trajectory_interpolation",
        "cluster_geometry_rows": int(len(positions)),
    }
    return result, pd.DataFrame(positions), metadata


def build_cache(output: Path, dt_run: str | Path | None = None) -> dict:
    fiber_path = ROOT / "Data" / "3Dfrac" / "光纤本井监测08.txt"
    pressure_path = ROOT / "Data" / "3Dfrac" / "JY84-Z1-stage08-f1.xls"
    trajectory_path = ROOT / "Data" / "3Dfrac" / "JY84-Z1HF-1011.csv"
    history_path, cluster_path, summary_path = _latest_dt_run(dt_run)

    fiber = load_frac_monitor_text(fiber_path)
    checked_controls, fiber_quality = validate_cluster_controls(fiber.controls, expected_clusters=6)
    valid_keys = checked_controls.loc[checked_controls["qc_valid"], ["step", "cluster_id"]].drop_duplicates()
    stage = fiber.stage_info.merge(valid_keys, on=["step", "cluster_id"], how="inner")
    fiber_by_step = (
        stage.groupby("step", as_index=False)
        .agg(
            cumulative_liquid_m3=("cumulative_liquid_volume_m3", "sum"),
            cumulative_sand_t=("cumulative_sand_mass_t", "sum"),
            balance_degree=("balance_degree", "mean"),
            cumulative_balance_degree=("cumulative_balance_degree", "mean"),
        )
        .sort_values("step")
    )

    trajectory = load_well_trajectory(trajectory_path)
    cluster_geometry_path = ROOT / "App" / "config" / "cluster_geometry.csv"
    cluster_geometry, geometry_meta = load_cluster_geometry(cluster_geometry_path, stage_id="08")
    vertical_depth = float(trajectory["vertical_depth_m"].max()) if not trajectory.empty else 3200.0
    measured_depth = float(trajectory["measured_depth_m"].max()) if not trajectory.empty else 5200.0
    pressure, pressure_meta = load_stage_pressure_schedule(
        pressure_path,
        config=load_pressure_model_config(ROOT / "App" / "config" / "pressure_calibration.json"),
        measured_depth_m=measured_depth,
        vertical_depth_m=vertical_depth,
    )

    history = _read_csv(history_path).sort_values("time_s").reset_index(drop=True)
    clusters = _read_csv(cluster_path).sort_values(["cluster_id", "time_s"]).reset_index(drop=True)
    dt_times = history["time_s"].to_numpy(dtype=float)
    pressure_times = pressure["time_s"].to_numpy(dtype=float)
    fiber_times = fiber_by_step["step"].to_numpy(dtype=float)
    max_time = int(max(pressure_times.max(), fiber_times.max(), dt_times.max()))
    # Use a one-second presentation axis. Source files may start at different
    # absolute timestamps, so each adapter is first converted to elapsed time.
    # The first displayed sample is 1 s, matching the acceptance-demo wording.
    timeline = np.arange(1, max_time + 1, dtype=float)

    def p(name: str) -> list[float]:
        return _interp(pressure[name], pressure_times, timeline)

    def f(name: str) -> list[float]:
        return _interp(fiber_by_step[name], fiber_times, timeline)

    def d(name: str, default: float = 0.0) -> list[float]:
        if name not in history:
            return [float(default)] * len(timeline)
        return _interp(history[name], dt_times, timeline)

    arrays: dict[str, list[float]] = {
        "surface_pressure_mpa": p("surface_pressure_mpa"),
        "bottomhole_pressure_mpa": p("bottomhole_pressure_mpa"),
        "observed_bhp_mpa": p("bottomhole_pressure_mpa"),
        "net_pressure_mpa": p("net_pressure_mpa"),
        "flow_rate_m3_min": p("flow_rate_m3_min"),
        "sand_ratio_percent": p("sand_ratio_percent"),
        "fiber_cumulative_liquid_m3": f("cumulative_liquid_m3"),
        "fiber_cumulative_sand_t": f("cumulative_sand_t"),
        "fiber_balance_degree": f("balance_degree"),
        "fiber_cumulative_balance_degree": f("cumulative_balance_degree"),
        "prior_eprime_gpa": d("prior_eprime_gpa"),
        "posterior_eprime_gpa": d("posterior_eprime_gpa"),
        "prior_leakoff_m_sqrt_s": d("prior_leakoff_m_sqrt_s"),
        "posterior_leakoff_m_sqrt_s": d("posterior_leakoff_m_sqrt_s"),
        "prior_viscosity_pa_s": d("prior_viscosity_pa_s"),
        "posterior_viscosity_pa_s": d("posterior_viscosity_pa_s"),
        "prior_min_stress_mpa": d("prior_min_stress_mpa"),
        "posterior_min_stress_mpa": d("posterior_min_stress_mpa"),
        "prior_bhp_mpa": d("prior_bottomhole_pressure_mpa"),
        "posterior_bhp_mpa": d("posterior_bottomhole_pressure_mpa"),
        "prior_liquid_error": d("prior_liquid_tvd"),
        "posterior_liquid_error": d("posterior_liquid_tvd"),
        "prior_sand_error": d("prior_sand_tvd"),
        "posterior_sand_error": d("posterior_sand_tvd"),
        "prior_bhp_error": d("prior_bhp_relative_error"),
        "posterior_bhp_error": d("posterior_bhp_relative_error"),
        "kalman_gain": d("mean_abs_kalman_gain"),
        "step_compute_ms": d("step_compute_ms"),
        "posterior_pkn_bhp_mpa": d("posterior_pkn_bottomhole_pressure_mpa"),
    }

    cluster_arrays: dict[str, dict[str, list[float]]] = {}
    for cluster_id, group in clusters.groupby("cluster_id"):
        group = group.sort_values("time_s")
        source = group["time_s"].to_numpy(dtype=float)
        cluster_arrays[str(int(cluster_id))] = {
            "observed_liquid_share": _interp(group["observed_liquid_share"], source, timeline),
            "observed_sand_share": _interp(group["observed_sand_share"], source, timeline),
            "prior_half_length_m": _interp(group["prior_half_length_m"], source, timeline) if "prior_half_length_m" in group else [],
            "posterior_half_length_m": _interp(group["posterior_half_length_m"], source, timeline),
            "posterior_cluster_factor": _interp(group["posterior_cluster_factor"], source, timeline),
            "posterior_sand_transport_factor": _interp(group["posterior_sand_transport_factor"], source, timeline),
        }
        if "fiber_liquid_allocation" in group:
            cluster_arrays[str(int(cluster_id))]["fiber_liquid_allocation"] = _interp(
                group["fiber_liquid_allocation"], source, timeline
            )

    # Build an auditable, PKN-consistent equivalent length estimate from the
    # measured liquid boundary condition. PKN length scales as q^0.6, so a
    # linear allocation would systematically exaggerate cluster differences.
    # Sand remains a separate transport channel and is not double-counted in
    # the hydraulic length estimate.
    cluster_ids = sorted(cluster_arrays, key=lambda value: int(value))
    posterior_matrix = np.asarray(
        [cluster_arrays[cluster_id]["posterior_half_length_m"] for cluster_id in cluster_ids], dtype=float
    )
    liquid_matrix = np.asarray(
        [cluster_arrays[cluster_id]["observed_liquid_share"] for cluster_id in cluster_ids], dtype=float
    )
    sand_matrix = np.asarray(
        [cluster_arrays[cluster_id]["observed_sand_share"] for cluster_id in cluster_ids], dtype=float
    )
    pkn_length_exponent = 0.6
    estimate_weight = np.power(np.maximum(liquid_matrix, 1.0e-12), pkn_length_exponent)
    weight_sum = np.where(estimate_weight.sum(axis=0) > 1.0e-9, estimate_weight.sum(axis=0), 1.0)
    estimated_matrix = posterior_matrix.sum(axis=0, keepdims=True) * estimate_weight / weight_sum
    length_error = np.abs(posterior_matrix - estimated_matrix).sum(axis=0) / np.maximum(
        estimated_matrix.sum(axis=0), 1.0e-9
    )
    for row_index, cluster_id in enumerate(cluster_ids):
        cluster_arrays[cluster_id]["estimated_half_length_m"] = estimated_matrix[row_index].astype(float).tolist()
    arrays["posterior_length_error"] = length_error.astype(float).tolist()

    display_trajectory, cluster_positions, display_geometry = _build_deep_display_geometry(trajectory, len(cluster_arrays), cluster_geometry)
    trajectory_records = display_trajectory.to_dict(orient="records")
    cluster_position_records = []
    for record in cluster_positions.to_dict(orient="records"):
        # The inversion history uses zero-based cluster IDs.
        record["cluster_id"] = int(record["cluster_id"]) - 1
        cluster_position_records.append(record)
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}

    payload = {
        "schema_version": 1,
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "alignment": {
            "method": "Each source is converted to elapsed seconds from its own first timestamp; values are linearly interpolated onto a common 1-second axis.",
            "common_time_axis": "relative seconds, 1 through max source duration",
            "fiber_end_s": float(fiber_times.max()),
            "pressure_end_s": float(pressure_times.max()),
            "dt_end_s": float(dt_times.max()),
            "fiber_after_end": "not part of DAS validation; UI may show last value with an explicit no-new-observation label",
            "dt_after_end": "hold last PKN/EnKF state for visual continuity; no new inversion step is claimed",
        },
        "sources": {
            "fiber": str(fiber_path),
            "pressure": str(pressure_path),
            "trajectory": str(trajectory_path),
            "dt_history": str(history_path),
            "dt_cluster_history": str(cluster_path),
            "dt_summary": str(summary_path),
        },
        "meta": {
            "well": fiber.meta.get("wellName"),
            "stage": str(fiber.meta.get("stage", "08")),
            "cluster_count": int(len(cluster_arrays)),
            "max_time_s": max_time,
            "fiber_rows": int(len(stage)),
            "pressure_rows": int(len(pressure)),
            "trajectory_rows": int(len(trajectory)),
            "pressure_layout": pressure_meta.get("layout_assumption", {}),
            "pressure_calibration_status": pressure_meta.get("calibration_status", "待校准"),
            "dt_metrics": summary.get("metrics", {}),
            "display_geometry": display_geometry,
            "cluster_geometry": geometry_meta,
            "observation_quality": fiber_quality.to_dict(),
            "length_estimate": {
                "method": "total posterior PKN length allocated by normalized cumulative liquid share^0.6",
                "pkn_length_exponent": 0.6,
                "error": "sum(abs(posterior_half_length - estimated_half_length)) / sum(estimated_half_length)",
                "scope": "PKN-consistent equivalent hydraulic estimate; not independent geometric truth",
            },
        },
        "timeline_s": timeline.astype(int).tolist(),
        "arrays": arrays,
        "clusters": cluster_arrays,
        "trajectory": trajectory_records,
        "cluster_positions": cluster_position_records,
    }
    # Keep both source-conditioned views in one cache so the APP can switch
    # scenarios without rebuilding or mixing their observation claims.
    das_payload = copy.deepcopy(payload)
    das_end = float(fiber_times.max()) if len(fiber_times) else 0.0
    das_indices = [i for i, value in enumerate(payload["timeline_s"]) if float(value) <= das_end]
    for key, values in list(das_payload["arrays"].items()):
        das_payload["arrays"][key] = [values[i] for i in das_indices if i < len(values)]
    das_payload["timeline_s"] = [payload["timeline_s"][i] for i in das_indices]
    das_payload["meta"]["max_time_s"] = das_end
    das_payload["meta"]["scenario_id"] = "das_cluster_observation"
    das_payload["meta"]["observation_mode"] = "pressure_plus_cluster"
    no_das_payload = copy.deepcopy(payload)
    no_das_payload["meta"]["scenario_id"] = "no_das_pressure_only"
    no_das_payload["meta"]["observation_mode"] = "pressure_only"
    no_das_payload["meta"]["cluster_observations"] = "not_available"
    no_das_payload["meta"]["cluster_geometry"] = {"status": "not_used_without_das"}
    no_das_payload["clusters"] = {}
    observed = np.asarray(no_das_payload["arrays"]["bottomhole_pressure_mpa"], dtype=float)
    prior = np.asarray(no_das_payload["arrays"].get("prior_bhp_mpa", observed), dtype=float)
    correction = run_pressure_only_correction(prior, observed, PressureOnlyConfig())
    no_das_payload["arrays"]["prior_bhp_mpa"] = correction["prior_bottomhole_mpa"].tolist()
    no_das_payload["arrays"]["posterior_bhp_mpa"] = correction["posterior_bottomhole_mpa"].tolist()
    no_das_payload["arrays"]["pressure_bias_mpa"] = correction["posterior_bias_mpa"].tolist()
    no_das_payload["alignment"]["scenario_note"] = "Pressure-only conversion and bounded pressure-bias correction; no cluster values inferred."
    payload["scenarios"] = {
        "das_cluster_observation": das_payload,
        "no_das_pressure_only": no_das_payload,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build APP DT synchronized playback cache")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "app" / "dt_realtime_cache.json"))
    parser.add_argument(
        "--dt-run",
        default=None,
        help="Explicit DT run directory or direct_observation_history.csv; otherwise use the registry path.",
    )
    args = parser.parse_args()
    data = build_cache(Path(args.output), dt_run=args.dt_run)
    print(json.dumps({"output": args.output, "max_time_s": data["meta"]["max_time_s"], "cluster_count": data["meta"]["cluster_count"]}, ensure_ascii=False, indent=2))
