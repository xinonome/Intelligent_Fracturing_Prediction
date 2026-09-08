"""Cluster response and balance metrics for fiber-informed allocation.

The inputs are interpreted DAS/FracMonitor cluster observations, not raw DAS
amplitude.  This module deliberately keeps observation quality and geometry
provenance visible.  A missing geometry or an invalid first-response event is
``unknown`` and cannot become a Piggy-Bank transfer signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResponseMetricConfig:
    """Configuration for first-response and balance metrics."""

    response_increment_threshold_m3: float = 1.0e-8
    minimum_cumulative_volume_m3: float = 1.0e-8
    slow_fast_score_threshold: float = 0.20
    target_shares: tuple[float, ...] | None = None
    reference_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _seconds_axis(values: Iterable[object]) -> np.ndarray:
    series = pd.Series(list(values))
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_values = numeric.to_numpy(dtype=float)
    # pandas stores datetime64 as nanoseconds when coerced to numeric.  Treat
    # only ordinary elapsed-second axes as numeric; otherwise parse datetime.
    finite_numeric = numeric_values[np.isfinite(numeric_values)]
    if len(finite_numeric) >= 2 and float(np.nanmax(np.abs(finite_numeric))) < 1.0e9:
        first = float(numeric_values[np.isfinite(numeric_values)][0])
        return numeric_values - first
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().sum() == 0:
        return np.full(len(parsed), np.nan, dtype=float)
    first = parsed.dropna().iloc[0]
    return (parsed - first).dt.total_seconds().to_numpy(dtype=float)


def _normalise(values: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total > 1.0e-12:
        return values / total
    if fallback is not None:
        fallback = np.asarray(fallback, dtype=float)
        fallback = np.clip(np.nan_to_num(fallback, nan=0.0), 0.0, None)
        fallback_total = float(fallback.sum())
        if fallback_total > 1.0e-12:
            return fallback / fallback_total
    return np.full(len(values), 1.0 / max(len(values), 1), dtype=float)


def balance_indices(shares: Iterable[float], target_shares: Iterable[float] | None = None) -> dict[str, float]:
    """Return interpretable balance indicators for a cluster share vector.

    ``balance_degree`` is one minus the total-variation distance to the target
    distribution.  It is therefore 1 when the vector equals the target and
    approaches 0 as the allocation concentrates away from it.  The target is
    not assumed to be equal splitting when one is supplied.
    """

    share = _normalise(np.asarray(list(shares), dtype=float))
    if target_shares is None:
        target = np.full(len(share), 1.0 / max(len(share), 1), dtype=float)
    else:
        target = _normalise(np.asarray(list(target_shares), dtype=float))
        if len(target) != len(share):
            raise ValueError("target_shares must have the same length as shares")
    absolute_deviation = float(np.abs(share - target).sum())
    balance_degree = float(np.clip(1.0 - 0.5 * absolute_deviation, 0.0, 1.0))
    gini = float(np.abs(share[:, None] - share[None, :]).sum() / (2.0 * len(share) * max(float(share.sum()), 1.0e-12)))
    positive_share = share[share > 0.0]
    entropy = float(-np.sum(positive_share * np.log(positive_share)))
    entropy_normalized = float(entropy / max(np.log(max(len(share), 2)), 1.0e-12))
    return {
        "balance_degree": balance_degree,
        "total_variation_to_target": 0.5 * absolute_deviation,
        "gini": gini,
        "entropy_normalized": float(np.clip(entropy_normalized, 0.0, 1.0)),
        "max_share": float(np.max(share, initial=0.0)),
        "min_share": float(np.min(share, initial=0.0)),
    }


def _geometry_map(geometry: pd.DataFrame | None) -> dict[int, dict[str, object]]:
    if geometry is None or geometry.empty:
        return {}
    result: dict[int, dict[str, object]] = {}
    for _, row in geometry.iterrows():
        try:
            cluster_id = int(row.get("cluster_id"))
        except (TypeError, ValueError):
            continue
        result[cluster_id] = row.to_dict()
    return result


def _cluster_distance(row: dict[str, object], reference_xyz: tuple[float, float, float]) -> tuple[float, str]:
    for key in ("distance_m", "distance_to_response_m", "response_distance_m"):
        value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
        if pd.notna(value) and float(value) >= 0.0:
            return float(value), "configured_distance"
    aliases = {
        "x": ("east_m", "x_m", "east", "x"),
        "y": ("north_m", "y_m", "north", "y"),
        "z": ("vertical_depth_m", "tvd_m", "z_m", "vertical_depth", "z"),
    }
    xyz: list[float] = []
    for axis in ("x", "y", "z"):
        value = np.nan
        for key in aliases[axis]:
            candidate = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
            if pd.notna(candidate):
                value = float(candidate)
                break
        xyz.append(value)
    if not all(np.isfinite(xyz)):
        return np.nan, "geometry_unavailable"
    return float(np.linalg.norm(np.asarray(xyz) - np.asarray(reference_xyz, dtype=float))), "configured_cluster_geometry"


def _target_for_clusters(cluster_ids: list[int], config: ResponseMetricConfig) -> np.ndarray:
    if config.target_shares is None:
        return np.full(len(cluster_ids), 1.0 / max(len(cluster_ids), 1), dtype=float)
    target = np.asarray(config.target_shares, dtype=float)
    if len(target) != len(cluster_ids):
        raise ValueError("ResponseMetricConfig.target_shares length must match cluster count")
    return _normalise(target)


def _control_score(efficiencies: np.ndarray, observed: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Positive score means slow/under-allocated; negative means fast/dominant."""

    finite = np.isfinite(efficiencies)
    score = np.full(len(observed), np.nan, dtype=float)
    if not finite.any():
        return score
    median = float(np.nanmedian(efficiencies[finite]))
    mad = float(np.nanmedian(np.abs(efficiencies[finite] - median)))
    scale = max(1.4826 * mad, abs(median) * 0.05, 1.0e-12)
    # High response efficiency means less future liquid is needed; it gets a
    # negative contribution.  The target-share term prevents a fast cluster
    # that is already under-allocated from being treated as a release source.
    efficiency_signal = -np.clip((efficiencies - median) / scale, -3.0, 3.0) / 3.0
    share_gap = np.clip((target - observed) / np.maximum(target, 1.0e-9), -1.0, 1.0)
    score[finite] = np.clip(0.65 * efficiency_signal[finite] + 0.35 * share_gap[finite], -1.0, 1.0)
    return score


def compute_segment_response_metrics(
    controls: pd.DataFrame,
    geometry: pd.DataFrame | None = None,
    *,
    config: ResponseMetricConfig | None = None,
    as_of_step: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Compute first-response metrics and a time series of balance metrics.

    Returns ``(cluster_metrics, balance_history, metadata)``.  The first table
    has one row per cluster.  The second table has one row per valid time step.
    When ``as_of_step`` is supplied, only data available up to that step is
    used; this supports leakage-free open-loop replay.
    """

    config = config or ResponseMetricConfig()
    if controls is None or controls.empty:
        empty = pd.DataFrame()
        return empty, empty, {"status": "empty", "reason": "no_cluster_controls"}
    required = {"step", "cluster_id", "liquid_volume_m3", "cumulative_liquid_volume_m3"}
    missing = sorted(required - set(controls.columns))
    if missing:
        raise ValueError(f"controls missing required response columns: {missing}")
    frame = controls.copy()
    frame["step"] = pd.to_numeric(frame["step"], errors="coerce")
    frame["cluster_id"] = pd.to_numeric(frame["cluster_id"], errors="coerce")
    frame = frame.dropna(subset=["step", "cluster_id"]).copy()
    frame["step"] = frame["step"].astype(int)
    frame["cluster_id"] = frame["cluster_id"].astype(int)
    if as_of_step is not None:
        frame = frame[frame["step"] <= int(as_of_step)].copy()
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame(), {"status": "empty", "reason": "no_data_before_as_of_step"}
    frame = frame.sort_values(["step", "cluster_id"]).reset_index(drop=True)
    cluster_ids = sorted(frame["cluster_id"].unique().tolist())
    target = _target_for_clusters(cluster_ids, config)
    target_by_cluster = dict(zip(cluster_ids, target))
    geometry_by_cluster = _geometry_map(geometry)
    # FracMonitor exports often number clusters 0..5 while configuration
    # tables and display geometry use 1..6.  Align this convention at the
    # module boundary and record the geometry source separately.
    if cluster_ids and not all(cluster_id in geometry_by_cluster for cluster_id in cluster_ids):
        one_based = {cluster_id - 1: value for cluster_id, value in geometry_by_cluster.items()}
        if all(cluster_id in one_based for cluster_id in cluster_ids):
            geometry_by_cluster = one_based
    time_axis = _seconds_axis(frame["time"] if "time" in frame.columns else frame["step"])
    frame["_time_seconds"] = time_axis
    events: list[dict[str, object]] = []
    for cluster_id in cluster_ids:
        group = frame[frame["cluster_id"] == cluster_id].sort_values("step").copy()
        cumulative = pd.to_numeric(group["cumulative_liquid_volume_m3"], errors="coerce").to_numpy(dtype=float)
        incremental = pd.to_numeric(group["liquid_volume_m3"], errors="coerce").to_numpy(dtype=float)
        diff = np.r_[cumulative[0] if len(cumulative) else 0.0, np.diff(cumulative)]
        signal = np.where(np.isfinite(incremental) & (incremental > 0.0), incremental, diff)
        valid = np.isfinite(signal) & (signal > float(config.response_increment_threshold_m3)) & np.isfinite(cumulative)
        valid &= cumulative >= float(config.minimum_cumulative_volume_m3)
        row: dict[str, object] = {
            "cluster_id": int(cluster_id),
            "target_share": float(target_by_cluster[cluster_id]),
            "first_response_available": bool(valid.any()),
            "first_response_step": np.nan,
            "first_response_time_s": np.nan,
            "first_response_cumulative_liquid_volume_m3": np.nan,
            "distance_to_response_m": np.nan,
            "distance_source": "geometry_unavailable",
            "volume_normalized_response_efficiency_m_per_m3": np.nan,
            "time_based_response_rate_m_per_s": np.nan,
            "geometry_status": "geometry_unavailable",
            "response_status": "unknown",
        }
        if valid.any():
            index = int(np.flatnonzero(valid)[0])
            selected = group.iloc[index]
            response_step = int(selected["step"])
            response_time = float(selected["_time_seconds"])
            geometry_row = geometry_by_cluster.get(int(cluster_id), {})
            distance, distance_source = _cluster_distance(geometry_row, config.reference_xyz_m)
            row.update({
                "first_response_step": response_step,
                "first_response_time_s": response_time,
                "first_response_cumulative_liquid_volume_m3": float(cumulative[index]),
                "distance_to_response_m": distance,
                "distance_source": distance_source,
                "geometry_status": "valid" if np.isfinite(distance) else "geometry_unavailable",
            })
            if np.isfinite(distance) and cumulative[index] > float(config.minimum_cumulative_volume_m3):
                row["volume_normalized_response_efficiency_m_per_m3"] = float(distance / cumulative[index])
                if response_time > 0.0:
                    row["time_based_response_rate_m_per_s"] = float(distance / response_time)
                row["response_status"] = "valid"
            else:
                row["response_status"] = "unknown"
        events.append(row)
    event_frame = pd.DataFrame(events)

    # The current observed share is the latest cumulative volume available at
    # the as-of point.  It is deliberately kept separate from model weights.
    latest = frame.sort_values("step").groupby("cluster_id", as_index=False).tail(1).set_index("cluster_id")
    cumulative_latest = pd.to_numeric(latest["cumulative_liquid_volume_m3"], errors="coerce").reindex(cluster_ids).to_numpy(dtype=float)
    observed = _normalise(cumulative_latest, fallback=target)
    event_frame["observed_liquid_share"] = [float(observed[cluster_ids.index(int(cid))]) for cid in event_frame["cluster_id"]]
    efficiencies = event_frame["volume_normalized_response_efficiency_m_per_m3"].to_numpy(dtype=float)
    scores = _control_score(efficiencies, observed, target)
    event_frame["control_score"] = scores
    threshold = float(config.slow_fast_score_threshold)
    labels: list[str] = []
    for score, status in zip(scores, event_frame["response_status"]):
        if status != "valid" or not np.isfinite(score):
            labels.append("unknown")
        elif score >= threshold:
            labels.append("slow")
        elif score <= -threshold:
            labels.append("fast")
        else:
            labels.append("neutral")
    event_frame["control_state"] = labels

    balance_rows: list[dict[str, object]] = []
    for step, group in frame.groupby("step", sort=True):
        cumulative = group.groupby("cluster_id")["cumulative_liquid_volume_m3"].last().reindex(cluster_ids).to_numpy(dtype=float)
        shares = _normalise(cumulative, fallback=target)
        indices = balance_indices(shares, target)
        step_time_values = pd.to_numeric(group["_time_seconds"], errors="coerce")
        row = {"step": int(step), "time_s": float(step_time_values.mean()) if step_time_values.notna().any() else float(step)}
        row.update(indices)
        row["cluster_count"] = len(cluster_ids)
        row["observed_total_liquid_volume_m3"] = float(np.nansum(cumulative))
        balance_rows.append(row)
    balance_frame = pd.DataFrame(balance_rows)
    metadata = {
        "status": "ok",
        "cluster_count": len(cluster_ids),
        "as_of_step": None if as_of_step is None else int(as_of_step),
        "cluster_geometry_available": bool(event_frame["geometry_status"].eq("valid").all()) if not event_frame.empty else False,
        "valid_response_count": int(event_frame["response_status"].eq("valid").sum()),
        "unknown_response_count": int(event_frame["response_status"].ne("valid").sum()),
        "target_share_source": "configured" if config.target_shares is not None else "equal_split_default",
        "metric_definition": "distance_to_response / first_response_cumulative_liquid_volume",
        "time_rate_definition": "distance_to_response / first_response_time",
        "observation_layer": "interpreted_DAS_or_FracMonitor_cluster_observation",
    }
    return event_frame, balance_frame, metadata


def response_metric_snapshot(
    events: pd.DataFrame,
    current_controls: pd.DataFrame,
    *,
    config: ResponseMetricConfig | None = None,
    as_of_step: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, object]]:
    """Create a fast leakage-free snapshot from precomputed response events.

    First-response geometry and cumulative volume are calculated once from the
    full table, but an event is only made available when its own response step
    is no later than ``as_of_step``.  This is equivalent to replaying the
    history up to the decision point without the quadratic dataframe work.
    """

    config = config or ResponseMetricConfig()
    if events is None or events.empty or current_controls is None or current_controls.empty:
        return pd.DataFrame(), {}, {"status": "empty"}
    metrics = events.copy().sort_values("cluster_id").reset_index(drop=True)
    cluster_ids = metrics["cluster_id"].astype(int).tolist()
    if config.target_shares is None:
        target = _normalise(metrics["target_share"].to_numpy(dtype=float))
    else:
        target = _target_for_clusters(cluster_ids, config)
    cumulative = current_controls.groupby("cluster_id")["cumulative_liquid_volume_m3"].last().reindex(cluster_ids).to_numpy(dtype=float)
    observed = _normalise(cumulative, fallback=target)
    metrics["observed_liquid_share"] = observed
    event_steps = pd.to_numeric(metrics["first_response_step"], errors="coerce").to_numpy(dtype=float)
    # Some pandas backends expose the result of ``to_numpy`` as a read-only
    # view.  ``available`` is intentionally narrowed in place below, so make
    # an owned writable array instead of mutating the DataFrame-backed view.
    available = metrics["response_status"].eq("valid").to_numpy(dtype=bool, copy=True)
    if as_of_step is not None:
        available &= np.isfinite(event_steps) & (event_steps <= float(as_of_step))
    efficiency = metrics["volume_normalized_response_efficiency_m_per_m3"].to_numpy(dtype=float, copy=True)
    efficiency[~available] = np.nan
    metrics["response_available_as_of_step"] = available
    metrics["control_score"] = _control_score(efficiency, observed, target)
    threshold = float(config.slow_fast_score_threshold)
    states: list[str] = []
    for available_flag, score in zip(available, metrics["control_score"]):
        if not available_flag or not np.isfinite(score):
            states.append("unknown")
        elif score >= threshold:
            states.append("slow")
        elif score <= -threshold:
            states.append("fast")
        else:
            states.append("neutral")
    metrics["control_state"] = states
    indices = balance_indices(observed, target)
    indices["observed_total_liquid_volume_m3"] = float(np.nansum(cumulative))
    return metrics, indices, {
        "status": "ok",
        "cluster_geometry_available": bool(metrics["geometry_status"].eq("valid").all()),
        "valid_response_count": int(available.sum()),
        "unknown_response_count": int((~available).sum()),
        "as_of_step": None if as_of_step is None else int(as_of_step),
    }


def build_response_metric_snapshots(
    controls: pd.DataFrame,
    geometry: pd.DataFrame | None = None,
    *,
    config: ResponseMetricConfig | None = None,
    steps: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build leakage-free per-step cluster snapshots for open-loop replay."""

    config = config or ResponseMetricConfig()
    if steps is None:
        steps = sorted(pd.to_numeric(controls["step"], errors="coerce").dropna().astype(int).unique().tolist())
    metric_rows: list[pd.DataFrame] = []
    balance_rows: list[pd.DataFrame] = []
    for step in steps:
        events, balance, _ = compute_segment_response_metrics(controls, geometry, config=config, as_of_step=int(step))
        if not events.empty:
            events = events.copy()
            events.insert(0, "as_of_step", int(step))
            metric_rows.append(events)
        if not balance.empty:
            current = balance.sort_values("step").tail(1).copy()
            current.insert(0, "as_of_step", int(step))
            balance_rows.append(current)
    return (
        pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame(),
        pd.concat(balance_rows, ignore_index=True) if balance_rows else pd.DataFrame(),
    )
