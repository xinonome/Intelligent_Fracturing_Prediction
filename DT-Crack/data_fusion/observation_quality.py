"""Quality checks for interpreted DAS/FracMonitor cluster observations.

The table in this module is an interpreted cluster result, not raw DAS
amplitude.  A failed check therefore removes a whole time step from the
current EnKF update and records the reason; it is never silently repaired by
forward filling before assimilation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ObservationQualityReport:
    expected_clusters: int
    observed_clusters: int
    total_steps: int
    valid_steps: int
    invalid_steps: int
    duplicate_rows: int
    duplicate_timestamp_steps: int
    non_monotonic_timestamp_steps: int
    missing_cluster_steps: int
    non_monotonic_cumulative_rows: int
    negative_value_rows: int
    max_gap_s: float
    gap_count: int
    overlap_start_s: float | None
    overlap_end_s: float | None
    pressure_overlap_ratio: float
    allocation_sum_mismatch_steps: int
    total_volume_mismatch_steps: int
    invalid_sample_ratio: float
    valid_ratio: float
    reasons: tuple[str, ...]
    reasons_by_step: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _numeric_axis(values: Any) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if np.isfinite(numeric).sum() >= 2:
        return numeric
    parsed = pd.to_datetime(pd.Series(values), errors="coerce")
    if parsed.notna().sum() < 2:
        return np.asarray([], dtype=float)
    first = parsed.dropna().iloc[0]
    return (parsed - first).dt.total_seconds().to_numpy(dtype=float)


def _column(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series | None:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return None


def validate_cluster_controls(
    controls: pd.DataFrame,
    expected_clusters: int = 6,
    pressure_times: np.ndarray | None = None,
    max_gap_s: float = 5.0,
    stage_totals: pd.DataFrame | None = None,
    volume_tolerance: float = 0.02,
) -> tuple[pd.DataFrame, ObservationQualityReport]:
    """Return per-row/per-step QC flags for EnKF assimilation.

    Checks include timestamp ordering and gaps, duplicate step-cluster rows,
    complete cluster sets, monotone cumulative volumes, non-negative values,
    optional stage-total consistency, allocation normalization and pressure
    overlap.  The returned ``qc_valid`` flag is the only input gate that the
    EnKF caller should use.
    """

    required_columns = {
        "step",
        "cluster_id",
        "time",
        "liquid_volume_m3",
        "sand_mass_t",
        "cumulative_liquid_volume_m3",
        "cumulative_sand_mass_t",
    }
    if controls.empty:
        empty = controls.copy()
        empty["qc_valid"] = pd.Series(dtype=bool)
        empty["qc_reason"] = pd.Series(dtype=str)
        report = ObservationQualityReport(
            expected_clusters=expected_clusters,
            observed_clusters=0,
            total_steps=0,
            valid_steps=0,
            invalid_steps=0,
            duplicate_rows=0,
            duplicate_timestamp_steps=0,
            non_monotonic_timestamp_steps=0,
            missing_cluster_steps=0,
            non_monotonic_cumulative_rows=0,
            negative_value_rows=0,
            max_gap_s=0.0,
            gap_count=0,
            overlap_start_s=None,
            overlap_end_s=None,
            pressure_overlap_ratio=0.0,
            allocation_sum_mismatch_steps=0,
            total_volume_mismatch_steps=0,
            invalid_sample_ratio=0.0,
            valid_ratio=0.0,
            reasons=("empty_observation_table",),
            reasons_by_step={},
        )
        return empty, report

    missing_columns = sorted(required_columns - set(controls.columns))
    if missing_columns:
        raise ValueError(f"cluster observation table is missing columns: {missing_columns}")

    frame = controls.copy().sort_values(["step", "cluster_id"]).reset_index(drop=True)
    frame["step"] = pd.to_numeric(frame["step"], errors="coerce")
    frame["cluster_id"] = pd.to_numeric(frame["cluster_id"], errors="coerce")
    step_values = frame["step"].dropna().astype(int)
    frame["step"] = frame["step"].fillna(-1).astype(int)
    frame["cluster_id"] = frame["cluster_id"].fillna(-1).astype(int)
    total_steps = int(frame["step"].nunique())
    observed_clusters = int(frame.loc[frame["cluster_id"] >= 0, "cluster_id"].nunique())
    reasons_by_step: dict[int, set[str]] = {}

    def add_reason(steps: Any, reason: str) -> None:
        for step in steps:
            try:
                step_int = int(step)
            except (TypeError, ValueError):
                step_int = -1
            reasons_by_step.setdefault(step_int, set()).add(reason)

    duplicate_mask = frame.duplicated(["step", "cluster_id"], keep=False)
    duplicate_rows = int(frame.duplicated(["step", "cluster_id"]).sum())
    add_reason(frame.loc[duplicate_mask, "step"], "duplicate_step_cluster")

    counts = frame.groupby("step")["cluster_id"].nunique()
    incomplete_steps = counts[counts != expected_clusters].index.to_numpy()
    missing_cluster_steps = int(len(incomplete_steps))
    add_reason(incomplete_steps, "incomplete_cluster_set")

    parsed_time = pd.to_datetime(frame["time"], errors="coerce")
    frame["_parsed_time"] = parsed_time
    missing_time_mask = parsed_time.isna()
    add_reason(frame.loc[missing_time_mask, "step"], "invalid_timestamp")
    step_times = (
        frame.loc[~missing_time_mask, ["step", "_parsed_time"]]
        .drop_duplicates("step")
        .sort_values("step")
    )
    time_diffs = step_times["_parsed_time"].diff().dt.total_seconds()
    duplicate_timestamp_mask = step_times["_parsed_time"].duplicated(keep=False)
    duplicate_timestamp_steps = int(duplicate_timestamp_mask.sum())
    add_reason(step_times.loc[duplicate_timestamp_mask, "step"], "duplicate_timestamp")
    non_monotonic_mask = time_diffs.notna() & (time_diffs <= 0.0)
    non_monotonic_timestamp_steps = int(non_monotonic_mask.sum())
    add_reason(step_times.loc[non_monotonic_mask, "step"], "non_monotonic_timestamp")
    gap_mask = time_diffs > float(max_gap_s)
    gap_count = int(gap_mask.sum())
    max_gap_s_value = float(time_diffs.dropna().max()) if time_diffs.notna().any() else 0.0
    if gap_mask.any():
        gap_positions = np.where(gap_mask.to_numpy())[0]
        # ``time_diffs[i]`` is the interval ending at row ``i``.
        add_reason(step_times.iloc[gap_positions]["step"], "time_gap_exceeded")

    non_monotonic_cumulative_rows = 0
    for _, group in frame.groupby("cluster_id", sort=False):
        group = group.sort_values("step")
        for column in ("cumulative_liquid_volume_m3", "cumulative_sand_mass_t"):
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            bad = np.where(np.diff(values) < -1.0e-9)[0] + 1
            non_monotonic_cumulative_rows += int(len(bad))
            add_reason(group.iloc[bad]["step"], "cumulative_value_decreased")

    numeric_value_columns = (
        "liquid_volume_m3",
        "sand_mass_t",
        "cumulative_liquid_volume_m3",
        "cumulative_sand_mass_t",
    )
    negative_mask = np.zeros(len(frame), dtype=bool)
    missing_value_mask = np.zeros(len(frame), dtype=bool)
    for column in numeric_value_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        negative_mask |= values.to_numpy(dtype=float) < 0.0
        missing_value_mask |= values.isna().to_numpy(dtype=bool)
    negative_value_rows = int(negative_mask.sum())
    add_reason(frame.loc[negative_mask, "step"], "negative_observation")
    add_reason(frame.loc[missing_value_mask, "step"], "missing_numeric_value")

    allocation_sum_mismatch_steps = 0
    if "allocation_weight" in frame.columns:
        allocation = pd.to_numeric(frame["allocation_weight"], errors="coerce")
        allocation_sum = allocation.groupby(frame["step"]).sum(min_count=1)
        allocation_bad = allocation_sum[(allocation_sum - 1.0).abs() > float(volume_tolerance)].index
        allocation_sum_mismatch_steps = int(len(allocation_bad))
        add_reason(allocation_bad, "allocation_sum_mismatch")

    total_volume_mismatch_steps = 0
    if stage_totals is not None and not stage_totals.empty:
        stage = stage_totals.copy()
        if "step" not in stage.columns:
            raise ValueError("stage_totals must contain a step column")
        stage["step"] = pd.to_numeric(stage["step"], errors="coerce").fillna(-1).astype(int)
        obs = frame.groupby("step", as_index=False).agg(
            cumulative_liquid_volume_m3=("cumulative_liquid_volume_m3", "sum"),
            cumulative_sand_mass_t=("cumulative_sand_mass_t", "sum"),
        )
        stage_liquid = _column(stage, ("cumulative_liquid_volume_m3", "cumulative_liquid_m3", "total_liquid_m3"))
        stage_sand = _column(stage, ("cumulative_sand_mass_t", "cumulative_sand_t", "total_sand_t"))
        if stage_liquid is not None:
            stage = stage.assign(_stage_liquid=stage_liquid)
            obs = obs.merge(stage[["step", "_stage_liquid"]], on="step", how="inner")
            mismatch = np.abs(obs["cumulative_liquid_volume_m3"] - obs["_stage_liquid"]) / np.maximum(np.abs(obs["_stage_liquid"]), 1.0e-9)
            bad = obs.loc[mismatch > float(volume_tolerance), "step"]
            add_reason(bad, "liquid_total_mismatch")
            total_volume_mismatch_steps += int(len(bad))
        if stage_sand is not None:
            stage = stage.assign(_stage_sand=stage_sand)
            obs = frame.groupby("step", as_index=False).agg(cumulative_sand_mass_t=("cumulative_sand_mass_t", "sum"))
            obs = obs.merge(stage[["step", "_stage_sand"]], on="step", how="inner")
            mismatch = np.abs(obs["cumulative_sand_mass_t"] - obs["_stage_sand"]) / np.maximum(np.abs(obs["_stage_sand"]), 1.0e-9)
            bad = obs.loc[mismatch > float(volume_tolerance), "step"]
            add_reason(bad, "sand_total_mismatch")
            total_volume_mismatch_steps += int(len(bad))

    overlap_start = overlap_end = None
    pressure_overlap_ratio = 0.0
    if pressure_times is not None and len(pressure_times):
        pressure_axis = _numeric_axis(pressure_times)
        observation_axis = frame["step"].to_numpy(dtype=float)
        if pressure_axis.size and np.isfinite(observation_axis).any():
            obs_start, obs_end = float(np.nanmin(observation_axis)), float(np.nanmax(observation_axis))
            pressure_start, pressure_end = float(np.nanmin(pressure_axis)), float(np.nanmax(pressure_axis))
            overlap_start = max(obs_start, pressure_start)
            overlap_end = min(obs_end, pressure_end)
            obs_span = max(obs_end - obs_start, 0.0)
            overlap_span = max(overlap_end - overlap_start, 0.0)
            pressure_overlap_ratio = 1.0 if obs_span == 0.0 and overlap_span == 0.0 else overlap_span / max(obs_span, 1.0e-9)
            if overlap_span <= 0.0:
                add_reason(frame["step"].unique(), "no_pressure_time_overlap")

    reasons = sorted({reason for values in reasons_by_step.values() for reason in values})
    invalid_step_set = {step for step, values in reasons_by_step.items() if values}
    frame["qc_valid"] = ~frame["step"].isin(invalid_step_set)
    frame["qc_reason"] = frame["step"].map(
        lambda value: "ok" if int(value) not in reasons_by_step else ";".join(sorted(reasons_by_step[int(value)]))
    )
    frame = frame.drop(columns=["_parsed_time"])
    valid_steps = int(frame.loc[frame["qc_valid"], "step"].nunique())
    invalid_steps = int(len(invalid_step_set & set(frame["step"].unique())))
    invalid_sample_ratio = float((~frame["qc_valid"]).mean()) if len(frame) else 0.0
    valid_ratio = valid_steps / max(total_steps, 1)
    report = ObservationQualityReport(
        expected_clusters=expected_clusters,
        observed_clusters=observed_clusters,
        total_steps=total_steps,
        valid_steps=valid_steps,
        invalid_steps=invalid_steps,
        duplicate_rows=duplicate_rows,
        duplicate_timestamp_steps=duplicate_timestamp_steps,
        non_monotonic_timestamp_steps=non_monotonic_timestamp_steps,
        missing_cluster_steps=missing_cluster_steps,
        non_monotonic_cumulative_rows=non_monotonic_cumulative_rows,
        negative_value_rows=negative_value_rows,
        max_gap_s=max_gap_s_value,
        gap_count=gap_count,
        overlap_start_s=overlap_start,
        overlap_end_s=overlap_end,
        pressure_overlap_ratio=float(np.clip(pressure_overlap_ratio, 0.0, 1.0)),
        allocation_sum_mismatch_steps=allocation_sum_mismatch_steps,
        total_volume_mismatch_steps=total_volume_mismatch_steps,
        invalid_sample_ratio=invalid_sample_ratio,
        valid_ratio=valid_ratio,
        reasons=tuple(reasons),
        reasons_by_step={str(step): tuple(sorted(values)) for step, values in sorted(reasons_by_step.items()) if values},
    )
    return frame, report
