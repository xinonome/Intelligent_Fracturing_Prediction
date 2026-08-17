"""Quality checks for interpreted DAS/FracMonitor cluster observations."""

from __future__ import annotations

from dataclasses import dataclass, asdict
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
    missing_cluster_steps: int
    non_monotonic_cumulative_rows: int
    negative_value_rows: int
    max_gap_s: float
    overlap_start_s: float | None
    overlap_end_s: float | None
    valid_ratio: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_cluster_controls(
    controls: pd.DataFrame,
    expected_clusters: int = 6,
    pressure_times: np.ndarray | None = None,
    max_gap_s: float = 5.0,
) -> tuple[pd.DataFrame, ObservationQualityReport]:
    """Return per-step QC flags; invalid steps must not enter EnKF assimilation."""
    if controls.empty:
        report = ObservationQualityReport(expected_clusters, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, None, None, 0.0, ("empty_observation_table",))
        return controls.assign(qc_valid=pd.Series(dtype=bool), qc_reason=pd.Series(dtype=str)), report
    frame = controls.copy().sort_values(["step", "cluster_id"]).reset_index(drop=True)
    duplicate_rows = int(frame.duplicated(["step", "cluster_id"]).sum())
    observed_clusters = int(frame["cluster_id"].nunique())
    counts = frame.groupby("step")["cluster_id"].nunique()
    missing_cluster_steps = int((counts != expected_clusters).sum())
    non_monotonic = 0
    for _, group in frame.groupby("cluster_id"):
        for column in ("cumulative_liquid_volume_m3", "cumulative_sand_mass_t"):
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            non_monotonic += int(np.sum(np.diff(values) < -1.0e-9))
    negative = 0
    for column in ("liquid_volume_m3", "sand_mass_t", "cumulative_liquid_volume_m3", "cumulative_sand_mass_t"):
        negative += int((pd.to_numeric(frame[column], errors="coerce") < 0).sum())
    step_times = frame.groupby("step")["time"].first().sort_index()
    numeric_time = pd.to_datetime(step_times, errors="coerce")
    gaps = numeric_time.diff().dt.total_seconds().dropna()
    max_gap = float(gaps.max()) if not gaps.empty else 0.0
    invalid_steps: set[int] = set(frame.loc[frame.duplicated(["step", "cluster_id"], keep=False), "step"].astype(int))
    invalid_steps.update(int(step) for step, count in counts.items() if count != expected_clusters)
    if max_gap > max_gap_s:
        invalid_steps.update(int(step) for step in step_times.index[1:][gaps.to_numpy() > max_gap_s])
    for _, group in frame.groupby("cluster_id"):
        for column in ("cumulative_liquid_volume_m3", "cumulative_sand_mass_t"):
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            bad = np.where(np.diff(values) < -1.0e-9)[0] + 1
            invalid_steps.update(int(value) for value in group.iloc[bad]["step"].to_numpy())
    negative_mask = False
    for column in ("liquid_volume_m3", "sand_mass_t", "cumulative_liquid_volume_m3", "cumulative_sand_mass_t"):
        negative_mask = negative_mask | (pd.to_numeric(frame[column], errors="coerce") < 0)
    invalid_steps.update(int(value) for value in frame.loc[negative_mask, "step"].to_numpy())
    frame["qc_valid"] = ~frame["step"].astype(int).isin(invalid_steps)
    frame["qc_reason"] = frame["step"].astype(int).map(lambda value: "ok" if value not in invalid_steps else "invalid_cluster_observation")
    valid_steps = int(frame.loc[frame["qc_valid"], "step"].nunique())
    total_steps = int(frame["step"].nunique())
    overlap_start = overlap_end = None
    if pressure_times is not None and len(pressure_times):
        overlap_start = float(max(frame["step"].min(), np.min(pressure_times)))
        overlap_end = float(min(frame["step"].max(), np.max(pressure_times)))
    reasons = []
    if duplicate_rows: reasons.append("duplicate_step_cluster")
    if missing_cluster_steps: reasons.append("incomplete_cluster_set")
    if non_monotonic: reasons.append("cumulative_value_decreased")
    if negative: reasons.append("negative_observation")
    if max_gap > max_gap_s: reasons.append("time_gap_exceeded")
    report = ObservationQualityReport(
        expected_clusters, observed_clusters, total_steps, valid_steps, total_steps - valid_steps,
        duplicate_rows, missing_cluster_steps, non_monotonic, negative, max_gap,
        overlap_start, overlap_end, valid_steps / max(total_steps, 1), tuple(reasons),
    )
    return frame, report
