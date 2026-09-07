from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_fusion.observation_quality import validate_cluster_controls
from data_fusion.pressure_schedule_adapter import PressureModelConfig, _resolve_columns
from inversion.pressure_only_enkf import pressure_observation_vector, run_pressure_only_correction
from inversion.validate_direct_observations import select_replay_steps


def _controls(steps: int = 3, clusters: int = 2) -> pd.DataFrame:
    rows = []
    for step in range(steps):
        for cluster in range(1, clusters + 1):
            rows.append(
                {
                    "step": step,
                    "time": pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=step),
                    "cluster_id": cluster,
                    "liquid_volume_m3": 1.0,
                    "sand_mass_t": 0.1,
                    "cumulative_liquid_volume_m3": float(step + 1),
                    "cumulative_sand_mass_t": float(step + 1) * 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_pressure_columns_prefer_named_fields_and_flow_column() -> None:
    frame = pd.DataFrame(columns=["序号", "liquid1", "liquid2", "liquid3", "liquid4", "sand1", "sand2", "sand3", "sand4", "泵压(MPa)", "总液量", "砂比", "排出排量"])
    resolved = _resolve_columns(frame)
    assert resolved["surface_pressure_mpa"] == "泵压(MPa)"
    assert resolved["flow_rate_m3_min"] == "排出排量"
    assert PressureModelConfig().calibration_status == "engineering_default"


def test_qc_marks_incomplete_cluster_step_invalid() -> None:
    frame = _controls()
    frame = frame[~((frame["step"] == 1) & (frame["cluster_id"] == 2))]
    checked, report = validate_cluster_controls(frame, expected_clusters=2)
    assert report.invalid_steps == 1
    assert not bool(checked.loc[checked["step"] == 1, "qc_valid"].any())


def test_qc_records_timestamp_gap_and_pressure_overlap_reasons() -> None:
    frame = _controls(steps=4, clusters=2)
    frame.loc[frame["step"] == 2, "time"] = pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=20)
    checked, report = validate_cluster_controls(
        frame,
        expected_clusters=2,
        pressure_times=np.arange(0.0, 4.0),
        max_gap_s=5.0,
    )
    assert report.gap_count == 1
    assert report.pressure_overlap_ratio == 1.0
    assert "time_gap_exceeded" in report.reasons
    assert not bool(checked.loc[checked["step"] == 2, "qc_valid"].any())


def test_qc_marks_duplicate_timestamps_without_forward_filling() -> None:
    frame = _controls(steps=3, clusters=2)
    frame.loc[frame["step"] == 2, "time"] = frame.loc[frame["step"] == 1, "time"].iloc[0]
    checked, report = validate_cluster_controls(frame, expected_clusters=2)
    assert report.duplicate_timestamp_steps == 2
    assert "duplicate_timestamp" in report.reasons
    assert checked.loc[checked["step"].isin([1, 2]), "qc_valid"].eq(False).all()


def test_pressure_only_has_one_observation_and_bounded_bias() -> None:
    assert pressure_observation_vector(88.0).tolist() == [88.0]
    result = run_pressure_only_correction(
        np.asarray([80.0, 81.0]),
        np.asarray([100.0, 101.0]),
    )
    assert result["posterior_bottomhole_mpa"].shape == (2,)
    assert np.all(result["posterior_bias_mpa"] <= 15.0)
    assert np.all(result["posterior_bias_mpa"] >= -15.0)
    assert result["metadata"]["cluster_observations"] == "not_available"


def test_realtime_schedule_skips_arrivals_during_busy_update() -> None:
    steps, meta = select_replay_steps(
        np.arange(1, 11),
        SimpleNamespace(realtime_budget_s=10.0, realtime_step_cost_s=1.32, max_steps=60),
    )
    assert steps.tolist() == [1, 2, 3, 4, 6, 7, 8]
    assert meta["enabled"] is True
    assert meta["skipped_source_steps"] == 3
