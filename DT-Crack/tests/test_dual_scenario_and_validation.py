from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_fusion.observation_quality import validate_cluster_controls
from data_fusion.pressure_schedule_adapter import PressureModelConfig, _resolve_columns
from inversion.pressure_only_enkf import pressure_observation_vector, run_pressure_only_correction


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
