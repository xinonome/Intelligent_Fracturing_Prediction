from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inversion.piggy_bank_allocator import PiggyBankAllocator, PiggyBankConfig
from inversion.segment_response_metrics import balance_indices, compute_segment_response_metrics


def _controls() -> pd.DataFrame:
    rows = []
    increments = [2.0, 1.0, 0.5, 0.5, 1.0, 2.0]
    for step in range(1, 4):
        for cluster_id, increment in enumerate(increments):
            rows.append(
                {
                    "step": step,
                    "time": pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=step),
                    "cluster_id": cluster_id,
                    "liquid_volume_m3": increment,
                    "sand_mass_t": increment * 0.01,
                    "cumulative_liquid_volume_m3": increment * step,
                    "cumulative_sand_mass_t": increment * step * 0.01,
                }
            )
    return pd.DataFrame(rows)


def _geometry_one_based() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster_id": range(1, 7),
            "east_m": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            "north_m": [0.0] * 6,
            "vertical_depth_m": [0.0] * 6,
        }
    )


def test_response_metric_uses_datetime_seconds_and_aligns_zero_based_clusters() -> None:
    events, balance, meta = compute_segment_response_metrics(_controls(), _geometry_one_based())
    assert meta["cluster_geometry_available"] is True
    assert events["geometry_status"].eq("valid").all()
    assert events["first_response_time_s"].min() == 0.0
    assert len(balance) == 3
    assert np.isfinite(events["volume_normalized_response_efficiency_m_per_m3"]).all()


def test_balance_degree_is_one_for_target_and_less_for_concentrated_share() -> None:
    equal = balance_indices([1.0, 1.0, 1.0])
    concentrated = balance_indices([1.0, 0.0, 0.0])
    assert equal["balance_degree"] == 1.0
    assert concentrated["balance_degree"] < equal["balance_degree"]
    assert 0.0 <= concentrated["entropy_normalized"] <= 1.0


def test_piggy_bank_recommends_stage_total_only_and_does_not_modify_physics() -> None:
    events, _, _ = compute_segment_response_metrics(_controls(), _geometry_one_based())
    allocator = PiggyBankAllocator(PiggyBankConfig(max_single_cluster_adjustment_fraction=0.10))
    result = allocator.allocate([1 / 6] * 6, 1.0, 60.0, events, geometry_valid=True)
    weights = np.asarray(result["recommended_weights"], dtype=float)
    transfer = np.asarray(result["transferred_volume_m3"], dtype=float)
    assert result["physical_parameters_modified"] is False
    assert result["control_layer"] == "stage_total_liquid_advisory"
    assert result["cluster_allocation_actionable"] is False
    assert abs(float(weights.sum()) - 1.0) < 1.0e-10
    assert np.allclose(transfer, 0.0)
    assert np.isfinite(float(result["recommended_stage_volume_m3"]))
    assert np.isfinite(float(result["recommended_stage_rate_m3_s"]))
    assert np.all(weights >= 0.0)


def test_piggy_bank_holds_when_geometry_or_observation_is_unknown() -> None:
    events, _, _ = compute_segment_response_metrics(_controls(), None)
    allocator = PiggyBankAllocator()
    result = allocator.allocate([1 / 6] * 6, 1.0, 60.0, events, geometry_valid=False)
    assert result["status"] == "geometry_quality_hold"
    assert result["transferred_volume_m3"] == [0.0] * 6


def test_piggy_bank_safety_hold_has_no_transfer() -> None:
    events, _, _ = compute_segment_response_metrics(_controls(), _geometry_one_based())
    allocator = PiggyBankAllocator()
    result = allocator.allocate([1 / 6] * 6, 1.0, 60.0, events, pressure_high=True, geometry_valid=True)
    assert result["status"] == "pressure_safety_hold"
    assert np.allclose(result["recommended_weights"], [1 / 6] * 6)


def test_piggy_bank_holds_when_cluster_signals_conflict() -> None:
    metrics = pd.DataFrame(
        {
            "cluster_id": range(6),
            "control_score": [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "control_state": ["fast", "slow", "neutral", "neutral", "neutral", "neutral"],
        }
    )
    allocator = PiggyBankAllocator(PiggyBankConfig(max_stage_adjustment_fraction=0.10))
    result = allocator.allocate([1 / 6] * 6, 1.0, 60.0, metrics, geometry_valid=True)
    assert result["status"] == "mixed_cluster_signal_total_rate_not_targetable"
    assert result["recommendation_direction"] == "hold"
    assert result["stage_liquid_delta_m3"] == 0.0


def test_piggy_bank_recommends_stage_decrease_for_unidirectional_fast_signal() -> None:
    metrics = pd.DataFrame(
        {
            "cluster_id": range(6),
            "control_score": [-0.8] * 6,
            "control_state": ["fast"] * 6,
        }
    )
    allocator = PiggyBankAllocator(PiggyBankConfig(max_stage_adjustment_fraction=0.10))
    result = allocator.allocate([1 / 6] * 6, 1.0, 60.0, metrics, geometry_valid=True)
    assert result["status"] == "stage_total_liquid_decrease_recommended"
    assert result["recommendation_direction"] == "decrease"
    assert result["stage_liquid_delta_m3"] < 0.0
    assert result["virtual_stage_reserve_m3"] > 0.0
