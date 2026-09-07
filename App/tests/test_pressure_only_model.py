from __future__ import annotations

import numpy as np
import pandas as pd

from App.dt_pressure_only_model import build_assumed_cluster_positions, estimate_pressure_only_multicluster


def _trajectory() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "measured_depth_m": np.linspace(3600.0, 5200.0, 9),
            "vertical_depth_m": np.linspace(3000.0, 3200.0, 9),
            "north_m": np.linspace(0.0, 800.0, 9),
            "east_m": np.linspace(0.0, 400.0, 9),
        }
    )


def test_assumed_positions_are_explicitly_marked() -> None:
    positions, meta = build_assumed_cluster_positions(_trajectory(), 6, 3800.0, 5000.0)
    assert len(positions) == 6
    assert meta["status"] == "estimated"
    assert all(item["geometry_provenance"].startswith("assumed_") for item in positions)


def test_pressure_only_estimate_conserves_stage_length_and_shows_edge_relief() -> None:
    positions, _ = build_assumed_cluster_positions(_trajectory(), 6, 3800.0, 5000.0)
    times = np.arange(1.0, 301.0)
    result = estimate_pressure_only_multicluster(
        timeline_s=times,
        flow_rate_m3_min=np.full(len(times), 10.0),
        cumulative_liquid_m3=np.linspace(0.2, 500.0, len(times)),
        corrected_bhp_mpa=np.linspace(70.0, 100.0, len(times)),
        min_horizontal_stress_mpa=60.0,
        cluster_positions=positions,
    )
    final_stage = result["stage"]["pkn_stage_half_length_m"][-1]
    final_lengths = [result["clusters"][str(i)]["posterior_half_length_m"][-1] for i in range(6)]
    assert np.isclose(sum(final_lengths), final_stage)
    assert final_lengths[0] > final_lengths[2]
    assert final_lengths[-1] > final_lengths[3]


def test_pressure_only_fracture_front_does_not_retreat_during_rate_drop() -> None:
    positions, _ = build_assumed_cluster_positions(_trajectory(), 6, 3800.0, 5000.0)
    times = np.arange(1.0, 121.0)
    flow = np.r_[np.full(60, 12.0), np.zeros(60)]
    cumulative = np.r_[np.linspace(0.2, 120.0, 60), np.full(60, 120.0)]
    result = estimate_pressure_only_multicluster(
        timeline_s=times,
        flow_rate_m3_min=flow,
        cumulative_liquid_m3=cumulative,
        corrected_bhp_mpa=np.full(len(times), 90.0),
        min_horizontal_stress_mpa=60.0,
        cluster_positions=positions,
    )
    stage = np.asarray(result["stage"]["pkn_stage_half_length_m"], dtype=float)
    assert np.all(np.diff(stage) >= -1.0e-9)
    assert np.isclose(stage[59], stage[-1])
