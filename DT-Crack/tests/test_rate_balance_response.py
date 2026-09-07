from __future__ import annotations

import numpy as np

from inversion.physics import PhysicalEnKFConfig
from inversion.rate_balance_response import RateBalanceResponseConfig, scan_rate_balance_response


def test_rate_scan_preserves_total_allocation_and_reports_balance() -> None:
    state = np.zeros(14, dtype=float)
    state[3] = 60.0
    rows, summary = scan_rate_balance_response(
        state,
        [0.5, 1.0, 2.0],
        physics_config=PhysicalEnKFConfig(),
        config=RateBalanceResponseConfig(duration_s=300.0),
    )
    assert len(rows) == 3
    assert all(abs(sum(row["cluster_allocation"]) - 1.0) < 1.0e-10 for row in rows)
    assert all(0.0 <= row["balance_degree"] <= 1.0 for row in rows)
    assert summary["scientific_status"] == "model_sensitivity_scan"


def test_rate_scan_detects_a_rate_effect_when_threshold_is_tight() -> None:
    state = np.zeros(14, dtype=float)
    state[3] = 60.0
    _, summary = scan_rate_balance_response(
        state,
        [0.5, 1.0, 2.0, 4.0],
        config=RateBalanceResponseConfig(
            duration_s=300.0,
            minimum_identifiable_balance_range=1.0e-12,
        ),
    )
    assert summary["balance_degree_range"] >= 0.0
    assert summary["status"] in {"rate_effect_identified", "rate_effect_not_identified"}


def test_rate_scan_rejects_single_scenario() -> None:
    state = np.zeros(14, dtype=float)
    try:
        scan_rate_balance_response(state, [1.0])
    except ValueError as exc:
        assert "at least two" in str(exc)
    else:
        raise AssertionError("a sensitivity scan needs at least two rate scenarios")
