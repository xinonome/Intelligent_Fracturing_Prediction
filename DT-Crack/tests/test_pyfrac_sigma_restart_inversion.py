from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inversion.run_pyfrac_sigma_restart_inversion import (  # noqa: E402
    _candidate_stresses,
    _pressure_objective,
)
from inversion.performance_tiers import (  # noqa: E402
    classify_throughput,
    evaluate_throughput_requirement,
)


def test_candidate_stresses_are_clipped_and_unique() -> None:
    values = _candidate_stresses(60.0, [0.0, -50.0, 50.0, 50.0], 20.0, 90.0)
    assert values == [60.0, 20.0, 90.0]


def test_pressure_objective_returns_absolute_error() -> None:
    assert np.isclose(_pressure_objective(92.0, 88.5), 3.5)
    assert _pressure_objective(float("nan"), 88.5) is None


def test_throughput_tiers_keep_physical_acceptance_separate() -> None:
    result = classify_throughput(100.0, 1000.0)
    assert result["tier"] == "10%"
    assert result["passed"] == {"10%": True, "20%": True, "30%": True}
    assert "physical acceptance" in result["interpretation"]

    result = classify_throughput(250.0, 1000.0)
    assert result["tier"] == "30%"
    assert result["passed"]["10%"] is False
    assert result["passed"]["20%"] is False
    assert result["passed"]["30%"] is True


def test_throughput_tier_reports_missing_timing() -> None:
    result = classify_throughput(None, 100.0)
    assert result["tier"] == "not_measured"
    assert result["ratio"] is None


def test_throughput_requirement_is_optional_and_configurable() -> None:
    report_only = evaluate_throughput_requirement(125.0, 1000.0)
    assert report_only["status"] == "not_required"
    assert report_only["passed"] is None

    accepted_at_twenty_percent = evaluate_throughput_requirement(125.0, 1000.0, 0.20)
    assert accepted_at_twenty_percent["status"] == "passed"
    assert accepted_at_twenty_percent["passed"] is True

    rejected_at_ten_percent = evaluate_throughput_requirement(125.0, 1000.0, 0.10)
    assert rejected_at_ten_percent["status"] == "not_met"
    assert rejected_at_ten_percent["passed"] is False
