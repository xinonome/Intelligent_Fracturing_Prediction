from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inversion.performance_tiers import classify_throughput  # noqa: E402
from inversion.run_pyfrac_sigma_online_restart import (  # noqa: E402
    assimilation_targets,
    update_sigma_from_innovation,
)


def test_restart_targets_include_final_time_once() -> None:
    assert assimilation_targets(1.0, 100.0, 30.0) == [31.0, 61.0, 91.0, 100.0]


def test_sigma_update_uses_innovation_without_per_step_clip() -> None:
    updated, sensitivity = update_sigma_from_innovation(
        60.0,
        110.0,
        90.0,
        gain=1.0,
        lower_mpa=20.0,
        upper_mpa=120.0,
    )
    assert sensitivity == 1.0
    assert updated == 80.0


def test_sigma_update_uses_secant_sensitivity() -> None:
    updated, sensitivity = update_sigma_from_innovation(
        60.0,
        100.0,
        90.0,
        previous_sigma_mpa=50.0,
        previous_predicted_mpa=80.0,
        gain=1.0,
        lower_mpa=20.0,
        upper_mpa=120.0,
    )
    assert np.isclose(sensitivity, 1.0)
    assert np.isclose(updated, 70.0)


def test_speed_tier_is_not_physical_acceptance() -> None:
    result = classify_throughput(25.0, 100.0)
    assert result["tier"] == "30%"
    assert "physical acceptance" in result["interpretation"]
