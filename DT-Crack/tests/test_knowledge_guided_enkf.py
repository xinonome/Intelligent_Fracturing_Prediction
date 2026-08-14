from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inversion.knowledge_guided_enkf import (  # noqa: E402
    KnowledgeGuidedPriorConfig,
    build_knowledge_guided_distribution,
    build_knowledge_guided_prior,
    project_knowledge_guided_update,
)


def test_knowledge_prior_only_widens_distribution_without_mean_shift() -> None:
    steps = np.asarray([0, 60, 120, 180], dtype=int)
    pressure = pd.DataFrame(
        {
            "step": steps,
            "time_s": steps.astype(float),
            "bottomhole_pressure_mpa": [75.0, 84.0, 96.0, 108.0],
            "sand_ratio_percent": [3.0, 13.5, 16.0, 17.0],
            "flow_rate_m3_min": [18.0, 16.0, 14.0, 12.0],
        }
    )
    controls = pd.DataFrame({"step": steps, "flow_rate_m3_min": [18.0, 16.0, 14.0, 12.0]})
    spread = np.ones(5)
    process = np.ones(5) * 0.1
    adjusted_spread, adjusted_process, meta = build_knowledge_guided_distribution(
        controls,
        pressure,
        steps,
        spread,
        process,
        KnowledgeGuidedPriorConfig(enabled=True, strength=0.35),
    )
    assert meta["mode"] == "uncertainty_only"
    assert meta["signals"]["risk_score"] > 0.0
    assert np.all(adjusted_spread >= spread)
    assert np.all(adjusted_process >= process)
    assert meta["prior_mean_shift"] == [0.0] * 5


def test_soft_correlated_prior_changes_covariance_not_observation_rule() -> None:
    steps = np.asarray([0, 60, 120, 180], dtype=int)
    pressure = pd.DataFrame(
        {
            "step": steps,
            "time_s": steps.astype(float),
            "bottomhole_pressure_mpa": [75.0, 84.0, 96.0, 108.0],
            "sand_ratio_percent": [3.0, 13.5, 16.0, 17.0],
            "flow_rate_m3_min": [18.0, 16.0, 14.0, 12.0],
        }
    )
    controls = pd.DataFrame({"step": steps, "flow_rate_m3_min": [18.0, 16.0, 14.0, 12.0]})
    prior_mean = np.asarray([0.0, 0.0, 0.0, 60.0, 0.0])
    result = build_knowledge_guided_prior(
        controls,
        pressure,
        steps,
        prior_mean,
        np.ones(5),
        np.ones(5) * 0.1,
        KnowledgeGuidedPriorConfig(enabled=True, mode="soft_correlated", strength=0.8),
    )
    meta = result["meta"]
    assert meta["mode"] == "soft_correlated"
    assert any(abs(value) > 0 for value in meta["prior_mean_shift"])
    covariance = np.asarray(result["covariance"], dtype=float)
    assert covariance[1, 3] != 0.0
    assert meta["observation_noise_multiplier"]["pressure"] > 1.0
    assert meta["max_update_scale"] is not None


def test_knowledge_update_projection_limits_only_parameter_jump() -> None:
    updated = np.asarray([[4.0, -4.0, 2.0, 10.0, 3.0]])
    reference = np.zeros_like(updated)
    projected = project_knowledge_guided_update(
        updated,
        reference,
        np.ones(5),
        {"max_update_scale": 1.5},
    )
    assert np.all(np.abs(projected) <= 1.5)
