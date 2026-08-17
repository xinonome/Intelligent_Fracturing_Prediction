"""Pressure-only online correction used when DAS is unavailable.

This module separates pressure-conversion uncertainty from the physical PKN
state. The bounded scalar pressure-bias ensemble can be used as an augmented
EnKF channel without fabricating cluster observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PressureOnlyConfig:
    process_std_mpa: float = 1.5
    observation_std_mpa: float = 3.5
    bias_lower_mpa: float = -15.0
    bias_upper_mpa: float = 15.0
    ensemble_size: int = 200
    seed: int = 20260814

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pressure_observation_vector(bottomhole_pressure_mpa: float) -> np.ndarray:
    """Return the only observation available in the no-DAS scenario."""
    return np.asarray([float(bottomhole_pressure_mpa)], dtype=float)


def run_pressure_only_correction(
    prior_bottomhole_mpa: np.ndarray,
    converted_bottomhole_mpa: np.ndarray,
    config: PressureOnlyConfig | None = None,
) -> dict[str, np.ndarray | dict[str, Any]]:
    """Assimilate converted BHP with a bounded random-walk pressure bias."""
    config = config or PressureOnlyConfig()
    prior = np.asarray(prior_bottomhole_mpa, dtype=float)
    observed = np.asarray(converted_bottomhole_mpa, dtype=float)
    if prior.shape != observed.shape:
        raise ValueError("prior and converted bottomhole pressure must have the same shape")
    rng = np.random.default_rng(config.seed)
    members = rng.normal(0.0, max(config.observation_std_mpa, 1.0e-6), size=max(int(config.ensemble_size), 2))
    posterior_bias = []
    prior_bias = []
    posterior = []
    for p_model, p_obs in zip(prior, observed):
        members = np.clip(
            members + rng.normal(0.0, max(config.process_std_mpa, 0.0), size=members.size),
            config.bias_lower_mpa,
            config.bias_upper_mpa,
        )
        prediction = p_model + members
        innovation = float(p_obs) - float(np.mean(prediction))
        covariance = float(np.var(members, ddof=1))
        gain = covariance / max(covariance + config.observation_std_mpa**2, 1.0e-12)
        members = np.clip(members + gain * innovation, config.bias_lower_mpa, config.bias_upper_mpa)
        bias = float(np.mean(members))
        prior_bias.append(float(np.mean(prediction - p_model)))
        posterior_bias.append(bias)
        posterior.append(float(p_model + bias))
    return {
        "prior_bottomhole_mpa": prior,
        "observed_bottomhole_mpa": observed,
        "posterior_bottomhole_mpa": np.asarray(posterior, dtype=float),
        "prior_bias_mpa": np.asarray(prior_bias, dtype=float),
        "posterior_bias_mpa": np.asarray(posterior_bias, dtype=float),
        "metadata": {
            "observation_mode": "pressure_only",
            "observation_vector": ["bottomhole_pressure_mpa"],
            "bias_state": "bounded_random_walk",
            "config": config.to_dict(),
            "cluster_observations": "not_available",
        },
    }
