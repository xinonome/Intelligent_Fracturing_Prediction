from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FiberObservationConfig:
    """Weights for converting fiber cluster monitoring into length observations."""

    cumulative_liquid_weight: float = 0.70
    cumulative_sand_weight: float = 0.30
    instant_liquid_weight: float = 0.0
    instant_sand_weight: float = 0.0
    balance_gain: float = 0.20
    min_factor: float = 0.85
    max_factor: float = 1.15


def build_fiber_length_observation(
    controls: pd.DataFrame,
    prior_half_length_m: np.ndarray,
    config: FiberObservationConfig | None = None,
) -> pd.DataFrame:
    """Convert cluster fluid/sand monitoring into DAS-equivalent length observation.

    The current real file provides interpreted cluster-level fluid/sand intake
    and fracture balance, not a raw DAS time-depth amplitude matrix. Therefore
    the observation operator is an engineering proxy:

    - cumulative liquid controls long-term fracture contribution;
    - cumulative sand indicates proppant-carrying active fracture growth;
    - instantaneous liquid/sand are kept as diagnostic signals and default to
      zero weight because short-term flow fluctuation should not make fracture
      length observations shrink;
    - balance degree adjusts how strongly the distribution deviates from equal.

    Output factors are normalized around 1.0 and multiplied by the PKN prior
    half-length to obtain an equivalent observed half-length. This value is not
    used as the final fracture length. It is the observation in the EnKF update:
    EnKF changes the PKN parameter factors, then the forward model is re-run to
    obtain the corrected fracture geometry.
    """

    config = config or FiberObservationConfig()
    if controls.empty:
        return pd.DataFrame(
            {
                "cluster_id": np.arange(len(prior_half_length_m)),
                "fiber_observation_factor": np.ones(len(prior_half_length_m)),
                "observed_half_length_m": prior_half_length_m,
            }
        )

    df = controls.sort_values("cluster_id").reset_index(drop=True).copy()
    n = len(df)
    prior = np.resize(np.asarray(prior_half_length_m, dtype=float), n)

    cum_liquid_score = _relative_score(df["cumulative_liquid_volume_m3"].to_numpy(dtype=float))
    cum_sand_score = _relative_score(df["cumulative_sand_mass_t"].to_numpy(dtype=float))
    inst_liquid_score = _relative_score(df["liquid_volume_m3"].to_numpy(dtype=float))
    inst_sand_score = _relative_score(df["sand_mass_t"].to_numpy(dtype=float))

    activity = (
        config.cumulative_liquid_weight * cum_liquid_score
        + config.cumulative_sand_weight * cum_sand_score
        + config.instant_liquid_weight * inst_liquid_score
        + config.instant_sand_weight * inst_sand_score
    )
    activity = _normalize_mean_one(activity)

    balance_degree = float(df["balance_degree"].dropna().mean()) if "balance_degree" in df else 1.0
    cumulative_balance_degree = (
        float(df["cumulative_balance_degree"].dropna().mean()) if "cumulative_balance_degree" in df else balance_degree
    )
    # balance close to 1 means clusters are more even. Lower balance should
    # amplify the measured activity differences because the fiber interpretation
    # indicates stronger non-uniform fracture development.
    imbalance = np.clip(1.0 - min(balance_degree, cumulative_balance_degree), 0.0, 1.0)
    factor = 1.0 + (activity - 1.0) * (1.0 + config.balance_gain * imbalance)
    factor = np.clip(factor, config.min_factor, config.max_factor)
    observed = prior * factor

    out = df[
        [
            "cluster_id",
            "liquid_volume_m3",
            "sand_mass_t",
            "cumulative_liquid_volume_m3",
            "cumulative_sand_mass_t",
            "balance_degree",
            "cumulative_balance_degree",
        ]
    ].copy()
    out["cum_liquid_score"] = cum_liquid_score
    out["cum_sand_score"] = cum_sand_score
    out["instant_liquid_score"] = inst_liquid_score
    out["instant_sand_score"] = inst_sand_score
    out["fiber_activity_score"] = activity
    out["fiber_balance_imbalance"] = imbalance
    out["fiber_observation_factor"] = factor
    out["prior_half_length_m"] = prior
    out["observed_half_length_m"] = observed
    return out


def _relative_score(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    if values.sum() <= 1e-12:
        return np.ones_like(values, dtype=float)
    share = values / values.sum()
    equal_share = 1.0 / max(len(values), 1)
    return share / equal_share


def _normalize_mean_one(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=1.0, posinf=1.0, neginf=1.0)
    mean = float(np.mean(values))
    if abs(mean) <= 1e-12:
        return np.ones_like(values)
    return values / mean
