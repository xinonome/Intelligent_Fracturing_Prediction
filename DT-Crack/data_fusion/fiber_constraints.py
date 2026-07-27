from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FiberConstraintResult:
    """Per-fracture controls inferred from DAS monitoring."""

    table: pd.DataFrame
    weights: np.ndarray


def make_synthetic_fiber_monitoring(fracture_count: int = 5, steps: int = 18, seed: int = 7) -> pd.DataFrame:
    """Create public-data-free synthetic DAS monitoring for demos.

    A legacy ``dts_anomaly`` column is still emitted as zeros for older scripts,
    but current inversion demos use DAS only.
    """

    rng = np.random.default_rng(seed)
    rows = []
    base_activity = np.linspace(0.65, 1.25, fracture_count)
    for step in range(steps):
        pulse = 1.0 + 0.35 * np.sin(step / 3.0 + np.arange(fracture_count) * 0.8)
        das = np.maximum(0.05, base_activity * pulse + rng.normal(0.0, 0.05, fracture_count))
        pressure = 45.0 + 0.35 * step + 1.1 * das + rng.normal(0.0, 0.25, fracture_count)
        for frac_id in range(fracture_count):
            rows.append(
                {
                    "step": step,
                    "fracture_id": frac_id + 1,
                    "das_energy": float(das[frac_id]),
                    "dts_anomaly": 0.0,
                    "pressure_proxy_mpa": float(pressure[frac_id]),
                }
            )
    return pd.DataFrame(rows)


def allocate_controls_from_fiber(
    fiber_window: pd.DataFrame,
    total_flow_rate_m3_min: float,
    total_liquid_volume_m3: float,
    total_sand_mass_t: float,
    das_weight: float = 1.0,
    dts_weight: float = 0.0,
) -> FiberConstraintResult:
    """Convert DAS responses to per-fracture flow/liquid/sand controls.

    ``dts_weight`` is retained only for backward compatibility. The default
    current project setting is DAS-only because no DTS data is available.
    """

    fiber_window = fiber_window.copy()
    if "dts_anomaly" not in fiber_window.columns:
        fiber_window["dts_anomaly"] = 0.0

    grouped = (
        fiber_window.groupby("fracture_id", as_index=False)
        .agg(
            das_energy=("das_energy", "mean"),
            dts_anomaly=("dts_anomaly", "mean"),
            pressure_proxy_mpa=("pressure_proxy_mpa", "mean"),
        )
        .sort_values("fracture_id")
    )
    das_score = _normalize_positive(grouped["das_energy"].to_numpy())
    dts_score = _normalize_positive(grouped["dts_anomaly"].to_numpy())
    weights = _normalize_positive(das_weight * das_score + dts_weight * dts_score)
    grouped["allocation_weight"] = weights
    grouped["flow_rate_m3_min"] = total_flow_rate_m3_min * weights
    grouped["liquid_volume_m3"] = total_liquid_volume_m3 * weights
    grouped["sand_mass_t"] = total_sand_mass_t * weights
    return FiberConstraintResult(grouped, weights)


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.maximum(values, 0.0)
    total = values.sum()
    if total <= 1e-12:
        return np.full(len(values), 1.0 / max(len(values), 1))
    return values / total
