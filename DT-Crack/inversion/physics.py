"""Fast PKN forward operator and parameter-space EnKF utilities.

The EnKF state contains physical and per-cluster intake parameters. Fracture
length is always recomputed by the forward operator after each update; it is
not directly overwritten by the filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicalEnKFConfig:
    base_eprime_pa: float = 3.2e10
    base_leakoff_m_sqrt_s: float = 1.0e-5
    base_viscosity_pa_s: float = 0.1
    base_min_stress_mpa: float = 60.0
    height_m: float = 30.0
    pressure_proxy_scale: float = 30.0
    min_effective_rate_fraction: float = 0.05
    max_leakoff_fraction: float = 0.85
    stress_shadow_strength: float = 0.06
    leakoff_iterations: int = 6
    hydraulic_coupling_mode: str = "coupled"
    conductance_exponent: float = 0.45
    shadow_decay_clusters: float = 1.25


STATE_GLOBAL_NAMES = [
    "log_eprime_scale",
    "log_leakoff_scale",
    "log_viscosity_scale",
    "min_horizontal_stress_mpa",
]


def physical_values(state: np.ndarray, cfg: PhysicalEnKFConfig, n_clusters: int) -> dict[str, np.ndarray | float]:
    state = np.asarray(state, dtype=float)
    return {
        "eprime_pa": cfg.base_eprime_pa * np.exp(state[0]),
        "leakoff_m_sqrt_s": cfg.base_leakoff_m_sqrt_s * np.exp(state[1]),
        "viscosity_pa_s": cfg.base_viscosity_pa_s * np.exp(state[2]),
        "min_horizontal_stress_mpa": state[3],
        "cluster_factors": state[4 : 4 + n_clusters],
    }


def clip_state(state: np.ndarray, n_clusters: int) -> np.ndarray:
    out = np.asarray(state, dtype=float).copy()
    out[..., 0] = np.clip(out[..., 0], np.log(0.45), np.log(2.2))
    out[..., 1] = np.clip(out[..., 1], np.log(0.1), np.log(8.0))
    out[..., 2] = np.clip(out[..., 2], np.log(0.2), np.log(5.0))
    out[..., 3] = np.clip(out[..., 3], 35.0, 90.0)
    out[..., 4 : 4 + n_clusters] = np.clip(out[..., 4 : 4 + n_clusters], 0.65, 1.35)
    return out


def pkn_with_carter_leakoff(
    state: np.ndarray,
    q_base_m3_s: np.ndarray,
    t_seconds: float,
    cfg: PhysicalEnKFConfig,
    q_current_m3_s: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Evaluate enhanced PKN with leakoff, hydraulic coupling and stress shadow.

    ``q_base_m3_s`` is the cumulative-volume-equivalent rate used for fracture
    growth. ``q_current_m3_s`` is the current boundary rate used for aperture
    and pressure, so late rate ramps are not treated as full-history inputs.
    """

    n_clusters = len(q_base_m3_s)
    values = physical_values(state, cfg, n_clusters)
    eprime = float(values["eprime_pa"])
    leakoff = float(values["leakoff_m_sqrt_s"])
    viscosity = float(values["viscosity_pa_s"])
    stress = float(values["min_horizontal_stress_mpa"])
    factors = np.asarray(values["cluster_factors"], dtype=float)
    t = max(float(t_seconds), 1.0)
    q_base = np.maximum(np.asarray(q_base_m3_s, dtype=float), 0.0)
    q_current = q_base if q_current_m3_s is None else np.maximum(np.asarray(q_current_m3_s, dtype=float), 0.0)
    if q_current.shape != q_base.shape:
        raise ValueError("q_current_m3_s must have the same cluster shape as q_base_m3_s")
    weighted_rate = q_base * np.maximum(factors, 1e-6)
    q_nominal = float(q_base.sum()) * weighted_rate / max(float(weighted_rate.sum()), 1e-12)
    weighted_current = q_current * np.maximum(factors, 1e-6)
    q_current_nominal = float(q_current.sum()) * weighted_current / max(float(weighted_current.sum()), 1e-12)
    q_nominal = np.maximum(q_nominal, 1e-9)
    q_current_nominal = np.maximum(q_current_nominal, 1e-9)
    q_effective = q_nominal.copy()
    q_current_effective = q_current_nominal.copy()

    injected_volume = float(q_nominal.sum()) * t
    leakoff_volume = np.zeros(n_clusters, dtype=float)
    length = np.zeros(n_clusters, dtype=float)
    for _ in range(max(int(cfg.leakoff_iterations), 1)):
        length = 0.68 * ((q_effective**3 * eprime) / (viscosity * cfg.height_m**4)) ** 0.2 * t**0.8
        length *= 0.95 + 0.10 * factors
        fracture_area = 4.0 * length * cfg.height_m
        raw_leakoff = 2.0 * leakoff * fracture_area * np.sqrt(t)
        max_total_leakoff = cfg.max_leakoff_fraction * injected_volume
        leakoff_volume = raw_leakoff * min(1.0, max_total_leakoff / max(float(raw_leakoff.sum()), 1e-12))
        q_effective = np.maximum(q_nominal - leakoff_volume / t, cfg.min_effective_rate_fraction * q_nominal)
        q_current_effective = np.maximum(
            q_current_nominal - leakoff_volume / t,
            cfg.min_effective_rate_fraction * q_current_nominal,
        )

        if cfg.hydraulic_coupling_mode == "coupled" and n_clusters > 1:
            aperture_for_flow = 2.5 * (
                (q_current_effective**3 * viscosity) / (eprime * cfg.height_m**3)
            ) ** 0.2 * t**0.2
            relative_aperture = aperture_for_flow / max(float(np.mean(aperture_for_flow)), 1e-12)
            conductance = np.maximum(factors, 1e-6) * np.power(
                np.clip(relative_aperture, 0.35, 2.5), cfg.conductance_exponent
            )
            q_nominal = float(q_base.sum()) * conductance / max(float(conductance.sum()), 1e-12)
            q_current_nominal = float(q_current.sum()) * conductance / max(float(conductance.sum()), 1e-12)
            q_nominal = np.maximum(q_nominal, 1e-9)
            q_current_nominal = np.maximum(q_current_nominal, 1e-9)

    if n_clusters > 1 and cfg.stress_shadow_strength > 0.0:
        distance = np.abs(np.arange(n_clusters)[:, None] - np.arange(n_clusters)[None, :])
        interaction = np.exp(-distance / max(cfg.shadow_decay_clusters, 1e-6))
        np.fill_diagonal(interaction, 0.0)
        interaction /= np.maximum(interaction.sum(axis=1, keepdims=True), 1e-12)
        normalized_rate = q_effective / max(float(np.mean(q_effective)), 1e-12)
        shadow = np.tanh(interaction @ normalized_rate)
        length *= np.clip(1.0 - cfg.stress_shadow_strength * shadow, 0.85, 1.0)

    aperture_m = 2.5 * ((q_current_effective**3 * viscosity) / (eprime * cfg.height_m**3)) ** 0.2 * t**0.2
    net_pressure_mpa = cfg.pressure_proxy_scale * eprime * float(np.mean(aperture_m / 2.0)) / cfg.height_m / 1.0e6
    return {
        "half_length_m": length,
        "max_aperture_mm": aperture_m * 1000.0,
        "q_nominal_m3_s": q_nominal,
        "q_effective_m3_s": q_effective,
        "q_current_nominal_m3_s": q_current_nominal,
        "q_current_effective_m3_s": q_current_effective,
        "injected_volume_m3": injected_volume,
        "leakoff_volume_m3": float(leakoff_volume.sum()),
        "leakoff_fraction": float(leakoff_volume.sum()) / max(injected_volume, 1e-12),
        "rate_conservation_error": abs(float(q_nominal.sum()) - float(q_base.sum())) / max(float(q_base.sum()), 1e-12),
        "net_pressure_mpa": net_pressure_mpa,
        "bottomhole_pressure_mpa": stress + net_pressure_mpa,
        **values,
    }


def enkf_update(
    ensemble: np.ndarray,
    predicted_obs: np.ndarray,
    observed_obs: np.ndarray,
    observation_std: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x_anom = ensemble - ensemble.mean(axis=0)
    y_anom = predicted_obs - predicted_obs.mean(axis=0)
    denom = max(len(ensemble) - 1, 1)
    p_xy = x_anom.T @ y_anom / denom
    p_yy = y_anom.T @ y_anom / denom + np.diag(observation_std**2)
    gain = p_xy @ np.linalg.pinv(p_yy)
    perturbed = observed_obs + rng.normal(0.0, observation_std, size=predicted_obs.shape)
    return ensemble + (perturbed - predicted_obs) @ gain.T, gain


def state_record(prefix: str, state: np.ndarray, cfg: PhysicalEnKFConfig, n_clusters: int) -> dict[str, float]:
    values = physical_values(state, cfg, n_clusters)
    record = {
        f"{prefix}_eprime_gpa": float(values["eprime_pa"]) / 1.0e9,
        f"{prefix}_leakoff_m_sqrt_s": float(values["leakoff_m_sqrt_s"]),
        f"{prefix}_viscosity_pa_s": float(values["viscosity_pa_s"]),
        f"{prefix}_min_stress_mpa": float(values["min_horizontal_stress_mpa"]),
    }
    for index, value in enumerate(np.asarray(values["cluster_factors"]), start=1):
        record[f"{prefix}_factor_c{index}"] = float(value)
    return record
