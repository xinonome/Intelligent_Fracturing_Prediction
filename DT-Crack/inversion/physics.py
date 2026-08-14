"""Fast PKN forward operator and parameter-space EnKF utilities.

The EnKF state contains physical parameters.  The historical optional
``4 + n_clusters`` layout is still accepted for old experiments, but the new
default layout is ``[log E', log C_L, log mu, sigma_min, log K_IC]``.  Fracture
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
    base_fracture_toughness_pa_sqrt_m: float = 5.0e5
    height_m: float = 30.0
    pressure_proxy_scale: float = 30.0
    min_effective_rate_fraction: float = 0.05
    max_leakoff_fraction: float = 0.85
    stress_shadow_strength: float = 0.06
    leakoff_iterations: int = 6
    hydraulic_coupling_mode: str = "coupled"
    conductance_exponent: float = 0.45
    shadow_decay_clusters: float = 1.25
    # Enhanced profile terms. Keep them at zero by default so the historical
    # baseline remains reproducible; the enhanced validation profile enables
    # the physically motivated feedback explicitly.
    stress_shadow_feedback: float = 0.0
    pressure_leakoff_exponent: float = 0.0
    pressure_leakoff_reference_mpa: float = 15.0
    log_cluster_factor_state: bool = False
    # Mild engineering closure for fracture toughness.  The full toughness
    # response is supplied by a higher-fidelity teacher/solver later; this
    # bounded exponent makes the new K_IC state observable in the PKN baseline.
    fracture_toughness_length_exponent: float = 0.08
    fracture_toughness_aperture_exponent: float = 0.05


STATE_GLOBAL_NAMES = [
    "log_eprime_scale",
    "log_leakoff_scale",
    "log_viscosity_scale",
    "min_horizontal_stress_mpa",
]


def physical_values(
    state: np.ndarray,
    cfg: PhysicalEnKFConfig,
    n_clusters: int,
    cluster_factors: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    state = np.asarray(state, dtype=float)
    # A 4+n state is the historical layout used by the existing tests and
    # archived runs.  It must remain readable even when the compatibility flag
    # is false.  The new five-state layout reserves index 4 for log K_IC.
    legacy_cluster_layout = bool(state.shape[-1] == 4 + n_clusters)
    if cluster_factors is None:
        # Four-state runs intentionally do not expose a free per-cluster growth
        # factor.  Keep the legacy state format readable, but default missing
        # cluster state to a neutral multiplier.
        if legacy_cluster_layout:
            cluster_state = state[4 : 4 + n_clusters]
            cluster_factors = np.exp(cluster_state) if cfg.log_cluster_factor_state else cluster_state
        else:
            cluster_factors = np.ones(n_clusters, dtype=float)
    cluster_factors = np.asarray(cluster_factors, dtype=float)
    return {
        "eprime_pa": cfg.base_eprime_pa * np.exp(state[0]),
        "leakoff_m_sqrt_s": cfg.base_leakoff_m_sqrt_s * np.exp(state[1]),
        "viscosity_pa_s": cfg.base_viscosity_pa_s * np.exp(state[2]),
        "min_horizontal_stress_mpa": state[3],
        "fracture_toughness_pa_sqrt_m": (
            cfg.base_fracture_toughness_pa_sqrt_m * np.exp(state[4])
            if state.shape[-1] >= 5 and not legacy_cluster_layout
            else cfg.base_fracture_toughness_pa_sqrt_m
        ),
        "cluster_factors": cluster_factors,
    }


def clip_state(state: np.ndarray, n_clusters: int) -> np.ndarray:
    out = np.asarray(state, dtype=float).copy()
    out[..., 0] = np.clip(out[..., 0], np.log(0.45), np.log(2.2))
    out[..., 1] = np.clip(out[..., 1], np.log(0.1), np.log(8.0))
    out[..., 2] = np.clip(out[..., 2], np.log(0.2), np.log(5.0))
    out[..., 3] = np.clip(out[..., 3], 35.0, 90.0)
    legacy_cluster_layout = out.shape[-1] == 4 + n_clusters
    if legacy_cluster_layout:
        out[..., 4 : 4 + n_clusters] = np.clip(out[..., 4 : 4 + n_clusters], 0.65, 1.35)
    elif out.shape[-1] >= 5:
        out[..., 4] = np.clip(out[..., 4], np.log(0.25), np.log(4.0))
        if out.shape[-1] >= 5 + n_clusters:
            out[..., 5 : 5 + n_clusters] = np.clip(out[..., 5 : 5 + n_clusters], 0.65, 1.35)
    return out


def pkn_with_carter_leakoff(
    state: np.ndarray,
    q_base_m3_s: np.ndarray,
    t_seconds: float,
    cfg: PhysicalEnKFConfig,
    q_current_m3_s: np.ndarray | None = None,
    cluster_allocation: np.ndarray | None = None,
    cluster_current_allocation: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Evaluate enhanced PKN with leakoff, hydraulic coupling and stress shadow.

    ``q_base_m3_s`` is the cumulative-volume-equivalent rate used for fracture
    growth. ``q_current_m3_s`` is the current boundary rate used for aperture
    and pressure, so late rate ramps are not treated as full-history inputs.
    """

    n_clusters = len(q_base_m3_s)
    measured_allocation = cluster_allocation is not None
    allocation = None
    current_allocation = None
    if measured_allocation:
        allocation = normalize_allocation(cluster_allocation)
        current_allocation = normalize_allocation(
            cluster_current_allocation if cluster_current_allocation is not None else allocation
        )
    values = physical_values(
        state,
        cfg,
        n_clusters,
        cluster_factors=np.ones(n_clusters, dtype=float) if measured_allocation else None,
    )
    eprime = float(values["eprime_pa"])
    leakoff = float(values["leakoff_m_sqrt_s"])
    viscosity = float(values["viscosity_pa_s"])
    stress = float(values["min_horizontal_stress_mpa"])
    factors = np.asarray(values["cluster_factors"], dtype=float)
    toughness_ratio = float(
        np.clip(
            values["fracture_toughness_pa_sqrt_m"] / max(cfg.base_fracture_toughness_pa_sqrt_m, 1.0),
            0.25,
            4.0,
        )
    )
    toughness_length_factor = toughness_ratio ** (-max(cfg.fracture_toughness_length_exponent, 0.0))
    toughness_aperture_factor = toughness_ratio ** (-max(cfg.fracture_toughness_aperture_exponent, 0.0))
    t = max(float(t_seconds), 1.0)
    q_base = np.maximum(np.asarray(q_base_m3_s, dtype=float), 0.0)
    q_current = q_base if q_current_m3_s is None else np.maximum(np.asarray(q_current_m3_s, dtype=float), 0.0)
    if q_current.shape != q_base.shape:
        raise ValueError("q_current_m3_s must have the same cluster shape as q_base_m3_s")
    if measured_allocation:
        # Fiber-derived liquid allocation is the boundary condition for PKN.
        # EnKF must not estimate the same allocation a second time.
        q_nominal = float(q_base.sum()) * allocation
        q_current_nominal = float(q_current.sum()) * current_allocation
    else:
        weighted_rate = q_base * np.maximum(factors, 1e-6)
        q_nominal = float(q_base.sum()) * weighted_rate / max(float(weighted_rate.sum()), 1e-12)
        weighted_current = q_current * np.maximum(factors, 1e-6)
        q_current_nominal = float(q_current.sum()) * weighted_current / max(float(weighted_current.sum()), 1e-12)
    q_nominal = np.maximum(q_nominal, 1e-9)
    q_current_nominal = np.maximum(q_current_nominal, 1e-9)
    q_effective = q_nominal.copy()
    q_current_effective = q_current_nominal.copy()

    def shadow_factor_from_rate(rate: np.ndarray) -> np.ndarray:
        if n_clusters <= 1 or cfg.stress_shadow_strength <= 0.0:
            return np.ones(n_clusters, dtype=float)
        distance = np.abs(np.arange(n_clusters)[:, None] - np.arange(n_clusters)[None, :])
        interaction = np.exp(-distance / max(cfg.shadow_decay_clusters, 1e-6))
        np.fill_diagonal(interaction, 0.0)
        interaction /= np.maximum(interaction.sum(axis=1, keepdims=True), 1e-12)
        normalized_rate = rate / max(float(np.mean(rate)), 1e-12)
        shadow = np.tanh(interaction @ normalized_rate)
        return np.clip(1.0 - cfg.stress_shadow_strength * shadow, 0.85, 1.0)

    injected_volume = float(q_nominal.sum()) * t
    leakoff_volume = np.zeros(n_clusters, dtype=float)
    length = np.zeros(n_clusters, dtype=float)
    for _ in range(max(int(cfg.leakoff_iterations), 1)):
        length = 0.68 * ((q_effective**3 * eprime) / (viscosity * cfg.height_m**4)) ** 0.2 * t**0.8
        # Keep the leakoff fixed-point iteration tied to hydraulic geometry.
        # Apply the bounded toughness penalty after the mass-balance solve so a
        # K_IC update cannot indirectly create a larger rate merely by
        # changing the leakoff area in an intermediate iteration.
        length *= 0.95 + 0.10 * factors
        fracture_area = 4.0 * length * cfg.height_m
        # Carter leakoff is weakly pressure dependent in the enhanced profile.
        # It remains bounded by the injected volume below, so this term cannot
        # create fluid mass that was not injected.
        aperture_probe = 2.5 * (
            (q_current_effective**3 * viscosity) / (eprime * cfg.height_m**3)
        ) ** 0.2 * t**0.2
        net_pressure_probe = cfg.pressure_proxy_scale * eprime * float(
            np.mean(aperture_probe / 2.0)
        ) / cfg.height_m / 1.0e6
        pressure_multiplier = (
            1.0 + max(net_pressure_probe, 0.0) / max(cfg.pressure_leakoff_reference_mpa, 1e-6)
        ) ** max(cfg.pressure_leakoff_exponent, 0.0)
        raw_leakoff = 2.0 * leakoff * pressure_multiplier * fracture_area * np.sqrt(t)
        max_total_leakoff = cfg.max_leakoff_fraction * injected_volume
        leakoff_volume = raw_leakoff * min(1.0, max_total_leakoff / max(float(raw_leakoff.sum()), 1e-12))
        if measured_allocation:
            # The fiber allocation is an observed boundary condition.  Apply
            # Carter loss as a common mass-conserving efficiency rather than
            # subtracting a different leakoff amount from each cluster; the
            # latter can manufacture artificial cluster-length collapse when a
            # small cluster has a temporary rate change.
            effective_fraction = max(
                1.0 - float(leakoff_volume.sum()) / max(injected_volume, 1.0e-12),
                cfg.min_effective_rate_fraction,
            )
            q_effective = q_nominal * effective_fraction
            q_current_effective = q_current_nominal * effective_fraction
        else:
            q_effective = np.maximum(q_nominal - leakoff_volume / t, cfg.min_effective_rate_fraction * q_nominal)
            q_current_effective = np.maximum(
                q_current_nominal - leakoff_volume / t,
                cfg.min_effective_rate_fraction * q_current_nominal,
            )

        if cfg.hydraulic_coupling_mode == "coupled" and n_clusters > 1 and not measured_allocation:
            aperture_for_flow = 2.5 * (
                (q_current_effective**3 * viscosity) / (eprime * cfg.height_m**3)
            ) ** 0.2 * t**0.2
            relative_aperture = aperture_for_flow / max(float(np.mean(aperture_for_flow)), 1e-12)
            shadow_factor = shadow_factor_from_rate(q_effective)
            conductance = np.maximum(factors, 1e-6) * np.power(
                np.clip(relative_aperture, 0.35, 2.5), cfg.conductance_exponent
            )
            conductance *= np.power(shadow_factor, max(cfg.stress_shadow_feedback, 0.0))
            q_nominal = float(q_base.sum()) * conductance / max(float(conductance.sum()), 1e-12)
            q_current_nominal = float(q_current.sum()) * conductance / max(float(conductance.sum()), 1e-12)
            q_nominal = np.maximum(q_nominal, 1e-9)
            q_current_nominal = np.maximum(q_current_nominal, 1e-9)

    final_shadow_factor = shadow_factor_from_rate(q_effective)
    if cfg.stress_shadow_strength > 0.0:
        length *= final_shadow_factor
    length *= toughness_length_factor

    aperture_m = 2.5 * ((q_current_effective**3 * viscosity) / (eprime * cfg.height_m**3)) ** 0.2 * t**0.2
    aperture_m *= toughness_aperture_factor
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
        "measured_allocation_used": bool(measured_allocation),
        "cluster_allocation": allocation if allocation is not None else normalize_allocation(q_nominal),
        "cluster_current_allocation": current_allocation if current_allocation is not None else normalize_allocation(q_current_nominal),
        "net_pressure_mpa": net_pressure_mpa,
        "bottomhole_pressure_mpa": stress + net_pressure_mpa,
        "stress_shadow_factor": final_shadow_factor,
        "pressure_leakoff_multiplier": float(pressure_multiplier),
        "toughness_ratio": toughness_ratio,
        "toughness_length_factor": toughness_length_factor,
        "toughness_aperture_factor": toughness_aperture_factor,
        **values,
    }


def normalize_allocation(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.nan_to_num(np.asarray(values, dtype=float), nan=0.0), 0.0, None)
    total = float(values.sum())
    return values / total if total > 1.0e-12 else np.full(len(values), 1.0 / max(len(values), 1))


def enkf_update(
    ensemble: np.ndarray,
    predicted_obs: np.ndarray,
    observed_obs: np.ndarray,
    observation_std: np.ndarray,
    rng: np.random.Generator,
    localization: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x_anom = ensemble - ensemble.mean(axis=0)
    y_anom = predicted_obs - predicted_obs.mean(axis=0)
    denom = max(len(ensemble) - 1, 1)
    p_xy = x_anom.T @ y_anom / denom
    p_yy = y_anom.T @ y_anom / denom + np.diag(observation_std**2)
    gain = p_xy @ np.linalg.pinv(p_yy)
    if localization is not None:
        weights = np.asarray(localization, dtype=float)
        if weights.shape != gain.shape:
            raise ValueError(f"localization shape {weights.shape} does not match gain {gain.shape}")
        gain *= weights
    perturbed = observed_obs + rng.normal(0.0, observation_std, size=predicted_obs.shape)
    return ensemble + (perturbed - predicted_obs) @ gain.T, gain


def denkf_update(
    ensemble: np.ndarray,
    predicted_obs: np.ndarray,
    observed_obs: np.ndarray,
    observation_std: np.ndarray,
    covariance_inflation: float = 1.0,
    localization: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic ensemble Kalman update with optional localization.

    The DEnKF anomaly update avoids adding artificial random observation noise.
    Localization suppresses physically implausible cross-updates, for example a
    sand-share residual directly moving elastic modulus only because of finite
    ensemble sampling noise.
    """

    x = np.asarray(ensemble, dtype=float)
    y = np.asarray(predicted_obs, dtype=float)
    observed = np.asarray(observed_obs, dtype=float)
    std = np.maximum(np.asarray(observation_std, dtype=float), 1.0e-9)
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_anom = x - x_mean
    y_anom = y - y_mean
    denom = max(len(x) - 1, 1)
    p_xy = x_anom.T @ y_anom / denom
    p_yy = y_anom.T @ y_anom / denom + np.diag(std**2)
    gain = p_xy @ np.linalg.pinv(p_yy)
    if localization is not None:
        weights = np.asarray(localization, dtype=float)
        if weights.shape != gain.shape:
            raise ValueError(f"localization shape {weights.shape} does not match gain {gain.shape}")
        gain *= weights

    analysis_mean = x_mean + gain @ (observed - y_mean)
    analysis_anom = x_anom - 0.5 * (y_anom @ gain.T)
    analysis_anom *= max(float(covariance_inflation), 1.0)
    return analysis_mean + analysis_anom, gain


def state_record(prefix: str, state: np.ndarray, cfg: PhysicalEnKFConfig, n_clusters: int) -> dict[str, float]:
    values = physical_values(state, cfg, n_clusters)
    record = {
        f"{prefix}_eprime_gpa": float(values["eprime_pa"]) / 1.0e9,
        f"{prefix}_leakoff_m_sqrt_s": float(values["leakoff_m_sqrt_s"]),
        f"{prefix}_viscosity_pa_s": float(values["viscosity_pa_s"]),
        f"{prefix}_min_stress_mpa": float(values["min_horizontal_stress_mpa"]),
        f"{prefix}_fracture_toughness_pa_sqrt_m": float(values["fracture_toughness_pa_sqrt_m"]),
    }
    for index, value in enumerate(np.asarray(values["cluster_factors"]), start=1):
        record[f"{prefix}_factor_c{index}"] = float(value)
    return record
