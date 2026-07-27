from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_fusion import (
    FiberObservationConfig,
    PressureModelConfig,
    build_fiber_length_observation,
    controls_for_step,
    load_stage_pressure_schedule,
    load_fiber_json,
    load_frac_monitor_text,
    load_well_trajectory,
    make_cluster_trajectory_positions,
    parse_fiber_api_payload,
    pressure_for_step,
)
from forward_models import LengthForwardModel, build_length_forward_model


def configure_matplotlib_chinese_font() -> None:
    for font_path in [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf"), Path(r"C:\Windows\Fonts\simsun.ttc")]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            return


configure_matplotlib_chinese_font()


def calc_eprime(young_modulus_pa: float, poisson_ratio: float) -> float:
    return young_modulus_pa / (1.0 - poisson_ratio**2)


def calc_pkn(flow_rate_m3_s: np.ndarray | float, viscosity_pa_s: float, e_prime_pa: float, height_m: float, t_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Same simplified PKN scaling form as the client pkn4.py."""

    q = np.maximum(np.asarray(flow_rate_m3_s, dtype=float), 1e-9)
    t = max(float(t_seconds), 1e-9)
    w_max = 2.5 * ((q**3 * viscosity_pa_s) / (e_prime_pa * height_m**3)) ** 0.2 * t**0.2
    half_length = 0.68 * ((q**3 * e_prime_pa) / (viscosity_pa_s * height_m**4)) ** 0.2 * t**0.8
    return w_max, half_length


def build_cluster_factors(n_clusters: int) -> np.ndarray:
    """Edge clusters expand slightly better; middle clusters are constrained."""

    if n_clusters == 1:
        return np.ones(1)
    idx = np.arange(n_clusters)
    center = (n_clusters - 1) / 2.0
    edge_weight = np.abs(idx - center) / center
    factors = 0.86 + 0.22 * edge_weight
    return factors / np.mean(factors)


def calc_fracture_area(half_length_m: np.ndarray, height_m: float) -> np.ndarray:
    return 4.0 * half_length_m * height_m


def calc_pkn_volume_approx(w_max_m: np.ndarray, half_length_m: np.ndarray, height_m: float) -> np.ndarray:
    return w_max_m * (2.0 * half_length_m / 1.25) * (np.pi * height_m / 4.0)


def pkn_observation_from_factors(
    factor_state: np.ndarray,
    cluster_x: np.ndarray,
    q_base: np.ndarray | float,
    viscosity_pa_s: float,
    e_prime_pa: float,
    height_m: float,
    t_seconds: float,
    forward_model: LengthForwardModel | None = None,
) -> pd.DataFrame:
    """Run the forward model with the current inverted PKN parameter factors.

    The EnKF state is not fracture length itself. It is a per-cluster effective
    growth / intake factor used by the PKN forward model. DAS-derived equivalent
    half-lengths are observations. EnKF updates the factors, then PKN is run again
    to produce half-length, aperture, area and volume.
    """

    model = forward_model or build_length_forward_model("pkn4")
    return model.simulate_lengths(
        factor_state=factor_state,
        cluster_x=cluster_x,
        q_base=q_base,
        viscosity_pa_s=viscosity_pa_s,
        e_prime_pa=e_prime_pa,
        height_m=height_m,
        t_seconds=t_seconds,
    ).table


def enforce_non_decreasing_by_parameter_update(
    factor_state: np.ndarray,
    previous_half_length: np.ndarray | None,
    cluster_x: np.ndarray,
    q_base: np.ndarray | float,
    viscosity_pa_s: float,
    e_prime_pa: float,
    height_m: float,
    t_seconds: float,
    forward_model: LengthForwardModel,
) -> np.ndarray:
    """Adjust inverted PKN factors instead of directly overwriting length.

    Earlier demos clipped posterior half-length directly for visual continuity.
    That is misleading for parameter inversion. This helper keeps the constraint
    in parameter space: if the updated PKN factors would make a cluster shrink
    relative to the previous frame, increase that cluster factor slightly and
    re-run PKN until the forward result is non-decreasing or bounds are reached.
    """

    if previous_half_length is None:
        return np.asarray(factor_state, dtype=float)
    factors = np.clip(np.asarray(factor_state, dtype=float).copy(), 0.65, 1.35)
    previous = np.asarray(previous_half_length, dtype=float)
    for _ in range(8):
        trial = pkn_observation_from_factors(
            factors,
            cluster_x,
            q_base,
            viscosity_pa_s,
            e_prime_pa,
            height_m,
            t_seconds,
            forward_model,
        )
        length = trial["half_length_m"].to_numpy(dtype=float)
        mask = length < previous
        if not np.any(mask):
            break
        ratio = previous[mask] / np.maximum(length[mask], 1e-9)
        factors[mask] *= np.clip(ratio**1.1, 1.0, 1.08)
        factors = np.clip(factors, 0.65, 1.35)
    return factors


def enkf_update(
    ensemble: np.ndarray,
    predicted_obs: np.ndarray,
    observed_obs: np.ndarray,
    obs_noise: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    state_mean = ensemble.mean(axis=0)
    obs_mean = predicted_obs.mean(axis=0)
    x_anom = ensemble - state_mean
    y_anom = predicted_obs - obs_mean
    denom = max(len(ensemble) - 1, 1)
    cov_xy = x_anom.T @ y_anom / denom
    cov_yy = y_anom.T @ y_anom / denom + np.diag(obs_noise**2)
    gain = cov_xy @ np.linalg.pinv(cov_yy)
    perturbed_obs = observed_obs.reshape(1, -1) + rng.normal(0.0, obs_noise, size=predicted_obs.shape)
    updated = ensemble + (perturbed_obs - predicted_obs) @ gain.T
    return np.clip(updated, 0.65, 1.35), gain


def pkn_proxy_net_pressure_mpa(table: pd.DataFrame, e_prime_pa: float, height_m: float, scale: float = 1.0) -> float:
    """Estimate net pressure from PKN aperture for joint EnKF assimilation.

    This is a lightweight proxy, not a full geomechanics pressure solver. It
    converts mean maximum aperture to net pressure using a linear elastic PKN
    style relation so pressure observations can influence the inverted PKN
    growth/intake factors.
    """

    aperture_m = table["max_aperture_mm"].to_numpy(dtype=float) / 2000.0
    mean_aperture = float(np.nanmean(np.clip(aperture_m, 0.0, None)))
    return max(float(scale) * e_prime_pa * mean_aperture / max(float(height_m), 1e-9) / 1.0e6, 0.0)


def relative_error(values: np.ndarray, reference: np.ndarray) -> float:
    return float(np.mean(np.abs(values - reference) / np.maximum(np.abs(reference), 1e-9)))


def das_scenario_profile(name: str, n_clusters: int, step_idx: int) -> np.ndarray:
    """Synthetic DAS response patterns used before real DAS is connected."""

    idx = np.arange(n_clusters, dtype=float)
    if name == "uniform":
        base = np.ones(n_clusters)
    elif name == "left_dominant":
        base = np.linspace(1.18, 0.86, n_clusters)
    elif name == "right_dominant":
        base = np.linspace(0.86, 1.18, n_clusters)
    elif name == "middle_dominant":
        center = (n_clusters - 1) / 2.0
        base = 0.88 + 0.34 * np.exp(-((idx - center) ** 2) / max(n_clusters / 2.0, 1.0))
    elif name == "alternating":
        base = 1.0 + 0.14 * np.where((idx.astype(int) % 2) == 0, 1.0, -1.0)
    else:
        raise ValueError(f"Unsupported DAS scenario: {name}")
    pulse = 1.0 + 0.018 * np.sin(step_idx * 0.85 + idx * 0.55)
    return base * pulse / np.mean(base * pulse)


def run_demo(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    e_prime = calc_eprime(args.young_modulus_pa, args.poisson_ratio)
    forward_model = build_length_forward_model(args.forward_model)
    q_base = args.total_flow_rate_m3_s / args.n_clusters
    fiber_controls = None
    fiber_meta: dict[str, object] = {}
    if args.fiber_json:
        fiber_payload = load_fiber_json(args.fiber_json)
        fiber_tables = parse_fiber_api_payload(fiber_payload, default_step_seconds=args.default_step_seconds)
        fiber_controls = fiber_tables.controls
        fiber_meta = fiber_tables.meta
        if not fiber_controls.empty:
            args.n_clusters = int(fiber_controls["cluster_id"].nunique())
    if args.frac_monitor_text:
        monitor_tables = load_frac_monitor_text(args.frac_monitor_text, default_step_seconds=args.default_step_seconds)
        fiber_controls = monitor_tables.controls
        fiber_meta = monitor_tables.meta
        if not fiber_controls.empty:
            args.n_clusters = int(fiber_controls["cluster_id"].nunique())
            if args.use_real_monitor_timeline:
                available_steps = np.asarray(sorted(fiber_controls["step"].unique()), dtype=int)
                sample_count = min(int(args.max_playback_steps), len(available_steps))
                sample_idx = np.linspace(0, len(available_steps) - 1, sample_count).round().astype(int)
                args.source_steps = available_steps[sample_idx].tolist()
                first_time = fiber_controls["time"].dropna().min()
                sampled_times = [
                    fiber_controls.loc[fiber_controls["step"] == step, "time"].dropna().iloc[0]
                    for step in args.source_steps
                ]
                args.time_steps = [max(float((time_value - first_time).total_seconds()), 1.0) for time_value in sampled_times]
    cluster_x = np.linspace(args.well_len_m * args.cluster_start, args.well_len_m * args.cluster_end, args.n_clusters)
    trajectory_positions = pd.DataFrame()
    if args.well_trajectory_csv:
        trajectory = load_well_trajectory(args.well_trajectory_csv)
        trajectory_positions = make_cluster_trajectory_positions(
            trajectory,
            args.n_clusters,
            stage_md_start_m=args.stage_md_start_m,
            stage_md_end_m=args.stage_md_end_m,
        )
        if not trajectory_positions.empty and trajectory_positions[["north_m", "east_m"]].notna().all().all():
            east = trajectory_positions["east_m"].to_numpy(dtype=float)
            north = trajectory_positions["north_m"].to_numpy(dtype=float)
            dist = np.zeros(len(east), dtype=float)
            if len(east) > 1:
                dist[1:] = np.cumsum(np.sqrt(np.diff(east) ** 2 + np.diff(north) ** 2))
            if dist.max() > 1e-9:
                cluster_x = args.well_len_m * (args.cluster_start + (args.cluster_end - args.cluster_start) * dist / dist.max())
    pressure_schedule = pd.DataFrame()
    pressure_meta: dict[str, object] = {}
    if args.construction_pressure_xls:
        pressure_measured_depth = args.pressure_measured_depth_m
        pressure_vertical_depth = args.pressure_vertical_depth_m
        if not trajectory_positions.empty:
            if pressure_measured_depth is None and "measured_depth_m" in trajectory_positions:
                pressure_measured_depth = float(trajectory_positions["measured_depth_m"].max())
            if pressure_vertical_depth is None and "vertical_depth_m" in trajectory_positions:
                pressure_vertical_depth = float(trajectory_positions["vertical_depth_m"].max())
        pressure_config = PressureModelConfig(
            fcd=args.pressure_fcd,
            dwell_m=args.pressure_dwell_m,
            proppant_density_kg_m3=args.pressure_proppant_density_kg_m3,
            base_fluid_density_kg_m3=args.pressure_base_fluid_density_kg_m3,
            jzl=args.pressure_jzl,
            perforation_count=args.pressure_perforation_count,
            perforation_diameter_m=args.pressure_perforation_diameter_m,
            perforation_erosion_coeff=args.pressure_perforation_erosion_coeff,
            perforation_flow_coeff=args.pressure_perforation_flow_coeff,
            perforation_flow_decay_coeff=args.pressure_perforation_flow_decay_coeff,
            min_horizontal_stress_mpa=args.pressure_min_horizontal_stress_mpa,
            rolling_window_seconds=args.pressure_rolling_window_seconds,
            step_seconds=args.pressure_step_seconds,
        )
        pressure_schedule, pressure_meta = load_stage_pressure_schedule(
            args.construction_pressure_xls,
            config=pressure_config,
            measured_depth_m=float(pressure_measured_depth or 5200.0),
            vertical_depth_m=float(pressure_vertical_depth or 3200.0),
        )
    base_factors = build_cluster_factors(args.n_clusters)
    true_state = base_factors * 1.035
    prior_state = base_factors * 0.94
    ensemble = np.stack([prior_state + rng.normal(0.0, 0.055, size=prior_state.shape) for _ in range(args.ensemble_size)])
    obs_noise = np.full(args.n_clusters, args.length_noise_m)
    process_noise = np.full(args.n_clusters, 0.012)
    observation_config = FiberObservationConfig(
        cumulative_liquid_weight=args.obs_cum_liquid_weight,
        cumulative_sand_weight=args.obs_cum_sand_weight,
        instant_liquid_weight=args.obs_instant_liquid_weight,
        instant_sand_weight=args.obs_instant_sand_weight,
        balance_gain=args.obs_balance_gain,
        min_factor=args.obs_min_factor,
        max_factor=args.obs_max_factor,
    )

    rows: list[pd.DataFrame] = []
    history: list[dict] = []
    previous_prior_flat: np.ndarray | None = None
    previous_observed_flat: np.ndarray | None = None
    previous_posterior_flat: np.ndarray | None = None
    previous_observation_factor: np.ndarray | None = None
    for step_idx, t_seconds in enumerate(args.time_steps):
        source_step = int(round(float(t_seconds) / max(float(args.default_step_seconds), 1e-9)))
        step_controls = None
        q_base_step: np.ndarray | float = q_base
        if fiber_controls is not None and not fiber_controls.empty:
            available_steps = sorted(fiber_controls["step"].unique())
            if hasattr(args, "source_steps"):
                source_step = args.source_steps[min(step_idx, len(args.source_steps) - 1)]
            else:
                source_step = available_steps[min(step_idx, len(available_steps) - 1)]
            step_controls = controls_for_step(fiber_controls, int(source_step))
            cumulative_liquid = step_controls["cumulative_liquid_volume_m3"].to_numpy(dtype=float)
            elapsed_seconds = max(float(t_seconds), float(args.default_step_seconds), 1.0)
            if np.nansum(cumulative_liquid) > 1e-9:
                # Fracture half-length is an accumulated state. Use cumulative
                # intake / elapsed time as the PKN effective rate so temporary
                # flow-rate fluctuation does not make the forward length shrink.
                q_base_step = np.maximum(cumulative_liquid / elapsed_seconds, 1e-9)
            else:
                q_base_step = np.maximum(step_controls["flow_rate_m3_min"].to_numpy() / 60.0, 1e-9)
        pressure_row = None
        if not pressure_schedule.empty:
            pressure_row = pressure_for_step(pressure_schedule, int(source_step))
        observed_net_pressure_mpa = float(pressure_row["net_pressure_mpa"]) if pressure_row is not None else None
        # Synthetic field condition: the effective cluster factors drift slowly
        # with time, representing stress shadow and uneven cluster intake.
        drift = 1.0 + 0.025 * np.sin(step_idx / 2.0 + np.arange(args.n_clusters) * 0.7)
        das_profile = das_scenario_profile(args.das_scenario, args.n_clusters, step_idx)
        truth_state_step = true_state.copy()
        truth_state_step *= drift * das_profile

        truth = pkn_observation_from_factors(
            truth_state_step,
            cluster_x,
            q_base_step,
            args.viscosity_pa_s,
            e_prime,
            args.height_m,
            t_seconds,
            forward_model,
        )

        ensemble = np.clip(ensemble + rng.normal(0.0, process_noise, size=ensemble.shape), 0.65, 1.35)
        pred_tables = [
            pkn_observation_from_factors(member, cluster_x, q_base, args.viscosity_pa_s, e_prime, args.height_m, t_seconds, forward_model)
            if step_controls is None
            else pkn_observation_from_factors(member, cluster_x, q_base_step, args.viscosity_pa_s, e_prime, args.height_m, t_seconds, forward_model)
            for member in ensemble
        ]
        predicted_lengths = np.asarray([tbl["half_length_m"].to_numpy() for tbl in pred_tables])
        if observed_net_pressure_mpa is not None and args.assimilate_net_pressure:
            predicted_pressure = np.asarray(
                [
                    pkn_proxy_net_pressure_mpa(tbl, e_prime, args.height_m, args.net_pressure_proxy_scale)
                    for tbl in pred_tables
                ],
                dtype=float,
            ).reshape(-1, 1)
            predicted_obs = np.hstack([predicted_lengths, predicted_pressure])
        else:
            predicted_obs = predicted_lengths
        prior_mean = ensemble.mean(axis=0)
        prior_parameter_factor = prior_mean.copy()
        prior = pkn_observation_from_factors(prior_mean, cluster_x, q_base_step, args.viscosity_pa_s, e_prime, args.height_m, t_seconds, forward_model)
        prior_flat = prior["half_length_m"].to_numpy()
        prior_proxy_net_pressure_mpa = pkn_proxy_net_pressure_mpa(
            prior,
            e_prime,
            args.height_m,
            args.net_pressure_proxy_scale,
        )
        observation_details = pd.DataFrame()
        if step_controls is not None and not step_controls.empty:
            observation_details = build_fiber_length_observation(step_controls, prior_flat, observation_config)
            observed_flat = observation_details["observed_half_length_m"].to_numpy()
        else:
            observed_flat = truth["half_length_m"].to_numpy() + rng.normal(0.0, args.length_noise_m, args.n_clusters)
        if not observation_details.empty and args.obs_factor_smoothing > 0.0:
            current_factor = observation_details["fiber_observation_factor"].to_numpy(dtype=float)
            if previous_observation_factor is None:
                smoothed_factor = current_factor
            else:
                alpha = float(np.clip(args.obs_factor_smoothing, 0.0, 1.0))
                smoothed_factor = alpha * previous_observation_factor + (1.0 - alpha) * current_factor
                smoothed_factor = np.clip(smoothed_factor, args.obs_min_factor, args.obs_max_factor)
            previous_observation_factor = smoothed_factor.copy()
            observation_details["fiber_observation_factor_raw"] = current_factor
            observation_details["fiber_observation_factor"] = smoothed_factor
            observation_details["observed_half_length_m"] = prior_flat * smoothed_factor
            observed_flat = observation_details["observed_half_length_m"].to_numpy()
        if args.enforce_non_decreasing_length:
            if previous_observed_flat is not None:
                observed_flat = np.maximum(observed_flat, previous_observed_flat)
        if not observation_details.empty:
            observation_details["observed_half_length_m"] = observed_flat
            observation_details["fiber_observation_factor_effective"] = observed_flat / np.maximum(prior_flat, 1e-9)
        if observed_net_pressure_mpa is not None and args.assimilate_net_pressure:
            observed_obs = np.r_[observed_flat, observed_net_pressure_mpa]
            obs_noise = np.r_[np.full(args.n_clusters, args.length_noise_m), args.net_pressure_noise_mpa]
        else:
            observed_obs = observed_flat
            obs_noise = np.full(args.n_clusters, args.length_noise_m)
        prior_error = relative_error(prior_flat, observed_flat)

        ensemble, gain = enkf_update(ensemble, predicted_obs, observed_obs, obs_noise, rng)
        raw_posterior_parameter_factor = ensemble.mean(axis=0)
        posterior_parameter_factor = raw_posterior_parameter_factor.copy()
        if args.enforce_non_decreasing_length:
            posterior_parameter_factor = enforce_non_decreasing_by_parameter_update(
                posterior_parameter_factor,
                previous_posterior_flat,
                cluster_x,
                q_base_step,
                args.viscosity_pa_s,
                e_prime,
                args.height_m,
                t_seconds,
                forward_model,
            )
            adjustment = posterior_parameter_factor - raw_posterior_parameter_factor
            ensemble = np.clip(ensemble + adjustment.reshape(1, -1), 0.65, 1.35)
        posterior = pkn_observation_from_factors(posterior_parameter_factor, cluster_x, q_base_step, args.viscosity_pa_s, e_prime, args.height_m, t_seconds, forward_model)
        posterior_flat = posterior["half_length_m"].to_numpy()
        posterior_proxy_net_pressure_mpa = pkn_proxy_net_pressure_mpa(
            posterior,
            e_prime,
            args.height_m,
            args.net_pressure_proxy_scale,
        )
        posterior_error = relative_error(posterior_flat, observed_flat)

        out = posterior.copy()
        out.insert(0, "time_s", t_seconds)
        out["prior_half_length_m"] = prior["half_length_m"].to_numpy()
        out["prior_max_aperture_mm"] = prior["max_aperture_mm"].to_numpy()
        out["prior_growth_factor"] = prior_parameter_factor
        out["posterior_growth_factor"] = posterior_parameter_factor
        out["parameter_update_delta"] = posterior_parameter_factor - prior_parameter_factor
        out["prior_Q_cluster_m3s"] = prior["Q_cluster_m3s"].to_numpy()
        out["posterior_Q_cluster_m3s"] = posterior["Q_cluster_m3s"].to_numpy()
        out["inverted_parameter"] = "per_cluster_effective_growth_factor"
        out["prior_proxy_net_pressure_mpa"] = prior_proxy_net_pressure_mpa
        out["posterior_proxy_net_pressure_mpa"] = posterior_proxy_net_pressure_mpa
        out["net_pressure_assimilated"] = bool(observed_net_pressure_mpa is not None and args.assimilate_net_pressure)
        out["observed_half_length_m"] = observed_flat
        out["das_scenario_factor"] = das_profile
        out["forward_model"] = forward_model.model_name
        if step_controls is not None:
            out["fiber_liquid_volume_m3"] = step_controls["liquid_volume_m3"].to_numpy()
            out["fiber_sand_mass_t"] = step_controls["sand_mass_t"].to_numpy()
            out["fiber_cumulative_liquid_volume_m3"] = step_controls["cumulative_liquid_volume_m3"].to_numpy()
            out["fiber_cumulative_sand_mass_t"] = step_controls["cumulative_sand_mass_t"].to_numpy()
            out["fiber_allocation_weight"] = step_controls["allocation_weight"].to_numpy()
            if not observation_details.empty:
                for column in [
                    "cum_liquid_score",
                    "cum_sand_score",
                    "instant_liquid_score",
                    "instant_sand_score",
                    "fiber_activity_score",
                    "fiber_balance_imbalance",
                    "fiber_observation_factor",
                    "fiber_observation_factor_effective",
                ]:
                    out[column] = observation_details[column].to_numpy()
                if "fiber_observation_factor_raw" in observation_details.columns:
                    out["fiber_observation_factor_raw"] = observation_details["fiber_observation_factor_raw"].to_numpy()
        if pressure_row is not None:
            for column in [
                "source_second",
                "surface_pressure_mpa",
                "cumulative_liquid_m3",
                "flow_rate_m3_min",
                "sand_ratio_percent",
                "slurry_density_kg_m3",
                "hydrostatic_pressure_mpa",
                "pipe_friction_mpa",
                "perforation_friction_mpa",
                "bottomhole_pressure_mpa",
                "net_pressure_raw_mpa",
                "net_pressure_mpa",
                "perforation_diameter_m",
                "perforation_flow_coeff",
            ]:
                out[column] = float(pressure_row[column])
            out["observed_net_pressure_mpa"] = observed_net_pressure_mpa
            out["prior_net_pressure_residual_mpa"] = prior_proxy_net_pressure_mpa - observed_net_pressure_mpa
            out["posterior_net_pressure_residual_mpa"] = posterior_proxy_net_pressure_mpa - observed_net_pressure_mpa
        rows.append(out)
        history_row = {
            "step": step_idx,
            "time_s": t_seconds,
            "prior_error": prior_error,
            "posterior_error": posterior_error,
            "within_15_percent": posterior_error <= 0.15,
            "mean_kalman_gain_abs": float(np.mean(np.abs(gain))),
            "mean_prior_growth_factor": float(np.mean(prior_parameter_factor)),
            "mean_posterior_growth_factor": float(np.mean(posterior_parameter_factor)),
            "mean_parameter_update_abs": float(np.mean(np.abs(posterior_parameter_factor - prior_parameter_factor))),
            "max_half_length_m": float(out["half_length_m"].max()),
            "max_aperture_mm": float(out["max_aperture_mm"].max()),
            "total_area_m2": float(out["area_m2"].sum()),
            "total_volume_m3": float(out["volume_m3"].sum()),
            "prior_proxy_net_pressure_mpa": prior_proxy_net_pressure_mpa,
            "posterior_proxy_net_pressure_mpa": posterior_proxy_net_pressure_mpa,
            "net_pressure_assimilated": bool(observed_net_pressure_mpa is not None and args.assimilate_net_pressure),
        }
        if observed_net_pressure_mpa is not None:
            history_row.update(
                {
                    "observed_net_pressure_mpa": observed_net_pressure_mpa,
                    "prior_net_pressure_abs_error_mpa": abs(prior_proxy_net_pressure_mpa - observed_net_pressure_mpa),
                    "posterior_net_pressure_abs_error_mpa": abs(posterior_proxy_net_pressure_mpa - observed_net_pressure_mpa),
                }
            )
        history.append(history_row)
        previous_prior_flat = prior_flat.copy()
        previous_observed_flat = observed_flat.copy()
        previous_posterior_flat = posterior_flat.copy()

    summary = pd.DataFrame(history)
    table = pd.concat(rows, ignore_index=True)
    if not trajectory_positions.empty:
        traj_cols = [
            "cluster_id",
            "measured_depth_m",
            "vertical_depth_m",
            "north_m",
            "east_m",
            "stage_md_start_m",
            "stage_md_end_m",
        ]
        table = table.merge(trajectory_positions[traj_cols], on="cluster_id", how="left")
    metrics = {
        "final_prior_error": float(summary["prior_error"].iloc[-1]),
        "final_posterior_error": float(summary["posterior_error"].iloc[-1]),
        "final_within_15_percent": bool(summary["within_15_percent"].iloc[-1]),
        "within_15_percent_rate": float(summary["within_15_percent"].mean()),
        "max_half_length_m": float(table["half_length_m"].max()),
        "max_aperture_mm_pkn_derived": float(table["max_aperture_mm"].max()),
    }
    return {
        "table": table,
        "history": summary,
        "metrics": metrics,
        "cluster_x": cluster_x,
        "e_prime": e_prime,
        "fiber_meta": fiber_meta,
        "trajectory_positions": trajectory_positions,
        "pressure_meta": pressure_meta,
        "pressure_schedule": pressure_schedule,
    }


def plot_montage(table: pd.DataFrame, history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for cid, sub in table.groupby("cluster_id"):
        axes[0, 0].plot(sub["time_s"], sub["half_length_m"], label=f"C{cid}")
        axes[0, 1].plot(sub["time_s"], sub["max_aperture_mm"], label=f"C{cid}")
    axes[0, 0].set_title("PKN re-forward half-length after EnKF parameter inversion")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Half-length (m)")
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[0, 0].grid(alpha=0.25)
    axes[0, 1].set_title("PKN-derived maximum aperture, not inversion target")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Aperture (mm)")
    axes[0, 1].legend(ncol=3, fontsize=8)
    axes[0, 1].grid(alpha=0.25)
    axes[1, 0].plot(history["time_s"], history["prior_error"] * 100, color="#F58518", label="PKN prior error")
    axes[1, 0].plot(history["time_s"], history["posterior_error"] * 100, color="#54A24B", label="Updated-parameter PKN error")
    axes[1, 0].axhline(15, color="#B22222", linestyle="--", label="15% target")
    axes[1, 0].set_title("DAS-equivalent length mismatch")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Relative error (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)
    pivot = table.pivot_table(index="cluster_id", columns="time_s", values="max_aperture_mm")
    im = axes[1, 1].imshow(pivot, aspect="auto", cmap="turbo", origin="lower")
    axes[1, 1].set_title("Derived aperture heatmap")
    axes[1, 1].set_xlabel("Time step")
    axes[1, 1].set_ylabel("Cluster")
    fig.colorbar(im, ax=axes[1, 1], label="mm")
    fig.suptitle("Client pkn4-style PKN Forward + DAS-only EnKF Parameter Inversion", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _fracture_surface_arrays(
    x_center: float,
    half_length_m: float,
    aperture_max_mm: float,
    height_m: float,
    gain: float,
    ny: int = 90,
    nz: int = 46,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.linspace(-half_length_m, half_length_m, ny)
    z = np.linspace(-height_m / 2.0, height_m / 2.0, nz)
    yg, zg = np.meshgrid(y, z, indexing="ij")
    length_decay = np.clip(1.0 - np.abs(yg) / max(half_length_m, 1e-9), 0.0, None) ** 0.25
    height_decay = np.sqrt(np.clip(1.0 - (2.0 * zg / height_m) ** 2, 0.0, None))
    aperture_mm = aperture_max_mm * length_decay * height_decay
    x_half_vis = (aperture_mm / 2000.0) * gain
    return x_center + x_half_vis, x_center - x_half_vis, yg, zg, aperture_mm


def render_pkn4_style_frame(
    frame_table: pd.DataFrame,
    history_row: pd.Series,
    args: argparse.Namespace,
    path: Path,
    global_max_length: float,
    global_max_aperture: float,
) -> None:
    """Render one pkn4.py-style 3D frame and overlay parameter inversion."""

    t_seconds = float(history_row["time_s"])
    max_aperture = max(float(global_max_aperture), 1e-6)
    max_length = max(float(global_max_length), 1.0)
    norm = Normalize(vmin=0.0, vmax=max_aperture)
    cmap = plt.get_cmap("turbo")

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # Horizontal wellbore.
    ax.plot([0, args.well_len_m], [0, 0], [0, 0], color="#111827", linewidth=5, label="水平井井筒")
    if {"north_m", "east_m", "measured_depth_m"}.issubset(frame_table.columns):
        trajectory = frame_table.sort_values("x_center_m")
        north = trajectory["north_m"].to_numpy(dtype=float)
        north_centered = north - np.nanmean(north)
        scale = (max_length * 0.10) / max(float(np.nanmax(np.abs(north_centered))), 1e-9)
        ax.plot(
            trajectory["x_center_m"].to_numpy(dtype=float),
            north_centered * scale,
            np.full(len(trajectory), -args.height_m * 0.72),
            color="#64748b",
            linewidth=3,
            linestyle=":",
            label="真实井轨迹投影",
        )
        ax.text(
            float(trajectory["x_center_m"].min()),
            float(np.nanmin(north_centered * scale)),
            -args.height_m * 0.86,
            "灰色虚线=井轨迹投影",
            color="#64748b",
            fontsize=9,
        )

    for _, row in frame_table.iterrows():
        x_center = float(row["x_center_m"])
        posterior_length = float(row["half_length_m"])
        prior_length = float(row["prior_half_length_m"])
        observed_length = float(row["observed_half_length_m"])
        aperture = float(row["max_aperture_mm"])

        xp, xn, yg, zg, aperture_grid = _fracture_surface_arrays(
            x_center=x_center,
            half_length_m=posterior_length,
            aperture_max_mm=aperture,
            height_m=args.height_m,
            gain=args.visual_aperture_gain,
        )
        colors = cmap(norm(aperture_grid))
        ax.plot_surface(xp, yg, zg, facecolors=colors, linewidth=0, antialiased=True, shade=False, alpha=0.90)
        ax.plot_surface(xn, yg, zg, facecolors=colors, linewidth=0, antialiased=True, shade=False, alpha=0.90)

        # Cluster marker.
        ax.plot([x_center, x_center], [0, 0], [-args.height_m * 0.75, args.height_m * 0.75], color="#ef4444", linewidth=2.8)
        label = f"C{int(row['cluster_id'])}"
        if "measured_depth_m" in frame_table.columns and pd.notna(row.get("measured_depth_m")):
            label += f"\nMD={float(row['measured_depth_m']):.0f}m"
        ax.text(x_center, -max_length * 0.08, args.height_m * 0.88, label, color="#ef4444", fontsize=10)

        # Prior and observed length overlays: these make parameter inversion visible.
        ax.plot(
            [x_center, x_center],
            [-prior_length, prior_length],
            [args.height_m * 0.58, args.height_m * 0.58],
            color="#f97316",
            linewidth=2.4,
            linestyle="--",
        )
        ax.plot(
            [x_center, x_center],
            [-observed_length, observed_length],
            [args.height_m * 0.72, args.height_m * 0.72],
            color="#111827",
            linewidth=2.2,
            linestyle="-.",
        )

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.62, pad=0.08)
    cbar.set_label("PKN派生裂缝开度 (mm)")

    ax.set_title(
        (
            f"PKN4风格 DAS-only 缝长反演闭环 | t={int(t_seconds)}s | "
            f"PKN先验误差={history_row['prior_error']*100:.2f}% | "
            f"参数更新后误差={history_row['posterior_error']*100:.2f}%"
        ),
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    ax.text2D(
        0.02,
        0.94,
        "黑色虚线=DAS等效观测缝长；橙色虚线=PKN先验缝长；彩色裂缝面=EnKF更新PKN参数后再正演结果",
        transform=ax.transAxes,
        fontsize=11,
        color="#111827",
    )
    ax.text2D(
        0.02,
        0.89,
        "流程：PKN参数先验 -> PKN正演 -> DAS观测残差 -> EnKF更新参数 -> PKN再正演 -> 下一时刻继续",
        transform=ax.transAxes,
        fontsize=10,
        color="#374151",
    )

    ax.set_xlabel("井筒方向 X (m)")
    ax.set_ylabel("裂缝两翼方向 Y (m)")
    ax.set_zlabel("固定裂缝高度 Z (m)")
    ax.set_xlim(-5, args.well_len_m + 10)
    ax.set_ylim(-max_length * 1.08, max_length * 1.08)
    ax.set_zlim(-args.height_m * 0.85, args.height_m * 0.85)
    ax.view_init(elev=22, azim=-58)
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def render_pkn4_style_frames(table: pd.DataFrame, history: pd.DataFrame, args: argparse.Namespace, figures_dir: Path) -> list[str]:
    frame_paths: list[str] = []
    global_max_length = float(table[["half_length_m", "prior_half_length_m", "observed_half_length_m"]].max().max())
    global_max_aperture = float(table[["max_aperture_mm", "prior_max_aperture_mm"]].max().max())
    for _, history_row in history.iterrows():
        t_seconds = float(history_row["time_s"])
        frame_table = table[table["time_s"] == t_seconds].copy()
        frame_path = figures_dir / f"pkn4_style_enkf_frame_t_{int(t_seconds)}s.png"
        render_pkn4_style_frame(frame_table, history_row, args, frame_path, global_max_length, global_max_aperture)
        frame_paths.append(str(Path("figures") / frame_path.name).replace("\\", "/"))
    return frame_paths


def write_playback_html(table: pd.DataFrame, history: pd.DataFrame, args: argparse.Namespace, path: Path, frame_paths: list[str]) -> None:
    payload = {
        "records": table.to_dict(orient="records"),
        "history": history.to_dict(orient="records"),
        "time_steps": list(map(float, args.time_steps)),
        "frames": frame_paths,
        "well_len": args.well_len_m,
        "height": args.height_m,
        "max_length": float(table[["half_length_m", "prior_half_length_m", "observed_half_length_m"]].max().max()),
        "max_aperture": float(table[["max_aperture_mm", "prior_max_aperture_mm"]].max().max()),
    }
    data = json.dumps(payload, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>PKN4风格 EnKF 参数反演闭环</title>
<style>
:root {{ --ink:#1f2933; --muted:#667085; --line:#d8dee8; --panel:#ffffff; --bg:#f3f5f8; --blue:#276ef1; --green:#14804a; --orange:#f97316; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:"Microsoft YaHei", "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }}
header {{ background:linear-gradient(135deg,#ffffff 0%,#edf4ff 100%); border-bottom:1px solid var(--line); padding:18px 28px 14px; }}
h1 {{ margin:0; font-size:26px; font-weight:800; letter-spacing:.2px; }}
.sub {{ margin-top:8px; color:var(--muted); font-size:14px; line-height:1.55; max-width:1400px; }}
.wrap {{ padding:18px 28px 28px; max-width:1780px; margin:0 auto; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:16px; margin-bottom:16px; box-shadow:0 10px 28px rgba(20,31,48,.08); }}
.toolbar {{ display:flex; gap:14px; align-items:center; flex-wrap:wrap; }}
button {{ background:var(--ink); color:white; border:0; border-radius:10px; padding:10px 18px; cursor:pointer; font-weight:700; }}
button:hover {{ background:#0f172a; }}
input[type=range] {{ width:min(860px,72vw); accent-color:var(--blue); }}
.metrics {{ display:grid; grid-template-columns:repeat(5,minmax(150px,1fr)); gap:12px; }}
.card {{ border:1px solid #e1e7ef; border-radius:14px; background:#fbfcff; padding:14px; }}
.card b {{ display:block; font-size:13px; color:#667085; }}
.card span {{ display:block; margin-top:8px; font-size:24px; font-weight:800; }}
.stage {{ display:grid; grid-template-columns:minmax(0,1fr) 360px; gap:16px; align-items:start; }}
.frameBox {{ background:white; border:1px solid #dbe3ee; border-radius:16px; overflow:hidden; min-height:620px; display:flex; align-items:center; justify-content:center; }}
.frameBox img {{ width:100%; height:auto; display:block; }}
.side h3 {{ margin:0 0 10px; font-size:18px; }}
.flow {{ display:grid; gap:9px; margin:12px 0 16px; }}
.flow div {{ padding:10px 12px; border-left:4px solid var(--blue); background:#f7faff; border-radius:10px; font-size:14px; }}
.legend {{ display:grid; gap:8px; color:#475467; font-size:14px; }}
.dot {{ display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:6px; vertical-align:-1px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ padding:8px 10px; border-bottom:1px solid #e6ebf2; text-align:right; white-space:nowrap; }}
th:first-child,td:first-child {{ text-align:left; }}
@media (max-width:1100px) {{ .stage {{ grid-template-columns:1fr; }} .metrics {{ grid-template-columns:repeat(2,1fr); }} }}
</style>
</head>
<body>
<header>
  <h1>PKN4风格：DAS-only 多簇裂缝 PKN 正演 - EnKF 参数反演 - 再正演闭环</h1>
  <div class="sub">展示重点不是直接用光纤等效缝长替代模型结果，而是用观测残差反演更新 PKN 的每簇有效增长/进液参数。流程为：PKN 参数先验进入正演模型，得到每簇缝长先验；DAS 等效观测到达后计算残差；EnKF 更新 PKN 参数；再用更新后的参数重新调用 PKN 正演得到裂缝几何。当前只使用 DAS/振幅与簇级进液进砂，不使用 DTS；缝高为 PKN 固定输入，开度/面积/体积为正演派生量。</div>
</header>
<div class="wrap">
  <section class="panel toolbar">
    <button id="play">播放闭环过程</button>
    <strong id="label">-</strong>
    <input id="slider" type="range" min="0" max="0" value="0">
  </section>

  <section class="panel metrics">
    <div class="card"><b>PKN先验缝长误差</b><span id="priorErr">-</span></div>
    <div class="card"><b>参数更新后PKN误差</b><span id="postErr">-</span></div>
    <div class="card"><b>Kalman Gain均值</b><span id="gain">-</span></div>
    <div class="card"><b>15%误差目标</b><span id="within">-</span></div>
    <div class="card"><b>PKN派生最大开度</b><span id="aperture">-</span></div>
  </section>

  <section class="stage">
    <div class="frameBox"><img id="frame" alt="pkn4-style frame"></div>
    <aside class="panel side">
      <h3>当前时间步闭环逻辑</h3>
      <div class="flow">
        <div>1. 上一时刻后验 PKN 参数进入当前时刻正演模型</div>
        <div>2. PKN 根据参数推测当前每簇缝长先验</div>
        <div>3. DAS 等效观测缝长到达</div>
        <div>4. 计算残差并由 EnKF 更新每簇 PKN 有效增长参数</div>
        <div>5. 使用更新后参数重新运行 PKN 正演，输出校正后的裂缝几何</div>
      </div>
      <div class="legend">
        <div><span class="dot" style="background:#111827"></span>黑色虚线：DAS等效观测缝长</div>
        <div><span class="dot" style="background:#f97316"></span>橙色虚线：PKN正演先验缝长</div>
        <div><span class="dot" style="background:#22c55e"></span>彩色裂缝面：EnKF更新参数后PKN再正演结果</div>
      </div>
    </aside>
  </section>

  <section class="panel">
    <table>
      <thead><tr><th>簇</th><th>X位置</th><th>先验参数</th><th>后验参数</th><th>参数更新</th><th>DAS观测缝长</th><th>PKN先验缝长</th><th>参数更新后PKN缝长</th><th>派生开度</th><th>面积</th><th>体积</th></tr></thead>
      <tbody id="tbody"></tbody>
    </table>
  </section>
</div>
<script>
const data={data};
const slider=document.getElementById('slider');
const frame=document.getElementById('frame');
let timer=null;
slider.max=data.time_steps.length-1;
function pct(v){{return (v*100).toFixed(2)+'%'}}
function n(v,d=2){{return Number(v).toFixed(d)}}
function render(i){{
  const t=data.time_steps[i];
  const rec=data.records.filter(r=>r.time_s===t);
  const h=data.history[i];
  frame.src=data.frames[i];
  document.getElementById('label').textContent=`时间步 t=${{t}}s / ${{data.time_steps[data.time_steps.length-1]}}s`;
  document.getElementById('priorErr').textContent=pct(h.prior_error);
  document.getElementById('postErr').textContent=pct(h.posterior_error);
  document.getElementById('gain').textContent=n(h.mean_kalman_gain_abs,4);
  document.getElementById('within').textContent=h.within_15_percent?'达标':'未达标';
  document.getElementById('aperture').textContent=n(h.max_aperture_mm,2)+' mm';
  document.getElementById('tbody').innerHTML=rec.map(r=>`<tr><td>C${{r.cluster_id}}</td><td>${{n(r.x_center_m,1)}}</td><td>${{n(r.prior_growth_factor,3)}}</td><td>${{n(r.posterior_growth_factor,3)}}</td><td>${{n(r.parameter_update_delta,4)}}</td><td>${{n(r.observed_half_length_m,1)}}</td><td>${{n(r.prior_half_length_m,1)}}</td><td>${{n(r.half_length_m,1)}}</td><td>${{n(r.max_aperture_mm,3)}}</td><td>${{n(r.area_m2,1)}}</td><td>${{n(r.volume_m3,2)}}</td></tr>`).join('');
}}
slider.addEventListener('input',()=>render(Number(slider.value)));
document.getElementById('play').onclick=()=>{{
  const btn=document.getElementById('play');
  if(timer){{clearInterval(timer);timer=null;btn.textContent='播放闭环过程';return;}}
  btn.textContent='暂停';
  timer=setInterval(()=>{{let v=Number(slider.value)+1;if(v>Number(slider.max))v=0;slider.value=v;render(v);}},1200);
}};
render(0);
</script>
</body></html>"""
    path.write_text(html, encoding="utf-8")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Client pkn4-style multi-cluster PKN + EnKF playback demo.")
    parser.add_argument(
        "--forward-model",
        choices=["pkn4", "bem_stub", "bem_reduced", "physics_hybrid", "data_surrogate"],
        default="pkn4",
        help="Forward model behind the EnKF loop.",
    )
    parser.add_argument(
        "--das-scenario",
        choices=["uniform", "left_dominant", "right_dominant", "middle_dominant", "alternating"],
        default="uniform",
        help="DAS response pattern used for differentiated simulation before real DAS is connected.",
    )
    parser.add_argument("--young-modulus-pa", type=float, default=3.0e10)
    parser.add_argument("--poisson-ratio", type=float, default=0.25)
    parser.add_argument("--height-m", type=float, default=30.0)
    parser.add_argument("--viscosity-pa-s", type=float, default=0.1)
    parser.add_argument("--total-flow-rate-m3-s", type=float, default=0.20)
    parser.add_argument("--time-steps", type=float, nargs="+", default=[300, 600, 900, 1200, 1500, 1800])
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--well-len-m", type=float, default=120.0)
    parser.add_argument("--cluster-start", type=float, default=0.15)
    parser.add_argument("--cluster-end", type=float, default=0.85)
    parser.add_argument("--ensemble-size", type=int, default=80)
    parser.add_argument("--length-noise-m", type=float, default=18.0)
    parser.add_argument("--aperture-noise-mm", type=float, default=0.035, help="Legacy argument kept for command compatibility; aperture is derived, not assimilated.")
    parser.add_argument("--visual-aperture-gain", type=float, default=450.0, help="Only controls 3D visual thickness. Smaller values make fractures visually narrower.")
    parser.add_argument("--fiber-json", default=None, help="Optional client-style fiber JSON file. If set, stageInfo.cluster controls PKN cluster rates.")
    parser.add_argument("--frac-monitor-text", default=None, help="Optional real FracMonitor text export. It overrides --fiber-json controls when provided.")
    parser.add_argument("--construction-pressure-xls", default=None, help="Optional stage construction pressure XLS. Current stage-08 layout is parsed by fixed column indexes.")
    parser.add_argument("--use-real-monitor-timeline", action="store_true", help="Sample playback time steps from the real FracMonitor timeline.")
    parser.add_argument("--max-playback-steps", type=int, default=16, help="Maximum sampled time steps for real FracMonitor playback.")
    parser.add_argument("--well-trajectory-csv", default=None, help="Optional four-column well trajectory CSV for 3D spatial display.")
    parser.add_argument("--stage-md-start-m", type=float, default=None, help="Optional measured-depth start of the displayed stage interval.")
    parser.add_argument("--stage-md-end-m", type=float, default=None, help="Optional measured-depth end of the displayed stage interval.")
    parser.add_argument("--pressure-measured-depth-m", type=float, default=None, help="Measured depth used by bottom-hole pressure conversion. Defaults to trajectory stage end if available.")
    parser.add_argument("--pressure-vertical-depth-m", type=float, default=None, help="Vertical depth used by hydrostatic pressure conversion. Defaults to trajectory stage end if available.")
    parser.add_argument("--pressure-fcd", type=float, default=1.0e-10, help="Empirical tubing friction coefficient. Placeholder default; calibrate with client data.")
    parser.add_argument("--pressure-dwell-m", type=float, default=0.10, help="Tubing inner diameter in meters.")
    parser.add_argument("--pressure-proppant-density-kg-m3", type=float, default=2650.0)
    parser.add_argument("--pressure-base-fluid-density-kg-m3", type=float, default=1000.0)
    parser.add_argument("--pressure-jzl", type=float, default=0.5, help="Tubing flow-rate correction coefficient.")
    parser.add_argument("--pressure-perforation-count", type=int, default=48)
    parser.add_argument("--pressure-perforation-diameter-m", type=float, default=0.010)
    parser.add_argument("--pressure-perforation-erosion-coeff", type=float, default=0.0)
    parser.add_argument("--pressure-perforation-flow-coeff", type=float, default=0.85)
    parser.add_argument("--pressure-perforation-flow-decay-coeff", type=float, default=0.0)
    parser.add_argument("--pressure-min-horizontal-stress-mpa", type=float, default=60.0)
    parser.add_argument("--pressure-rolling-window-seconds", type=int, default=30)
    parser.add_argument("--pressure-step-seconds", type=float, default=1.0)
    parser.add_argument(
        "--assimilate-net-pressure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If construction pressure is provided, append net pressure to the EnKF observation vector.",
    )
    parser.add_argument(
        "--net-pressure-noise-mpa",
        type=float,
        default=10.0,
        help="Observation noise for net-pressure assimilation. Larger values make pressure correction weaker.",
    )
    parser.add_argument(
        "--net-pressure-proxy-scale",
        type=float,
        default=30.0,
        help="Scale for the lightweight PKN aperture-to-net-pressure proxy used inside the EnKF observation operator.",
    )
    parser.add_argument("--obs-cum-liquid-weight", type=float, default=0.70, help="Observation operator weight for cumulative liquid volume.")
    parser.add_argument("--obs-cum-sand-weight", type=float, default=0.30, help="Observation operator weight for cumulative sand mass.")
    parser.add_argument("--obs-instant-liquid-weight", type=float, default=0.0, help="Optional observation weight for instantaneous liquid volume. Defaults to 0 to avoid nonphysical length shrinkage.")
    parser.add_argument("--obs-instant-sand-weight", type=float, default=0.0, help="Optional observation weight for instantaneous sand mass. Defaults to 0 to avoid nonphysical length shrinkage.")
    parser.add_argument("--obs-balance-gain", type=float, default=0.20, help="How strongly low balance degree amplifies cluster differences.")
    parser.add_argument("--obs-min-factor", type=float, default=0.85, help="Lower bound of fiber observation length factor.")
    parser.add_argument("--obs-max-factor", type=float, default=1.15, help="Upper bound of fiber observation length factor.")
    parser.add_argument("--obs-factor-smoothing", type=float, default=0.65, help="Exponential smoothing for fiber observation factor. 0 disables smoothing; larger values retain more previous state.")
    parser.add_argument("--enforce-non-decreasing-length", action=argparse.BooleanOptionalAction, default=True, help="Enforce non-decreasing prior/observed/posterior half-length by cluster.")
    parser.add_argument("--default-step-seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run-dir", default=str(ROOT.parent / "outputs" / "dt" / "playback"))
    parser.add_argument("--open", action="store_true", help="Open the generated playback HTML in the default browser.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_root = Path(args.run_dir).resolve() / time.strftime("%Y%m%d_%H%M%S")
    figures = run_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    result = run_demo(args)
    table = result["table"]
    history = result["history"]
    table.to_csv(run_root / "summary_all_time_steps.csv", index=False, encoding="utf-8-sig")
    history.to_csv(run_root / "enkf_history.csv", index=False, encoding="utf-8-sig")
    montage = figures / "pkn4_style_montage.png"
    playback = run_root / "pkn4_style_enkf_playback.html"
    plot_montage(table, history, montage)
    frame_paths = render_pkn4_style_frames(table, history, args, figures)
    write_playback_html(table, history, args, playback, frame_paths)
    summary = {
        "demo": "pkn4_style_pkn_das_only_parameter_inversion_playback",
        "run_root": str(run_root),
        "target": "Client pkn4-style horizontal-well multi-cluster PKN propagation with DAS-only EnKF PKN-parameter inversion",
        "model_scope": {
            "forward_model": args.forward_model,
            "forward_model_interface": "LengthForwardModel.simulate_lengths(...)",
            "das_scenario": args.das_scenario,
            "real_frac_monitor_text": args.frac_monitor_text,
            "real_well_trajectory_csv": args.well_trajectory_csv,
            "fiber_observation_used": "DAS amplitude / amp response and stageInfo cluster liquid-sand controls",
            "fiber_observation_not_used": "DTS temperature is not used in the current model",
            "inversion_target": "Per-cluster effective PKN growth/intake parameter. Fracture half-length is produced by re-running PKN after parameter update.",
            "not_inversion_targets": ["direct fracture half-length overwrite", "fracture height", "aperture/width"],
            "fixed_pkn_height_m": args.height_m,
            "derived_outputs": ["PKN aperture", "area", "volume"],
        },
        "parameter_inversion": {
            "state_vector": "ensemble_size x n_clusters effective_growth_factor",
            "forward_operator": "LengthForwardModel.simulate_lengths(factor_state) -> half_length_m; PKN aperture proxy -> net_pressure_mpa when pressure is available",
            "observation": "DAS/fiber-derived equivalent half_length_m plus optional construction-pressure-derived net_pressure_mpa",
            "update_logic": "EnKF updates effective_growth_factor, then PKN is re-run with posterior factors.",
            "csv_columns": [
                "prior_growth_factor",
                "posterior_growth_factor",
                "parameter_update_delta",
                "prior_half_length_m",
                "observed_half_length_m",
                "half_length_m",
            ],
        },
        "net_pressure_assimilation": {
            "enabled": bool(args.assimilate_net_pressure and args.construction_pressure_xls),
            "construction_pressure_xls": args.construction_pressure_xls,
            "observation": "net_pressure_mpa = bottomhole_pressure_mpa - min_horizontal_stress_mpa",
            "bottomhole_formula": "surface_pressure + hydrostatic_pressure - pipe_friction - perforation_friction",
            "proxy_formula": "proxy_net_pressure = scale * E_prime * mean_aperture / fixed_height",
            "proxy_scale": args.net_pressure_proxy_scale,
            "observation_noise_mpa": args.net_pressure_noise_mpa,
            "state_updated": "per_cluster_effective_growth_factor",
            "not_directly_updated": ["net_pressure", "half_length"],
            "csv_columns": [
                "observed_net_pressure_mpa",
                "prior_proxy_net_pressure_mpa",
                "posterior_proxy_net_pressure_mpa",
                "prior_net_pressure_residual_mpa",
                "posterior_net_pressure_residual_mpa",
                "net_pressure_assimilated",
            ],
        },
        "fiber_observation_operator": {
            "formula": "observed_half_length = prior_half_length * clipped(weighted_liquid_sand_activity * balance_adjustment)",
            "inputs": [
                "cumulative_liquid_volume_m3",
                "cumulative_sand_mass_t",
                "liquid_volume_m3",
                "sand_mass_t",
                "balance_degree",
                "cumulative_balance_degree",
            ],
            "weights": {
                "cumulative_liquid": args.obs_cum_liquid_weight,
                "cumulative_sand": args.obs_cum_sand_weight,
                "instant_liquid": args.obs_instant_liquid_weight,
                "instant_sand": args.obs_instant_sand_weight,
                "balance_gain": args.obs_balance_gain,
            },
            "factor_bounds": [args.obs_min_factor, args.obs_max_factor],
            "csv_columns": [
                "cum_liquid_score",
                "cum_sand_score",
                "instant_liquid_score",
                "instant_sand_score",
                "fiber_activity_score",
                "fiber_balance_imbalance",
                "fiber_observation_factor",
                "fiber_observation_factor_effective",
            ],
        },
        "pkn4_alignment": {
            "horizontal_well": True,
            "multi_cluster": True,
            "time_steps": args.time_steps,
            "cluster_interference_factor": True,
            "outputs_half_length_aperture_area_volume": True,
            "parameter_update_then_forward": True,
            "pyvista_mesh_not_required": "This version uses browser/SVG playback for portability; client pkn4.py can still provide PyVista rendering.",
            "fiber_json_supported": True,
            "frac_monitor_text_supported": True,
            "well_trajectory_csv_supported": True,
        },
        "real_data_meta": {
            "fiber_meta": result.get("fiber_meta", {}),
            "trajectory_cluster_positions": result.get("trajectory_positions", pd.DataFrame()).to_dict(orient="records"),
            "pressure_meta": result.get("pressure_meta", {}),
        },
        "metrics": result["metrics"],
        "outputs": {
            "summary_all_time_steps": str(run_root / "summary_all_time_steps.csv"),
            "enkf_history": str(run_root / "enkf_history.csv"),
            "playback_html": str(playback),
        },
        "figures": {"pkn4_style_montage": str(montage)},
        "pkn4_style_frames": [str(run_root / frame_path) for frame_path in frame_paths],
        "args": vars(args),
    }
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_root / "metrics.json").write_text(json.dumps(result["metrics"], ensure_ascii=False, indent=2), encoding="utf-8")
    if args.open:
        webbrowser.open(playback.resolve().as_uri())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
