"""Continuous native-PyFrac forward model with periodic EnKF inversion.

This experiment is deliberately separate from the production PKN+EnKF
validation entry point.  It answers a narrower but important question:

    Can a retained native PyFrac fracture state be advanced through 4,000+
    seconds while an EnKF repeatedly updates the physical parameters used by
    the next PyFrac window?

The current implementation is a single planar fracture / single-cluster
experiment driven by the pressure schedule.  It is not a native six-cluster
horizontal-well solver.  A small ensemble is used because every member is a
real native PyFrac forward run; this is intentionally an offline feasibility
and inversion experiment, not an online 15-second path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DT_ROOT.parent
if str(DT_ROOT) not in sys.path:
    sys.path.insert(0, str(DT_ROOT))

from data_fusion.pressure_schedule_adapter import PressureModelConfig, load_stage_pressure_schedule  # noqa: E402
from forward_models.pyfrac_adapter import PyFracAdapter, PyFracNativeSession  # noqa: E402
from forward_models.pyfrac_config import PyFracConfig  # noqa: E402
from forward_models.pyfrac_robustness import relaxed_update  # noqa: E402
from inversion.physics import PhysicalEnKFConfig, clip_state, denkf_update, physical_values, pkn_with_carter_leakoff, state_record  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction-pressure-xls", default="Data/3Dfrac/JY84-Z1-stage08-f1.xls")
    parser.add_argument("--pressure-calibration-config", default=None)
    parser.add_argument("--output-dir", default="outputs/dt/pyfrac_enkf_full_flow")
    parser.add_argument("--target-time-s", type=float, default=4435.0)
    parser.add_argument("--warm-start-time-s", type=float, default=360.0)
    parser.add_argument("--assimilation-interval-s", type=float, default=600.0)
    parser.add_argument("--injection-schedule-step-s", type=float, default=30.0)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--ensemble-pressure-noise-mpa", type=float, default=3.5)
    parser.add_argument("--base-eprime-pa", type=float, default=3.2e10)
    parser.add_argument("--base-leakoff-m-sqrt-s", type=float, default=1.0e-5)
    parser.add_argument("--base-viscosity-pa-s", type=float, default=0.1)
    parser.add_argument("--base-min-stress-mpa", type=float, default=60.0)
    parser.add_argument("--height-m", type=float, default=30.0)
    parser.add_argument("--mesh-nx", type=int, default=81)
    parser.add_argument("--mesh-ny", type=int, default=61)
    parser.add_argument(
        "--mesh-domain-time-s",
        type=float,
        default=None,
        help="time scale used to size the retained native mesh; defaults to target time",
    )
    parser.add_argument("--max-time-steps-per-window", type=int, default=80)
    parser.add_argument("--dynamic-step-limit-s", type=float, default=30.0)
    parser.add_argument("--max-native-retries", type=int, default=2)
    parser.add_argument("--retry-time-step-factor", type=float, default=0.5)
    parser.add_argument("--min-dynamic-step-s", type=float, default=0.5)
    parser.add_argument("--assimilation-relaxation", type=float, default=0.35)
    parser.add_argument(
        "--assimilation-max-parameter-step",
        type=float,
        nargs=5,
        default=[0.08, 0.20, 0.12, 3.0, 0.12],
        metavar=("D_LOGE", "D_LOGCL", "D_LOGMU", "D_STRESS", "D_LOGK"),
    )
    parser.add_argument("--disable-adaptive-mesh", action="store_true")
    parser.add_argument("--disable-pyfrac-remeshing", action="store_true")
    parser.add_argument("--seed", type=int, default=20260819)
    return parser


def derive_injection_rate(pressure: pd.DataFrame) -> np.ndarray:
    """Build a non-negative m3/s rate from the construction boundary.

    The measured pump-rate channel is the primary boundary.  Some early rows
    in this export have a zero pump-rate value while cumulative liquid still
    increases, so the cumulative-liquid finite difference is used only for
    those rows.  Using the cumulative derivative for every row overestimates
    the late-stage rate because the source cumulative values are rounded.
    """

    cumulative = pd.to_numeric(pressure["cumulative_liquid_m3"], errors="coerce").ffill().fillna(0.0)
    times = pd.to_numeric(pressure["time_s"], errors="coerce").to_numpy(dtype=float)
    values = cumulative.to_numpy(dtype=float)
    dt = np.diff(times, prepend=times[0])
    dt[0] = dt[1] if len(dt) > 1 and dt[1] > 0 else 1.0
    cumulative_rate = np.maximum(np.diff(values, prepend=values[0]) / np.maximum(dt, 1.0e-9), 0.0)
    measured_series = pd.to_numeric(pressure["flow_rate_m3_s"], errors="coerce")
    measured_rate = measured_series.to_numpy(dtype=float)

    # The measured pump channel is authoritative once it has produced its
    # first valid positive value.  In this export the cumulative-liquid
    # column continues to increase during several real shut-in intervals;
    # using its finite difference for every zero pump-rate row would turn a
    # shut-in into artificial injection and destabilise native PyFrac.
    rate = np.where(np.isfinite(measured_rate), np.maximum(measured_rate, 0.0), cumulative_rate)
    positive = np.flatnonzero(rate > 1.0e-9)
    if positive.size:
        first_positive = int(positive[0])
        if first_positive > 0:
            # A short leading section of this file has no valid pump-rate
            # samples.  Preserve the historical cumulative-difference
            # fallback only for that leading section.
            rate[:first_positive] = cumulative_rate[:first_positive]
        # Missing values after the first measured injection are filled from
        # the nearest measured value, while explicit measured zeros remain
        # zero and therefore preserve shut-in boundaries.
        missing = ~np.isfinite(measured_rate)
        rate[missing & (np.arange(len(rate)) >= first_positive)] = cumulative_rate[
            missing & (np.arange(len(rate)) >= first_positive)
        ]
    else:
        # If a source has no valid measured pump channel at all, use the
        # cumulative-volume derivative as the documented fallback.
        rate = np.maximum(cumulative_rate, 0.0)
    if len(rate) > 1 and rate[0] <= 1.0e-9:
        rate[0] = rate[1]
    return np.maximum(rate, 0.0)


def make_schedule(
    pressure: pd.DataFrame,
    rate: np.ndarray,
    target_time_s: float,
    step_s: float,
    rate_change_threshold_m3_s: float = 0.02,
) -> np.ndarray:
    """Build a compact native schedule without turning sensor jitter into 1 s steps.

    The source pressure table is still retained at its native resolution for
    observations.  PyFrac only needs rate-regime transitions and material
    rate changes.  Keeping every tiny measured fluctuation as a ``timeToHit``
    event forces the legacy controller to take one-second steps across the
    whole run, defeating the long-run performance target.  Zero/non-zero
    transitions are always retained; non-zero changes are retained only when
    they exceed ``rate_change_threshold_m3_s``.
    """
    source_times = pressure["time_s"].to_numpy(dtype=float)
    target = float(target_time_s)
    step = max(float(step_s), 1.0)
    count = max(int(np.ceil(target / step)), 1)
    regular_times = np.linspace(0.0, target, count + 1)

    # Keep the compact schedule for ordinary segments.  Always retain
    # positive/zero regime transitions so a 30 s schedule cannot interpolate
    # across a one-second shut-in boundary and feed the wrong rate to PyFrac.
    # Small non-zero changes are sensor/rounding jitter for the native model;
    # retaining them would create hundreds of unnecessary one-second events.
    source_rate = np.asarray(rate, dtype=float)
    change_mask = np.zeros(len(source_times), dtype=bool)
    if len(source_times) > 1:
        finite_rate = np.where(np.isfinite(source_rate), source_rate, 0.0)
        rate_delta = np.abs(np.diff(finite_rate))
        regime_change = (finite_rate[1:] > 1.0e-12) != (finite_rate[:-1] > 1.0e-12)
        change_mask[1:] = regime_change | (rate_delta >= max(float(rate_change_threshold_m3_s), 0.0))
    event_times = source_times[change_mask & (source_times > 0.0) & (source_times < target)]
    times = np.unique(np.concatenate([regular_times, event_times, np.asarray([target])]))
    q = np.interp(times, source_times, source_rate)
    if len(q) > 1 and q[0] <= 1.0e-9:
        q[0] = q[1]
    # Zero is a valid rate: it represents a true shut-in.  The first point is
    # kept positive above because the warm-start constructor needs an initial
    # injection rate to define the analytical state.
    return np.vstack([times, np.maximum(q, 0.0)])


def observation_at(pressure: pd.DataFrame, target_time_s: float, column: str) -> float:
    return float(
        np.interp(
            float(target_time_s),
            pressure["time_s"].to_numpy(dtype=float),
            pressure[column].to_numpy(dtype=float),
        )
    )


def finite_or_none(value: object) -> float | None:
    """Convert NumPy NaN/Inf to JSON null instead of emitting invalid JSON."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def write_progress(
    output: Path,
    history: list[dict[str, object]],
    member_audit: list[dict[str, object]],
    *,
    status: str,
    current_window: int,
    current_time_s: float,
    targets: list[float],
) -> None:
    """Persist every completed member/window so long native runs are auditable."""

    pd.DataFrame(history).to_csv(output / "pyfrac_enkf_history_progress.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(member_audit).to_csv(output / "pyfrac_enkf_member_audit_progress.csv", index=False, encoding="utf-8-sig")
    progress = {
        "status": status,
        "current_window": int(current_window),
        "current_time_s": float(current_time_s),
        "assimilation_times_s": [float(value) for value in targets],
        "completed_windows": int(len(history)),
        "completed_member_windows": int(len(member_audit)),
    }
    (output / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def state_parameters(state: np.ndarray, cfg: PhysicalEnKFConfig) -> dict[str, float]:
    values = physical_values(state, cfg, 1)
    return {
        "height_m": float(cfg.height_m),
        "viscosity_pa_s": float(values["viscosity_pa_s"]),
        "e_prime_pa": float(values["eprime_pa"]),
        "leakoff_coefficient_m_sqrt_s": float(values["leakoff_m_sqrt_s"]),
        "min_horizontal_stress_pa": float(values["min_horizontal_stress_mpa"]) * 1.0e6,
        "fracture_toughness_pa_sqrt_m": float(values["fracture_toughness_pa_sqrt_m"]),
    }


def pkn_pressure(state: np.ndarray, cfg: PhysicalEnKFConfig, q_history: np.ndarray, target_time_s: float) -> dict:
    q_total = float(np.trapz(q_history[1], q_history[0]) / max(float(target_time_s), 1.0))
    q_current = float(q_history[1, -1])
    return pkn_with_carter_leakoff(
        state,
        np.asarray([max(q_total, 1.0e-9)]),
        float(target_time_s),
        cfg,
        q_current_m3_s=np.asarray([max(q_current, 1.0e-9)]),
        cluster_allocation=np.asarray([1.0]),
        cluster_current_allocation=np.asarray([1.0]),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pressure, pressure_meta = load_stage_pressure_schedule(
        args.construction_pressure_xls,
        config=PressureModelConfig(step_seconds=1.0),
    )
    max_source_time = float(pressure["time_s"].max())
    target = min(float(args.target_time_s), max_source_time)
    warm = max(float(args.warm_start_time_s), 1.0)
    if target <= warm:
        raise ValueError("target-time-s must be later than warm-start-time-s")

    rate = derive_injection_rate(pressure)
    full_schedule = make_schedule(pressure, rate, target, args.injection_schedule_step_s)
    cfg = PhysicalEnKFConfig(
        base_eprime_pa=args.base_eprime_pa,
        base_leakoff_m_sqrt_s=args.base_leakoff_m_sqrt_s,
        base_viscosity_pa_s=args.base_viscosity_pa_s,
        base_min_stress_mpa=args.base_min_stress_mpa,
        height_m=args.height_m,
        pressure_proxy_scale=30.0,
        max_leakoff_fraction=0.85,
        hydraulic_coupling_mode="coupled",
    )
    pyfrac_config = PyFracConfig(
        height_m=args.height_m,
        viscosity_pa_s=args.base_viscosity_pa_s,
        min_horizontal_stress_pa=args.base_min_stress_mpa * 1.0e6,
        mesh_nx=args.mesh_nx,
        mesh_ny=args.mesh_ny,
        native_start_time_s=warm,
        max_time_steps=args.max_time_steps_per_window,
        dynamic_step_limit_s=args.dynamic_step_limit_s,
        adaptive_mesh_enabled=not args.disable_adaptive_mesh,
        enable_pyfrac_remeshing=not args.disable_pyfrac_remeshing,
        max_retries=args.max_native_retries,
        retry_time_step_factor=args.retry_time_step_factor,
        min_dynamic_step_s=args.min_dynamic_step_s,
        assimilation_relaxation=args.assimilation_relaxation,
        assimilation_max_parameter_step=tuple(args.assimilation_max_parameter_step),
    )
    adapter = PyFracAdapter(pyfrac_config, project_root=PROJECT_ROOT)
    rng = np.random.default_rng(args.seed)
    prior_mean = np.asarray([0.0, 0.0, 0.0, args.base_min_stress_mpa, 0.0], dtype=float)
    # Keep the initial physical ensemble inside a numerically plausible
    # neighbourhood.  A very wide viscosity/leakoff perturbation can make the
    # legacy front-reconstruction solver non-convergent before the first
    # observation; uncertainty is still expanded by the covariance inflation
    # and updated after each window.
    spread = np.asarray([0.06, 0.12, 0.08, 1.0, 0.08], dtype=float)
    ensemble = clip_state(rng.normal(prior_mean, spread, size=(max(int(args.ensemble_size), 3), 5)), 1)

    sessions: list[PyFracNativeSession] = []
    mesh_domain_time = float(args.mesh_domain_time_s or warm)
    for member in ensemble:
        sessions.append(
            PyFracNativeSession(
                adapter,
                full_schedule,
                warm,
                **state_parameters(member, cfg),
                max_time_steps=args.max_time_steps_per_window,
                domain_time_s=mesh_domain_time,
            )
        )

    targets = list(np.arange(warm + args.assimilation_interval_s, target, args.assimilation_interval_s))
    if not targets or targets[-1] < target - 1.0e-6:
        targets.append(target)
    targets = [float(value) for value in targets]

    history: list[dict[str, object]] = []
    member_audit: list[dict[str, object]] = []
    last_target = warm
    for window_index, target_time in enumerate(targets):
        window_started = time.perf_counter()
        schedule = make_schedule(pressure, rate, target_time, args.injection_schedule_step_s)
        observed_bhp = observation_at(pressure, target_time, "bottomhole_pressure_mpa")
        prior_state = ensemble.mean(axis=0)
        prior_pkn = pkn_pressure(prior_state, cfg, schedule, target_time)
        results = []
        for member_index, (member_state, session) in enumerate(zip(ensemble, sessions)):
            result = session.advance_to(target_time, schedule, **state_parameters(member_state, cfg))
            results.append(result)
            member_audit.append(
                {
                    "window_index": window_index,
                    "member_index": member_index,
                    "time_s": target_time,
                    "success": bool(result.success),
                    "target_reached": bool(result.target_reached),
                    "final_time_s": result.final_time_s,
                    "runtime_s": result.runtime_seconds,
                    "successful_time_steps": result.successful_time_steps,
                    "failed_time_steps": result.failed_time_steps,
                    "mass_balance_relative_error": result.mass_balance_relative_error,
                    "mesh_level": result.mesh_level,
                    "mesh_nx": result.mesh_nx,
                    "mesh_ny": result.mesh_ny,
                    "mesh_cells_across_front": result.mesh_cells_across_front,
                    "adaptive_mesh_status": result.adaptive_mesh_status,
                    "checkpoint_id": result.checkpoint_id or "",
                    "rollback_applied": bool(result.rollback_applied),
                    "retry_count": result.retry_count,
                    "time_step_limit_s": result.time_step_limit_s,
                    "error": result.error or "",
                }
            )
            # Native PyFrac can take minutes on a single member.  Persist the
            # audit immediately so an interrupted run never looks like an
            # all-or-nothing black box.
            write_progress(
                output,
                history,
                member_audit,
                status="running",
                current_window=window_index,
                current_time_s=target_time,
                targets=targets,
            )

        valid = np.asarray([bool(item.success and np.isfinite(item.bottomhole_pressure_mpa)) for item in results])
        predicted = np.asarray([item.bottomhole_pressure_mpa for item in results], dtype=float)
        innovation = float(observed_bhp - np.nanmean(predicted[valid])) if valid.any() else float("nan")
        gain_mean = float("nan")
        update_status = "not_updated"
        relaxation_result = None
        if int(valid.sum()) >= 3:
            updated_valid, gain = denkf_update(
                ensemble[valid],
                predicted[valid, None],
                np.asarray([observed_bhp]),
                np.asarray([args.ensemble_pressure_noise_mpa]),
                covariance_inflation=1.01,
            )
            analysis = clip_state(updated_valid, 1)
            relaxation_result = relaxed_update(
                ensemble[valid],
                analysis,
                relaxation=pyfrac_config.assimilation_relaxation,
                max_step=np.asarray(pyfrac_config.assimilation_max_parameter_step, dtype=float),
            )
            ensemble[valid] = clip_state(relaxation_result.state, 1)
            gain_mean = float(np.mean(np.abs(gain)))
            update_status = "updated_relaxed"
        else:
            update_status = "insufficient_valid_native_members"

        posterior_state = ensemble.mean(axis=0)
        posterior_pkn = pkn_pressure(posterior_state, cfg, schedule, target_time)
        successful_results = [item for item in results if item.success]
        pyfrac_mean_bhp = float(np.mean([item.bottomhole_pressure_mpa for item in successful_results])) if successful_results else float("nan")
        pyfrac_mean_length = float(np.mean([item.half_length_m for item in successful_results])) if successful_results else float("nan")
        pyfrac_mean_width = float(np.mean([item.max_aperture_mm for item in successful_results])) if successful_results else float("nan")
        pyfrac_mass_error = float(np.nanmax([item.mass_balance_relative_error for item in successful_results])) if successful_results else float("nan")
        row = {
            "window_index": window_index,
            "time_s": target_time,
            "window_start_s": last_target,
            "observed_bottomhole_pressure_mpa": observed_bhp,
            "pyfrac_prior_mean_bottomhole_pressure_mpa": pyfrac_mean_bhp,
            "pyfrac_prior_pressure_error_mpa": abs(pyfrac_mean_bhp - observed_bhp) if np.isfinite(pyfrac_mean_bhp) else np.nan,
            "pkn_prior_bottomhole_pressure_mpa": float(prior_pkn["bottomhole_pressure_mpa"]),
            "pkn_prior_pressure_error_mpa": abs(float(prior_pkn["bottomhole_pressure_mpa"]) - observed_bhp),
            "pkn_posterior_bottomhole_pressure_mpa": float(posterior_pkn["bottomhole_pressure_mpa"]),
            "pkn_posterior_pressure_error_mpa": abs(float(posterior_pkn["bottomhole_pressure_mpa"]) - observed_bhp),
            "pyfrac_mean_half_length_m": pyfrac_mean_length,
            "pyfrac_mean_max_aperture_mm": pyfrac_mean_width,
            "pyfrac_max_mass_balance_relative_error": pyfrac_mass_error,
            "native_members": int(len(results)),
            "native_success_members": int(valid.sum()),
            "native_failed_members": int((~valid).sum()),
            "enkf_update_status": update_status,
            "enkf_innovation_mpa": innovation,
            "mean_abs_kalman_gain": gain_mean,
            "enkf_relaxation": float(relaxation_result.relaxation) if relaxation_result else np.nan,
            "enkf_raw_update_norm": float(relaxation_result.raw_update_norm) if relaxation_result else np.nan,
            "enkf_applied_update_norm": float(relaxation_result.applied_update_norm) if relaxation_result else np.nan,
            "enkf_step_clipped_components": int(relaxation_result.clipped_components) if relaxation_result else 0,
            "rollback_members": int(sum(bool(item.rollback_applied) for item in results)),
            "retry_count_max": int(max((item.retry_count for item in results), default=0)),
            "mesh_level_max": int(max((item.mesh_level for item in results), default=0)),
            "window_runtime_s": time.perf_counter() - window_started,
            **state_record("prior", prior_state, cfg, 1),
            **state_record("posterior", posterior_state, cfg, 1),
        }
        history.append(row)
        last_target = target_time
        write_progress(
            output,
            history,
            member_audit,
            status="window_completed",
            current_window=window_index,
            current_time_s=target_time,
            targets=targets,
        )

    history_frame = pd.DataFrame(history)
    audit_frame = pd.DataFrame(member_audit)
    history_frame.to_csv(output / "pyfrac_enkf_history.csv", index=False, encoding="utf-8-sig")
    audit_frame.to_csv(output / "pyfrac_enkf_member_audit.csv", index=False, encoding="utf-8-sig")
    write_figure(history_frame, output / "pyfrac_enkf_history.png")
    success_rate = float(audit_frame["success"].mean()) if not audit_frame.empty else 0.0
    successful_audit = audit_frame[audit_frame["success"] == True] if not audit_frame.empty else audit_frame
    max_native_time = (
        finite_or_none(successful_audit["final_time_s"].max()) if not successful_audit.empty else None
    )
    successful_windows = (
        int(history_frame.loc[history_frame["native_success_members"] > 0, "window_index"].nunique())
        if not history_frame.empty
        else 0
    )
    summary = {
        "demo": "continuous_native_pyfrac_with_periodic_enkf_parameter_inversion",
        "status": "offline_experimental",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": time.perf_counter() - started,
        "target_time_s": target,
        "warm_start_time_s": warm,
        "mesh_domain_time_s": mesh_domain_time,
        "assimilation_times_s": targets,
        "continuous_state": True,
        "native_pyfrac_forward": True,
        "native_pyfrac_scope": "single_planar_fracture_single_cluster",
        "ensemble_size": int(len(ensemble)),
        "initial_ensemble_spread": [0.06, 0.12, 0.08, 1.0, 0.08],
        "native_member_success_rate": success_rate,
        "native_max_successful_time_s": max_native_time,
        "windows_with_native_success": successful_windows,
        "enkf_update_count": int(history_frame["enkf_update_status"].isin(["updated", "updated_relaxed"]).sum()) if not history_frame.empty else 0,
        "enkf_relaxed_update_count": int((history_frame["enkf_update_status"] == "updated_relaxed").sum()) if not history_frame.empty else 0,
        "checkpoint_save_count": int(sum(sum(event.get("event") == "saved" for event in session.checkpoints.events) for session in sessions)),
        "rollback_event_count": int(sum(sum(event.get("event") == "restored" for event in session.checkpoints.events) for session in sessions)),
        "rollback_window_count": int(history_frame["rollback_members"].gt(0).sum()) if not history_frame.empty else 0,
        "adaptive_mesh_enabled": bool(pyfrac_config.adaptive_mesh_enabled),
        "pyfrac_remeshing_enabled": bool(pyfrac_config.enable_pyfrac_remeshing),
        "assimilation_relaxation": float(pyfrac_config.assimilation_relaxation),
        "assimilation_max_parameter_step": list(pyfrac_config.assimilation_max_parameter_step),
        "native_total_successful_steps": int(sum(session.total_successful_steps for session in sessions)),
        "native_total_failed_steps": int(sum(session.total_failed_steps for session in sessions)),
        "target_reached": bool(
            not history_frame.empty
            and history_frame["time_s"].max() >= target
            and int(history_frame.iloc[-1]["native_success_members"]) > 0
        ),
        "final_pyfrac_pressure_error_mpa": finite_or_none(history_frame.iloc[-1]["pyfrac_prior_pressure_error_mpa"]) if not history_frame.empty else None,
        "final_pkn_posterior_pressure_error_mpa": finite_or_none(history_frame.iloc[-1]["pkn_posterior_pressure_error_mpa"]) if not history_frame.empty else None,
        "max_native_mass_balance_relative_error": finite_or_none(history_frame["pyfrac_max_mass_balance_relative_error"].max()) if not history_frame.empty else None,
        "physical_state": ["E_prime", "C_L", "mu", "sigma_min", "K_IC"],
        "pressure_observation": "converted bottomhole pressure from construction schedule",
        "injection_boundary": "measured pump rate, with cumulative-liquid derivative only where the measured rate is zero; piecewise-linearized for native windows",
        "pyfrac_config": asdict(pyfrac_config),
        "pressure_meta": pressure_meta,
        "pyfrac_installation": adapter.verify_installation(),
        "outputs": {
            "history": str(output / "pyfrac_enkf_history.csv"),
            "member_audit": str(output / "pyfrac_enkf_member_audit.csv"),
            "figure": str(output / "pyfrac_enkf_history.png"),
            "checkpoint_events": str(output / "pyfrac_checkpoint_events.csv"),
        },
        "limitations": [
            "当前为单簇平面 PyFrac 原生连续状态，不是六簇原生水平井求解。",
            "EnKF 每个成员均调用原生 PyFrac，属于离线实验，不能进入在线 15 秒链路。",
            "注入曲线优先使用施工排量，排量为零但累计液量增加时用差分补齐，再按时间步分段线性化。",
            "网格按暖启动前缘分辨率自适应选择，并允许 PyFrac 原生扩展；若扩展和重试仍不能达到目标时间，则检查点回滚并将成员标记为失败。",
            "参数反演当前只使用井底压力；没有把 DAS 分簇观测直接作为 PyFrac 内部观测量。",
            "模型间压力差异受黏度、滤失、应力和井筒压力换算标定影响，不能直接当作现场误差。",
        ],
    }
    write_progress(
        output,
        history,
        member_audit,
        status="completed",
        current_window=len(targets) - 1,
        current_time_s=target,
        targets=targets,
    )
    checkpoint_events = []
    for session_index, session in enumerate(sessions):
        for event in session.checkpoints.events:
            checkpoint_events.append({"member_index": session_index, **event})
    pd.DataFrame(checkpoint_events).to_csv(output / "pyfrac_checkpoint_events.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_figure(history: pd.DataFrame, path: Path) -> None:
    if history.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    time_min = history["time_s"] / 60.0
    axes[0, 0].plot(time_min, history["observed_bottomhole_pressure_mpa"], "k-o", label="观测井底压力")
    axes[0, 0].plot(time_min, history["pyfrac_prior_mean_bottomhole_pressure_mpa"], "C3-o", label="PyFrac原生先验")
    axes[0, 0].plot(time_min, history["pkn_posterior_bottomhole_pressure_mpa"], "C0-o", label="PKN后验")
    axes[0, 0].set_ylabel("MPa")
    axes[0, 0].set_title("连续原生 PyFrac + EnKF 压力反演")
    axes[0, 0].legend()
    axes[0, 1].plot(time_min, history["pyfrac_prior_pressure_error_mpa"], "C3-o", label="PyFrac先验误差")
    axes[0, 1].plot(time_min, history["pkn_posterior_pressure_error_mpa"], "C0-o", label="PKN后验误差")
    axes[0, 1].set_ylabel("绝对误差 / MPa")
    axes[0, 1].set_title("同化前后压力误差")
    axes[0, 1].legend()
    axes[1, 0].plot(time_min, history["posterior_eprime_gpa"], "C2-o", label="E' / GPa")
    axes[1, 0].plot(time_min, history["posterior_min_stress_mpa"], "C1-o", label="σmin / MPa")
    axes[1, 0].set_ylabel("参数值")
    axes[1, 0].set_title("EnKF 后验参数轨迹")
    axes[1, 0].legend()
    axes[1, 1].plot(time_min, history["pyfrac_mean_half_length_m"], "C4-o", label="PyFrac半缝长 / m")
    axes[1, 1].plot(time_min, history["pyfrac_mean_max_aperture_mm"], "C5-o", label="PyFrac最大开度 / mm")
    axes[1, 1].set_title("连续 PyFrac 状态")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("时间 / min")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
