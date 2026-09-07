"""Checkpointed single-native-PyFrac + multi-fidelity EnKF long experiment.

One native PyFrac state is advanced continuously.  A lightweight parameter
ensemble obtains pressure sensitivity from PKN and is anchored to the native
PyFrac pressure at every assimilation node.  Partial native advances are kept
as durable checkpoints.  This is an offline parameter-inversion experiment,
not a wall-clock real-time replay and not a native-PyFrac ensemble.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Legacy PyFrac opens blocking diagnostic figures when front reconstruction
# fails.  A checkpointed batch run must never wait for a GUI window.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd


DT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DT_ROOT.parent
if str(DT_ROOT) not in sys.path:
    sys.path.insert(0, str(DT_ROOT))

from data_fusion.pressure_schedule_adapter import PressureModelConfig, load_stage_pressure_schedule  # noqa: E402
from forward_models.pyfrac_adapter import (  # noqa: E402
    PyFracAdapter,
    PyFracNativeSession,
    PyFracRunResult,
    target_time_reached,
)
from forward_models.pyfrac_config import PyFracConfig  # noqa: E402
from forward_models.pyfrac_robustness import relaxed_update  # noqa: E402
from inversion.physics import PhysicalEnKFConfig, clip_state, denkf_update, state_record  # noqa: E402
from inversion.run_pyfrac_enkf import (  # noqa: E402
    derive_injection_rate,
    make_schedule,
    observation_at,
    pkn_pressure,
    state_parameters,
)
from inversion.run_pyfrac_enkf_realtime import (  # noqa: E402
    _finite,
    _load_latest_checkpoint,
    _process_memory_snapshot,
    _record_result,
)
from inversion.performance_tiers import (  # noqa: E402
    DEFAULT_THROUGHPUT_TIERS,
    classify_throughput,
    evaluate_throughput_requirement,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction-pressure-xls", default="Data/3Dfrac/JY84-Z1-stage08-f1.xls")
    parser.add_argument("--output-dir", default="outputs/dt/pyfrac_enkf_synchronized4435")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--resume-time-cap-s", type=float, default=None)
    parser.add_argument("--resume-max-step-limit-s", type=float, default=None)
    parser.add_argument("--target-time-s", type=float, default=4435.0)
    parser.add_argument(
        "--warm-start-time-s",
        type=float,
        default=1.0,
        help="compatibility name only; default 1 s means no analytical warm start",
    )
    parser.add_argument("--assimilation-interval-s", type=float, default=600.0)
    parser.add_argument("--injection-schedule-step-s", type=float, default=30.0)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument(
        "--native-member-count",
        type=int,
        choices=[1],
        default=1,
        help="number of continuously advanced native PyFrac states; fixed to one",
    )
    parser.add_argument("--ensemble-pressure-noise-mpa", type=float, default=3.5)
    parser.add_argument("--base-eprime-pa", type=float, default=3.2e10)
    parser.add_argument("--base-leakoff-m-sqrt-s", type=float, default=1.0e-5)
    parser.add_argument("--base-viscosity-pa-s", type=float, default=0.1)
    parser.add_argument("--base-min-stress-mpa", type=float, default=60.0)
    parser.add_argument("--height-m", type=float, default=30.0)
    parser.add_argument("--mesh-nx", type=int, default=61)
    parser.add_argument("--mesh-ny", type=int, default=9)
    parser.add_argument(
        "--mesh-half-length-m",
        type=float,
        default=2.0,
        help="initial horizontal half-domain; keep the t=1 s front resolved by several cells",
    )
    parser.add_argument(
        "--mesh-half-height-m",
        type=float,
        default=90.0,
        help="fixed/initial PyFrac vertical half-domain; enlarge it for long continuation windows",
    )
    parser.add_argument("--mesh-domain-time-s", type=float, default=None)
    parser.add_argument("--max-time-steps-per-attempt", type=int, default=40)
    parser.add_argument("--dynamic-step-limit-s", type=float, default=30.0)
    parser.add_argument("--initial-step-limit-s", type=float, default=1.0)
    parser.add_argument("--max-native-retries", type=int, default=2)
    parser.add_argument("--retry-time-step-factor", type=float, default=0.5)
    parser.add_argument("--min-dynamic-step-s", type=float, default=0.5)
    parser.add_argument("--max-failed-attempts-per-member-node", type=int, default=12)
    parser.add_argument("--backtrack-after-consecutive-failures", type=int, default=3)
    parser.add_argument("--max-backtracks-per-member-node", type=int, default=6)
    parser.add_argument(
        "--assimilation-relaxation",
        type=float,
        default=1.0,
        help="analysis relaxation; default 1.0 keeps the unbounded-update control",
    )
    parser.add_argument(
        "--native-parameter-transition",
        choices=["time_linear", "instant"],
        default="time_linear",
        help="apply the full EnKF posterior instantly or by solver homotopy over the next native window",
    )
    parser.add_argument(
        "--assimilation-max-parameter-step",
        type=float,
        nargs=5,
        default=[0.0, 0.0, 0.0, 0.0, 0.0],
        metavar=("D_LOGE", "D_LOGCL", "D_LOGMU", "D_STRESS", "D_LOGK"),
        help="optional numerical jump limits; all zeros means no single-update limit",
    )
    parser.add_argument("--checkpoint-interval-s", type=float, default=30.0)
    parser.add_argument("--max-mesh-elements", type=int, default=6000)
    parser.add_argument("--max-process-commit-gb", type=float, default=4.0)
    parser.add_argument(
        "--wall-clock-budget-s",
        type=float,
        default=0.0,
        help="safe-boundary run budget; 0 means no budget limit",
    )
    parser.add_argument("--disable-adaptive-mesh", action="store_true")
    parser.add_argument("--disable-pyfrac-remeshing", action="store_true")
    parser.add_argument(
        "--enable-mesh-extension",
        action="store_true",
        help="allow native PyFrac to extend its domain after a boundary event",
    )
    parser.add_argument(
        "--mesh-extension-all-directions",
        action="store_true",
        help="extend bottom/top/left/right when native PyFrac requests remeshing",
    )
    parser.add_argument("--mesh-extension-factor", type=float, default=1.5)
    parser.add_argument(
        "--front-advancing",
        choices=("predictor-corrector", "implicit", "explicit"),
        default="implicit",
    )
    parser.add_argument(
        "--projection-method",
        choices=("ILSA_orig", "LS_grad", "LS_continousfront"),
        default="ILSA_orig",
    )
    parser.add_argument("--pyfrac-max-solver-iters", type=int, default=140)
    parser.add_argument("--pyfrac-max-front-iters", type=int, default=25)
    parser.add_argument("--pyfrac-max-reattempts", type=int, default=8)
    parser.add_argument("--injection-volume-step-fraction", type=float, default=0.10)
    parser.add_argument("--positive-volume-change-step-fraction", type=float, default=0.12)
    parser.add_argument("--negative-volume-change-step-fraction", type=float, default=0.05)
    parser.add_argument("--time-step-time-fraction", type=float, default=0.15)
    parser.add_argument("--cell-traversal-fraction", type=float, default=3.0)
    parser.add_argument(
        "--ehl-solver",
        choices=("implicit_Anderson", "implicit_Picard", "RKL2"),
        default="implicit_Anderson",
    )
    parser.add_argument("--front-cfl", type=float, default=0.80)
    parser.add_argument("--front-length-fraction", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--throughput-target-ratios",
        type=float,
        nargs="+",
        default=list(DEFAULT_THROUGHPUT_TIERS),
        metavar="RATIO",
        help="native wall-runtime/simulated-time tiers; defaults to 0.10 0.20 0.30",
    )
    parser.add_argument(
        "--required-throughput-ratio",
        type=float,
        default=None,
        help="optional formal runtime requirement; omit to report tiers only",
    )
    return parser


def assimilation_targets(warm: float, target: float, interval: float) -> list[float]:
    values = list(np.arange(warm + interval, target, interval))
    if not values or values[-1] < target - 1.0e-8:
        values.append(target)
    return [float(value) for value in values]


def _read_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    return frame.replace({np.nan: None}).to_dict("records")


def _initial_ensemble(args: argparse.Namespace, resume_root: Path | None) -> np.ndarray:
    if resume_root is not None:
        path = resume_root / "ensemble_state_progress.npy"
        if path.is_file():
            ensemble = np.asarray(np.load(path), dtype=float)
            if ensemble.shape == (max(int(args.ensemble_size), 3), 5):
                return ensemble
            raise ValueError("resume ensemble shape does not match --ensemble-size")
    rng = np.random.default_rng(args.seed)
    mean = np.asarray([0.0, 0.0, 0.0, args.base_min_stress_mpa, 0.0], dtype=float)
    spread = np.asarray([0.06, 0.12, 0.08, 1.0, 0.08], dtype=float)
    return clip_state(rng.normal(mean, spread, size=(max(int(args.ensemble_size), 3), 5)), 1)


def _write_progress(
    output: Path,
    *,
    status: str,
    target_time_s: float,
    current_node_s: float,
    sessions: list[PyFracNativeSession],
    ensemble: np.ndarray,
    history: list[dict[str, object]],
    attempts: list[dict[str, object]],
    memory: list[dict[str, object]],
    started: float,
) -> None:
    pd.DataFrame(history).to_csv(output / "assimilation_history_progress.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(attempts).to_csv(output / "member_attempts_progress.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(memory).to_csv(output / "memory_progress.csv", index=False, encoding="utf-8-sig")
    np.save(output / "ensemble_state_progress.npy", ensemble)
    member_times = [float(session.fracture.time) for session in sessions]
    payload = {
        "status": status,
        "target_time_s": float(target_time_s),
        "current_assimilation_node_s": float(current_node_s),
        "completed_assimilation_nodes": int(len(history)),
        "member_model_times_s": member_times,
        "minimum_member_model_time_s": float(min(member_times)),
        "maximum_member_model_time_s": float(max(member_times)),
        "elapsed_wall_clock_s": float(time.perf_counter() - started),
        "updated_at_epoch_s": time.time(),
    }
    (output / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _memory_exceeded(snapshot: dict[str, float], limit_bytes: float) -> bool:
    value = float(snapshot.get("commit_bytes", float("nan")))
    return limit_bytes > 0.0 and np.isfinite(value) and value > limit_bytes


def _result_at_current_state(session: PyFracNativeSession, target: float, schedule: np.ndarray, parameters: dict[str, float]) -> PyFracRunResult:
    return session.advance_to(target, schedule, **parameters, allow_partial=False)


def anchored_shadow_pressures(
    ensemble: np.ndarray,
    prior_mean: np.ndarray,
    native_bottomhole_pressure_mpa: float,
    physical_cfg: PhysicalEnKFConfig,
    schedule: np.ndarray,
    target_time_s: float,
) -> np.ndarray:
    """Build a cheap EnKF pressure ensemble around one native PyFrac result.

    PKN supplies only the local pressure deltas between parameter samples.  The
    absolute pressure anchor remains the continuously advanced native PyFrac
    state, so this helper does not present the shadow members as native runs.
    """

    pkn_anchor = float(pkn_pressure(prior_mean, physical_cfg, schedule, target_time_s)["bottomhole_pressure_mpa"])
    predictions = []
    for member in np.asarray(ensemble, dtype=float):
        member_pressure = float(pkn_pressure(member, physical_cfg, schedule, target_time_s)["bottomhole_pressure_mpa"])
        predictions.append(float(native_bottomhole_pressure_mpa) + member_pressure - pkn_anchor)
    return np.asarray(predictions, dtype=float)


def interpolate_native_parameters(
    start: dict[str, float],
    target: dict[str, float],
    fraction: float,
) -> dict[str, float]:
    """Interpolate solver parameters without changing the EnKF posterior.

    Positive scale parameters are blended in log space; stress and height are
    blended linearly.  ``fraction=1`` is exactly the requested posterior.
    """

    alpha = float(np.clip(fraction, 0.0, 1.0))
    if alpha >= 1.0:
        return {key: float(value) for key, value in target.items()}
    if alpha <= 0.0:
        return {key: float(start.get(key, value)) for key, value in target.items()}
    log_keys = {
        "viscosity_pa_s",
        "e_prime_pa",
        "leakoff_coefficient_m_sqrt_s",
        "fracture_toughness_pa_sqrt_m",
    }
    result: dict[str, float] = {}
    for key, target_value in target.items():
        start_value = float(start.get(key, target_value))
        target_number = float(target_value)
        if key in log_keys and start_value > 0.0 and target_number > 0.0:
            result[key] = float(np.exp((1.0 - alpha) * np.log(start_value) + alpha * np.log(target_number)))
        else:
            result[key] = float((1.0 - alpha) * start_value + alpha * target_number)
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    resume_root = Path(args.resume_from).resolve() if args.resume_from else None
    if resume_root is not None and not resume_root.is_dir():
        raise FileNotFoundError(f"resume directory does not exist: {resume_root}")

    started = time.perf_counter()
    pressure, pressure_meta = load_stage_pressure_schedule(
        args.construction_pressure_xls,
        config=PressureModelConfig(step_seconds=1.0),
    )
    source_end = min(float(args.target_time_s), float(pressure["time_s"].max()))
    warm = max(float(args.warm_start_time_s), 1.0)
    if source_end <= warm:
        raise ValueError("target-time-s must be later than warm-start-time-s")
    targets = assimilation_targets(warm, source_end, float(args.assimilation_interval_s))
    rate = derive_injection_rate(pressure)

    physical_cfg = PhysicalEnKFConfig(
        base_eprime_pa=args.base_eprime_pa,
        base_leakoff_m_sqrt_s=args.base_leakoff_m_sqrt_s,
        base_viscosity_pa_s=args.base_viscosity_pa_s,
        base_min_stress_mpa=args.base_min_stress_mpa,
        height_m=args.height_m,
        pressure_proxy_scale=30.0,
        max_leakoff_fraction=0.85,
        hydraulic_coupling_mode="coupled",
    )
    pyfrac_cfg = PyFracConfig(
        height_m=args.height_m,
        viscosity_pa_s=args.base_viscosity_pa_s,
        min_horizontal_stress_pa=args.base_min_stress_mpa * 1.0e6,
        mesh_nx=args.mesh_nx,
        mesh_ny=args.mesh_ny,
        mesh_half_height_m=args.mesh_half_height_m,
        mesh_half_length_m=max(float(args.mesh_half_length_m), 1.0),
        native_start_time_s=warm,
        initial_step_limit_s=max(float(args.initial_step_limit_s), 1.0e-6),
        max_time_steps=args.max_time_steps_per_attempt,
        dynamic_step_limit_s=args.dynamic_step_limit_s,
        injection_volume_step_fraction=max(float(args.injection_volume_step_fraction), 1.0e-6),
        positive_volume_change_step_fraction=max(float(args.positive_volume_change_step_fraction), 1.0e-6),
        negative_volume_change_step_fraction=max(float(args.negative_volume_change_step_fraction), 1.0e-6),
        time_step_time_fraction=max(float(args.time_step_time_fraction), 1.0e-6),
        cell_traversal_fraction=max(float(args.cell_traversal_fraction), 1.0e-6),
        elastohydr_solver=str(args.ehl_solver),
        front_cfl=max(float(args.front_cfl), 0.0),
        front_length_fraction=max(float(args.front_length_fraction), 0.0),
        front_advancing=args.front_advancing,
        projection_method=args.projection_method,
        adaptive_mesh_enabled=not args.disable_adaptive_mesh,
        enable_pyfrac_remeshing=not args.disable_pyfrac_remeshing,
        mesh_extension_enabled=args.enable_mesh_extension,
        mesh_extension_all_directions=args.mesh_extension_all_directions,
        mesh_extension_factor=args.mesh_extension_factor,
        max_solver_iterations=args.pyfrac_max_solver_iters,
        max_front_iterations=args.pyfrac_max_front_iters,
        max_pyfrac_reattempts=args.pyfrac_max_reattempts,
        max_retries=args.max_native_retries,
        retry_time_step_factor=args.retry_time_step_factor,
        min_dynamic_step_s=args.min_dynamic_step_s,
        assimilation_relaxation=args.assimilation_relaxation,
        assimilation_max_parameter_step=tuple(args.assimilation_max_parameter_step),
    )
    adapter = PyFracAdapter(pyfrac_cfg, project_root=PROJECT_ROOT)
    ensemble = _initial_ensemble(args, resume_root)
    warm_schedule = make_schedule(pressure, rate, warm, args.injection_schedule_step_s)
    mesh_domain_time = float(args.mesh_domain_time_s or min(source_end, warm + 600.0))

    sessions: list[PyFracNativeSession] = []
    native_initial_state = ensemble.mean(axis=0)
    for member_index in range(int(args.native_member_count)):
        payload = None
        if resume_root is not None:
            payload = _load_latest_checkpoint(
                resume_root / "checkpoints" / f"member_{member_index:03d}",
                max_mesh_elements=args.max_mesh_elements,
                max_model_time_s=args.resume_time_cap_s,
            )
        resumed_step_limit = float(payload.get("continuation_step_limit_s", args.dynamic_step_limit_s)) if payload else None
        if resumed_step_limit is not None and args.resume_max_step_limit_s is not None:
            resumed_step_limit = min(resumed_step_limit, float(args.resume_max_step_limit_s))
        sessions.append(
            PyFracNativeSession(
                adapter,
                warm_schedule,
                warm,
                **state_parameters(native_initial_state, physical_cfg),
                max_time_steps=args.max_time_steps_per_attempt,
                domain_time_s=mesh_domain_time,
                checkpoint_dir=checkpoint_root / f"member_{member_index:03d}",
                checkpoint_interval_s=args.checkpoint_interval_s,
                initial_fracture=payload.get("fracture") if payload else None,
                initial_parameters=payload.get("last_parameters") if payload else None,
                initial_successful_steps=int(payload.get("successful_steps", 0)) if payload else 0,
                initial_failed_steps=int(payload.get("failed_steps", 0)) if payload else 0,
                initial_step_limit_s=resumed_step_limit,
                initial_consecutive_failures=int(payload.get("consecutive_failures", 0)) if payload else 0,
                initial_success_streak=int(payload.get("continuation_success_streak", 0)) if payload else 0,
                initial_front_metadata_repair_count=int(payload.get("front_metadata_repair_count", 0)) if payload else 0,
            )
        )

    history = _read_records(resume_root / "assimilation_history_progress.csv") if resume_root else []
    attempts = _read_records(resume_root / "member_attempts_progress.csv") if resume_root else []
    memory_samples = _read_records(resume_root / "memory_progress.csv") if resume_root else []
    # A historical runner accepted 99.9% of a target as complete.  Never let
    # such a record suppress the missing native advance during resume.
    resume_native_time = min(float(session.fracture.time) for session in sessions)
    history = [
        row for row in history
        if row.get("time_s") is not None and target_time_reached(resume_native_time, float(row["time_s"]))
    ]
    completed_time = max((float(row["time_s"]) for row in history if row.get("enkf_update_status") == "updated"), default=warm)
    pending_targets = [target for target in targets if target > completed_time + 1.0e-8]
    limit_bytes = max(float(args.max_process_commit_gb), 0.0) * 1024.0**3
    wall_budget = max(float(args.wall_clock_budget_s), 0.0)
    status = "completed"
    peak_commit = max((float(row.get("commit_bytes") or 0.0) for row in memory_samples), default=0.0)

    for node_index, target_time in enumerate(pending_targets, start=len(history)):
        schedule = make_schedule(pressure, rate, target_time, args.injection_schedule_step_s)
        prior_state = ensemble.mean(axis=0)
        prior_pkn = pkn_pressure(prior_state, physical_cfg, schedule, target_time)
        node_results: list[PyFracRunResult] = []
        native_parameters = state_parameters(prior_state, physical_cfg)

        for member_index, session in enumerate(sessions):
            node_native_start_time = float(session.fracture.time)
            node_native_start_parameters = dict(session.last_parameters or native_parameters)
            # An EnKF analysis is a candidate for the next native continuation.
            # If that candidate cannot cross this node, retry once with the
            # last parameters that produced a healthy native state. This is a
            # solver-health fallback, not a parameter jump limiter.
            safe_parameters = dict(node_native_start_parameters)
            parameter_fallback_used = False
            failed_attempts = 0
            backtrack_count = 0
            attempt_index = 0
            while not target_time_reached(float(session.fracture.time), target_time):
                if wall_budget > 0.0 and time.perf_counter() - started >= wall_budget:
                    status = "wall_clock_budget_exhausted"
                    _write_progress(
                        output,
                        status=status,
                        target_time_s=source_end,
                        current_node_s=target_time,
                        sessions=sessions,
                        ensemble=ensemble,
                        history=history,
                        attempts=attempts,
                        memory=memory_samples,
                        started=started,
                    )
                    return _finalize(output, status, source_end, sessions, history, attempts, memory_samples, peak_commit, args, pressure_meta, started)

                before_time = float(session.fracture.time)
                if parameter_fallback_used:
                    transition_fraction = 1.0
                    applied_parameters = dict(safe_parameters)
                elif args.native_parameter_transition == "time_linear":
                    expected_horizon = float(session._continuation_step_limit_s) * int(args.max_time_steps_per_attempt)
                    expected_end = min(target_time, before_time + max(expected_horizon, 0.0))
                    transition_fraction = (expected_end - node_native_start_time) / max(
                        target_time - node_native_start_time,
                        1.0e-12,
                    )
                    applied_parameters = interpolate_native_parameters(
                        node_native_start_parameters,
                        native_parameters,
                        transition_fraction,
                    )
                else:
                    transition_fraction = 1.0
                    applied_parameters = native_parameters
                result = session.advance_to(
                    target_time,
                    schedule,
                    **applied_parameters,
                    allow_partial=True,
                )
                attempt_index += 1
                record = {
                    "node_index": int(node_index),
                    "attempt_index": int(attempt_index),
                    "node_time_s": float(target_time),
                    "start_time_s": before_time,
                    "persistent_step_limit_s": float(session._continuation_step_limit_s),
                    "consecutive_failures": int(session.consecutive_failures),
                    "front_metadata_repair_count": int(session.front_metadata_repair_count),
                    "last_front_metadata_repairs": " | ".join(session.last_front_metadata_repairs),
                    "native_parameter_transition": str(args.native_parameter_transition),
                    "native_parameter_transition_fraction": float(np.clip(transition_fraction, 0.0, 1.0)),
                    **_record_result(result, member_index, target_time),
                }
                attempts.append(record)
                if not result.success:
                    failed_attempts += 1
                elif float(result.final_time_s) <= before_time + 1.0e-8:
                    failed_attempts += 1
                else:
                    failed_attempts = 0

                snapshot = _process_memory_snapshot()
                peak_commit = max(peak_commit, float(snapshot.get("commit_bytes", 0.0) or 0.0))
                memory_samples.append(
                    {
                        "node_index": int(node_index),
                        "member_index": int(member_index),
                        "attempt_index": int(attempt_index),
                        "member_model_time_s": float(session.fracture.time),
                        "wall_clock_elapsed_s": float(time.perf_counter() - started),
                        **snapshot,
                    }
                )
                _write_progress(
                    output,
                    status="running",
                    target_time_s=source_end,
                    current_node_s=target_time,
                    sessions=sessions,
                    ensemble=ensemble,
                    history=history,
                    attempts=attempts,
                    memory=memory_samples,
                    started=started,
                )
                if _memory_exceeded(snapshot, limit_bytes):
                    status = "memory_guard_triggered"
                    _write_progress(
                        output,
                        status=status,
                        target_time_s=source_end,
                        current_node_s=target_time,
                        sessions=sessions,
                        ensemble=ensemble,
                        history=history,
                        attempts=attempts,
                        memory=memory_samples,
                        started=started,
                    )
                    return _finalize(output, status, source_end, sessions, history, attempts, memory_samples, peak_commit, args, pressure_meta, started)
                if failed_attempts >= int(args.backtrack_after_consecutive_failures):
                    backtracked = session.rollback_to_previous_accepted(
                        reason=f"{failed_attempts} consecutive failures before node {target_time:g}s"
                    )
                    if backtracked:
                        backtrack_count += 1
                        attempts.append(
                            {
                                "node_index": int(node_index),
                                "attempt_index": int(attempt_index),
                                "node_time_s": float(target_time),
                                "member_index": int(member_index),
                                "orchestration_event": "backtracked_to_previous_accepted",
                                "final_time_s": float(session.fracture.time),
                                "persistent_step_limit_s": float(session._continuation_step_limit_s),
                                "backtrack_count": int(backtrack_count),
                                "success": False,
                            }
                        )
                        failed_attempts = 0
                        _write_progress(
                            output,
                            status="running",
                            target_time_s=source_end,
                            current_node_s=target_time,
                            sessions=sessions,
                            ensemble=ensemble,
                            history=history,
                            attempts=attempts,
                            memory=memory_samples,
                            started=started,
                        )
                        if backtrack_count > int(args.max_backtracks_per_member_node):
                            if not parameter_fallback_used:
                                parameter_fallback_used = True
                                safe_parameters = dict(session.last_parameters or safe_parameters)
                                native_parameters = dict(safe_parameters)
                                node_native_start_parameters = dict(safe_parameters)
                                failed_attempts = 0
                                backtrack_count = 0
                                attempts.append(
                                    {
                                        "node_index": int(node_index),
                                        "attempt_index": int(attempt_index),
                                        "node_time_s": float(target_time),
                                        "member_index": int(member_index),
                                        "orchestration_event": "enkf_parameter_rejected_native_fallback",
                                        "fallback_reason": "candidate_failed_native_continuation_after_rollback",
                                        "final_time_s": float(session.fracture.time),
                                        "fallback_parameters": dict(safe_parameters),
                                    }
                                )
                                continue
                            status = "member_backtrack_limit_exhausted"
                            return _finalize(output, status, source_end, sessions, history, attempts, memory_samples, peak_commit, args, pressure_meta, started)
                        continue
                if failed_attempts >= int(args.max_failed_attempts_per_member_node):
                    status = "member_retry_limit_exhausted"
                    _write_progress(
                        output,
                        status=status,
                        target_time_s=source_end,
                        current_node_s=target_time,
                        sessions=sessions,
                        ensemble=ensemble,
                        history=history,
                        attempts=attempts,
                        memory=memory_samples,
                        started=started,
                    )
                    return _finalize(output, status, source_end, sessions, history, attempts, memory_samples, peak_commit, args, pressure_meta, started)

            node_results.append(
                _result_at_current_state(
                    session,
                    target_time,
                    schedule,
                    dict(session.last_parameters or native_parameters),
                )
            )

        synchronized = all(result.success and result.target_reached for result in node_results)
        native_pressure = float(node_results[0].bottomhole_pressure_mpa) if node_results else float("nan")
        predicted = anchored_shadow_pressures(
            ensemble,
            prior_state,
            native_pressure,
            physical_cfg,
            schedule,
            target_time,
        ) if synchronized and np.isfinite(native_pressure) else np.full(len(ensemble), np.nan)
        finite_predictions = np.isfinite(predicted)
        observed = observation_at(pressure, target_time, "bottomhole_pressure_mpa")
        update_status = "not_updated"
        gain_mean = float("nan")
        relaxation_result = None
        if synchronized and bool(finite_predictions.all()) and len(predicted) >= 3:
            analysis, gain = denkf_update(
                ensemble,
                predicted[:, None],
                np.asarray([observed]),
                np.asarray([args.ensemble_pressure_noise_mpa]),
                covariance_inflation=1.01,
            )
            relaxation_result = relaxed_update(
                ensemble,
                clip_state(analysis, 1),
                relaxation=pyfrac_cfg.assimilation_relaxation,
                max_step=(
                    None
                    if not np.any(np.asarray(pyfrac_cfg.assimilation_max_parameter_step, dtype=float) > 0.0)
                    else np.asarray(pyfrac_cfg.assimilation_max_parameter_step, dtype=float)
                ),
            )
            ensemble = clip_state(relaxation_result.state, 1)
            gain_mean = float(np.mean(np.abs(gain)))
            update_status = "updated"
        else:
            status = "synchronization_or_prediction_failure"

        posterior_state = ensemble.mean(axis=0)
        posterior_pkn = pkn_pressure(posterior_state, physical_cfg, schedule, target_time)
        history.append(
            {
                "node_index": int(node_index),
                "time_s": float(target_time),
                "native_model_time_s": float(node_results[0].final_time_s),
                "all_members_synchronized": bool(synchronized),
                "member_time_spread_s": float(max(result.final_time_s for result in node_results) - min(result.final_time_s for result in node_results)),
                "native_member_count": int(len(node_results)),
                "shadow_ensemble_size": int(len(predicted)),
                "prediction_operator": "single_native_pyfrac_anchor_plus_pkn_local_delta",
                "observed_bottomhole_pressure_mpa": float(observed),
                "native_pyfrac_bottomhole_pressure_mpa": native_pressure,
                "native_pyfrac_pressure_error_mpa": float(abs(native_pressure - observed)),
                "shadow_prior_mean_bottomhole_pressure_mpa": float(np.mean(predicted)),
                "shadow_prior_pressure_spread_mpa": float(np.std(predicted, ddof=1)),
                "shadow_prior_pressure_error_mpa": float(abs(np.mean(predicted) - observed)),
                "pyfrac_mean_half_length_m": float(np.mean([result.half_length_m for result in node_results])),
                "pyfrac_mean_maximum_width_m": float(np.mean([result.maximum_width_m for result in node_results])),
                "pyfrac_max_mass_balance_relative_error": float(np.nanmax([result.mass_balance_relative_error for result in node_results])),
                "pkn_prior_bottomhole_pressure_mpa": float(prior_pkn["bottomhole_pressure_mpa"]),
                "pkn_posterior_bottomhole_pressure_mpa": float(posterior_pkn["bottomhole_pressure_mpa"]),
                "enkf_update_status": update_status,
                "native_parameter_fallback_used": bool(parameter_fallback_used),
                "enkf_innovation_mpa": float(observed - np.mean(predicted)),
                "mean_abs_kalman_gain": gain_mean,
                "enkf_relaxation": float(relaxation_result.relaxation) if relaxation_result else np.nan,
                "enkf_raw_update_norm": float(relaxation_result.raw_update_norm) if relaxation_result else np.nan,
                "enkf_applied_update_norm": float(relaxation_result.applied_update_norm) if relaxation_result else np.nan,
                **state_record("prior", prior_state, physical_cfg, 1),
                **state_record("posterior", posterior_state, physical_cfg, 1),
            }
        )
        _write_progress(
            output,
            status="running" if status == "completed" else status,
            target_time_s=source_end,
            current_node_s=target_time,
            sessions=sessions,
            ensemble=ensemble,
            history=history,
            attempts=attempts,
            memory=memory_samples,
            started=started,
        )
        if status != "completed":
            break

    member_times = [float(session.fracture.time) for session in sessions]
    if not member_times or not target_time_reached(min(member_times), source_end):
        status = "incomplete"
    elif not history or float(history[-1]["time_s"]) < source_end - 1.0e-8:
        status = "incomplete"
    elif history[-1].get("enkf_update_status") != "updated":
        status = "incomplete"
    else:
        status = "completed"
    _write_progress(
        output,
        status=status,
        target_time_s=source_end,
        current_node_s=source_end,
        sessions=sessions,
        ensemble=ensemble,
        history=history,
        attempts=attempts,
        memory=memory_samples,
        started=started,
    )
    return _finalize(output, status, source_end, sessions, history, attempts, memory_samples, peak_commit, args, pressure_meta, started)


def _finalize(
    output: Path,
    status: str,
    target: float,
    sessions: list[PyFracNativeSession],
    history: list[dict[str, object]],
    attempts: list[dict[str, object]],
    memory: list[dict[str, object]],
    peak_commit_bytes: float,
    args: argparse.Namespace,
    pressure_meta: dict[str, object],
    started: float,
) -> dict[str, object]:
    pd.DataFrame(history).to_csv(output / "assimilation_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(attempts).to_csv(output / "member_attempts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(memory).to_csv(output / "memory.csv", index=False, encoding="utf-8-sig")
    member_times = [float(session.fracture.time) for session in sessions]
    summary = {
        "status": status,
        "experiment_mode": "single_native_pyfrac_multifidelity_enkf",
        "native_member_count": int(len(sessions)),
        "parameter_ensemble_size": int(max(int(args.ensemble_size), 3)),
        "prediction_operator": "single_native_pyfrac_anchor_plus_pkn_local_delta",
        "native_geometry_source": "one_continuously_advanced_native_pyfrac_state",
        "enkf_covariance_source": "pkn_local_parameter_deltas_anchored_to_native_pressure",
        "native_parameter_transition": str(args.native_parameter_transition),
        "parameter_update_limits_enabled": bool(
            np.any(np.asarray(args.assimilation_max_parameter_step, dtype=float) > 0.0)
        ),
        "target_time_s": float(target),
        "target_reached": bool(member_times and target_time_reached(min(member_times), target) and status == "completed"),
        "member_model_times_s": member_times,
        "minimum_member_model_time_s": float(min(member_times)) if member_times else None,
        "maximum_member_model_time_s": float(max(member_times)) if member_times else None,
        "assimilation_update_count": int(sum(row.get("enkf_update_status") == "updated" for row in history)),
        "all_updates_synchronized": bool(history and all(bool(row.get("all_members_synchronized")) for row in history)),
        "attempt_count": int(len(attempts)),
        "failed_attempt_count": int(sum(not bool(row.get("success")) for row in attempts)),
        "partial_advance_count": int(sum(bool(row.get("partial_progress")) for row in attempts)),
        "rollback_count": int(sum(bool(row.get("rollback_applied")) for row in attempts)),
        "peak_process_commit_gb": float(peak_commit_bytes / 1024.0**3),
        "memory_guard_limit_gb": float(args.max_process_commit_gb),
        "max_mesh_elements": int(args.max_mesh_elements),
        "elapsed_wall_clock_s": float(time.perf_counter() - started),
        "performance": {
            "gate_definition": "native wall-clock runtime / minimum native simulated time; lower is better",
            "target_ratios": [float(value) for value in args.throughput_target_ratios],
            "native_throughput": classify_throughput(
                time.perf_counter() - started,
                min(member_times) if member_times else None,
                args.throughput_target_ratios,
            ),
            "required_throughput_ratio": args.required_throughput_ratio,
            "formal_gate": evaluate_throughput_requirement(
                time.perf_counter() - started,
                min(member_times) if member_times else None,
                args.required_throughput_ratio,
            ),
        },
        "warm_start_time_s": float(args.warm_start_time_s),
        "warm_start_is_analytical": bool(float(args.warm_start_time_s) > 1.0),
        "pressure_source": pressure_meta,
        "outputs": {
            "assimilation_history": str(output / "assimilation_history.csv"),
            "member_attempts": str(output / "member_attempts.csv"),
            "memory": str(output / "memory.csv"),
            "checkpoints": str(output / "checkpoints"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
