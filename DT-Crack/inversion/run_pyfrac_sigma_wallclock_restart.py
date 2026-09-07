"""Wall-clock experiment for restart-from-initial PyFrac inversion.

This runner implements the experiment definition used by the project review:

* the experiment budget is wall-clock time (default: 4435 s);
* at the beginning of an attempt, the newest integer source point that has
  arrived is selected;
* a fresh native PyFrac run starts at t=1 s and advances to that point using
  the current ``sigma_min``;
* source points that arrive while PyFrac is busy are not backfilled;
* when a valid run finishes, the pressure innovation at the point represented
  by that run updates ``sigma_min`` for the next attempt;
* a failed or timed-out run is kept in the audit trail, does not assimilate an
  observation, and selects a different deterministic ``sigma_min`` candidate
  for the next attempt.  This prevents a failed candidate from being retried
  indefinitely with exactly the same parameter.

The distinction between ``requested_target_time_s`` and
``latest_source_time_at_completion_s`` is intentional.  A result produced for
one simulated time cannot be compared with a later observation without a
second forward solve.  Therefore later points are recorded as skipped rather
than silently being used with a mismatched model state.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


DT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DT_ROOT.parent
if str(DT_ROOT) not in sys.path:
    sys.path.insert(0, str(DT_ROOT))

from data_fusion.pressure_schedule_adapter import PressureModelConfig, load_stage_pressure_schedule  # noqa: E402
from forward_models.pyfrac_config import PyFracConfig  # noqa: E402
from inversion.performance_tiers import DEFAULT_THROUGHPUT_TIERS, classify_throughput  # noqa: E402
from inversion.run_pyfrac_enkf import derive_injection_rate, make_schedule, observation_at  # noqa: E402
from inversion.run_pyfrac_sigma_online_restart import (  # noqa: E402
    _as_bool,
    _evaluate_candidate,
    _json_default,
    update_sigma_from_innovation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction-pressure-xls", default="Data/3Dfrac/JY84-Z1-stage08-f1.xls")
    parser.add_argument("--output-dir", default="outputs/dt/pyfrac_sigma_wallclock4435")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--target-time-s", type=float, default=4435.0)
    parser.add_argument("--wall-clock-budget-s", type=float, default=4435.0)
    parser.add_argument(
        "--candidate-timeout-s",
        type=float,
        default=0.0,
        help=(
            "optional per-candidate hard timeout in seconds; 0 disables it "
            "and leaves the 4435 s experiment budget as the outer limit"
        ),
    )
    parser.add_argument(
        "--watchdog-start-s",
        type=float,
        default=60.0,
        help="do not evaluate the slow-progress watchdog before this wall time",
    )
    parser.add_argument(
        "--watchdog-ratio",
        type=float,
        default=3.0,
        help=(
            "terminate when simulated seconds per wall-clock second fall below "
            "this throughput threshold"
        ),
    )
    parser.add_argument("--watchdog-poll-s", type=float, default=1.0)
    parser.add_argument("--watchdog-min-sim-time-s", type=float, default=5.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.25)
    parser.add_argument("--sigma-min-mpa", type=float, default=60.0)
    parser.add_argument("--sigma-lower-mpa", type=float, default=20.0)
    parser.add_argument("--sigma-upper-mpa", type=float, default=120.0)
    parser.add_argument("--sigma-update-gain", type=float, default=0.5)
    parser.add_argument("--sigma-sensitivity-floor", type=float, default=0.10)
    parser.add_argument(
        "--failure-fallback-step-mpa",
        type=float,
        default=5.0,
        help="fallback step used only when no previous non-zero sigma update direction exists",
    )
    parser.add_argument(
        "--failure-contraction",
        type=float,
        default=0.8,
        help="contraction applied to the latest successful sigma update direction after failure",
    )
    parser.add_argument("--height-m", type=float, default=30.0)
    parser.add_argument("--base-eprime-pa", type=float, default=3.2e10)
    parser.add_argument("--base-leakoff-m-sqrt-s", type=float, default=1.0e-5)
    parser.add_argument("--base-viscosity-pa-s", type=float, default=0.1)
    parser.add_argument("--fracture-toughness-pa-sqrt-m", type=float, default=5.0e5)
    parser.add_argument("--mesh-nx", type=int, default=61)
    parser.add_argument("--mesh-ny", type=int, default=9)
    parser.add_argument("--mesh-half-length-m", type=float, default=10.0)
    parser.add_argument("--mesh-half-height-m", type=float, default=25.0)
    parser.add_argument("--max-time-steps", type=int, default=240)
    parser.add_argument("--dynamic-step-limit-s", type=float, default=30.0)
    parser.add_argument("--initial-step-limit-s", type=float, default=1.0)
    parser.add_argument("--injection-volume-step-fraction", type=float, default=2.0)
    parser.add_argument("--positive-volume-change-step-fraction", type=float, default=2.0)
    parser.add_argument("--negative-volume-change-step-fraction", type=float, default=1.0)
    parser.add_argument("--time-step-time-fraction", type=float, default=1.0)
    parser.add_argument("--cell-traversal-fraction", type=float, default=3.0)
    parser.add_argument("--ehl-solver", choices=("implicit_Anderson", "implicit_Picard", "RKL2"), default="implicit_Anderson")
    parser.add_argument("--ehl-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--ehl-relaxation", type=float, default=1.0)
    parser.add_argument("--front-cfl", type=float, default=100.0)
    parser.add_argument("--front-length-fraction", type=float, default=20.0)
    parser.add_argument("--front-advancing", choices=("predictor-corrector", "implicit", "explicit"), default="implicit")
    parser.add_argument("--projection-method", choices=("ILSA_orig", "LS_grad", "LS_continousfront"), default="ILSA_orig")
    parser.add_argument("--max-native-retries", type=int, default=8)
    parser.add_argument("--retry-time-step-factor", type=float, default=0.5)
    parser.add_argument("--min-dynamic-step-s", type=float, default=0.5)
    parser.add_argument("--max-solver-iterations", type=int, default=140)
    parser.add_argument("--max-front-iterations", type=int, default=25)
    parser.add_argument("--max-pyfrac-reattempts", type=int, default=8)
    parser.add_argument(
        "--enable-volume-balance-projection",
        action="store_true",
        help="enable the explicitly labelled conservative volume projection",
    )
    parser.add_argument("--disable-adaptive-mesh", action="store_true")
    parser.add_argument("--disable-pyfrac-remeshing", action="store_true")
    parser.add_argument("--disable-mesh-extension", action="store_true")
    return parser


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    return frame.replace({np.nan: None}).to_dict("records")


def _build_config(args: argparse.Namespace) -> PyFracConfig:
    return PyFracConfig(
        height_m=float(args.height_m),
        young_modulus_pa=float(args.base_eprime_pa),
        fracture_toughness_pa_sqrt_m=float(args.fracture_toughness_pa_sqrt_m),
        leakoff_coefficient_m_sqrt_s=float(args.base_leakoff_m_sqrt_s),
        viscosity_pa_s=float(args.base_viscosity_pa_s),
        min_horizontal_stress_pa=float(args.sigma_min_mpa) * 1.0e6,
        initial_time_s=1.0,
        native_start_time_s=1.0,
        initial_step_limit_s=max(float(args.initial_step_limit_s), 1.0e-6),
        mesh_half_length_m=max(float(args.mesh_half_length_m), 1.0),
        mesh_half_height_m=max(float(args.mesh_half_height_m), 1.0),
        mesh_nx=max(int(args.mesh_nx), 31),
        mesh_ny=max(int(args.mesh_ny), 9),
        max_time_steps=max(int(args.max_time_steps), 1),
        dynamic_step_limit_s=max(float(args.dynamic_step_limit_s), 0.0),
        injection_volume_step_fraction=max(float(args.injection_volume_step_fraction), 1.0e-6),
        positive_volume_change_step_fraction=max(float(args.positive_volume_change_step_fraction), 1.0e-6),
        negative_volume_change_step_fraction=max(float(args.negative_volume_change_step_fraction), 1.0e-6),
        time_step_time_fraction=max(float(args.time_step_time_fraction), 1.0e-6),
        cell_traversal_fraction=max(float(args.cell_traversal_fraction), 1.0e-6),
        elastohydr_solver=str(args.ehl_solver),
        ehl_tolerance=max(float(args.ehl_tolerance), 1.0e-12),
        ehl_relaxation=max(float(args.ehl_relaxation), 1.0e-6),
        front_cfl=max(float(args.front_cfl), 0.0),
        front_length_fraction=max(float(args.front_length_fraction), 0.0),
        front_advancing=args.front_advancing,
        projection_method=args.projection_method,
        adaptive_mesh_enabled=not args.disable_adaptive_mesh,
        enable_pyfrac_remeshing=not args.disable_pyfrac_remeshing,
        mesh_extension_enabled=not args.disable_mesh_extension,
        max_retries=max(int(args.max_native_retries), 0),
        retry_time_step_factor=float(args.retry_time_step_factor),
        min_dynamic_step_s=max(float(args.min_dynamic_step_s), 1.0e-6),
        max_solver_iterations=max(int(args.max_solver_iterations), 1),
        max_front_iterations=max(int(args.max_front_iterations), 1),
        max_pyfrac_reattempts=max(int(args.max_pyfrac_reattempts), 0),
        enable_volume_balance_projection=bool(args.enable_volume_balance_projection),
        checkpoint_enabled=False,
    )


def _write_checkpoint(
    output: Path,
    *,
    status: str,
    started: float,
    target: float,
    last_requested: float | None,
    sigma: float,
    history: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    candidate_audits: list[dict[str, Any]],
) -> None:
    payload = {
        "status": status,
        "target_time_s": float(target),
        "last_requested_target_time_s": last_requested,
        "current_sigma_min_mpa": float(sigma),
        "completed_attempt_count": len(history),
        "skipped_source_point_count": len(skipped),
        "parameter_update_count": len(updates),
        "failure_reselection_count": sum(
            row.get("update_status") == "reselected_after_failure" for row in history
        ),
        "candidate_audit_count": len(candidate_audits),
        "elapsed_wall_clock_s": float(time.perf_counter() - started),
        "updated_at_epoch_s": time.time(),
    }
    (output / "checkpoint.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    pd.DataFrame(history).to_csv(output / "wallclock_history_progress.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(skipped).to_csv(output / "skipped_source_points_progress.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(updates).to_csv(output / "sigma_update_audit_progress.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(candidate_audits).to_csv(output / "sigma_candidate_audit_progress.csv", index=False, encoding="utf-8-sig")


def _source_point(start_source: float, target: float, started: float) -> int:
    """Return the newest integer source point available at this wall time."""

    elapsed = max(time.perf_counter() - started, 0.0)
    return int(min(float(target), np.floor(float(start_source) + elapsed)))


def _record_skipped(
    skipped: list[dict[str, Any]],
    *,
    previous: int,
    current: int,
    attempt_index: int,
    reason: str,
    wall_elapsed: float,
) -> None:
    if current <= previous + 1:
        return
    for source_time in range(previous + 1, current):
        skipped.append(
            {
                "attempt_index": int(attempt_index),
                "source_time_s": int(source_time),
                "reason": reason,
                "wall_elapsed_s": float(wall_elapsed),
            }
        )


def _failure_reason(row: dict[str, Any]) -> str:
    """Classify a rejected candidate without calling every timeout a solver error."""

    error = str(row.get("error") or "").lower()
    if "watchdog" in error:
        return "sim_to_wall_ratio_below_threshold"
    if "timeout" in error:
        return "timeout_before_target"
    progress = row.get("last_progress")
    if isinstance(progress, str):
        try:
            progress = json.loads(progress)
        except (TypeError, json.JSONDecodeError):
            progress = None
    if isinstance(progress, dict):
        try:
            if int(progress.get("failed_time_steps", 0) or 0) > 0:
                return "native_failed_time_step"
        except (TypeError, ValueError):
            pass
        if progress.get("time_s") is not None and progress.get("final_time_s") is not None:
            try:
                if float(progress["time_s"]) < float(progress["final_time_s"]) - 1.0e-8:
                    return "native_incomplete_before_target"
            except (TypeError, ValueError):
                pass
    if "nan" in error or "inf" in error:
        return "non_finite_native_result"
    if error:
        return "worker_exception_or_missing_result"
    return "native_acceptance_rejected"


def _next_failure_candidate(
    current_sigma_mpa: float,
    *,
    anchor_sigma_mpa: float | None,
    update_delta_mpa: float | None,
    failure_streak: int,
    lower_mpa: float,
    upper_mpa: float,
    contraction: float,
    fallback_step_mpa: float,
) -> float:
    """Contract the last successful update direction after a failed run.

    ``anchor + contraction**failure_streak * delta`` implements the requested
    recovery policy. It does not constrain a successful observation update; it
    is only used after a candidate failed, timed out, or was stopped by the
    slow-progress watchdog. If no previous non-zero update direction exists,
    a small deterministic fallback step is used to guarantee a new candidate.
    """

    lower = float(min(lower_mpa, upper_mpa))
    upper = float(max(lower_mpa, upper_mpa))
    current = float(np.clip(current_sigma_mpa, lower, upper))
    streak = max(int(failure_streak), 1)
    factor = float(np.clip(contraction, 1.0e-6, 0.999999))
    epsilon = 1.0e-8

    if anchor_sigma_mpa is not None and update_delta_mpa is not None:
        try:
            anchor = float(anchor_sigma_mpa)
            delta = float(update_delta_mpa)
        except (TypeError, ValueError):
            anchor = float("nan")
            delta = 0.0
        if np.isfinite(anchor) and np.isfinite(delta) and abs(delta) > epsilon:
            candidate = float(np.clip(anchor + (factor**streak) * delta, lower, upper))
            if abs(candidate - current) > epsilon:
                return candidate
            # Clipping can collapse a candidate onto the failed boundary. Try
            # one more contraction before using the bounded fallback.
            for extra in range(1, 8):
                candidate = float(np.clip(anchor + (factor ** (streak + extra)) * delta, lower, upper))
                if abs(candidate - current) > epsilon:
                    return candidate

    step = max(abs(float(fallback_step_mpa)), 1.0e-6) * (factor ** max(streak - 1, 0))
    direction = 1.0 if update_delta_mpa is None or float(update_delta_mpa or 0.0) >= 0.0 else -1.0
    fallback_candidates = [current + direction * step, current - direction * step, current + step, current - step]
    for candidate in fallback_candidates:
        candidate = float(np.clip(candidate, lower, upper))
        if abs(candidate - current) > epsilon:
            return candidate
    # This only occurs when the bounds collapse to one value. The repeat is
    # unavoidable and remains visible in the candidate audit.
    return current


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    schedule_dir = output / "schedules"
    schedule_dir.mkdir(parents=True, exist_ok=True)

    pressure, pressure_meta = load_stage_pressure_schedule(
        args.construction_pressure_xls,
        config=PressureModelConfig(step_seconds=1.0),
    )
    source_start = max(float(pressure["time_s"].min()), 1.0)
    source_end = min(float(args.target_time_s), float(pressure["time_s"].max()))
    rate = derive_injection_rate(pressure)
    config = _build_config(args)

    history = _read_rows(Path(args.resume_from).resolve() / "wallclock_history.csv") if args.resume_from else []
    skipped = _read_rows(Path(args.resume_from).resolve() / "skipped_source_points.csv") if args.resume_from else []
    updates = _read_rows(Path(args.resume_from).resolve() / "sigma_update_audit.csv") if args.resume_from else []
    candidate_audits = _read_rows(Path(args.resume_from).resolve() / "sigma_candidate_audit.csv") if args.resume_from else []
    sigma = float(args.sigma_min_mpa)
    if history:
        last_sigma = history[-1].get("next_sigma_min_mpa")
        if last_sigma is not None and np.isfinite(float(last_sigma)):
            sigma = float(last_sigma)
    last_requested = int(max((int(row["requested_target_time_s"]) for row in history if row.get("requested_target_time_s") is not None), default=0))
    previous_success: dict[str, float] | None = None
    for row in reversed(history):
        if _as_bool(row.get("native_acceptance")) and row.get("predicted_bottomhole_pressure_mpa") is not None:
            previous_success = {
                "sigma_mpa": float(row["sigma_min_mpa"]),
                "predicted_mpa": float(row["predicted_bottomhole_pressure_mpa"]),
            }
            break
    failure_streak = 0
    for row in history:
        if row.get("update_status") == "reselected_after_failure":
            failure_streak += 1
        elif row.get("update_status") == "updated":
            failure_streak = 0

    # On resume, restore the direction of the latest successful observation
    # update. Failed attempts contract this direction; they do not invent a
    # new grid of unrelated sigma values.
    update_anchor_sigma: float | None = None
    update_delta_sigma: float | None = None
    if updates:
        latest_update = updates[-1]
        try:
            update_anchor_sigma = float(latest_update["sigma_before_mpa"])
            sigma_after = float(latest_update["sigma_after_mpa"])
            update_delta_sigma = sigma_after - update_anchor_sigma
            if not np.isfinite(update_anchor_sigma) or not np.isfinite(update_delta_sigma):
                update_anchor_sigma = None
                update_delta_sigma = None
        except (KeyError, TypeError, ValueError):
            update_anchor_sigma = None
            update_delta_sigma = None

    budget = max(float(args.wall_clock_budget_s), 0.0)
    status = "running"
    attempt_index = len(history)
    while True:
        elapsed_before = time.perf_counter() - started
        if budget > 0.0 and elapsed_before >= budget:
            status = "wall_clock_budget_exhausted"
            break
        request_time = _source_point(source_start, source_end, started)
        if request_time <= last_requested:
            sleep_for = min(max(float(args.poll_interval_s), 0.01), max(budget - elapsed_before, 0.01))
            time.sleep(sleep_for)
            continue

        _record_skipped(
            skipped,
            previous=last_requested,
            current=request_time,
            attempt_index=attempt_index,
            reason="solver_busy_or_stale_point",
            wall_elapsed=elapsed_before,
        )
        schedule = make_schedule(pressure, rate, float(request_time), 30.0)
        schedule_path = schedule_dir / f"target_{request_time:05d}.npy"
        np.save(schedule_path, schedule, allow_pickle=False)
        observed = float(observation_at(pressure, float(request_time), "bottomhole_pressure_mpa"))
        remaining = budget - elapsed_before if budget > 0.0 else 0.0
        configured_timeout = max(float(args.candidate_timeout_s), 0.0)
        if configured_timeout > 0.0 and remaining > 0.0:
            timeout = min(configured_timeout, remaining)
        elif configured_timeout > 0.0:
            timeout = configured_timeout
        else:
            # No arbitrary per-candidate cap.  When a finite experiment
            # budget exists, the candidate is bounded only by the remaining
            # wall-clock budget; otherwise the worker is watchdog-controlled.
            timeout = remaining if remaining > 0.0 else None
        attempt_started = time.perf_counter()
        row = _evaluate_candidate(
            output=output / f"attempt_{attempt_index:05d}_t{request_time:05d}",
            candidate_id=f"wallclock_{attempt_index:05d}",
            sigma_min_mpa=sigma,
            target_time_s=float(request_time),
            observed_pressure_mpa=observed,
            schedule_path=schedule_path,
            pyfrac_config=config,
            timeout_s=timeout,
            watchdog_start_s=float(args.watchdog_start_s),
            watchdog_ratio=float(args.watchdog_ratio),
            watchdog_poll_s=float(args.watchdog_poll_s),
            watchdog_min_sim_time_s=float(args.watchdog_min_sim_time_s),
        )
        attempt_completed_elapsed = time.perf_counter() - started
        within_budget = budget <= 0.0 or attempt_completed_elapsed <= budget
        row.update(
            {
                "attempt_index": int(attempt_index),
                "requested_target_time_s": int(request_time),
                "sigma_min_mpa": float(sigma),
                "wall_clock_started_s": float(elapsed_before),
                "wall_clock_completed_s": float(attempt_completed_elapsed),
                "attempt_wall_runtime_s": float(time.perf_counter() - attempt_started),
                "latest_source_time_at_completion_s": int(_source_point(source_start, source_end, started)),
                "completed_within_wall_clock_budget": bool(within_budget),
                "skipped_before_attempt_count": int(max(request_time - last_requested - 1, 0)),
            }
        )
        latest_at_completion = int(row["latest_source_time_at_completion_s"])
        row["latest_observation_bottomhole_pressure_mpa"] = float(
            observation_at(pressure, float(latest_at_completion), "bottomhole_pressure_mpa")
        )
        row["observation_alignment"] = (
            "requested_target_time_aligned"
            if latest_at_completion <= request_time
            else "later_points_skipped_not_used_for_mismatched_update"
        )
        if not within_budget:
            row.update(
                {
                    "update_status": "not_counted_after_wall_clock_budget",
                    "next_sigma_min_mpa": float(sigma),
                    "native_acceptance_for_experiment": False,
                    "failure_reason": "wall_clock_budget_exhausted_during_attempt",
                    "candidate_selection_status": "no_next_candidate_after_budget",
                }
            )
            history.append(row)
            status = "wall_clock_budget_exhausted_during_attempt"
            candidate_audits.append(
                {
                    "attempt_index": int(attempt_index),
                    "requested_target_time_s": int(request_time),
                    "sigma_before_attempt_mpa": float(sigma),
                    "outcome": "out_of_budget",
                    "failure_reason": "wall_clock_budget_exhausted_during_attempt",
                    "next_sigma_candidate_mpa": None,
                    "failure_streak": int(failure_streak),
                }
            )
            _write_checkpoint(output, status=status, started=started, target=source_end, last_requested=request_time, sigma=sigma, history=history, skipped=skipped, updates=updates, candidate_audits=candidate_audits)
            break

        accepted = _as_bool(row.get("native_acceptance")) and row.get("predicted_bottomhole_pressure_mpa") is not None
        if accepted:
            predicted = float(row["predicted_bottomhole_pressure_mpa"])
            previous_sigma = previous_success["sigma_mpa"] if previous_success else None
            previous_predicted = previous_success["predicted_mpa"] if previous_success else None
            next_sigma, sensitivity = update_sigma_from_innovation(
                sigma,
                observed,
                predicted,
                previous_sigma_mpa=previous_sigma,
                previous_predicted_mpa=previous_predicted,
                gain=float(args.sigma_update_gain),
                sensitivity_floor=float(args.sigma_sensitivity_floor),
                lower_mpa=float(args.sigma_lower_mpa),
                upper_mpa=float(args.sigma_upper_mpa),
            )
            innovation = observed - predicted
            update = {
                "attempt_index": int(attempt_index),
                "requested_target_time_s": int(request_time),
                "observation_time_s": int(request_time),
                "observed_bottomhole_pressure_mpa": observed,
                "predicted_bottomhole_pressure_mpa": predicted,
                "innovation_mpa": float(innovation),
                "sigma_before_mpa": float(sigma),
                "sigma_after_mpa": float(next_sigma),
                "sensitivity_mpa_per_mpa": float(sensitivity),
                "update_wall_clock_s": float(attempt_completed_elapsed),
                "update_status": "updated",
            }
            updates.append(update)
            update_anchor_sigma = float(sigma)
            update_delta_sigma = float(next_sigma - sigma)
            row.update(
                {
                    "innovation_mpa": float(innovation),
                    "sigma_sensitivity_mpa_per_mpa": float(sensitivity),
                    "next_sigma_min_mpa": float(next_sigma),
                    "update_status": "updated",
                    "native_acceptance_for_experiment": True,
                    "failure_reason": None,
                    "failure_streak": 0,
                    "candidate_selection_status": "observation_assimilated",
                }
            )
            previous_success = {"sigma_mpa": sigma, "predicted_mpa": predicted}
            candidate_audits.append(
                {
                    "attempt_index": int(attempt_index),
                    "requested_target_time_s": int(request_time),
                    "sigma_before_attempt_mpa": float(sigma),
                    "outcome": "accepted",
                    "failure_reason": None,
                    "next_sigma_candidate_mpa": float(next_sigma),
                    "failure_streak": 0,
                    "selection_policy": "observation_assimilation",
                }
            )
            failure_streak = 0
            sigma = next_sigma
        else:
            failure_reason = _failure_reason(row)
            failure_streak += 1
            next_sigma = _next_failure_candidate(
                sigma,
                anchor_sigma_mpa=update_anchor_sigma,
                update_delta_mpa=update_delta_sigma,
                failure_streak=failure_streak,
                lower_mpa=float(args.sigma_lower_mpa),
                upper_mpa=float(args.sigma_upper_mpa),
                contraction=float(args.failure_contraction),
                fallback_step_mpa=float(args.failure_fallback_step_mpa),
            )
            row.update(
                {
                    "innovation_mpa": None,
                    "sigma_sensitivity_mpa_per_mpa": None,
                    "next_sigma_min_mpa": float(next_sigma),
                    "update_status": "reselected_after_failure",
                    "native_acceptance_for_experiment": False,
                    "failure_reason": failure_reason,
                    "failure_streak": int(failure_streak),
                    "candidate_selection_status": "different_sigma_selected_after_failure",
                    "sigma_update_anchor_mpa": update_anchor_sigma,
                    "sigma_update_delta_mpa": update_delta_sigma,
                    "failure_contraction": float(args.failure_contraction),
                }
            )
            candidate_audits.append(
                {
                    "attempt_index": int(attempt_index),
                    "requested_target_time_s": int(request_time),
                    "sigma_before_attempt_mpa": float(sigma),
                    "outcome": "failed_or_incomplete",
                    "failure_reason": failure_reason,
                    "next_sigma_candidate_mpa": float(next_sigma),
                    "failure_streak": int(failure_streak),
                    "selection_policy": "last_update_direction_contraction",
                    "update_anchor_sigma_mpa": update_anchor_sigma,
                    "update_delta_sigma_mpa": update_delta_sigma,
                    "contraction": float(args.failure_contraction),
                }
            )
            sigma = next_sigma
        row["throughput_tier"] = classify_throughput(
            row.get("solver_runtime_s"),
            row.get("final_native_time_s"),
            DEFAULT_THROUGHPUT_TIERS,
        )["tier"]
        history.append(row)
        last_requested = request_time
        attempt_index += 1
        _write_checkpoint(output, status=status, started=started, target=source_end, last_requested=last_requested, sigma=sigma, history=history, skipped=skipped, updates=updates, candidate_audits=candidate_audits)

    history_frame = pd.DataFrame(history)
    skipped_frame = pd.DataFrame(skipped)
    updates_frame = pd.DataFrame(updates)
    candidate_audits_frame = pd.DataFrame(candidate_audits)
    history_frame.to_csv(output / "wallclock_history.csv", index=False, encoding="utf-8-sig")
    skipped_frame.to_csv(output / "skipped_source_points.csv", index=False, encoding="utf-8-sig")
    updates_frame.to_csv(output / "sigma_update_audit.csv", index=False, encoding="utf-8-sig")
    candidate_audits_frame.to_csv(output / "sigma_candidate_audit.csv", index=False, encoding="utf-8-sig")
    valid = history_frame[history_frame.get("native_acceptance_for_experiment", pd.Series(dtype=bool)).map(_as_bool)] if not history_frame.empty else history_frame
    counted = history_frame[history_frame.get("completed_within_wall_clock_budget", pd.Series(dtype=bool)).map(_as_bool)] if not history_frame.empty else history_frame
    summary: dict[str, Any] = {
        "experiment": "pyfrac_sigma_wallclock_restart_inversion",
        "status": status,
        "wall_clock_budget_s": budget,
        "elapsed_wall_clock_s": float(time.perf_counter() - started),
        "source_window_s": [float(source_start), float(source_end)],
        "requested_semantics": {
            "wall_clock_is_experiment_duration": True,
            "fresh_restart_from_1s_after_each_sigma_update": True,
            "newest_point_at_attempt_start_is_used": True,
            "points_arriving_while_solver_is_busy_are_skipped": True,
            "failed_attempts_are_audited_without_observation_assimilation": True,
            "failed_attempts_select_a_different_sigma_candidate": True,
            "no_backfill_or_snapshot_substitution": True,
        },
        "completed_attempt_count": int(len(history)),
        "completed_within_wall_clock_budget_count": int(len(counted)),
        "out_of_budget_result_count": int(len(history) - len(counted)),
        "successful_native_attempt_count": int(len(valid)),
        "parameter_update_count": int(len(updates)),
        "skipped_source_point_count": int(len(skipped)),
        "failed_or_invalid_attempt_count": int(sum(row.get("native_acceptance_for_experiment") is not True for row in history)),
        "failure_reselection_count": int(sum(row.get("update_status") == "reselected_after_failure" for row in history)),
        "last_requested_target_time_s": int(last_requested) if last_requested else None,
        "final_sigma_min_mpa": float(sigma),
        "pressure_error": {
            "mean_abs_mpa": float(pd.to_numeric(valid.get("objective_abs_pressure_error_mpa"), errors="coerce").dropna().mean()) if not valid.empty and "objective_abs_pressure_error_mpa" in valid else None,
            "max_abs_mpa": float(pd.to_numeric(valid.get("objective_abs_pressure_error_mpa"), errors="coerce").dropna().max()) if not valid.empty and "objective_abs_pressure_error_mpa" in valid else None,
            "last_abs_mpa": float(valid.iloc[-1]["objective_abs_pressure_error_mpa"]) if not valid.empty and pd.notna(valid.iloc[-1].get("objective_abs_pressure_error_mpa")) else None,
        },
        "pressure_meta": pressure_meta,
        "pyfrac_config": asdict(config),
        "outputs": {
            "history": str(output / "wallclock_history.csv"),
            "skipped_source_points": str(output / "skipped_source_points.csv"),
            "sigma_update_audit": str(output / "sigma_update_audit.csv"),
            "sigma_candidate_audit": str(output / "sigma_candidate_audit.csv"),
            "checkpoint": str(output / "checkpoint.json"),
        },
        "limitations": [
            "本实验只反演 sigma_min；其他参数保持基础配置不变。",
            "更新使用与本次重演终点时间一致的最新可处理观测；计算期间到达的后续点明确记为跳过。",
            "墙钟实验的完成点数量不等于 4435 个；数量由每次从 1 s 重演的实际耗时决定。",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _write_checkpoint(output, status=status, started=started, target=source_end, last_requested=last_requested or None, sigma=sigma, history=history, skipped=skipped, updates=updates, candidate_audits=candidate_audits)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2, default=_json_default))
