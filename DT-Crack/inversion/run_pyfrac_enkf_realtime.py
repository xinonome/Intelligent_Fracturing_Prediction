"""Run a wall-clock bounded native-PyFrac + EnKF experiment.

This is the PyFrac counterpart of the project's real-time PKN replay.  The
source window is 1--4435 s, but the process is given a 4435 s wall-clock
budget.  A source point that arrives while the native solver is busy is
recorded as skipped; it is never back-filled or interpolated into EnKF.

PyFrac remains the forward model.  EnKF only updates the parameter ensemble
used by the next native continuation window.  Accepted native states are
periodically archived as dill checkpoints, while the in-memory checkpoint is
kept for immediate rollback/retry.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import asdict, fields
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
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
    native_state_health_errors,
)
from forward_models.pyfrac_config import PyFracConfig  # noqa: E402
from forward_models.pyfrac_robustness import relaxed_update  # noqa: E402
from inversion.physics import PhysicalEnKFConfig, clip_state, denkf_update, physical_values, state_record  # noqa: E402
from inversion.run_pyfrac_enkf import derive_injection_rate, make_schedule, observation_at  # noqa: E402
from inversion.performance_tiers import (  # noqa: E402
    DEFAULT_THROUGHPUT_TIERS,
    classify_throughput,
    evaluate_throughput_requirement,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction-pressure-xls", default="Data/3Dfrac/JY84-Z1-stage08-f1.xls")
    parser.add_argument("--output-dir", default="outputs/dt/pyfrac_enkf_realtime4435")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="resume from a previous realtime output directory using its latest durable checkpoint",
    )
    parser.add_argument("--target-time-s", type=float, default=4435.0)
    parser.add_argument("--wall-clock-budget-s", type=float, default=4435.0)
    parser.add_argument(
        "--native-initial-time-s",
        type=float,
        default=1.0,
        help="native initial condition time; the formal full-process run requires exactly 1 s",
    )
    # Keep the old option only to produce a clear migration error for copied
    # commands. It must never silently re-enable the former 360 s shortcut.
    parser.add_argument("--warm-start-time-s", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--source-point-interval-s", type=float, default=1.0)
    parser.add_argument("--assimilation-interval-s", type=float, default=600.0)
    parser.add_argument("--injection-schedule-step-s", type=float, default=30.0)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument(
        "--allow-partial-ensemble-update",
        action="store_true",
        help=(
            "allow exploratory EnKF updates from valid synchronized members; "
            "disabled by default for formal all-member mode"
        ),
    )
    parser.add_argument(
        "--min-valid-members",
        type=int,
        default=3,
        help="minimum valid members required by exploratory partial update",
    )
    parser.add_argument(
        "--max-consecutive-member-failures",
        type=int,
        default=3,
        help="deactivate a member after this many consecutive native failures",
    )
    parser.add_argument(
        "--disable-failed-member-recovery",
        action="store_true",
        help=(
            "do not restore a failed member to its latest valid parameter vector; "
            "diagnostic option only"
        ),
    )
    parser.add_argument(
        "--member-timeout-s",
        type=float,
        default=30.0,
        help=(
            "hard wall-clock limit for one native member window; timeout only "
            "invalidates that member window and preserves the last checkpoint"
        ),
    )
    parser.add_argument(
        "--disable-member-process-isolation",
        action="store_true",
        help="diagnostic compatibility mode; run members in the parent process",
    )
    parser.add_argument("--ensemble-pressure-noise-mpa", type=float, default=3.5)
    parser.add_argument(
        "--ensemble-spread-scale",
        type=float,
        default=1.0,
        help=(
            "scale the initial physical-parameter ensemble spread; this only "
            "controls initial numerical admissibility and never clips EnKF updates"
        ),
    )
    parser.add_argument("--base-eprime-pa", type=float, default=3.2e10)
    parser.add_argument("--base-leakoff-m-sqrt-s", type=float, default=1.0e-5)
    parser.add_argument("--base-viscosity-pa-s", type=float, default=0.1)
    parser.add_argument("--base-min-stress-mpa", type=float, default=60.0)
    parser.add_argument("--height-m", type=float, default=30.0)
    parser.add_argument("--mesh-nx", type=int, default=241)
    parser.add_argument("--mesh-ny", type=int, default=21)
    parser.add_argument(
        "--mesh-domain-time-s",
        type=float,
        default=None,
        help="legacy mesh-horizon option; fresh native runs resolve the t=1 s initial state first",
    )
    parser.add_argument("--initial-time-step-s", type=float, default=1.0)
    parser.add_argument("--mesh-half-length-m", type=float, default=10.0)
    parser.add_argument("--max-time-steps-per-window", type=int, default=40)
    parser.add_argument("--dynamic-step-limit-s", type=float, default=30.0)
    parser.add_argument(
        "--front-advancing",
        choices=("predictor-corrector", "implicit", "explicit"),
        default="implicit",
        help="native PyFrac front solver used by the formal full-process run",
    )
    parser.add_argument(
        "--projection-method",
        choices=("ILSA_orig", "LS_grad", "LS_continousfront"),
        default="ILSA_orig",
        help="native PyFrac level-set projection method used by the formal full-process run",
    )
    parser.add_argument("--max-native-retries", type=int, default=2)
    parser.add_argument("--retry-time-step-factor", type=float, default=0.5)
    parser.add_argument("--min-dynamic-step-s", type=float, default=0.05)
    parser.add_argument(
        "--assimilation-relaxation",
        type=float,
        default=1.0,
        help="formal comparison uses the complete EnKF analysis; kept for compatibility",
    )
    parser.add_argument(
        "--assimilation-max-parameter-step",
        type=float,
        nargs=5,
        default=[0.0, 0.0, 0.0, 0.0, 0.0],
        metavar=("D_LOGE", "D_LOGCL", "D_LOGMU", "D_STRESS", "D_LOGK"),
    )
    parser.add_argument("--checkpoint-interval-s", type=float, default=60.0)
    parser.add_argument(
        "--max-mesh-elements",
        type=int,
        default=6000,
        help="reject oversized restart meshes before allocating the dense elasticity matrix",
    )
    parser.add_argument(
        "--max-process-commit-gb",
        type=float,
        default=8.0,
        help="stop at the next safe boundary when process private/commit memory exceeds this value",
    )
    parser.add_argument("--disable-adaptive-mesh", action="store_true")
    parser.add_argument("--disable-pyfrac-remeshing", action="store_true")
    parser.add_argument(
        "--enable-mesh-extension",
        action="store_true",
        help="compatibility alias; mesh extension is enabled by default for the t=1 s run",
    )
    parser.add_argument(
        "--disable-mesh-extension",
        action="store_true",
        help="disable native mesh extension (only for diagnostic experiments)",
    )
    parser.add_argument(
        "--mesh-extension-all-directions",
        action="store_true",
        help="extend all four domain sides; default extension is horizontal left/right only",
    )
    parser.add_argument("--mesh-extension-factor", type=float, default=1.10)
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


def _load_latest_checkpoint(
    member_dir: Path,
    max_mesh_elements: int = 6000,
    max_model_time_s: float | None = None,
) -> dict[str, object] | None:
    """Load the newest durable checkpoint for one ensemble member."""

    candidates = sorted(
        [
            *member_dir.glob("accepted_*.dill"),
            *member_dir.glob("rollback_*.dill"),
            *member_dir.glob("resume_*.dill"),
            *member_dir.glob("initial_*.dill"),
            *member_dir.glob("backtrack_*.dill"),
        ],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        return None
    # The vendored PyFrac classes are imported as top-level legacy modules
    # (``fracture``, ``mesh``...). Load them before dill resolves the class
    # names from a checkpoint created by another process.
    PyFracAdapter(project_root=PROJECT_ROOT)._load_modules()
    import dill

    for candidate in reversed(candidates):
        with candidate.open("rb") as handle:
            payload = dill.load(handle)
        if max_model_time_s is not None and float(payload.get("time_s", float("inf"))) > float(max_model_time_s) + 1.0e-8:
            continue
        fracture = payload.get("fracture")
        mesh = getattr(fracture, "mesh", None)
        elements = int(getattr(mesh, "NumberOfElts", 0) or 0)
        if max_mesh_elements > 0 and elements > int(max_mesh_elements):
            continue
        if fracture is not None and native_state_health_errors(fracture):
            continue
        payload["checkpoint_path"] = str(candidate)
        payload["checkpoint_mesh_elements"] = elements
        return payload
    return None


def _load_resume_state(
    resume_root: Path,
    ensemble_size: int,
    seed: int,
    base_min_stress_mpa: float,
    max_mesh_elements: int,
    source_point_interval_s: float = 1.0,
) -> tuple[list[dict[str, object] | None], np.ndarray, float, float, float]:
    progress_path = resume_root / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
    ensemble_path = resume_root / "ensemble_state_progress.npy"
    if ensemble_path.is_file():
        ensemble = np.asarray(np.load(ensemble_path), dtype=float)
    else:
        rng = np.random.default_rng(seed)
        prior_mean = np.asarray([0.0, 0.0, 0.0, base_min_stress_mpa, 0.0], dtype=float)
        spread = np.asarray([0.06, 0.12, 0.08, 1.0, 0.08], dtype=float)
        ensemble = clip_state(rng.normal(prior_mean, spread, size=(max(int(ensemble_size), 1), 5)), 1)
    if ensemble.shape[0] != max(int(ensemble_size), 1):
        raise ValueError("resume ensemble size does not match --ensemble-size")
    members = []
    for member_index in range(ensemble.shape[0]):
        members.append(
            _load_latest_checkpoint(
                resume_root / "checkpoints" / f"member_{member_index:03d}",
                max_mesh_elements=max_mesh_elements,
            )
        )
    checkpoint_times = [
        float(payload.get("time_s", 0.0))
        for payload in members
        if payload is not None and np.isfinite(float(payload.get("time_s", np.nan)))
    ]
    max_checkpoint_time = max(checkpoint_times, default=0.0)
    requested_next_time = float(progress.get("next_source_time_s", float("nan")))
    interval = max(float(source_point_interval_s), 1.0)
    if not np.isfinite(requested_next_time) or requested_next_time <= max_checkpoint_time + 1.0e-8:
        # A copied checkpoint directory may not contain progress.json.  Never
        # replay source points before the retained native state; resume from
        # the first source point after the latest common checkpoint instead.
        next_source_time = max_checkpoint_time + interval
    else:
        next_source_time = requested_next_time
    last_assimilation_time = 1.0
    update_path = resume_root / "enkf_update_audit_progress.csv"
    if update_path.is_file():
        try:
            updates = pd.read_csv(update_path)
        except pd.errors.EmptyDataError:
            updates = pd.DataFrame()
        accepted = updates[updates.get("status", pd.Series(dtype=str)) == "updated"] if not updates.empty else updates
        if not accepted.empty:
            last_assimilation_time = float(accepted["time_s"].max())
    previous_elapsed = float(progress.get("elapsed_wall_clock_s", 0.0))
    return members, ensemble, next_source_time, last_assimilation_time, previous_elapsed


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _process_memory_snapshot() -> dict[str, float]:
    """Return process resident and private/commit memory without hard dependency."""

    try:
        import psutil

        info = psutil.Process().memory_info()
        private = float(getattr(info, "private", getattr(info, "pagefile", info.vms)))
        return {
            "rss_bytes": float(info.rss),
            "commit_bytes": private,
        }
    except Exception:
        return {"rss_bytes": float("nan"), "commit_bytes": float("nan")}


def _state_parameters(state: np.ndarray, cfg: PhysicalEnKFConfig) -> dict[str, float]:
    values = physical_values(state, cfg, 1)
    return {
        "height_m": float(cfg.height_m),
        "viscosity_pa_s": float(values["viscosity_pa_s"]),
        "e_prime_pa": float(values["eprime_pa"]),
        "leakoff_coefficient_m_sqrt_s": float(values["leakoff_m_sqrt_s"]),
        "min_horizontal_stress_pa": float(values["min_horizontal_stress_mpa"]) * 1.0e6,
        "fracture_toughness_pa_sqrt_m": float(values["fracture_toughness_pa_sqrt_m"]),
    }


def _record_result(result: PyFracRunResult, member_index: int, requested_time_s: float) -> dict[str, object]:
    return {
        "member_index": int(member_index),
        "requested_time_s": float(requested_time_s),
        "success": bool(result.success),
        "target_reached": bool(result.target_reached),
        "partial_progress": bool(result.partial_progress),
        "final_time_s": _finite(result.final_time_s),
        "runtime_s": _finite(result.runtime_seconds),
        "successful_time_steps": int(result.successful_time_steps),
        "failed_time_steps": int(result.failed_time_steps),
        "half_length_m": _finite(result.half_length_m),
        "maximum_width_m": _finite(result.maximum_width_m),
        "bottomhole_pressure_mpa": _finite(result.bottomhole_pressure_mpa),
        "net_pressure_mpa": _finite(result.net_pressure_mpa),
        "injected_volume_m3": _finite(result.injected_volume_m3),
        "fracture_volume_m3": _finite(result.fracture_volume_m3),
        "leakoff_volume_m3": _finite(result.leakoff_volume_m3),
        "mass_balance_relative_error": _finite(result.mass_balance_relative_error),
        "efficiency": _finite(result.efficiency),
        "checkpoint_id": result.checkpoint_id or "",
        "rollback_applied": bool(result.rollback_applied),
        "retry_count": int(result.retry_count),
        "error": result.error or "",
    }


def _failed_worker_result(session: PyFracNativeSession, started: float, error: str) -> PyFracRunResult:
    """Create a failure result without mutating the parent's native state."""

    return PyFracRunResult(
        half_length_m=float("nan"),
        max_aperture_mm=float("nan"),
        area_m2=float("nan"),
        volume_m3=float("nan"),
        net_pressure_mpa=float("nan"),
        bottomhole_pressure_mpa=float("nan"),
        front_geometry=[],
        runtime_seconds=time.perf_counter() - started,
        model_name="PyFrac",
        engine_mode="pyfrac_native_dynamic_isolated_worker",
        success=False,
        error=error,
        final_time_s=float(session.fracture.time),
        successful_time_steps=0,
        failed_time_steps=0,
        target_reached=False,
        partial_progress=False,
    )


def _result_from_worker_payload(payload: dict[str, object]) -> PyFracRunResult:
    """Rehydrate only the declared result fields from a worker JSON file."""

    names = {field.name for field in fields(PyFracRunResult)}
    values = {name: payload[name] for name in names if name in payload}
    values.setdefault("front_geometry", [])
    values.setdefault("model_name", "PyFrac")
    values.setdefault("engine_mode", "pyfrac_native_dynamic_isolated_worker")
    values.setdefault("success", False)
    values.setdefault("runtime_seconds", float(payload.get("worker_runtime_s", 0.0)))
    values.setdefault("error", None)
    return PyFracRunResult(**values)


def _dump_dill(path: Path, value: object) -> None:
    import dill

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        dill.dump(value, handle, -1)


def _load_dill(path: Path) -> object:
    import dill

    with path.open("rb") as handle:
        return dill.load(handle)


def _rebuild_session_from_worker_checkpoint(
    *,
    payload: dict[str, object],
    checkpoint_path: Path,
    adapter: PyFracAdapter,
    schedule: np.ndarray,
    native_initial_time_s: float,
    state_parameters: dict[str, float],
    args: argparse.Namespace,
    checkpoint_dir: Path,
) -> PyFracNativeSession:
    """Replace a parent session with the worker's last accepted state."""

    fracture_payload = _load_dill(checkpoint_path)
    if not isinstance(fracture_payload, dict) or fracture_payload.get("fracture") is None:
        raise RuntimeError(f"worker checkpoint has no fracture state: {checkpoint_path}")
    fracture = fracture_payload["fracture"]
    session = PyFracNativeSession(
        adapter,
        schedule,
        native_initial_time_s,
        **state_parameters,
        max_time_steps=args.max_time_steps_per_window,
        domain_time_s=float(args.mesh_domain_time_s or native_initial_time_s),
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval_s=args.checkpoint_interval_s,
        initial_fracture=fracture,
        initial_parameters=(
            dict(fracture_payload.get("last_parameters") or {})
            if isinstance(fracture_payload.get("last_parameters"), dict)
            else state_parameters
        ),
        initial_successful_steps=int(fracture_payload.get("successful_steps", 0)),
        initial_failed_steps=int(fracture_payload.get("failed_steps", 0)),
        initial_step_limit_s=float(
            fracture_payload.get("continuation_step_limit_s", args.initial_time_step_s)
        ),
        initial_consecutive_failures=int(fracture_payload.get("consecutive_failures", 0)),
        initial_success_streak=int(fracture_payload.get("continuation_success_streak", 0)),
        initial_front_metadata_repair_count=int(
            fracture_payload.get("front_metadata_repair_count", 0)
        ),
    )
    # Preserve the durable worker artifact in the final manifest.  The
    # constructor creates a new resume checkpoint as the in-memory anchor.
    session.disk_checkpoint_paths.append(str(checkpoint_path))
    return session


def _run_member_isolated(
    *,
    session: PyFracNativeSession,
    member_index: int,
    cycle_index: int,
    requested_time_s: float,
    schedule: np.ndarray,
    state_parameters: dict[str, float],
    adapter: PyFracAdapter,
    pyfrac_cfg: PyFracConfig,
    args: argparse.Namespace,
    native_initial_time_s: float,
    output: Path,
) -> tuple[PyFracRunResult, PyFracNativeSession]:
    """Advance one member with a killable process boundary.

    A worker timeout or exception only invalidates this member for this
    source point.  The caller decides whether to deactivate it after the
    configured consecutive-failure threshold.
    """

    started = time.perf_counter()
    worker_dir = output / "workers" / f"member_{member_index:03d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    initial_path = worker_dir / "initial_fracture.dill"
    schedule_path = worker_dir / "schedule.npy"
    spec_path = worker_dir / "request.json"
    result_path = worker_dir / "result.json"
    log_path = worker_dir / "worker.log"
    _dump_dill(initial_path, session.fracture)
    np.save(schedule_path, schedule)
    spec = {
        "project_root": str(PROJECT_ROOT),
        "pyfrac_config": asdict(pyfrac_cfg),
        "initial_fracture_path": str(initial_path),
        "schedule_path": str(schedule_path),
        "checkpoint_dir": str(output / "checkpoints" / f"member_{member_index:03d}"),
        "result_path": str(result_path),
        "target_time_s": float(requested_time_s),
        "native_initial_time_s": float(native_initial_time_s),
        "state_parameters": state_parameters,
        "initial_parameters": dict(session.last_parameters),
        "initial_successful_steps": int(session.total_successful_steps),
        "initial_failed_steps": int(session.total_failed_steps),
        "continuation_step_limit_s": float(session._continuation_step_limit_s),
        "consecutive_failures": int(session.consecutive_failures),
        "continuation_success_streak": int(session._continuation_success_streak),
        "front_metadata_repair_count": int(session.front_metadata_repair_count),
        "max_time_steps": int(args.max_time_steps_per_window),
        "domain_time_s": float(args.mesh_domain_time_s or native_initial_time_s),
        "member_index": int(member_index),
        "cycle_index": int(cycle_index),
    }
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    result_path.unlink(missing_ok=True)
    worker_script = Path(__file__).resolve().with_name("pyfrac_member_worker.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(DT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = None
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                [sys.executable, "-u", str(worker_script), "--spec", str(spec_path)],
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            try:
                return_code = process.wait(timeout=max(float(args.member_timeout_s), 0.1))
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
                return (
                    _failed_worker_result(
                        session,
                        started,
                        f"member worker timeout after {float(args.member_timeout_s):g}s; last checkpoint retained",
                    ),
                    session,
                )
        if not result_path.is_file():
            return (
                _failed_worker_result(
                    session,
                    started,
                    f"member worker exited with code {return_code}, result.json missing; see {log_path}",
                ),
                session,
            )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("worker result is not an object")
        if not bool(payload.get("success", False)):
            return (
                _result_from_worker_payload(payload),
                session,
            )
        checkpoint_value = payload.get("checkpoint_path")
        if not checkpoint_value:
            raise RuntimeError("worker reported success without an accepted checkpoint")
        checkpoint_path = Path(str(checkpoint_value)).resolve()
        if not checkpoint_path.is_file():
            raise RuntimeError(f"worker accepted checkpoint does not exist: {checkpoint_path}")
        rebuilt = _rebuild_session_from_worker_checkpoint(
            payload=payload,
            checkpoint_path=checkpoint_path,
            adapter=adapter,
            schedule=schedule,
            native_initial_time_s=native_initial_time_s,
            state_parameters=state_parameters,
            args=args,
            checkpoint_dir=output / "checkpoints" / f"member_{member_index:03d}",
        )
        return _result_from_worker_payload(payload), rebuilt
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        return (
            _failed_worker_result(session, started, f"isolated worker exception: {type(exc).__name__}: {exc}"),
            session,
        )


def _write_progress(
    output: Path,
    history: list[dict[str, object]],
    member_audit: list[dict[str, object]],
    skipped: list[dict[str, object]],
    updates: list[dict[str, object]],
    *,
    status: str,
    next_source_time_s: float,
    elapsed_s: float,
    target_time_s: float,
) -> None:
    pd.DataFrame(history).to_csv(output / "realtime_history_progress.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(member_audit).to_csv(output / "member_audit_progress.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(skipped).to_csv(output / "skipped_source_points_progress.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(updates).to_csv(output / "enkf_update_audit_progress.csv", index=False, encoding="utf-8-sig")
    payload = json.dumps(
        {
            "status": status,
            "next_source_time_s": float(next_source_time_s),
            "elapsed_wall_clock_s": float(elapsed_s),
            "target_time_s": float(target_time_s),
            "computed_points": int(len(history)),
            "skipped_points": int(len(skipped)),
            "accepted_enkf_updates": int(
                sum(
                    item.get("status") in {"updated", "updated_partial_valid_members"}
                    for item in updates
                )
            ),
            "checkpoint_dirs": sorted(str(path) for path in output.glob("checkpoints/member_*")),
        },
        ensure_ascii=False,
        indent=2,
        default=_json_default,
    )
    progress_path = output / "progress.json"
    # Windows Defender/indexers can briefly hold a file immediately after the
    # CSV writes above.  Do not lose a long experiment because a direct
    # write_text hit a transient sharing violation: write a sibling temp file,
    # replace atomically, and keep a readable fallback if the target remains
    # locked.  The checkpoint files remain the authoritative restart state.
    temporary_path = output / "progress.json.tmp"
    written = False
    for attempt in range(8):
        try:
            temporary_path.write_text(payload, encoding="utf-8")
            os.replace(temporary_path, progress_path)
            written = True
            break
        except PermissionError:
            temporary_path.unlink(missing_ok=True)
            if attempt < 7:
                time.sleep(0.05 * (attempt + 1))
    if not written:
        fallback_path = output / "progress_fallback.json"
        fallback_path.write_text(payload, encoding="utf-8")


def _write_figure(history: pd.DataFrame, path: Path) -> None:
    if history.empty:
        return
    # Matplotlib does not reliably choose a CJK font on Windows.  Register a
    # local fallback explicitly so Chinese titles, legends and axis labels do
    # not render as tofu boxes in exported figures.
    for font_path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False

    def column(name: str) -> pd.Series:
        # A real-time run may contain valid partial PyFrac points before the
        # first EnKF update.  Those rows legitimately have no posterior
        # parameter columns; plotting must not turn that into a fatal error.
        if name in history.columns:
            return pd.to_numeric(history[name], errors="coerce")
        return pd.Series(np.nan, index=history.index, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    t = column("computed_time_s")
    axes[0, 0].plot(t, column("observed_bottomhole_pressure_mpa"), "k.", label="观测井底压力")
    axes[0, 0].plot(t, column("pyfrac_prior_bottomhole_pressure_mpa"), "C3-o", label="PyFrac先验")
    axes[0, 0].set_ylabel("MPa")
    axes[0, 0].set_title("实时连续 PyFrac 压力")
    axes[0, 0].legend()
    axes[0, 1].plot(t, column("pyfrac_prior_bottomhole_pressure_error_mpa"), "C1-o", label="压力绝对误差")
    axes[0, 1].set_ylabel("MPa")
    axes[0, 1].set_title("PyFrac 观测残差")
    axes[0, 1].legend()
    axes[1, 0].plot(t, column("posterior_eprime_gpa"), "C2-o", label="E' / GPa")
    axes[1, 0].plot(t, column("posterior_min_stress_mpa"), "C0-o", label="最小水平应力 / MPa")
    axes[1, 0].set_title("EnKF 后验参数")
    axes[1, 0].legend()
    axes[1, 1].plot(t, column("pyfrac_half_length_m"), "C4-o", label="半缝长 / m")
    axes[1, 1].plot(t, column("pyfrac_max_width_m"), "C5-o", label="最大缝宽 / m")
    axes[1, 1].set_title("连续 PyFrac 状态")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("模型时间 / s")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    experiment_started = time.perf_counter()

    pressure, pressure_meta = load_stage_pressure_schedule(
        args.construction_pressure_xls,
        config=PressureModelConfig(step_seconds=1.0),
    )
    source_end = min(float(args.target_time_s), float(pressure["time_s"].max()))
    native_initial_time_s = float(args.native_initial_time_s)
    if args.warm_start_time_s is not None and not np.isclose(args.warm_start_time_s, native_initial_time_s):
        raise ValueError(
            "正式全流程已禁用暖启动：--warm-start-time-s 只能与原生初始时刻 1 s 一致"
        )
    if not np.isclose(native_initial_time_s, 1.0):
        raise ValueError("正式全流程原生初始时刻固定为 1 s，不允许用暖启动替代")
    if source_end < native_initial_time_s:
        raise ValueError("target-time-s must be at least the native initial time 1 s")
    # Fresh sessions deliberately build and resolve the t=1 s mesh first.
    # Later growth is handled by native PyFrac mesh extension; the old
    # warm-start-plus-600 s rolling-domain shortcut is not used here.
    mesh_domain_time_s = float(args.mesh_domain_time_s or native_initial_time_s)
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
        native_start_time_s=native_initial_time_s,
        initial_step_limit_s=max(float(args.initial_time_step_s), 1.0e-6),
        mesh_half_length_m=max(float(args.mesh_half_length_m), 1.0),
        max_time_steps=args.max_time_steps_per_window,
        dynamic_step_limit_s=args.dynamic_step_limit_s,
        front_advancing=args.front_advancing,
        projection_method=args.projection_method,
        adaptive_mesh_enabled=not args.disable_adaptive_mesh,
        enable_pyfrac_remeshing=not args.disable_pyfrac_remeshing,
        mesh_extension_enabled=bool(args.enable_mesh_extension or not args.disable_mesh_extension),
        mesh_extension_all_directions=bool(args.mesh_extension_all_directions),
        mesh_extension_factor=float(args.mesh_extension_factor),
        max_retries=args.max_native_retries,
        retry_time_step_factor=args.retry_time_step_factor,
        min_dynamic_step_s=args.min_dynamic_step_s,
        assimilation_relaxation=args.assimilation_relaxation,
        assimilation_max_parameter_step=tuple(args.assimilation_max_parameter_step),
    )
    adapter = PyFracAdapter(pyfrac_cfg, project_root=PROJECT_ROOT)
    resume_root = Path(args.resume_from).resolve() if args.resume_from else None
    requested_ensemble_size = int(args.ensemble_size)
    if requested_ensemble_size < 3:
        raise ValueError(
            "正式 PyFrac-EnKF 全流程至少需要 3 个同步有效成员；"
            "单成员只能进行前向计算，不能计算有效协方差或执行 EnKF 更新"
        )
    min_valid_members = int(args.min_valid_members)
    if min_valid_members < 3:
        raise ValueError("--min-valid-members must be at least 3")
    if min_valid_members > requested_ensemble_size:
        raise ValueError("--min-valid-members cannot exceed --ensemble-size")
    max_consecutive_member_failures = max(int(args.max_consecutive_member_failures), 1)
    resume_payloads: list[dict[str, object] | None] = [None] * requested_ensemble_size
    previous_elapsed = 0.0
    if resume_root is not None:
        if not resume_root.is_dir():
            raise FileNotFoundError(f"resume directory does not exist: {resume_root}")
        resume_payloads, ensemble, resumed_next_time, resumed_last_assimilation, previous_elapsed = _load_resume_state(
            resume_root,
            args.ensemble_size,
            args.seed,
            args.base_min_stress_mpa,
            args.max_mesh_elements,
            args.source_point_interval_s,
        )
    else:
        rng = np.random.default_rng(args.seed)
        prior_mean = np.asarray([0.0, 0.0, 0.0, args.base_min_stress_mpa, 0.0], dtype=float)
        spread = float(args.ensemble_spread_scale) * np.asarray(
            [0.06, 0.12, 0.08, 1.0, 0.08], dtype=float
        )
        if not np.isfinite(spread).all() or np.any(spread < 0.0):
            raise ValueError("--ensemble-spread-scale must be finite and non-negative")
        ensemble = clip_state(
            rng.normal(prior_mean, spread, size=(requested_ensemble_size, 5)),
            1,
        )
        resumed_next_time = float(native_initial_time_s + max(args.source_point_interval_s, 1.0))
        resumed_last_assimilation = float(native_initial_time_s)

    initial_schedule = make_schedule(pressure, rate, native_initial_time_s, args.injection_schedule_step_s)
    sessions: list[PyFracNativeSession] = []
    initialization_failures: list[dict[str, object]] = []
    for member_index, member in enumerate(ensemble):
        payload = resume_payloads[member_index] if member_index < len(resume_payloads) else None
        initial_fracture = payload.get("fracture") if payload else None
        initial_parameters = payload.get("last_parameters") if payload else None
        try:
            sessions.append(
                PyFracNativeSession(
                    adapter,
                    initial_schedule,
                    native_initial_time_s,
                    **_state_parameters(member, physical_cfg),
                    max_time_steps=args.max_time_steps_per_window,
                    domain_time_s=mesh_domain_time_s,
                    checkpoint_dir=checkpoint_root / f"member_{member_index:03d}",
                    checkpoint_interval_s=args.checkpoint_interval_s,
                    initial_fracture=initial_fracture,
                    initial_parameters=initial_parameters if isinstance(initial_parameters, dict) else None,
                    initial_successful_steps=int(payload.get("successful_steps", 0)) if payload else 0,
                    initial_failed_steps=int(payload.get("failed_steps", 0)) if payload else 0,
                    initial_step_limit_s=max(float(args.initial_time_step_s), 1.0e-6),
                )
            )
        except (Exception, SystemExit) as exc:
            initialization_failures.append(
                {
                    "member_index": int(member_index),
                    "status": "native_initialization_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if initialization_failures:
        # Do not continue with a partial ensemble and do not substitute a
        # 360 s checkpoint. A failed t=1 s native start is a failed full-run
        # initialization and must be visible to downstream reports.
        failure_summary = {
            "demo": "wall_clock_bounded_native_pyfrac_with_synchronized_enkf",
            "status": "native_full_run_failed",
            "initialization_status": "原生全流程启动失败",
            "warm_start_used": False,
            "native_initial_time_s": native_initial_time_s,
            "source_window_s": [1.0, source_end],
            "ensemble_size": requested_ensemble_size,
            "synchronized_member_requirement": 3,
            "initialization_failures": initialization_failures,
            "snapshot_used": False,
            "target_time_reached_by_all_members": False,
            "pyfrac_config": asdict(pyfrac_cfg),
            "pressure_meta": pressure_meta,
            "pyfrac_installation": adapter.verify_installation(),
            "outputs": {"summary": str(output / "summary.json")},
            "limitations": [
                "原生 PyFrac 未能从 t=1 s 建立完整成员集合；未使用暖启动结果替代。",
                "由于没有形成至少 3 个同步有效成员，本次未执行 EnKF 协方差更新。",
            ],
        }
        (output / "initialization_failures.json").write_text(
            json.dumps(initialization_failures, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        failure_summary["outputs"]["initialization_failures"] = str(output / "initialization_failures.json")
        (output / "summary.json").write_text(
            json.dumps(failure_summary, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return failure_summary

    # The live clock starts after the t=1 s native initial states and ensemble
    # have been initialized. Initialization cost is reported separately.
    run_started = time.perf_counter()
    next_source_time = float(resumed_next_time)
    # A resumed realtime run continues its source clock from the next
    # unprocessed point. Anchoring this to the initial state would make the runner
    # process almost every historical point after a restart instead of
    # explicitly skipping points that arrive while PyFrac is busy.
    source_clock_origin = float(next_source_time)
    last_assimilation_time = float(resumed_last_assimilation)
    history: list[dict[str, object]] = []
    member_audit: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    updates: list[dict[str, object]] = []
    memory_samples: list[dict[str, object]] = []
    # One ensemble member is one complete native PyFrac realization under one
    # perturbed parameter vector, not one scalar parameter.  Partial mode
    # keeps failed members out of the current covariance calculation while
    # retaining their audit trail.  Formal mode still requires every member.
    member_active = np.ones(requested_ensemble_size, dtype=bool)
    member_failure_streak = np.zeros(requested_ensemble_size, dtype=int)
    # A posterior is only a candidate until it successfully drives the native
    # solver through one continuation window.  Retain the latest parameter
    # vector that produced a valid native state so an invalid unbounded EnKF
    # analysis can be rejected by state health and retried from the retained
    # native checkpoint.  This is not a parameter-jump limiter.
    member_last_valid_parameters = np.asarray(ensemble, dtype=float).copy()
    member_parameter_rollback_count = np.zeros(requested_ensemble_size, dtype=int)
    peak_process_commit_bytes = 0.0
    memory_guard_limit_bytes = max(float(args.max_process_commit_gb), 0.0) * (1024.0**3)
    if resume_root is not None:
        for filename, target in (
            ("realtime_history_progress.csv", history),
            ("member_audit_progress.csv", member_audit),
            ("skipped_source_points_progress.csv", skipped),
            ("enkf_update_audit_progress.csv", updates),
        ):
            source = resume_root / filename
            if source.is_file():
                try:
                    frame = pd.read_csv(source)
                except pd.errors.EmptyDataError:
                    frame = pd.DataFrame()
                if not frame.empty:
                    target.extend(frame.replace({np.nan: None}).to_dict("records"))
    status = "completed"
    cycle_index = int(max((int(row.get("cycle_index", -1)) for row in history), default=-1) + 1)
    total_wall_clock_budget = float(args.wall_clock_budget_s)
    remaining_wall_clock_budget = max(total_wall_clock_budget - previous_elapsed, 0.0) if resume_root else total_wall_clock_budget
    np.save(output / "ensemble_state_progress.npy", ensemble)

    try:
        while next_source_time <= source_end + 1.0e-8:
            elapsed_before = time.perf_counter() - run_started
            if elapsed_before >= remaining_wall_clock_budget:
                status = "wall_clock_budget_exhausted"
                break
            requested_time = float(next_source_time)
            cycle_started = time.perf_counter()
            schedule = make_schedule(pressure, rate, requested_time, args.injection_schedule_step_s)
            results: list[tuple[int, PyFracRunResult]] = []
            memory_guard_triggered = False

            active_indices = [int(index) for index in np.flatnonzero(member_active)]

            def _advance_one_member(
                member_index: int,
            ) -> tuple[int, PyFracRunResult, PyFracNativeSession, float]:
                """Advance one member; isolated workers make this concurrent-safe."""

                member = ensemble[member_index]
                session = sessions[member_index]
                # A real-time source clock can be far ahead of the retained
                # native state because PyFrac is slower than one source
                # second.  Never pass that backlog directly to the legacy
                # Controller: a one-step/partial run would then be reported
                # as "stopped before target" and repeatedly deactivate a
                # healthy member.  The source time remains the observation
                # timestamp; the native target is only the next computable
                # continuation window.
                current_native_time = float(session.fracture.time)
                native_window = max(
                    float(session._continuation_step_limit_s),
                    float(args.min_dynamic_step_s),
                    1.0e-6,
                ) * max(int(args.max_time_steps_per_window), 1)
                native_target_time = min(
                    requested_time,
                    current_native_time + native_window,
                )
                if native_target_time <= current_native_time + 1.0e-8:
                    native_target_time = current_native_time + max(
                        float(args.min_dynamic_step_s), 1.0e-6
                    )
                member_parameters = _state_parameters(member, physical_cfg)
                if args.disable_member_process_isolation:
                    result = session.advance_to(
                        native_target_time,
                        schedule,
                        **member_parameters,
                        allow_partial=True,
                    )
                    updated_session = session
                else:
                    result, updated_session = _run_member_isolated(
                        session=session,
                        member_index=member_index,
                        cycle_index=cycle_index,
                        requested_time_s=native_target_time,
                        schedule=schedule,
                        state_parameters=member_parameters,
                        adapter=adapter,
                        pyfrac_cfg=pyfrac_cfg,
                        args=args,
                        native_initial_time_s=native_initial_time_s,
                        output=output,
                    )
                    # The parent never mutates the native object while the
                    # worker is running. Replace it only after an accepted
                    # checkpoint has been validated and reloaded.
                    pass
                return member_index, result, updated_session, float(native_target_time)

            # Native PyFrac is the expensive part.  Every isolated worker has
            # its own serialized fracture state and worker directory, so the
            # ensemble can be advanced concurrently.  Parent-process mode is
            # intentionally retained as sequential diagnostic behavior.
            member_outcomes: dict[int, tuple[PyFracRunResult, PyFracNativeSession, float]] = {}
            if args.disable_member_process_isolation or len(active_indices) <= 1:
                for member_index in active_indices:
                    try:
                        _, result, updated_session, native_target_time = _advance_one_member(member_index)
                    except BaseException as exc:
                        result = _failed_worker_result(
                            sessions[member_index],
                            time.perf_counter(),
                            f"member wrapper exception: {type(exc).__name__}: {exc}",
                        )
                        updated_session = sessions[member_index]
                        native_target_time = float(sessions[member_index].fracture.time)
                    member_outcomes[member_index] = (result, updated_session, native_target_time)
            else:
                with ThreadPoolExecutor(max_workers=len(active_indices)) as executor:
                    futures = {
                        executor.submit(_advance_one_member, member_index): member_index
                        for member_index in active_indices
                    }
                    for future in as_completed(futures):
                        member_index = futures[future]
                        try:
                            _, result, updated_session, native_target_time = future.result()
                        except BaseException as exc:
                            result = _failed_worker_result(
                                sessions[member_index],
                                time.perf_counter(),
                                f"member wrapper exception: {type(exc).__name__}: {exc}",
                            )
                            updated_session = sessions[member_index]
                            native_target_time = float(sessions[member_index].fracture.time)
                        member_outcomes[member_index] = (result, updated_session, native_target_time)

            for member_index in active_indices:
                if member_index not in member_outcomes:
                    continue
                result, updated_session, native_target_time = member_outcomes[member_index]
                sessions[member_index] = updated_session
                results.append((member_index, result))
                audit_row = _record_result(result, member_index, requested_time)
                audit_row["execution_mode"] = (
                    "parent_process" if args.disable_member_process_isolation else "isolated_worker"
                )
                audit_row["native_target_time_s"] = float(native_target_time)
                audit_row["source_observation_time_s"] = float(requested_time)
                audit_row["worker_timeout_s"] = float(args.member_timeout_s)
                parameter_rollback_applied = False
                if result.success:
                    member_failure_streak[member_index] = 0
                    # The current posterior (if any) has now survived one
                    # native continuation window and becomes the rollback
                    # anchor for this member.
                    member_last_valid_parameters[member_index] = ensemble[member_index].copy()
                else:
                    member_failure_streak[member_index] += 1
                    candidate = np.asarray(ensemble[member_index], dtype=float).copy()
                    safe = np.asarray(member_last_valid_parameters[member_index], dtype=float)
                    if (
                        not args.disable_failed_member_recovery
                        and np.isfinite(safe).all()
                        and not np.allclose(candidate, safe, rtol=1.0e-12, atol=1.0e-15)
                    ):
                        # The worker never mutates the parent session on
                        # failure. Restoring the last valid parameters is
                        # therefore enough to retry from that checkpoint.
                        ensemble[member_index] = safe
                        member_parameter_rollback_count[member_index] += 1
                        parameter_rollback_applied = True
                        audit_row["failure_recovery"] = "restored_last_valid_parameters"
                    else:
                        audit_row["failure_recovery"] = "none"
                    # An isolated worker is a copy of the parent session.  A
                    # retry-time-step reduction performed inside that worker
                    # is therefore lost when the worker times out.  Carry the
                    # reduced continuation limit back to the parent so the
                    # next retry actually starts with an easier native step.
                    # This is numerical continuation recovery, not an EnKF
                    # parameter jump limiter: parameter updates remain
                    # unbounded and are only rejected if they fail native
                    # state-health validation.
                    recovery_session = sessions[member_index]
                    current_limit = max(
                        float(getattr(recovery_session, "_continuation_step_limit_s", 0.0)),
                        float(args.min_dynamic_step_s),
                        1.0e-6,
                    )
                    floor = max(float(args.min_dynamic_step_s), 1.0e-6)
                    reduced_limit = max(
                        floor,
                        current_limit * float(args.retry_time_step_factor),
                    )
                    if reduced_limit < current_limit - 1.0e-15:
                        recovery_session._continuation_step_limit_s = reduced_limit
                        audit_row["failure_recovery"] = (
                            f"{audit_row['failure_recovery']};reduced_native_step_limit"
                        )
                    # Give the reduced-step recovery enough attempts to reach
                    # its configured floor.  A member is only deactivated
                    # after it is still failing at that floor, so one timeout
                    # does not permanently remove an otherwise recoverable
                    # ensemble member.
                    at_floor = reduced_limit <= floor + 1.0e-15
                    if (
                        member_failure_streak[member_index] >= max_consecutive_member_failures
                        and at_floor
                    ):
                        member_active[member_index] = False
                audit_row["parameter_rollback_applied"] = parameter_rollback_applied
                audit_row["member_failure_streak"] = int(member_failure_streak[member_index])
                member_audit.append(audit_row)
                memory = _process_memory_snapshot()
                peak_process_commit_bytes = max(
                    peak_process_commit_bytes,
                    float(memory.get("commit_bytes", 0.0)) if np.isfinite(memory.get("commit_bytes", np.nan)) else 0.0,
                )
                memory_samples.append(
                    {
                        "cycle_index": int(cycle_index),
                        "member_index": int(member_index),
                        "requested_time_s": requested_time,
                        "wall_clock_elapsed_s": float(time.perf_counter() - run_started),
                        **memory,
                    }
                )
                if (
                    memory_guard_limit_bytes > 0.0
                    and np.isfinite(memory.get("commit_bytes", np.nan))
                    and float(memory["commit_bytes"]) > memory_guard_limit_bytes
                ):
                    status = "memory_guard_triggered"
                    memory_guard_triggered = True
                    break

            pd.DataFrame(memory_samples).to_csv(
                output / "memory_progress.csv", index=False, encoding="utf-8-sig"
            )
            if memory_guard_triggered:
                _write_progress(
                    output,
                    history,
                    member_audit,
                    skipped,
                    updates,
                    status=status,
                    next_source_time_s=requested_time,
                    elapsed_s=previous_elapsed + (time.perf_counter() - run_started),
                    target_time_s=source_end,
                )
                break

            successful = [
                (member_index, item)
                for member_index, item in results
                if item.success and np.isfinite(item.bottomhole_pressure_mpa)
            ]
            if successful:
                # A point is represented by the earliest valid member state so
                # no member is allowed to use a future state to claim coverage.
                computed_time = float(min(item.final_time_s for _, item in successful))
                comparable = [
                    (member_index, item)
                    for member_index, item in successful
                    if abs(float(item.final_time_s) - computed_time)
                    <= max(float(args.source_point_interval_s), 1.0)
                ]
                result_times = [float(item.final_time_s) for _, item in results]
                result_pressures = [float(item.bottomhole_pressure_mpa) for _, item in results]
                synchronized = bool(
                    len(results) == requested_ensemble_size
                    and len(results) == len(sessions)
                    and len(successful) == requested_ensemble_size
                    and all(np.isfinite(value) for value in result_times)
                    and np.isfinite(
                        float(np.mean(result_pressures))
                    )
                    and np.isfinite(
                        float(
                            np.mean(
                                [
                                    observation_at(
                                        pressure,
                                        float(item.final_time_s),
                                        "bottomhole_pressure_mpa",
                                    )
                                    for _, item in results
                                ]
                            )
                        )
                    )
                    and max(result_times) - min(result_times) <= 1.0e-6
                )
                observed = float(observation_at(pressure, computed_time, "bottomhole_pressure_mpa"))
                prior_bhp = float(np.mean([item.bottomhole_pressure_mpa for _, item in comparable])) if comparable else float("nan")
                prior_length = float(np.mean([item.half_length_m for _, item in comparable])) if comparable else float("nan")
                prior_width = float(np.mean([item.maximum_width_m for _, item in comparable])) if comparable else float("nan")
                prior_mass = float(np.nanmax([item.mass_balance_relative_error for _, item in comparable])) if comparable else float("nan")
                row: dict[str, object] = {
                    "cycle_index": int(cycle_index),
                    "requested_source_time_s": requested_time,
                    "computed_time_s": computed_time,
                    "wall_clock_elapsed_s": float(time.perf_counter() - run_started),
                    "cycle_runtime_s": float(time.perf_counter() - cycle_started),
                    "computed_point": True,
                    "partial_member_count": int(sum(item.partial_progress for _, item in comparable)),
                    "native_success_members": int(len(comparable)),
                    "native_active_members": int(member_active.sum()),
                    "native_synchronized_members": int(len(results) if synchronized else 0),
                    "native_members_synchronized": synchronized,
                    "observed_bottomhole_pressure_mpa": observed,
                    "pyfrac_prior_bottomhole_pressure_mpa": prior_bhp,
                    "pyfrac_prior_bottomhole_pressure_error_mpa": abs(prior_bhp - observed) if np.isfinite(prior_bhp) else np.nan,
                    "pyfrac_half_length_m": prior_length,
                    "pyfrac_max_width_m": prior_width,
                    "pyfrac_mass_balance_relative_error": prior_mass,
                    "enkf_update_status": "not_due",
                    "enkf_innovation_mpa": np.nan,
                    "enkf_accepted": False,
                    **state_record(
                        "prior",
                        ensemble[member_active].mean(axis=0) if member_active.any() else ensemble.mean(axis=0),
                        physical_cfg,
                        1,
                    ),
                }

                due = computed_time - last_assimilation_time >= float(args.assimilation_interval_s)
                if due:
                    partial_eligible = bool(
                        args.allow_partial_ensemble_update
                        and len(comparable) >= min_valid_members
                    )
                    update_eligible = bool(synchronized or partial_eligible)
                    if update_eligible:
                        update_members = results if synchronized else comparable
                        update_indices = np.asarray(
                            [member_index for member_index, _ in update_members],
                            dtype=int,
                        )
                        predicted = np.asarray(
                            [item.bottomhole_pressure_mpa for _, item in update_members],
                            dtype=float,
                        )
                        observations = np.asarray(
                            [
                                observation_at(
                                    pressure,
                                    float(item.final_time_s),
                                    "bottomhole_pressure_mpa",
                                )
                                for _, item in update_members
                            ],
                            dtype=float,
                        )
                        # Exploratory partial mode uses only finite members
                        # that are close to the same native time. Failed
                        # members are never filled with zero or copied as if
                        # they had produced a valid observation.
                        prior_for_update = ensemble[update_indices].copy()
                        updated, gain = denkf_update(
                            prior_for_update,
                            predicted[:, None],
                            np.asarray([float(np.mean(observations))]),
                            np.asarray([args.ensemble_pressure_noise_mpa]),
                            covariance_inflation=1.01,
                        )
                        relaxed = relaxed_update(
                            prior_for_update,
                            clip_state(updated, 1),
                            # This run is the requested unbounded-update
                            # control: do not limit a single parameter jump
                            # and do not blend the analysis back toward the
                            # prior. ``clip_state`` only prevents physically
                            # invalid/NaN states; it is not a jump limiter.
                            relaxation=1.0,
                            max_step=None,
                        )
                        posterior = clip_state(relaxed.state, 1)
                        innovation = float(np.mean(observations - predicted))
                        last_assimilation_time = computed_time
                        effective_rank = int(
                            np.linalg.matrix_rank(
                                prior_for_update - prior_for_update.mean(axis=0)
                            )
                        )
                        update_status = (
                            "updated" if synchronized else "updated_partial_valid_members"
                        )
                        update_row = {
                            "cycle_index": cycle_index,
                            "time_s": computed_time,
                            "status": update_status,
                            "valid_members": int(len(update_members)),
                            "requested_members": int(requested_ensemble_size),
                            "member_indices": [int(index) for index in update_indices],
                            "inactive_member_indices": [
                                int(index) for index in np.flatnonzero(~member_active)
                            ],
                            "effective_covariance_rank": effective_rank,
                            "innovation_mpa": innovation,
                            "prior_mean_abs_error_mpa": float(
                                abs(np.mean(predicted) - np.mean(observations))
                            ),
                            "parameter_update_mode": "unbounded_control",
                            "mean_abs_kalman_gain": float(np.mean(np.abs(gain))),
                            "raw_update_norm": float(relaxed.raw_update_norm),
                            "applied_update_norm": float(relaxed.applied_update_norm),
                            "step_clipped_components": int(relaxed.clipped_components),
                        }
                        updates.append(update_row)
                        ensemble[update_indices] = posterior
                        row.update(
                            {
                                "enkf_update_status": update_status,
                                "enkf_innovation_mpa": innovation,
                                "enkf_accepted": True,
                                "enkf_valid_members": int(len(update_members)),
                                "enkf_effective_covariance_rank": effective_rank,
                                **state_record(
                                    "posterior",
                                    ensemble[member_active].mean(axis=0)
                                    if member_active.any()
                                    else ensemble.mean(axis=0),
                                    physical_cfg,
                                    1,
                                ),
                            }
                        )
                    else:
                        updates.append(
                            {
                                "cycle_index": cycle_index,
                                "time_s": computed_time,
                                "status": "skipped_not_all_members_synchronized",
                                "valid_members": len(comparable),
                                "requested_members": requested_ensemble_size,
                                "member_final_times_s": {
                                    str(member_index): _finite(item.final_time_s)
                                    for member_index, item in results
                                },
                                "partial_mode_enabled": bool(args.allow_partial_ensemble_update),
                                "min_valid_members": int(min_valid_members),
                            }
                        )
                        row["enkf_update_status"] = "skipped_not_all_members_synchronized"
                else:
                    row.update(
                        state_record(
                            "posterior",
                            ensemble[member_active].mean(axis=0)
                            if member_active.any()
                            else ensemble.mean(axis=0),
                            physical_cfg,
                            1,
                        )
                    )
                history.append(row)
            else:
                history.append(
                    {
                        "cycle_index": int(cycle_index),
                        "requested_source_time_s": requested_time,
                        "computed_time_s": np.nan,
                        "wall_clock_elapsed_s": float(time.perf_counter() - run_started),
                        "cycle_runtime_s": float(time.perf_counter() - cycle_started),
                        "computed_point": False,
                        "native_success_members": 0,
                        "native_active_members": int(member_active.sum()),
                        "enkf_update_status": "not_updated_no_valid_native_state",
                    }
                )

            # Once every member has been disabled after its configured
            # consecutive-failure threshold, there is no forward state left
            # to compute.  Do not let the wall-clock source cursor race through
            # the remaining historical timestamps and make the run look like
            # it covered the source window.  Persist the exact stopping point
            # so a resume or report can distinguish a solver exhaustion from
            # a completed 4435 s experiment.
            if not member_active.any():
                status = "native_no_active_members"
                elapsed_after = time.perf_counter() - run_started
                _write_progress(
                    output,
                    history,
                    member_audit,
                    skipped,
                    updates,
                    status=status,
                    next_source_time_s=requested_time + max(float(args.source_point_interval_s), 1.0),
                    elapsed_s=previous_elapsed + elapsed_after,
                    target_time_s=source_end,
                )
                np.save(output / "ensemble_state_progress.npy", ensemble)
                break

            # Source arrivals are tied to the wall clock.  Points arriving
            # while the ensemble is being advanced are explicitly skipped.
            elapsed_after = time.perf_counter() - run_started
            next_by_clock = source_clock_origin + max(args.source_point_interval_s, 1.0) + np.floor(
                elapsed_after / max(float(args.source_point_interval_s), 1.0e-9)
            ) * max(float(args.source_point_interval_s), 1.0)
            next_by_clock = max(float(next_by_clock), requested_time + max(float(args.source_point_interval_s), 1.0))
            skipped_end = min(float(next_by_clock), source_end + max(float(args.source_point_interval_s), 1.0))
            skipped_start = requested_time + max(float(args.source_point_interval_s), 1.0)
            cursor = skipped_start
            while cursor < skipped_end - 1.0e-8 and cursor <= source_end + 1.0e-8:
                skipped.append(
                    {
                        "source_time_s": float(cursor),
                        "reason": "solver_busy_when_source_point_arrived",
                        "wall_clock_elapsed_s": float(elapsed_after),
                    }
                )
                cursor += max(float(args.source_point_interval_s), 1.0)
            next_source_time = float(next_by_clock)
            cycle_index += 1
            _write_progress(
                output,
                history,
                member_audit,
                skipped,
                updates,
                status="running",
                next_source_time_s=next_source_time,
                elapsed_s=previous_elapsed + elapsed_after,
                target_time_s=source_end,
            )
            np.save(output / "ensemble_state_progress.npy", ensemble)
    except KeyboardInterrupt:
        status = "interrupted"

    elapsed = time.perf_counter() - run_started
    total_elapsed = previous_elapsed + elapsed
    history_frame = pd.DataFrame(history)
    member_frame = pd.DataFrame(member_audit)
    skipped_frame = pd.DataFrame(skipped)
    update_frame = pd.DataFrame(updates)
    history_frame.to_csv(output / "realtime_history.csv", index=False, encoding="utf-8-sig")
    member_frame.to_csv(output / "member_audit.csv", index=False, encoding="utf-8-sig")
    skipped_frame.to_csv(output / "skipped_source_points.csv", index=False, encoding="utf-8-sig")
    update_frame.to_csv(output / "enkf_update_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(memory_samples).to_csv(output / "memory.csv", index=False, encoding="utf-8-sig")
    _write_figure(history_frame[history_frame.get("computed_point", pd.Series(dtype=bool)) == True] if not history_frame.empty else history_frame, output / "pyfrac_enkf_realtime.png")

    checkpoint_events: list[dict[str, object]] = []
    checkpoint_paths: list[str] = []
    for member_index, session in enumerate(sessions):
        checkpoint_paths.extend(session.disk_checkpoint_paths)
        checkpoint_events.extend({"member_index": member_index, **event} for event in session.checkpoints.events)
    pd.DataFrame(checkpoint_events).to_csv(output / "checkpoint_events.csv", index=False, encoding="utf-8-sig")
    (output / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "engine_mode": "pyfrac_native_dynamic_continuation",
                "checkpoint_count": len(checkpoint_paths),
                "paths": checkpoint_paths,
                "restart_note": "Each dill file contains a retained native PyFrac Fracture state and parameter metadata; snapshot mode is not used.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    final_times = [float(session.fracture.time) for session in sessions]
    native_time_reached_by_all_members = bool(
        final_times and min(final_times) >= source_end - 1.0e-6
    )
    # The experiment has two clocks by design: the source/field clock and the
    # retained PyFrac native clock.  A real-time run may finish its source
    # window while the native solver has computed only a sparse sequence of
    # accepted states.  Preserve that distinction in the result instead of
    # calling a valid wall-clock experiment an unexplained "incomplete" run.
    source_window_completed_by_wall_clock = bool(
        status == "completed"
        or total_elapsed >= max(float(source_end) - 1.0, 0.0)
    )
    if not native_time_reached_by_all_members:
        if source_window_completed_by_wall_clock and status in {
            "completed",
            "wall_clock_budget_exhausted",
        }:
            status = "source_window_completed_native_partial"
        elif status == "completed":
            status = "native_full_run_incomplete"
    computed = history_frame[history_frame.get("computed_point", pd.Series(dtype=bool)) == True] if not history_frame.empty else history_frame
    native_throughput = classify_throughput(
        total_elapsed,
        min(final_times) if final_times else None,
        args.throughput_target_ratios,
    )
    summary: dict[str, object] = {
        "demo": "wall_clock_bounded_native_pyfrac_with_synchronized_enkf",
        "status": status,
        "initialization_status": "原生全流程启动成功",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "startup_seconds": float(run_started - experiment_started),
        "elapsed_wall_clock_seconds": float(total_elapsed),
        "wall_clock_budget_s": float(args.wall_clock_budget_s),
        "remaining_wall_clock_budget_s": float(max(remaining_wall_clock_budget - elapsed, 0.0)),
        "resumed_from": str(resume_root) if resume_root is not None else None,
        "source_window_s": [1.0, source_end],
        "source_window_completed_by_wall_clock": source_window_completed_by_wall_clock,
        "native_time_reached_by_all_members": native_time_reached_by_all_members,
        "clock_semantics": (
            "source window follows wall clock; busy source points are skipped; "
            "only accepted native PyFrac states enter the history"
        ),
        "warm_start_used": False,
        "native_initial_time_s": native_initial_time_s,
        "mesh_domain_time_s": mesh_domain_time_s,
        "native_dynamic_from_initial_time": True,
        "initial_time_step_limit_s": float(args.initial_time_step_s),
        "parameter_update_limits_enabled": False,
        "snapshot_used": False,
        "member_process_isolation": not bool(args.disable_member_process_isolation),
        "member_timeout_s": float(args.member_timeout_s),
        "worker_note": (
            "每个原生成员在可终止 worker 中推进；超时只保留该成员最近检查点，"
            "不会阻塞其他成员。"
            if not args.disable_member_process_isolation
            else "诊断模式：成员在主进程内推进。"
        ),
        "computed_point_count": int(len(computed)),
        "skipped_source_point_count": int(len(skipped)),
        "first_computed_time_s": _finite(computed["computed_time_s"].min()) if not computed.empty else None,
        "last_computed_time_s": _finite(computed["computed_time_s"].max()) if not computed.empty else None,
        "final_native_time_by_member_s": final_times,
        "ensemble_size": int(len(ensemble)),
        "initial_ensemble_spread_scale": float(args.ensemble_spread_scale),
        "partial_ensemble_update_enabled": bool(args.allow_partial_ensemble_update),
        "min_valid_members": int(min_valid_members),
        "max_consecutive_member_failures": int(max_consecutive_member_failures),
        "failed_member_recovery_enabled": not bool(args.disable_failed_member_recovery),
        "member_parameter_rollback_count": int(member_parameter_rollback_count.sum()),
        "member_parameter_rollback_count_by_member": [
            int(value) for value in member_parameter_rollback_count
        ],
        "final_active_member_count": int(member_active.sum()),
        "final_inactive_member_indices": [
            int(index) for index in np.flatnonzero(~member_active)
        ],
        "enkf_update_attempt_count": int(len(updates)),
        "enkf_accepted_update_count": int(
            sum(
                item.get("status")
                in {"updated", "updated_partial_valid_members"}
                for item in updates
            )
        ),
        "enkf_partial_update_count": int(
            sum(item.get("status") == "updated_partial_valid_members" for item in updates)
        ),
        "checkpoint_count": int(len(checkpoint_paths)),
        "native_success_rate": float(member_frame["success"].mean()) if not member_frame.empty else 0.0,
        "rollback_event_count": int(sum(event.get("event") == "restored" for event in checkpoint_events)),
        "disk_checkpoint_save_failures": int(sum(event.get("event") == "disk_save_failed" for event in checkpoint_events)),
        "target_time_reached_by_all_members": native_time_reached_by_all_members,
        "performance": {
            "gate_definition": "native wall-clock runtime / minimum retained native simulated time; lower is better",
            "target_ratios": [float(value) for value in args.throughput_target_ratios],
            "native_throughput": native_throughput,
            "required_throughput_ratio": args.required_throughput_ratio,
            "formal_gate": evaluate_throughput_requirement(
                total_elapsed,
                min(final_times) if final_times else None,
                args.required_throughput_ratio,
            ),
            "realtime_source_clock_ratio": classify_throughput(
                total_elapsed,
                source_end,
                args.throughput_target_ratios,
            ),
        },
        "max_mesh_elements": int(args.max_mesh_elements),
        "memory_guard_limit_gb": float(args.max_process_commit_gb),
        "peak_process_commit_gb": float(peak_process_commit_bytes / (1024.0**3)),
        "physical_state": ["E_prime", "C_L", "mu", "sigma_min", "K_IC"],
        "pyfrac_config": asdict(pyfrac_cfg),
        "pressure_meta": pressure_meta,
        "pyfrac_installation": adapter.verify_installation(),
        "outputs": {
            "history": str(output / "realtime_history.csv"),
            "member_audit": str(output / "member_audit.csv"),
            "skipped_source_points": str(output / "skipped_source_points.csv"),
            "enkf_update_audit": str(output / "enkf_update_audit.csv"),
            "figure": str(output / "pyfrac_enkf_realtime.png"),
            "checkpoint_events": str(output / "checkpoint_events.csv"),
            "checkpoint_manifest": str(output / "checkpoint_manifest.json"),
            "memory": str(output / "memory.csv"),
        },
        "limitations": [
            "当前为单簇平面 PyFrac 原生连续状态，不是六簇原生水平井求解。",
            "PyFrac 从 t=1 s 原生初始状态开始；不使用暖启动结果替代全流程。",
            "运行按 4435 s 墙钟预算执行，忙时到达的源数据点被记录为跳过，不回放。",
            (
                "正式模式要求全部成员同步；探索模式可在满足最小有效成员数且时间一致性门控后，"
                "仅使用有效成员更新，失败成员不填充、不伪造。"
            ),
            "本版本取消单步参数跳变限幅和更新松弛，保留物理有效性裁剪；每次更新前后残差写入审计。",
            "正式全流程从 t=1 s 建立原生初始状态；若无法启动则报告原生全流程启动失败，不使用暖启动结果替代。",
            "当前 EnKF 观测为井底压力；DAS 分簇观测尚未进入原生 PyFrac 内部状态方程。",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _write_progress(
        output,
        history,
        member_audit,
        skipped,
        updates,
        status=status,
        next_source_time_s=next_source_time,
        elapsed_s=total_elapsed,
        target_time_s=source_end,
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2, default=_json_default))
