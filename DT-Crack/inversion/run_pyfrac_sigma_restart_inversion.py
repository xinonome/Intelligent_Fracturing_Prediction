"""Restart-from-initial PyFrac inversion for one physical parameter.

This entry point deliberately answers a narrower question than the realtime
EnKF runner:

    Can native PyFrac be used as a forward operator in a one-parameter
    inversion when every candidate is rebuilt from the explicit t=1 s state?

The inverted parameter is only ``sigma_min``.  Each candidate is evaluated in
an isolated worker and receives a fresh :class:`PyFracAdapter`; no candidate
inherits a fracture, pressure field, leak-off history, or front metadata from
another candidate.  A candidate is valid only if native PyFrac reaches the
requested target time.  Partial native states are retained as diagnostics,
but are never scored as an inversion result.

The default wall-clock budget is ten minutes.  This is a diagnostic budget,
not a claim that PyFrac has completed ten minutes of physical time.  Set
``--wall-clock-budget-s`` explicitly for a longer experiment.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DT_ROOT.parent
if str(DT_ROOT) not in sys.path:
    sys.path.insert(0, str(DT_ROOT))

from data_fusion.pressure_schedule_adapter import PressureModelConfig, load_stage_pressure_schedule  # noqa: E402
from forward_models.pyfrac_adapter import PyFracAdapter  # noqa: E402
from forward_models.pyfrac_config import PyFracConfig  # noqa: E402
from inversion.run_pyfrac_enkf import derive_injection_rate, make_schedule, observation_at  # noqa: E402
from inversion.performance_tiers import (  # noqa: E402
    DEFAULT_THROUGHPUT_TIERS,
    classify_throughput,
    evaluate_throughput_requirement,
)


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _candidate_stresses(
    base_stress_mpa: float,
    offsets_mpa: list[float],
    lower_mpa: float,
    upper_mpa: float,
) -> list[float]:
    """Return deterministic, unique candidate stresses in MPa."""

    if not np.isfinite([base_stress_mpa, lower_mpa, upper_mpa]).all():
        raise ValueError("stress search values must be finite")
    if lower_mpa >= upper_mpa:
        raise ValueError("stress lower bound must be smaller than upper bound")
    values: list[float] = []
    for offset in offsets_mpa:
        candidate = float(np.clip(base_stress_mpa + float(offset), lower_mpa, upper_mpa))
        if not any(np.isclose(candidate, previous, rtol=0.0, atol=1.0e-9) for previous in values):
            values.append(candidate)
    return values


def _pressure_objective(predicted_mpa: object, observed_mpa: float) -> float | None:
    """Absolute bottom-hole-pressure error used by the scalar inversion."""

    try:
        predicted = float(predicted_mpa)
        observed = float(observed_mpa)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(predicted) or not np.isfinite(observed):
        return None
    return abs(predicted - observed)


def _normalise_config_values(values: dict[str, Any]) -> dict[str, Any]:
    # ``PyFracConfig.to_dict`` also exposes the derived ``e_prime_pa`` value;
    # it is useful metadata but is not a dataclass constructor argument.
    allowed = set(PyFracConfig.__dataclass_fields__)
    result = {key: value for key, value in values.items() if key in allowed}
    if "mesh_extension_directions" in result:
        result["mesh_extension_directions"] = tuple(bool(value) for value in result["mesh_extension_directions"])
    if "assimilation_max_parameter_step" in result:
        result["assimilation_max_parameter_step"] = tuple(
            float(value) for value in result["assimilation_max_parameter_step"]
        )
    return result


def _run_worker(spec_path: Path) -> int:
    """Evaluate one fresh candidate in a process that can be terminated."""

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    result_path = Path(spec["result_path"]).resolve()
    started = time.perf_counter()
    try:
        config = PyFracConfig(**_normalise_config_values(dict(spec["pyfrac_config"])))
        adapter = PyFracAdapter(config, project_root=Path(spec["project_root"]).resolve())
        schedule = np.load(Path(spec["schedule_path"]).resolve(), allow_pickle=False)
        stress_mpa = float(spec["sigma_min_mpa"])
        result = adapter.run(
            injection_rate_m3_s=float(schedule[1, 0]),
            time_s=float(spec["target_time_s"]),
            mode="native",
            height_m=float(spec["height_m"]),
            viscosity_pa_s=float(spec["viscosity_pa_s"]),
            e_prime_pa=float(spec["e_prime_pa"]),
            leakoff_coefficient_m_sqrt_s=float(spec["leakoff_coefficient_m_sqrt_s"]),
            min_horizontal_stress_pa=stress_mpa * 1.0e6,
            fracture_toughness_pa_sqrt_m=float(spec["fracture_toughness_pa_sqrt_m"]),
            injection_rate_history=schedule,
        )
        payload = {
            "candidate_id": spec["candidate_id"],
            "sigma_min_mpa": stress_mpa,
            "worker_status": "completed",
            "worker_runtime_s": time.perf_counter() - started,
            "result": result.to_dict(),
        }
        _write_json(result_path, payload)
        return 0 if bool(result.success and result.target_reached) else 2
    except BaseException as exc:
        _write_json(
            result_path,
            {
                "candidate_id": spec.get("candidate_id"),
                "sigma_min_mpa": spec.get("sigma_min_mpa"),
                "worker_status": "exception",
                "worker_runtime_s": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return 3


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def _read_progress_snapshot(progress_path: Path) -> dict[str, Any] | None:
    """Read the worker heartbeat without making a candidate result valid."""

    if not progress_path.is_file():
        return None
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _evaluate_candidate(
    *,
    output: Path,
    candidate_id: str,
    sigma_min_mpa: float,
    target_time_s: float,
    observed_pressure_mpa: float,
    schedule_path: Path,
    pyfrac_config: PyFracConfig,
    timeout_s: float | None,
    watchdog_start_s: float | None = None,
    watchdog_ratio: float | None = None,
    watchdog_poll_s: float = 1.0,
    watchdog_min_sim_time_s: float = 5.0,
) -> dict[str, Any]:
    worker_dir = output / "workers" / candidate_id
    worker_dir.mkdir(parents=True, exist_ok=True)
    spec_path = worker_dir / "spec.json"
    result_path = worker_dir / "result.json"
    log_path = worker_dir / "worker.log"
    progress_path = worker_dir / "progress.json"
    progress_history_path = worker_dir / "progress_history.jsonl"
    spec = {
        "candidate_id": candidate_id,
        "project_root": str(PROJECT_ROOT),
        "result_path": str(result_path),
        "schedule_path": str(schedule_path),
        "sigma_min_mpa": float(sigma_min_mpa),
        "target_time_s": float(target_time_s),
        "height_m": float(pyfrac_config.height_m),
        "viscosity_pa_s": float(pyfrac_config.viscosity_pa_s),
        "e_prime_pa": float(pyfrac_config.e_prime_pa),
        "leakoff_coefficient_m_sqrt_s": float(pyfrac_config.leakoff_coefficient_m_sqrt_s),
        "fracture_toughness_pa_sqrt_m": float(pyfrac_config.fracture_toughness_pa_sqrt_m),
        "pyfrac_config": pyfrac_config.to_dict(),
    }
    _write_json(spec_path, spec)
    result_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    progress_history_path.unlink(missing_ok=True)
    worker_script = Path(__file__).resolve()
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    started = time.perf_counter()
    process: subprocess.Popen[str] | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                [sys.executable, "-u", str(worker_script), "--worker-spec", str(spec_path)],
                cwd=str(PROJECT_ROOT),
                env={
                    **os.environ,
                    "PYTHONPATH": str(DT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                    "PYFRAC_PROGRESS_FILE": str(progress_path),
                    "PYFRAC_PROGRESS_HISTORY_FILE": str(progress_history_path),
                },
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            # A per-candidate hard timeout is optional.  The wall-clock
            # experiment must be able to spend more than 120 s on a long
            # target when the measured simulated-time throughput is healthy
            # (for example, 1000 simulated seconds at 5x needs about 200 s).
            # The outer runner still supplies the remaining experiment budget
            # as the ultimate boundary.
            configured_timeout = float(timeout_s) if timeout_s is not None else 0.0
            timeout_limit = configured_timeout if configured_timeout > 0.0 else float("inf")
            watchdog_enabled = (
                watchdog_start_s is not None
                and watchdog_ratio is not None
                and float(watchdog_start_s) >= 0.0
                and float(watchdog_ratio) > 0.0
            )
            watchdog_triggered = False
            watchdog_ratio_observed: float | None = None
            watchdog_progress: dict[str, Any] | None = None
            return_code: int | None = None
            while process.poll() is None:
                elapsed = time.perf_counter() - started
                remaining = timeout_limit - elapsed
                if remaining <= 0.0:
                    break
                try:
                    process.wait(timeout=min(max(float(watchdog_poll_s), 0.1), remaining))
                    break
                except subprocess.TimeoutExpired:
                    if not watchdog_enabled or elapsed < float(watchdog_start_s):
                        continue
                    snapshot = _read_progress_snapshot(progress_path)
                    if not snapshot:
                        continue
                    try:
                        simulated_time = float(snapshot.get("time_s"))
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(simulated_time) or simulated_time < max(float(watchdog_min_sim_time_s), 1.0):
                        continue
                    try:
                        target_time = float(snapshot.get("final_time_s", target_time_s))
                    except (TypeError, ValueError):
                        target_time = float(target_time_s)
                    if target_time <= simulated_time + 1.0e-8:
                        continue
                    # The project acceptance rule is a throughput rule:
                    # simulated seconds per wall-clock second must stay at
                    # or above the configured threshold.  For example, with
                    # ratio=3, 100 wall-clock seconds must advance at least
                    # 300 simulated seconds.  Do not invert this ratio: the
                    # old wall/sim test incorrectly rejected fast runs.
                    ratio = simulated_time / max(elapsed, 1.0e-9)
                    if ratio < float(watchdog_ratio):
                        watchdog_triggered = True
                        watchdog_ratio_observed = float(ratio)
                        watchdog_progress = snapshot
                        break

            if process.poll() is None:
                _terminate_process(process)
                last_progress = watchdog_progress or _read_progress_snapshot(progress_path)
                if watchdog_triggered:
                    error = (
                        "candidate worker watchdog terminated slow progress; "
                        f"sim_to_wall_ratio={float(watchdog_ratio_observed):g}; "
                        f"required>={float(watchdog_ratio):g}"
                    )
                else:
                    if np.isfinite(timeout_limit):
                        error = f"candidate worker timeout after {timeout_limit:g}s; fresh run discarded"
                    else:
                        error = "candidate worker stopped without a watchdog reason; fresh run discarded"
                return {
                    "candidate_id": candidate_id,
                    "sigma_min_mpa": float(sigma_min_mpa),
                    "success": False,
                    "target_reached": False,
                    "runtime_s": time.perf_counter() - started,
                    "observed_bottomhole_pressure_mpa": float(observed_pressure_mpa),
                    "objective_abs_pressure_error_mpa": None,
                    "final_native_time_s": None,
                    "error": error,
                    "worker_result_path": str(result_path),
                    "progress_path": str(progress_path),
                    "last_progress": last_progress,
                    "watchdog_triggered": bool(watchdog_triggered),
                    "watchdog_metric": "simulated_seconds_per_wall_clock_second",
                    "watchdog_required_sim_to_wall_ratio": (
                        float(watchdog_ratio) if watchdog_ratio is not None else None
                    ),
                    "watchdog_sim_to_wall_ratio": watchdog_ratio_observed,
                }
            return_code = process.poll()
        if not result_path.is_file():
            return {
                "candidate_id": candidate_id,
                "sigma_min_mpa": float(sigma_min_mpa),
                "success": False,
                "target_reached": False,
                "runtime_s": time.perf_counter() - started,
                "observed_bottomhole_pressure_mpa": float(observed_pressure_mpa),
                "objective_abs_pressure_error_mpa": None,
                "final_native_time_s": None,
                "error": f"worker exited with code {return_code}; result.json missing",
                "worker_result_path": str(result_path),
            }
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict):
            error = payload.get("error", "worker result has no normalized PyFrac result") if isinstance(payload, dict) else "invalid worker result"
            return {
                "candidate_id": candidate_id,
                "sigma_min_mpa": float(sigma_min_mpa),
                "success": False,
                "target_reached": False,
                "runtime_s": time.perf_counter() - started,
                "observed_bottomhole_pressure_mpa": float(observed_pressure_mpa),
                "objective_abs_pressure_error_mpa": None,
                "final_native_time_s": None,
                "error": str(error),
                "worker_result_path": str(result_path),
            }
        predicted = result.get("bottomhole_pressure_mpa")
        target_reached = bool(result.get("target_reached", False))
        success = bool(result.get("success", False) and target_reached)
        mass_balance_error = result.get("mass_balance_relative_error")
        mass_balance_passed = bool(
            mass_balance_error is not None
            and np.isfinite(float(mass_balance_error))
            and float(mass_balance_error) <= float(pyfrac_config.mass_balance_tolerance)
        )
        # Do not accept a numerically completed but structurally collapsed
        # state.  In legacy PyFrac a crack footprint can remain populated
        # after the active front has disappeared at the domain boundary;
        # that state may even satisfy the volume ledger while pressure has
        # become nonphysical.  Native acceptance requires a finite front.
        front_geometry = result.get("front_geometry")
        boundary_limited = bool(
            result.get("boundary_limited", False)
            or not isinstance(front_geometry, list)
            or len(front_geometry) == 0
        )
        time_step_settings = result.get("time_step_settings") or {}
        zero_injection_jumps = time_step_settings.get("zero_injection_jumps") or []
        fully_dynamic = not bool(zero_injection_jumps)
        throughput = classify_throughput(
            result.get("runtime_seconds"),
            result.get("final_time_s"),
            DEFAULT_THROUGHPUT_TIERS,
        )
        return {
            "candidate_id": candidate_id,
            "sigma_min_mpa": float(sigma_min_mpa),
            "success": success,
            "native_acceptance": bool(
                success and mass_balance_passed and fully_dynamic and not boundary_limited
            ),
            "target_reached": target_reached,
            "runtime_s": time.perf_counter() - started,
            "solver_runtime_s": result.get("runtime_seconds"),
            "throughput_ratio": (
                float(result.get("runtime_seconds")) / float(result.get("final_time_s"))
                if result.get("runtime_seconds") is not None
                and result.get("final_time_s") is not None
                and np.isfinite(float(result.get("runtime_seconds")))
                and np.isfinite(float(result.get("final_time_s")))
                and float(result.get("final_time_s")) > 0.0
                else None
            ),
            "throughput_tier": throughput["tier"],
            "throughput_10_passed": bool(throughput["passed"].get("10%", False)),
            "throughput_20_passed": bool(throughput["passed"].get("20%", False)),
            "throughput_30_passed": bool(throughput["passed"].get("30%", False)),
            "observed_bottomhole_pressure_mpa": float(observed_pressure_mpa),
            "predicted_bottomhole_pressure_mpa": predicted,
            "objective_abs_pressure_error_mpa": _pressure_objective(predicted, observed_pressure_mpa) if success else None,
            "final_native_time_s": result.get("final_time_s"),
            "successful_time_steps": result.get("successful_time_steps"),
            "failed_time_steps": result.get("failed_time_steps"),
            "mass_balance_relative_error": mass_balance_error,
            "mass_balance_passed": mass_balance_passed,
            "fully_dynamic": fully_dynamic,
            "boundary_limited": boundary_limited,
            "zero_injection_jumps": zero_injection_jumps,
            "engine_mode": result.get("engine_mode"),
            "error": result.get("error"),
            "worker_result_path": str(result_path),
            "normalized_result": result,
        }
    except BaseException as exc:
        if process is not None:
            _terminate_process(process)
        return {
            "candidate_id": candidate_id,
            "sigma_min_mpa": float(sigma_min_mpa),
            "success": False,
            "target_reached": False,
            "runtime_s": time.perf_counter() - started,
            "observed_bottomhole_pressure_mpa": float(observed_pressure_mpa),
            "objective_abs_pressure_error_mpa": None,
            "final_native_time_s": None,
            "error": f"parent evaluation exception: {type(exc).__name__}: {exc}",
            "worker_result_path": str(result_path),
            "progress_path": str(progress_path),
            "last_progress": (
                json.loads(progress_path.read_text(encoding="utf-8"))
                if progress_path.is_file()
                else None
            ),
        }


def _plot_candidates(frame: pd.DataFrame, path: Path, observed_pressure_mpa: float) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    if not frame.empty:
        valid = frame[frame["objective_abs_pressure_error_mpa"].notna()].copy()
        if not valid.empty:
            axis.plot(valid["sigma_min_mpa"], valid["objective_abs_pressure_error_mpa"], "o-", label="压力绝对误差")
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        for _, row in frame.iterrows():
            label = "通过" if bool(row.get("success", False)) else "失败/超时"
            axis.annotate(label, (float(row["sigma_min_mpa"]), 0.0 if pd.isna(row.get("objective_abs_pressure_error_mpa")) else float(row["objective_abs_pressure_error_mpa"])), fontsize=8)
    axis.set_xlabel("最小水平应力 σ_min / MPa")
    axis.set_ylabel("目标时刻井底压力绝对误差 / MPa")
    axis.set_title(f"重启式 σ_min 反演；观测井底压力 {observed_pressure_mpa:.2f} MPa")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-spec", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--construction-pressure-xls", default="Data/3Dfrac/JY84-Z1-stage08-f1.xls")
    parser.add_argument("--output-dir", default="outputs/dt/pyfrac_sigma_restart_inversion_4435")
    parser.add_argument("--target-time-s", type=float, default=4435.0)
    parser.add_argument("--wall-clock-budget-s", type=float, default=600.0)
    parser.add_argument("--candidate-timeout-s", type=float, default=540.0)
    parser.add_argument("--base-min-stress-mpa", type=float, default=60.0)
    parser.add_argument("--stress-offsets-mpa", type=float, nargs="+", default=[0.0, -10.0, 10.0])
    parser.add_argument("--stress-lower-mpa", type=float, default=20.0)
    parser.add_argument("--stress-upper-mpa", type=float, default=120.0)
    parser.add_argument("--height-m", type=float, default=30.0)
    parser.add_argument("--base-eprime-pa", type=float, default=3.2e10)
    parser.add_argument("--base-leakoff-m-sqrt-s", type=float, default=1.0e-5)
    parser.add_argument("--base-viscosity-pa-s", type=float, default=0.1)
    parser.add_argument("--fracture-toughness-pa-sqrt-m", type=float, default=5.0e5)
    parser.add_argument("--mesh-nx", type=int, default=61)
    parser.add_argument("--mesh-ny", type=int, default=31)
    # The explicit t=1 s initialization must occupy several cells.  With the
    # 61x31 diagnostic mesh, 10 m x 25 m is the smallest already-validated
    # domain; an 80 m half-length makes the initial fracture sub-cell-sized.
    parser.add_argument("--mesh-half-length-m", type=float, default=10.0)
    parser.add_argument("--mesh-half-height-m", type=float, default=25.0)
    parser.add_argument("--max-time-steps", type=int, default=240)
    parser.add_argument("--dynamic-step-limit-s", type=float, default=30.0)
    parser.add_argument(
        "--injection-volume-step-fraction",
        type=float,
        default=0.10,
        help="fraction of current fracture volume allowed as injected volume per adaptive step",
    )
    parser.add_argument(
        "--positive-volume-change-step-fraction",
        type=float,
        default=0.12,
        help="positive relative fracture-volume-change fraction used by the adaptive step controller",
    )
    parser.add_argument(
        "--negative-volume-change-step-fraction",
        type=float,
        default=0.05,
        help="negative relative fracture-volume-change fraction used by the adaptive step controller",
    )
    parser.add_argument(
        "--time-step-time-fraction",
        type=float,
        default=0.15,
        help="maximum fraction of current simulated time used by the adaptive step controller",
    )
    parser.add_argument(
        "--cell-traversal-fraction",
        type=float,
        default=1.0,
        help="multiplier on the velocity/cell traversal criterion; >1 is an explicit throughput experiment",
    )
    parser.add_argument("--initial-step-limit-s", type=float, default=1.0)
    parser.add_argument(
        "--ehl-solver",
        choices=("implicit_Anderson", "implicit_Picard", "RKL2"),
        default="implicit_Anderson",
        help="native PyFrac elastohydrodynamic solver used for the benchmark",
    )
    parser.add_argument(
        "--absolute-pressure-solve",
        action="store_true",
        help="diagnostic: solve native viscous pressure directly instead of pressure increments",
    )
    parser.add_argument(
        "--enable-volume-balance-projection",
        action="store_true",
        help="experimental: project accepted opening onto injected minus leak-off volume and recompute pressure",
    )
    parser.add_argument(
        "--front-cfl",
        type=float,
        default=0.80,
        help="maximum front cell-traversal ratio used by the native stability guard",
    )
    parser.add_argument(
        "--front-length-fraction",
        type=float,
        default=0.80,
        help="fraction of current fracture length allowed per step; front CFL remains the cell-traversal guard",
    )
    parser.add_argument("--front-advancing", choices=("predictor-corrector", "implicit", "explicit"), default="implicit")
    # The restart inversion is intentionally conservative about front
    # reconstruction.  ``ILSA_orig`` is the native PyFrac projection that
    # passed the explicit t=1 -> t=2 s smoke test; the legacy continuous-front
    # projection is retained as an explicit diagnostic option.
    parser.add_argument("--projection-method", choices=("ILSA_orig", "LS_grad", "LS_continousfront"), default="ILSA_orig")
    parser.add_argument("--max-native-retries", type=int, default=8)
    parser.add_argument("--retry-time-step-factor", type=float, default=0.5)
    parser.add_argument("--min-dynamic-step-s", type=float, default=0.5)
    parser.add_argument("--max-solver-iterations", type=int, default=140)
    parser.add_argument(
        "--ehl-tolerance",
        type=float,
        default=1.0e-4,
        help="relative nonlinear EHL convergence tolerance; record non-default values as throughput experiments",
    )
    parser.add_argument(
        "--ehl-relaxation",
        type=float,
        default=1.0,
        help="Anderson damping factor in [0,1]; 1.0 preserves the legacy undamped iteration",
    )
    parser.add_argument("--max-front-iterations", type=int, default=25)
    parser.add_argument("--max-pyfrac-reattempts", type=int, default=8)
    parser.add_argument("--disable-adaptive-mesh", action="store_true")
    parser.add_argument("--disable-pyfrac-remeshing", action="store_true")
    parser.add_argument("--disable-mesh-extension", action="store_true")
    parser.add_argument("--mesh-extension-all-directions", action="store_true")
    parser.add_argument(
        "--mesh-extension-horizontal-only",
        action="store_true",
        help="extend only along the fracture/primary-flow direction; do not expand the contained height direction",
    )
    parser.add_argument("--mesh-extension-factor", type=float, default=1.25)
    parser.add_argument(
        "--remesh-factor",
        type=float,
        default=10.0,
        help="domain expansion factor used by the legacy compression/remesh path when extension is disabled",
    )
    parser.add_argument(
        "--skip-zero-injection-intervals",
        action="store_true",
        help="diagnostic only: jump zero-rate intervals to the next positive-injection event and record the skipped interval",
    )
    parser.add_argument(
        "--disable-injection-regime-clipping",
        action="store_true",
        help="allow a native time step to cross positive/zero injection transitions; use only for controlled comparison",
    )
    parser.add_argument(
        "--expand-domain-on-boundary",
        action="store_true",
        help="regrid to a larger horizontal domain at a boundary while keeping the current cell count",
    )
    parser.add_argument(
        "--domain-expansion-factor",
        type=float,
        default=2.0,
        help="horizontal domain multiplier for the fixed-cell regrid path",
    )
    parser.add_argument(
        "--throughput-target-ratios",
        type=float,
        nargs="+",
        default=list(DEFAULT_THROUGHPUT_TIERS),
        metavar="RATIO",
        help="ordered wall-runtime/simulated-time tiers; defaults to 0.10 0.20 0.30",
    )
    parser.add_argument(
        "--required-throughput-ratio",
        type=float,
        default=None,
        help="optional formal runtime requirement; omit to report 10/20/30%% tiers without making 10%% the sole gate",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pressure, pressure_meta = load_stage_pressure_schedule(
        args.construction_pressure_xls,
        config=PressureModelConfig(step_seconds=1.0),
    )
    target = min(float(args.target_time_s), float(pressure["time_s"].max()))
    if target < 1.0:
        raise ValueError("target time must be at least 1 s")
    observed = observation_at(pressure, target, "bottomhole_pressure_mpa")
    rate = derive_injection_rate(pressure)
    schedule = make_schedule(pressure, rate, target, 30.0)
    schedule_path = output / "injection_schedule.npy"
    np.save(schedule_path, schedule, allow_pickle=False)

    config = PyFracConfig(
        height_m=float(args.height_m),
        young_modulus_pa=float(args.base_eprime_pa),
        fracture_toughness_pa_sqrt_m=float(args.fracture_toughness_pa_sqrt_m),
        leakoff_coefficient_m_sqrt_s=float(args.base_leakoff_m_sqrt_s),
        viscosity_pa_s=float(args.base_viscosity_pa_s),
        min_horizontal_stress_pa=float(args.base_min_stress_mpa) * 1.0e6,
        initial_time_s=1.0,
        native_start_time_s=1.0,
        initial_step_limit_s=max(float(args.initial_step_limit_s), 1.0e-6),
        mesh_half_length_m=max(float(args.mesh_half_length_m), 1.0),
        mesh_half_height_m=max(float(args.mesh_half_height_m), 1.0),
        mesh_nx=max(int(args.mesh_nx), 31),
        mesh_ny=max(int(args.mesh_ny), 5),
        max_time_steps=max(int(args.max_time_steps), 1),
        dynamic_step_limit_s=max(float(args.dynamic_step_limit_s), 0.0),
        injection_volume_step_fraction=max(float(args.injection_volume_step_fraction), 1.0e-6),
        positive_volume_change_step_fraction=max(float(args.positive_volume_change_step_fraction), 1.0e-6),
        negative_volume_change_step_fraction=max(float(args.negative_volume_change_step_fraction), 1.0e-6),
        time_step_time_fraction=max(float(args.time_step_time_fraction), 1.0e-6),
        cell_traversal_fraction=max(float(args.cell_traversal_fraction), 1.0e-6),
        elastohydr_solver=str(args.ehl_solver),
        solve_delta_p=not bool(args.absolute_pressure_solve),
        enable_volume_balance_projection=bool(args.enable_volume_balance_projection),
        front_cfl=max(float(args.front_cfl), 0.0),
        front_length_fraction=max(float(args.front_length_fraction), 0.0),
        front_advancing=args.front_advancing,
        projection_method=args.projection_method,
        adaptive_mesh_enabled=not args.disable_adaptive_mesh,
        enable_pyfrac_remeshing=not args.disable_pyfrac_remeshing,
        mesh_extension_enabled=not args.disable_mesh_extension,
        mesh_extension_all_directions=bool(args.mesh_extension_all_directions),
        mesh_extension_directions=(False, False, True, True)
        if args.mesh_extension_horizontal_only and not args.mesh_extension_all_directions
        else (True, True, True, True),
        mesh_extension_factor=float(args.mesh_extension_factor),
        remesh_factor=max(float(args.remesh_factor), 1.01),
        skip_zero_injection_intervals=bool(args.skip_zero_injection_intervals),
        clip_to_injection_regime_events=not args.disable_injection_regime_clipping,
        expand_domain_on_boundary=bool(args.expand_domain_on_boundary),
        domain_expansion_factor=max(float(args.domain_expansion_factor), 1.05),
        max_retries=max(int(args.max_native_retries), 0),
        retry_time_step_factor=float(args.retry_time_step_factor),
        min_dynamic_step_s=max(float(args.min_dynamic_step_s), 1.0e-6),
        max_solver_iterations=max(int(args.max_solver_iterations), 1),
        ehl_tolerance=max(float(args.ehl_tolerance), 1.0e-12),
        ehl_relaxation=min(max(float(args.ehl_relaxation), 0.0), 1.0),
        max_front_iterations=max(int(args.max_front_iterations), 1),
        max_pyfrac_reattempts=max(int(args.max_pyfrac_reattempts), 0),
        checkpoint_enabled=False,
    )
    candidates = _candidate_stresses(
        float(args.base_min_stress_mpa),
        [float(value) for value in args.stress_offsets_mpa],
        float(args.stress_lower_mpa),
        float(args.stress_upper_mpa),
    )
    rows: list[dict[str, Any]] = []
    remaining_budget = max(float(args.wall_clock_budget_s), 0.0)
    for index, stress in enumerate(candidates):
        elapsed = time.perf_counter() - started
        remaining = remaining_budget - elapsed
        if remaining <= 0.0:
            rows.append(
                {
                    "candidate_id": f"candidate_{index:02d}",
                    "sigma_min_mpa": stress,
                    "success": False,
                    "target_reached": False,
                    "runtime_s": 0.0,
                    "observed_bottomhole_pressure_mpa": observed,
                    "objective_abs_pressure_error_mpa": None,
                    "final_native_time_s": None,
                    "error": "wall-clock budget exhausted before fresh restart",
                }
            )
            continue
        timeout = min(max(float(args.candidate_timeout_s), 0.1), remaining)
        row = _evaluate_candidate(
            output=output,
            candidate_id=f"candidate_{index:02d}",
            sigma_min_mpa=stress,
            target_time_s=target,
            observed_pressure_mpa=observed,
            schedule_path=schedule_path,
            pyfrac_config=config,
            timeout_s=timeout,
        )
        rows.append(row)
        _write_json(
            output / "inversion_progress.json",
            {
                "status": "candidate_completed",
                "target_time_s": target,
                "observed_bottomhole_pressure_mpa": observed,
                "completed_candidates": len(rows),
                "candidate_count": len(candidates),
                "elapsed_wall_clock_s": time.perf_counter() - started,
                "latest": row,
            },
        )

    frame = pd.DataFrame(
        [
            {key: value for key, value in row.items() if key != "normalized_result"}
            for row in rows
        ]
    )
    frame.to_csv(output / "candidate_evaluations.csv", index=False, encoding="utf-8-sig")
    valid = [
        row for row in rows
        if bool(row.get("native_acceptance"))
        and bool(row.get("target_reached"))
        and row.get("objective_abs_pressure_error_mpa") is not None
    ]
    best = min(valid, key=lambda row: float(row["objective_abs_pressure_error_mpa"])) if valid else None
    baseline = next((row for row in rows if np.isclose(float(row["sigma_min_mpa"]), float(args.base_min_stress_mpa))), None)

    validation: dict[str, Any] | None = None
    if best is not None:
        elapsed = time.perf_counter() - started
        remaining = remaining_budget - elapsed
        if remaining > 0.1:
            validation = _evaluate_candidate(
                output=output,
                candidate_id="posterior_validation",
                sigma_min_mpa=float(best["sigma_min_mpa"]),
                target_time_s=target,
                observed_pressure_mpa=observed,
                schedule_path=schedule_path,
                pyfrac_config=config,
                timeout_s=min(max(float(args.candidate_timeout_s), 0.1), remaining),
            )
            _write_json(output / "posterior_validation.json", validation)

    figure_frame = frame.copy()
    if not figure_frame.empty and "objective_abs_pressure_error_mpa" in figure_frame:
        figure_frame["objective_abs_pressure_error_mpa"] = pd.to_numeric(
            figure_frame["objective_abs_pressure_error_mpa"], errors="coerce"
        )
    _plot_candidates(figure_frame, output / "sigma_objective.png", observed)

    # Solver completion and formal native acceptance are different states.
    # A fast run with a conservation failure may still have ``success=True``
    # at the numerical-solver level, but it must not be promoted to the
    # posterior used by the inversion report.
    validation_accepted = bool(validation and validation.get("native_acceptance"))
    posterior = validation if validation_accepted else best
    status = "posterior_validated" if validation_accepted else (
        "candidate_found_without_validation" if best is not None else "no_valid_native_candidate"
    )
    baseline_error = baseline.get("objective_abs_pressure_error_mpa") if baseline else None
    posterior_error = posterior.get("objective_abs_pressure_error_mpa") if posterior else None
    improved = (
        baseline_error is not None
        and posterior_error is not None
        and float(posterior_error) < float(baseline_error)
    )
    summary = {
        "experiment": "pyfrac_restart_from_initial_sigma_min_inversion",
        "status": status,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_wall_clock_s": time.perf_counter() - started,
        "wall_clock_budget_s": float(args.wall_clock_budget_s),
        "target_time_s": target,
        "source_window": [1.0, target],
        "observed_bottomhole_pressure_mpa": observed,
        "inversion_parameter": "min_horizontal_stress_mpa",
        "forward_operator": "native PyFrac",
        "restart_policy": "every candidate creates a fresh adapter and starts from explicit native t=1 s; no prior fracture state is reused",
        "parameter_update_method": "derivative-free scalar candidate search; no EnKF required",
        "candidate_stresses_mpa": candidates,
        "candidate_count": len(candidates),
        "valid_native_candidate_count": len(valid),
        "baseline_sigma_min_mpa": float(args.base_min_stress_mpa),
        "baseline_pressure_error_mpa": baseline_error,
        "posterior_sigma_min_mpa": posterior.get("sigma_min_mpa") if posterior else None,
        "posterior_pressure_error_mpa": posterior_error,
        "pressure_error_improved": bool(improved),
        "performance": {
            "gate_definition": "native solver wall runtime / simulated final time; lower is better",
            "target_ratios": [float(value) for value in args.throughput_target_ratios],
            "tier": (
                classify_throughput(
                    posterior.get("solver_runtime_s") if posterior else None,
                    posterior.get("final_native_time_s") if posterior else None,
                    args.throughput_target_ratios,
                )["tier"]
                if posterior
                else "not_measured"
            ),
            "required_throughput_ratio": args.required_throughput_ratio,
            "baseline_throughput_ratio": baseline.get("throughput_ratio") if baseline else None,
            "posterior_throughput_ratio": posterior.get("throughput_ratio") if posterior else None,
            "baseline_tiers": (
                classify_throughput(
                    baseline.get("solver_runtime_s"),
                    baseline.get("final_native_time_s"),
                    args.throughput_target_ratios,
                )
                if baseline
                else classify_throughput(None, None, args.throughput_target_ratios)
            ),
            "posterior_tiers": (
                classify_throughput(
                    posterior.get("solver_runtime_s"),
                    posterior.get("final_native_time_s"),
                    args.throughput_target_ratios,
                )
                if posterior
                else classify_throughput(None, None, args.throughput_target_ratios)
            ),
            "baseline_throughput_gate": evaluate_throughput_requirement(
                baseline.get("solver_runtime_s") if baseline else None,
                baseline.get("final_native_time_s") if baseline else None,
                args.required_throughput_ratio,
            ),
            "posterior_throughput_gate": evaluate_throughput_requirement(
                posterior.get("solver_runtime_s") if posterior else None,
                posterior.get("final_native_time_s") if posterior else None,
                args.required_throughput_ratio,
            ),
        },
        "baseline": baseline,
        "posterior": posterior,
        "pressure_meta": pressure_meta,
        "pyfrac_config": config.to_dict(),
        "outputs": {
            "candidate_evaluations": str(output / "candidate_evaluations.csv"),
            "sigma_objective_figure": str(output / "sigma_objective.png"),
            "posterior_validation": str(output / "posterior_validation.json") if validation else None,
            "injection_schedule": str(schedule_path),
        },
        "limitations": [
            "本实验只反演最小水平应力，不代表多参数可辨识。",
            "只有在 native PyFrac 达到目标时间时，候选结果才进入误差排序。",
            "如果候选超时或未达到目标时间，只记录为失败，不用部分状态冒充完整反演。",
            "sigma_min 的搜索上下限是搜索区间，不是 EnKF 单步跳变限制。",
        ],
    }
    _write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    if args.worker_spec:
        return _run_worker(Path(args.worker_spec).resolve())
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    return 0 if summary["status"] in {"posterior_validated", "candidate_found_without_validation"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
