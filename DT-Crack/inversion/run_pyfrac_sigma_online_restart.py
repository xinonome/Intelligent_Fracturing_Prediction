"""Budgeted restart-from-initial PyFrac parameter inversion.

At each assimilation node this runner creates a fresh native PyFrac state at
t=1 s and advances it to the node with the current ``sigma_min``.  The
pressure innovation is then used to update that scalar parameter for the next
node.  No old front, pressure field, leak-off history, or front metadata is
reused after an update.  A failed or timed-out node is recorded and skipped;
it is never replaced by an interpolated or snapshot result.

This is intentionally a transparent scalar restart experiment, not a claim
that one parameter makes all PyFrac physics identifiable.  It is a safe
bridge for the current legacy solver while continuous state restart remains
unstable.
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
from inversion.performance_tiers import (  # noqa: E402
    DEFAULT_THROUGHPUT_TIERS,
    classify_throughput,
    evaluate_throughput_requirement,
)
from inversion.run_pyfrac_enkf import derive_injection_rate, make_schedule, observation_at  # noqa: E402
from inversion.run_pyfrac_sigma_restart_inversion import _evaluate_candidate, _json_default  # noqa: E402


def assimilation_targets(start: float, target: float, interval: float) -> list[float]:
    """Return monotonically increasing restart targets including the end."""

    if interval <= 0.0:
        raise ValueError("assimilation interval must be positive")
    values = list(np.arange(start + interval, target, interval, dtype=float))
    if not values or values[-1] < target - 1.0e-8:
        values.append(float(target))
    return [float(value) for value in values]


def update_sigma_from_innovation(
    sigma_mpa: float,
    observed_mpa: float,
    predicted_mpa: float,
    *,
    previous_sigma_mpa: float | None = None,
    previous_predicted_mpa: float | None = None,
    gain: float = 0.5,
    sensitivity_floor: float = 0.10,
    lower_mpa: float = 20.0,
    upper_mpa: float = 120.0,
) -> tuple[float, float]:
    """Apply a scalar secant/proportional update without a per-step limiter.

    The only clipping is the explicit global search interval.  This prevents
    an invalid material value while preserving the requested control
    experiment in which the update itself is not artificially capped.
    """

    innovation = float(observed_mpa) - float(predicted_mpa)
    sensitivity = 1.0
    if previous_sigma_mpa is not None and previous_predicted_mpa is not None:
        denominator = float(sigma_mpa) - float(previous_sigma_mpa)
        if abs(denominator) > 1.0e-9:
            measured = (float(predicted_mpa) - float(previous_predicted_mpa)) / denominator
            if np.isfinite(measured) and abs(measured) >= float(sensitivity_floor):
                sensitivity = measured
    delta = float(gain) * innovation / sensitivity
    updated = float(np.clip(float(sigma_mpa) + delta, float(lower_mpa), float(upper_mpa)))
    return updated, sensitivity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction-pressure-xls", default="Data/3Dfrac/JY84-Z1-stage08-f1.xls")
    parser.add_argument("--output-dir", default="outputs/dt/pyfrac_sigma_online_restart4435")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--target-time-s", type=float, default=4435.0)
    parser.add_argument("--assimilation-interval-s", type=float, default=600.0)
    parser.add_argument("--wall-clock-budget-s", type=float, default=600.0)
    parser.add_argument("--candidate-timeout-s", type=float, default=120.0)
    parser.add_argument("--sigma-min-mpa", type=float, default=60.0)
    parser.add_argument("--sigma-lower-mpa", type=float, default=20.0)
    parser.add_argument("--sigma-upper-mpa", type=float, default=120.0)
    parser.add_argument("--sigma-update-gain", type=float, default=0.5)
    parser.add_argument("--sigma-sensitivity-floor", type=float, default=0.10)
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
    parser.add_argument("--disable-adaptive-mesh", action="store_true")
    parser.add_argument("--disable-pyfrac-remeshing", action="store_true")
    parser.add_argument("--disable-mesh-extension", action="store_true")
    parser.add_argument("--throughput-target-ratios", type=float, nargs="+", default=list(DEFAULT_THROUGHPUT_TIERS), metavar="RATIO")
    parser.add_argument(
        "--required-throughput-ratio",
        type=float,
        default=None,
        help="optional formal runtime requirement; omit to report tiers only",
    )
    return parser


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    return frame.replace({np.nan: None}).to_dict("records")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "通过"}


def _write_progress(output: Path, *, status: str, target: float, current: float, sigma: float, history: list[dict[str, Any]], started: float) -> None:
    payload = {
        "status": status,
        "target_time_s": float(target),
        "current_target_time_s": float(current),
        "current_sigma_min_mpa": float(sigma),
        "completed_nodes": int(len(history)),
        "elapsed_wall_clock_s": float(time.perf_counter() - started),
        "updated_at_epoch_s": time.time(),
    }
    (output / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    pd.DataFrame(history).to_csv(output / "restart_history_progress.csv", index=False, encoding="utf-8-sig")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pressure, pressure_meta = load_stage_pressure_schedule(args.construction_pressure_xls, config=PressureModelConfig(step_seconds=1.0))
    target = min(float(args.target_time_s), float(pressure["time_s"].max()))
    targets = assimilation_targets(1.0, target, float(args.assimilation_interval_s))
    rate = derive_injection_rate(pressure)
    config = PyFracConfig(
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
        checkpoint_enabled=False,
    )
    history = _read_history(Path(args.resume_from).resolve() / "restart_history_progress.csv") if args.resume_from else []
    sigma = float(args.sigma_min_mpa)
    if history and history[-1].get("next_sigma_min_mpa") is not None:
        sigma = float(history[-1]["next_sigma_min_mpa"])
    completed_targets = {
        float(row["target_time_s"])
        for row in history
        if row.get("target_time_s") is not None and _as_bool(row.get("native_acceptance"))
    }
    previous_success: dict[str, float] | None = None
    for row in reversed(history):
        if _as_bool(row.get("native_acceptance")) and row.get("predicted_bottomhole_pressure_mpa") is not None:
            previous_success = {
                "sigma_mpa": float(row["sigma_min_mpa"]),
                "predicted_mpa": float(row["predicted_bottomhole_pressure_mpa"]),
            }
            break
    schedule_dir = output / "schedules"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    status = "completed"
    budget = max(float(args.wall_clock_budget_s), 0.0)
    for node_index, target_time in enumerate(targets):
        if target_time in completed_targets:
            continue
        elapsed = time.perf_counter() - started
        if budget > 0.0 and elapsed >= budget:
            status = "wall_clock_budget_exhausted"
            break
        schedule = make_schedule(pressure, rate, target_time, 30.0)
        schedule_path = schedule_dir / f"target_{int(round(target_time)):05d}.npy"
        np.save(schedule_path, schedule, allow_pickle=False)
        observed = float(observation_at(pressure, target_time, "bottomhole_pressure_mpa"))
        remaining = budget - elapsed if budget > 0.0 else float(args.candidate_timeout_s)
        timeout = min(max(float(args.candidate_timeout_s), 0.1), max(remaining, 0.1))
        row = _evaluate_candidate(
            output=output / f"node_{int(round(target_time)):05d}",
            candidate_id=f"node_{node_index:03d}",
            sigma_min_mpa=sigma,
            target_time_s=target_time,
            observed_pressure_mpa=observed,
            schedule_path=schedule_path,
            pyfrac_config=config,
            timeout_s=timeout,
        )
        row.update({"node_index": int(node_index), "target_time_s": float(target_time), "sigma_min_mpa": float(sigma)})
        if _as_bool(row.get("native_acceptance")) and row.get("predicted_bottomhole_pressure_mpa") is not None:
            previous_sigma = previous_success["sigma_mpa"] if previous_success else None
            previous_predicted = previous_success["predicted_mpa"] if previous_success else None
            next_sigma, sensitivity = update_sigma_from_innovation(
                sigma,
                observed,
                float(row["predicted_bottomhole_pressure_mpa"]),
                previous_sigma_mpa=previous_sigma,
                previous_predicted_mpa=previous_predicted,
                gain=float(args.sigma_update_gain),
                sensitivity_floor=float(args.sigma_sensitivity_floor),
                lower_mpa=float(args.sigma_lower_mpa),
                upper_mpa=float(args.sigma_upper_mpa),
            )
            row.update({
                "innovation_mpa": observed - float(row["predicted_bottomhole_pressure_mpa"]),
                "sigma_sensitivity_mpa_per_mpa": float(sensitivity),
                "next_sigma_min_mpa": float(next_sigma),
                "update_status": "updated",
            })
            previous_success = {"sigma_mpa": sigma, "predicted_mpa": float(row["predicted_bottomhole_pressure_mpa"])}
            sigma = next_sigma
        else:
            row.update({"innovation_mpa": None, "sigma_sensitivity_mpa_per_mpa": None, "next_sigma_min_mpa": float(sigma), "update_status": "skipped_native_invalid"})
        row["throughput_tier"] = classify_throughput(row.get("solver_runtime_s"), row.get("final_native_time_s"), args.throughput_target_ratios)["tier"]
        history.append(row)
        _write_progress(output, status="running", target=target, current=target_time, sigma=sigma, history=history, started=started)
    completed_node_times = {
        float(row.get("target_time_s"))
        for row in history
        if row.get("target_time_s") is not None and _as_bool(row.get("native_acceptance"))
    }
    if len(completed_node_times) < len(targets):
        status = status if status != "completed" else "incomplete_native_nodes"
    frame = pd.DataFrame(history)
    frame.to_csv(output / "restart_history.csv", index=False, encoding="utf-8-sig")
    if not frame.empty and "native_acceptance" in frame.columns:
        valid = frame[frame["native_acceptance"].map(_as_bool)]
    else:
        valid = frame.iloc[0:0]
    ratios = [float(value) for value in pd.to_numeric(valid.get("throughput_ratio"), errors="coerce").dropna()] if not valid.empty and "throughput_ratio" in valid else []
    summary: dict[str, Any] = {
        "experiment": "pyfrac_scalar_online_restart_inversion",
        "status": status,
        "target_time_s": target,
        "source_window_s": [1.0, target],
        "assimilation_interval_s": float(args.assimilation_interval_s),
        "completed_node_count": int(len(history)),
        "successful_native_node_count": int(len(valid)),
        "updated_node_count": int(sum(row.get("update_status") == "updated" for row in history)),
        "final_sigma_min_mpa": float(sigma),
        "restart_policy": "fresh native PyFrac from explicit t=1 s at every node; no retained fracture state is reused after parameter update",
        "parameter_update_method": "scalar innovation/secant update; no per-step jump limiter; global search interval only",
        "performance": {
            "target_ratios": [float(value) for value in args.throughput_target_ratios],
            "required_throughput_ratio": args.required_throughput_ratio,
            "node_throughput_tiers": {str(row.get("target_time_s")): row.get("throughput_tier", "not_measured") for row in history},
            "best_observed_tier": (
                min(
                    (
                        classify_throughput(
                            row.get("solver_runtime_s"),
                            row.get("final_native_time_s"),
                            args.throughput_target_ratios,
                        )
                        for row in history
                    ),
                    key=lambda item: item["ratio"] if item["ratio"] is not None else float("inf"),
                )["tier"]
                if history
                else "not_measured"
            ),
            "formal_gate": (
                {
                    "required_ratio": float(args.required_throughput_ratio),
                    "passed": bool(ratios and all(ratio <= float(args.required_throughput_ratio) for ratio in ratios)),
                    "status": (
                        "passed"
                        if ratios and all(ratio <= float(args.required_throughput_ratio) for ratio in ratios)
                        else "not_met"
                    ),
                    "ratio": max(ratios) if ratios else None,
                    "scope": "all successfully completed restart nodes",
                }
                if args.required_throughput_ratio is not None
                else evaluate_throughput_requirement(None, None, None)
            ),
        },
        "pressure_meta": pressure_meta,
        "pyfrac_config": asdict(config),
        "outputs": {
            "history": str(output / "restart_history.csv"),
            "progress": str(output / "progress.json"),
            "schedules": str(schedule_dir),
        },
        "limitations": [
            "当前只反演 sigma_min，不代表多参数 PyFrac 反演已经完成。",
            "每个节点都是从 t=1 s 的全新原生运行，计算量随节点数增加。",
            "速度档位与物理验收分开统计；压力误差、守恒和目标时间仍是独立门槛。",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _write_progress(output, status=status, target=target, current=float(history[-1]["target_time_s"]) if history else 1.0, sigma=sigma, history=history, started=started)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2, default=_json_default))
