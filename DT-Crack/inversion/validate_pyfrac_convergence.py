"""Native PyFrac grid, time-step and mass-balance acceptance runner.

This command intentionally runs the real native time marcher.  A snapshot
result is never counted as a convergence or conservation pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


DT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DT_ROOT.parent
if str(DT_ROOT) not in sys.path:
    sys.path.insert(0, str(DT_ROOT))

from forward_models.pyfrac_adapter import PyFracAdapter  # noqa: E402
from forward_models.pyfrac_config import PyFracConfig  # noqa: E402
from forward_models.pyfrac_robustness import evaluate_convergence  # noqa: E402


METRICS = (
    "half_length_m",
    "fracture_height_m",
    "maximum_width_m",
    "fracture_volume_m3",
    "leakoff_volume_m3",
    "net_pressure_mpa",
    "efficiency",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/dt/pyfrac_convergence")
    parser.add_argument("--target-time-s", type=float, default=4435.0)
    parser.add_argument("--warm-start-time-s", type=float, default=120.0)
    parser.add_argument("--injection-rate-m3-s", type=float, default=0.1)
    parser.add_argument("--height-m", type=float, default=30.0)
    parser.add_argument("--e-prime-pa", type=float, default=3.2e10)
    parser.add_argument("--viscosity-pa-s", type=float, default=0.1)
    parser.add_argument("--leakoff-coefficient-m-sqrt-s", type=float, default=1.0e-5)
    parser.add_argument("--min-stress-mpa", type=float, default=60.0)
    parser.add_argument("--mesh-nx", type=int, default=61)
    parser.add_argument("--mesh-ny", type=int, default=41)
    parser.add_argument("--max-time-steps", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--skip-native", action="store_true", help="write an explicit not-run report")
    return parser


def _schedule(target: float, q: float) -> np.ndarray:
    return np.vstack(([0.0, float(target)], [max(float(q), 1.0e-9)] * 2))


def _record(name: str, result) -> dict[str, object]:
    values = result.to_dict()
    return {
        "name": name,
        "success": bool(result.success),
        "target_reached": bool(result.target_reached),
        "failed_time_steps": int(result.failed_time_steps),
        "successful_time_steps": int(result.successful_time_steps),
        "final_time_s": result.final_time_s,
        "half_length_m": result.half_length_m,
        # Native PyFrac's current normalized result exposes planar area and
        # volume. Height/width are kept explicit as unavailable unless the
        # solver output contains those fields, never silently set to zero.
        "fracture_height_m": values.get("fracture_height_m", np.nan),
        "maximum_width_m": float(result.max_aperture_mm) / 1000.0,
        "fracture_volume_m3": result.fracture_volume_m3,
        "leakoff_volume_m3": result.leakoff_volume_m3,
        "net_pressure_mpa": result.net_pressure_mpa,
        "efficiency": result.efficiency,
        "mass_balance_residual_m3": result.mass_balance_residual_m3,
        "mass_balance_relative_error": result.mass_balance_relative_error,
        "runtime_s": result.runtime_seconds,
        "mesh_level": result.mesh_level,
        "mesh_nx": result.mesh_nx,
        "mesh_ny": result.mesh_ny,
        "time_step_limit_s": result.time_step_limit_s,
        "error": result.error or "",
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = float(args.target_time_s)
    schedule = _schedule(target, args.injection_rate_m3_s)
    mesh_specs = {
        "coarse": (max(31, int(args.mesh_nx * 0.67) | 1), max(17, int(args.mesh_ny * 0.67) | 1)),
        "medium": (int(args.mesh_nx) | 1, int(args.mesh_ny) | 1),
        "fine": (min(241, int(args.mesh_nx * 1.5) | 1), min(121, int(args.mesh_ny * 1.5) | 1)),
    }
    time_specs = {"coarse": 30.0, "medium": 15.0, "fine": 7.5}
    grid_records: list[dict[str, object]] = []
    time_records: list[dict[str, object]] = []
    if args.skip_native:
        grid_records = [{"name": name, "success": False, "target_reached": False, "error": "native run skipped"} for name in mesh_specs]
        time_records = [{"name": name, "success": False, "target_reached": False, "error": "native run skipped"} for name in time_specs]
    else:
        for name, (nx, ny) in mesh_specs.items():
            config = PyFracConfig(
                height_m=args.height_m,
                viscosity_pa_s=args.viscosity_pa_s,
                min_horizontal_stress_pa=args.min_stress_mpa * 1.0e6,
                mesh_nx=nx,
                mesh_ny=ny,
                native_start_time_s=args.warm_start_time_s,
                max_time_steps=args.max_time_steps,
                dynamic_step_limit_s=15.0,
                max_retries=args.max_retries,
            )
            result = PyFracAdapter(config, project_root=PROJECT_ROOT).run(
                injection_rate_m3_s=args.injection_rate_m3_s,
                injection_rate_history=schedule,
                time_s=target,
                mode="native",
                height_m=args.height_m,
                viscosity_pa_s=args.viscosity_pa_s,
                e_prime_pa=args.e_prime_pa,
                leakoff_coefficient_m_sqrt_s=args.leakoff_coefficient_m_sqrt_s,
                min_horizontal_stress_pa=args.min_stress_mpa * 1.0e6,
            )
            grid_records.append(_record(name, result))
        for name, step_limit in time_specs.items():
            config = PyFracConfig(
                height_m=args.height_m,
                viscosity_pa_s=args.viscosity_pa_s,
                min_horizontal_stress_pa=args.min_stress_mpa * 1.0e6,
                mesh_nx=int(args.mesh_nx) | 1,
                mesh_ny=int(args.mesh_ny) | 1,
                native_start_time_s=args.warm_start_time_s,
                max_time_steps=args.max_time_steps,
                dynamic_step_limit_s=step_limit,
                max_retries=args.max_retries,
            )
            result = PyFracAdapter(config, project_root=PROJECT_ROOT).run(
                injection_rate_m3_s=args.injection_rate_m3_s,
                injection_rate_history=schedule,
                time_s=target,
                mode="native",
                height_m=args.height_m,
                viscosity_pa_s=args.viscosity_pa_s,
                e_prime_pa=args.e_prime_pa,
                leakoff_coefficient_m_sqrt_s=args.leakoff_coefficient_m_sqrt_s,
                min_horizontal_stress_pa=args.min_stress_mpa * 1.0e6,
            )
            time_records.append(_record(name, result))

    grid_eval = evaluate_convergence(grid_records, reference_name="fine", comparison_name="medium", metrics=METRICS)
    time_eval = evaluate_convergence(time_records, reference_name="fine", comparison_name="medium", metrics=METRICS)
    grid_frame = pd.DataFrame(grid_records)
    time_frame = pd.DataFrame(time_records)
    grid_frame.to_csv(output / "grid_convergence.csv", index=False, encoding="utf-8-sig")
    time_frame.to_csv(output / "time_step_convergence.csv", index=False, encoding="utf-8-sig")
    summary = {
        "status": "completed",
        "native_dynamic_required": True,
        "snapshot_allowed_for_acceptance": False,
        "target_time_s": target,
        "grid": grid_eval,
        "time_step": time_eval,
        "mass_balance_tolerance": 0.10,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            "grid": str(output / "grid_convergence.csv"),
            "time_step": str(output / "time_step_convergence.csv"),
            "summary": str(output / "convergence_summary.json"),
            "acceptance": str(output / "acceptance.json"),
        },
    }
    acceptance = {
        "grid_convergence_pass": bool(grid_eval["passed"]),
        "time_step_convergence_pass": bool(time_eval["passed"]),
        "mass_balance_pass": bool(
            all(
                float(row.get("mass_balance_relative_error", np.inf)) <= 0.10
                for row in [*grid_records, *time_records]
                if row.get("success", False)
            )
            and any(row.get("success", False) for row in [*grid_records, *time_records])
        ),
        "passed": bool(grid_eval["passed"] and time_eval["passed"]),
        "reason": "native dynamic convergence and conservation criteria" if not args.skip_native else "native run skipped; acceptance is false",
    }
    (output / "convergence_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    (output / "acceptance.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return {"summary": summary, "acceptance": acceptance}


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2, default=_json_default))
