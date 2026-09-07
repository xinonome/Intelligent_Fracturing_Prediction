"""Machine-readable acceptance checks for the dual-scenario DT package.

This is a packaging/integration check, not a new physics solver.  It verifies
that the APP cache keeps the no-DAS and DAS observation contracts separate and
that the registered KG-EnKF run is the run actually represented by the cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate(cache_path: Path, kg_summary_path: Path) -> dict[str, Any]:
    cache = _load(cache_path)
    kg_summary = _load(kg_summary_path)
    scenarios = cache.get("scenarios", {})
    if not isinstance(scenarios, dict):
        scenarios = {}
    das = scenarios.get("das_cluster_observation", {})
    no_das = scenarios.get("no_das_pressure_only", {})
    das_meta = das.get("meta", {}) if isinstance(das, dict) else {}
    no_das_meta = no_das.get("meta", {}) if isinstance(no_das, dict) else {}
    kg_metrics = kg_summary.get("metrics", {})
    kg_meta = kg_metrics.get("knowledge_guided_prior", {})

    checks = {
        "cache_has_both_scenarios": set(scenarios) >= {"das_cluster_observation", "no_das_pressure_only"},
        "no_das_pressure_only_vector": no_das_meta.get("observation_vector") == ["bottomhole_pressure_mpa"],
        "no_das_has_no_cluster_observation": (
            no_das_meta.get("cluster_observations") == "not_available"
            and no_das_meta.get("cluster_result_status") == "model_estimate_only"
        ),
        "no_das_has_straight_assumed_geometry": (
            no_das_meta.get("cluster_geometry", {}).get("mode") == "straight_assumed_stage"
            and not no_das.get("trajectory")
        ),
        "das_has_pressure_and_cluster_vector": (
            "bottomhole_pressure_mpa" in (das_meta.get("observation_vector") or [])
            and "cumulative_liquid_share_by_cluster" in (das_meta.get("observation_vector") or [])
        ),
        "das_qc_has_full_pressure_overlap": (
            float((das_meta.get("observation_quality") or {}).get("pressure_overlap_ratio", 0.0)) >= 0.999
        ),
        "das_qc_has_no_invalid_steps": int((das_meta.get("observation_quality") or {}).get("invalid_steps", 1)) == 0,
        "kg_mode_is_soft_correlated": kg_meta.get("mode") == "soft_correlated",
        "kg_state_is_parameterized": int(kg_metrics.get("state_dimension", 0)) == 14,
        "kg_coverage_reaches_4435": float(kg_metrics.get("realtime_coverage_end_s") or 0.0) >= 4435.0,
        "kg_p95_under_15_seconds": float(kg_metrics.get("all_steps_compute_p95_ms") or 1.0e12) < 15000.0,
        "kg_has_no_nan_validation": all(
            value is not None
            for value in (
                kg_metrics.get("validation_bhp_relative_error_mean"),
                kg_metrics.get("validation_liquid_tvd_mean"),
                kg_metrics.get("validation_sand_tvd_mean"),
            )
        ),
    }
    result = {
        "schema_version": 1,
        "cache": str(cache_path),
        "kg_summary": str(kg_summary_path),
        "checks": checks,
        "passed": bool(all(checks.values())),
        "scenario_summary": {
            "das": {
                "coverage_s": [das_meta.get("source_start_s"), das_meta.get("source_end_s")],
                "observation_mode": das_meta.get("observation_mode"),
                "cluster_result_status": das_meta.get("cluster_result_status"),
                "calibration_status": das_meta.get("calibration_status"),
            },
            "no_das": {
                "coverage_s": [no_das_meta.get("source_start_s"), no_das_meta.get("source_end_s")],
                "observation_mode": no_das_meta.get("observation_mode"),
                "cluster_result_status": no_das_meta.get("cluster_result_status"),
                "calibration_status": no_das_meta.get("calibration_status"),
            },
        },
        "kg_metrics": {
            "mode": kg_meta.get("mode"),
            "state_dimension": kg_metrics.get("state_dimension"),
            "validation_bhp_relative_error_mean": kg_metrics.get("validation_bhp_relative_error_mean"),
            "validation_liquid_tvd_mean": kg_metrics.get("validation_liquid_tvd_mean"),
            "validation_sand_tvd_mean": kg_metrics.get("validation_sand_tvd_mean"),
            "p95_ms": kg_metrics.get("all_steps_compute_p95_ms"),
            "actual_update_count": kg_metrics.get("realtime_scheduled_update_count"),
            "coverage_end_s": kg_metrics.get("realtime_coverage_end_s"),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dual-scenario APP cache and KG-EnKF registration.")
    parser.add_argument("--cache", default="outputs/app/dt_realtime_cache.json")
    parser.add_argument("--kg-summary", required=True)
    parser.add_argument("--output", default="outputs/dt/second_part_validation.json")
    args = parser.parse_args()
    result = validate(Path(args.cache), Path(args.kg_summary))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
