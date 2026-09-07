"""Adapter for DT history and the single synchronized APP cache."""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


def read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=8)
def _read_cache_json(path_value: str, modified_ns: int, size: int) -> dict[str, Any]:
    """Parse one DT cache once per file version.

    A scenario refresh creates several DTLoader instances.  The cache is
    large, so repeated JSON parsing on the Qt thread made the UI appear to
    freeze during scenario switching.
    """

    del modified_ns, size
    try:
        value = json.loads(Path(path_value).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


class DTLoader:
    def __init__(self, registry_loader) -> None:
        self.registry = registry_loader
        self.module = registry_loader.module("dt")
        self.summary = registry_loader.summary("dt")
        self.history_path = registry_loader.table("dt", "direct_observation_history.csv")
        self.cluster_path = registry_loader.table("dt", "cluster_share_history.csv")
        self.cache_path = registry_loader.frame_source()
        self.history = read_csv(self.history_path)
        self.clusters = read_csv(self.cluster_path)
        if registry_loader.dataset().get("adapter") == "raw_frac_construction":
            # Registered reference histories belong to the DAS reference
            # stage. A raw stage owns its cache, never that reference history.
            self.history = []
            self.clusters = []
        # ``at()`` is called once for every replay node.  Scanning the full
        # cluster history for every node made the GUI spend tens of seconds
        # constructing an otherwise small 600-frame replay.  Index once by
        # timestamp and keep the hot path O(1).
        self._clusters_by_time: dict[float, list[dict[str, str]]] = {}
        for item in self.clusters:
            timestamp = number(item.get("time_s"), None)
            if timestamp is not None:
                self._clusters_by_time.setdefault(round(timestamp, 6), []).append(item)
        for rows in self._clusters_by_time.values():
            rows.sort(key=lambda value: number(value.get("cluster_id"), 0.0) or 0.0)
        self.cache = self._load_cache()

    def _load_cache(self) -> dict[str, Any]:
        if not self.cache_path or not self.cache_path.exists():
            return {}
        try:
            stat = self.cache_path.stat()
            value = _read_cache_json(str(self.cache_path.resolve()), stat.st_mtime_ns, stat.st_size)
            if not isinstance(value, dict):
                return {}
            scenarios = value.get("scenarios", {})
            scenario_id = getattr(self.registry, "scenario_id", None)
            if scenario_id and isinstance(scenarios, dict) and scenario_id in scenarios:
                selected = scenarios[scenario_id]
                if isinstance(selected, dict):
                    return selected
            return value
        except (OSError, json.JSONDecodeError):
            return {}

    def timeline_length(self) -> int:
        return len(self.cache.get("timeline_s", []) or [])

    def hmi_available(self) -> bool:
        return bool(self.cache) and bool(self.cache.get("meta", {}).get("hmi_available", True))

    def at(self, index: int, normalized_count: int | None = None) -> dict[str, Any]:
        pressure_only_cache = self.cache.get("meta", {}).get("observation_mode") == "pressure_only"
        # A raw_frac single-stage dataset intentionally has a cache-only
        # replay: it has construction/pressure data but no DT EnKF history or
        # HMI action table.  Continue into the pressure-only cache branch so
        # its stage-level PKN evolution can actually be played.
        if not self.history and not pressure_only_cache:
            return {}

        def cache_number(name: str, time_value: float) -> float | None:
            timeline = self.cache.get("timeline_s", []) or []
            values = self.cache.get("arrays", {}).get(name, []) or []
            if not timeline or not values:
                return None
            source_index = min(
                max(round((float(time_value) - float(timeline[0])) / max(float(timeline[-1]) - float(timeline[0]), 1.0) * (len(values) - 1)), 0),
                len(values) - 1,
            )
            return number(values[source_index])
        if pressure_only_cache:
            timeline = self.cache.get("timeline_s", [])
            count = normalized_count or len(timeline)
            cache_index = round(index / max(count - 1, 1) * max(len(timeline) - 1, 0)) if timeline else 0
            arrays = self.cache.get("arrays", {})
            get = lambda name: number((arrays.get(name) or [None])[min(max(cache_index, 0), len(arrays.get(name, []) or [None]) - 1)])
            estimated_clusters = []
            for cluster_id, record in sorted((self.cache.get("clusters", {}) or {}).items(), key=lambda item: int(item[0])):
                estimated_clusters.append({
                    "id": int(cluster_id),
                    "prior_length": number((record.get("prior_half_length_m") or [None])[min(max(cache_index, 0), len(record.get("prior_half_length_m", []) or [None]) - 1)]),
                    "length": number((record.get("posterior_half_length_m") or [None])[min(max(cache_index, 0), len(record.get("posterior_half_length_m", []) or [None]) - 1)]),
                    "liquid": number((record.get("estimated_liquid_share") or [None])[min(max(cache_index, 0), len(record.get("estimated_liquid_share", []) or [None]) - 1)]),
                    "sand": None,
                })
            scenario_meta = self.registry.scenario("no_das_pressure_only")
            cache_meta = self.cache.get("meta", {}) or {}
            return {
                "time_s": get("timeline_s") if arrays.get("timeline_s") else (float(timeline[cache_index]) if timeline else float(index + 1)),
                "surface_pressure_mpa": get("surface_pressure_mpa"),
                "bottomhole_pressure_mpa": get("posterior_bhp_mpa"),
                "observed_bottomhole_pressure_mpa": get("observed_bhp_mpa"),
                "prior_bottomhole_pressure_mpa": get("prior_bhp_mpa"),
                "net_pressure_mpa": get("net_pressure_mpa"),
                "prior_error": None,
                "posterior_error": None,
                "prior_pressure_error": None,
                "posterior_pressure_error": None,
                "prior_parameters": {},
                "posterior_parameters": {},
                "prior_half_lengths_m": [item["prior_length"] for item in estimated_clusters],
                "posterior_half_lengths_m": [item["length"] for item in estimated_clusters],
                "cluster_balance_degree": None,
                "fracture_length_m": cache_number("pkn_stage_half_length_m", float(timeline[cache_index])) if timeline else None,
                "fracture_width_m": (
                    cache_number("pkn_stage_max_aperture_mm", float(timeline[cache_index])) / 1000.0
                    if cache_number("pkn_stage_max_aperture_mm", float(timeline[cache_index])) is not None
                    else None
                ) if timeline else None,
                "clusters": estimated_clusters,
                "quality": {
                    "source": scenario_meta.get("pressure_source"),
                    "valid": True,
                    "observation_mode": "pressure_only",
                    "observation_vector": cache_meta.get("observation_vector", ["bottomhole_pressure_mpa"]),
                    "cluster_observations": "not_available",
                    "cluster_estimate": "assumed_count_model",
                    "cluster_result_status": cache_meta.get("cluster_result_status", "model_estimate_only"),
                    "calibration_status": cache_meta.get("calibration_status", scenario_meta.get("calibration_status", "待校准")),
                    "source_start_s": cache_meta.get("source_start_s", scenario_meta.get("source_start_s")),
                    "source_end_s": cache_meta.get("source_end_s", scenario_meta.get("source_end_s")),
                    "pressure_formula": cache_meta.get("pressure_formula", ""),
                    "pressure_bias_filter": cache_meta.get("pressure_bias_filter", {}),
                },
            }

        count = normalized_count or len(self.history)
        source_index = round(index / max(count - 1, 1) * max(len(self.history) - 1, 0))
        row = self.history[min(max(source_index, 0), len(self.history) - 1)]
        time_s = number(row.get("time_s"), 0.0) or 0.0
        cluster_rows = self._clusters_by_time.get(round(time_s, 6), [])
        clusters = []
        for item in cluster_rows:
            clusters.append({
                "id": int(number(item.get("cluster_id"), 0.0) or 0),
                "prior_length": number(item.get("prior_half_length_m"), 0.0) or 0.0,
                "length": number(item.get("posterior_half_length_m"), 0.0) or 0.0,
                "liquid": number(item.get("posterior_liquid_share"), 0.0) or 0.0,
                "sand": number(item.get("posterior_sand_share"), 0.0) or 0.0,
                "observed_liquid": number(item.get("observed_liquid_share"), 0.0) or 0.0,
                "observed_sand": number(item.get("observed_sand_share"), 0.0) or 0.0,
                "allocation_source": item.get("allocation_source", "unknown"),
            })
        return {
            "time_s": time_s,
            "phase": row.get("phase", "unknown"),
            "surface_pressure_mpa": number(row.get("surface_pressure_mpa")),
            "bottomhole_pressure_mpa": number(row.get("posterior_bottomhole_pressure_mpa")),
            "observed_bottomhole_pressure_mpa": number(row.get("observed_bottomhole_pressure_mpa")),
            "prior_bottomhole_pressure_mpa": number(row.get("prior_bottomhole_pressure_mpa")),
            "net_pressure_mpa": number(row.get("posterior_net_pressure_mpa")),
            "cluster_balance_degree": cache_number("fiber_balance_degree", time_s),
            "fracture_length_m": sum(item["length"] for item in clusters) if clusters else None,
            "fracture_width_m": (
                cache_number("pkn_stage_max_aperture_mm", time_s) / 1000.0
                if cache_number("pkn_stage_max_aperture_mm", time_s) is not None
                else None
            ),
            "prior_error": number(row.get("prior_bhp_relative_error")),
            "posterior_error": number(row.get("posterior_bhp_relative_error")),
            "prior_pressure_error": number(row.get("prior_bhp_relative_error")),
            "posterior_pressure_error": number(row.get("posterior_bhp_relative_error")),
            "prior_liquid_error": number(row.get("prior_liquid_tvd")),
            "posterior_liquid_error": number(row.get("posterior_liquid_tvd")),
            "prior_sand_error": number(row.get("prior_sand_tvd")),
            "posterior_sand_error": number(row.get("posterior_sand_tvd")),
            "kalman_gain": number(row.get("mean_abs_kalman_gain"), 0.0),
            "runtime_ms": number(row.get("step_compute_ms")),
            "prior_parameters": {
                "E_prime_gpa": number(row.get("prior_eprime_gpa")),
                "C_L_m_sqrt_s": number(row.get("prior_leakoff_m_sqrt_s")),
                "mu_pa_s": number(row.get("prior_viscosity_pa_s")),
                "sigma_min_mpa": number(row.get("prior_min_stress_mpa")),
                "K_IC_pa_sqrt_m": number(row.get("prior_fracture_toughness_pa_sqrt_m")),
                "stress_shadow_scale": number(row.get("prior_stress_shadow_scale")),
                "boundary_relief_scale": number(row.get("prior_boundary_relief_scale")),
                "allocation_exponent": number(row.get("prior_allocation_exponent")),
                **{
                    f"intake_capacity_C{cluster_id + 1}": number(row.get(f"prior_factor_c{cluster_id + 1}"))
                    for cluster_id in range(6)
                    if row.get(f"prior_factor_c{cluster_id + 1}") is not None
                },
            },
            "posterior_parameters": {
                "E_prime_gpa": number(row.get("posterior_eprime_gpa")),
                "C_L_m_sqrt_s": number(row.get("posterior_leakoff_m_sqrt_s")),
                "mu_pa_s": number(row.get("posterior_viscosity_pa_s")),
                "sigma_min_mpa": number(row.get("posterior_min_stress_mpa")),
                "K_IC_pa_sqrt_m": number(row.get("posterior_fracture_toughness_pa_sqrt_m")),
                "stress_shadow_scale": number(row.get("posterior_stress_shadow_scale")),
                "boundary_relief_scale": number(row.get("posterior_boundary_relief_scale")),
                "allocation_exponent": number(row.get("posterior_allocation_exponent")),
                **{
                    f"intake_capacity_C{cluster_id + 1}": number(row.get(f"posterior_factor_c{cluster_id + 1}"))
                    for cluster_id in range(6)
                    if row.get(f"posterior_factor_c{cluster_id + 1}") is not None
                },
            },
            "prior_half_lengths_m": [item["prior_length"] for item in clusters],
            "posterior_half_lengths_m": [item["length"] for item in clusters],
            "clusters": clusters,
            "within_15": str(row.get("posterior_all_observations_within_15_percent", "false")).lower() == "true",
            "allocation_mode": row.get("allocation_mode", "unknown"),
            "parameterized_allocation": str(row.get("parameterized_allocation", "false")).lower() == "true",
            "quality": {
                "source": str(self.history_path) if self.history_path else "missing",
                "valid": bool(row),
                "observation_mode": "pressure_plus_cluster",
                "cluster_result_status": (self.cache.get("meta", {}) or {}).get("cluster_result_status", "interpreted_observation"),
                "calibration_status": (self.cache.get("meta", {}) or {}).get("calibration_status", "待校准"),
                "source_start_s": (self.cache.get("meta", {}) or {}).get("source_start_s"),
                "source_end_s": (self.cache.get("meta", {}) or {}).get("source_end_s"),
                "pressure_formula": (self.cache.get("meta", {}) or {}).get("pressure_formula", ""),
                "observation_quality": (self.cache.get("meta", {}) or {}).get("observation_quality", {}),
            },
        }

    def snapshot(self) -> dict[str, Any]:
        return {"summary": self.summary, "history": self.history, "clusters": self.clusters, "cache": self.cache}
