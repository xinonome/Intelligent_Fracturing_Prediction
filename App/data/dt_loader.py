"""Adapter for DT history and the single synchronized APP cache."""

from __future__ import annotations

import csv
import json
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
        self.cache = self._load_cache()

    def _load_cache(self) -> dict[str, Any]:
        if not self.cache_path or not self.cache_path.exists():
            return {}
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def at(self, index: int, normalized_count: int | None = None) -> dict[str, Any]:
        if not self.history:
            return {}
        count = normalized_count or len(self.history)
        source_index = round(index / max(count - 1, 1) * max(len(self.history) - 1, 0))
        row = self.history[min(max(source_index, 0), len(self.history) - 1)]
        time_s = number(row.get("time_s"), 0.0) or 0.0
        cluster_rows = [item for item in self.clusters if number(item.get("time_s"), -1.0) == time_s]
        clusters = []
        for item in sorted(cluster_rows, key=lambda value: number(value.get("cluster_id"), 0.0) or 0.0):
            clusters.append({
                "id": int(number(item.get("cluster_id"), 0.0) or 0),
                "prior_length": number(item.get("prior_half_length_m"), 0.0) or 0.0,
                "length": number(item.get("posterior_half_length_m"), 0.0) or 0.0,
                "liquid": number(item.get("posterior_liquid_share"), 0.0) or 0.0,
                "sand": number(item.get("posterior_sand_share"), 0.0) or 0.0,
            })
        return {
            "time_s": time_s,
            "phase": row.get("phase", "unknown"),
            "surface_pressure_mpa": number(row.get("surface_pressure_mpa")),
            "bottomhole_pressure_mpa": number(row.get("posterior_bottomhole_pressure_mpa")),
            "observed_bottomhole_pressure_mpa": number(row.get("observed_bottomhole_pressure_mpa")),
            "prior_bottomhole_pressure_mpa": number(row.get("prior_bottomhole_pressure_mpa")),
            "net_pressure_mpa": number(row.get("posterior_net_pressure_mpa")),
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
            },
            "posterior_parameters": {
                "E_prime_gpa": number(row.get("posterior_eprime_gpa")),
                "C_L_m_sqrt_s": number(row.get("posterior_leakoff_m_sqrt_s")),
                "mu_pa_s": number(row.get("posterior_viscosity_pa_s")),
                "sigma_min_mpa": number(row.get("posterior_min_stress_mpa")),
            },
            "prior_half_lengths_m": [item["prior_length"] for item in clusters],
            "posterior_half_lengths_m": [item["length"] for item in clusters],
            "clusters": clusters,
            "within_15": str(row.get("posterior_all_observations_within_15_percent", "false")).lower() == "true",
            "quality": {"source": str(self.history_path) if self.history_path else "missing", "valid": bool(row)},
        }

    def snapshot(self) -> dict[str, Any]:
        return {"summary": self.summary, "history": self.history, "clusters": self.clusters, "cache": self.cache}
