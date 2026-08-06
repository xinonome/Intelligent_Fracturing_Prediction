"""Small, explicit DT -> HMI hand-off for the acceptance application."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .artifacts import resolve_project_path


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_dt_hmi_payload(
    summary: dict[str, Any],
    history_path: Path | None = None,
    cluster_history_path: Path | None = None,
) -> dict[str, Any]:
    """Convert the latest DT posterior state into the stable HMI input schema.

    The current DT output does not produce abnormal probabilities.  Those
    fields are therefore explicitly marked as unavailable rather than being
    silently invented from the geometric state.
    """

    history = _read_rows(history_path)
    clusters = _read_rows(cluster_history_path)
    latest = history[-1] if history else {}
    latest_time = _number(latest.get("time_s"), 0.0) or 0.0
    latest_clusters = [row for row in clusters if _number(row.get("time_s"), -1.0) == latest_time]
    if not latest_clusters and clusters:
        latest_time = _number(clusters[-1].get("time_s"), latest_time) or latest_time
        latest_clusters = [row for row in clusters if _number(row.get("time_s"), -1.0) == latest_time]

    posterior_lengths = [
        _number(row.get("posterior_half_length_m"), 0.0) or 0.0 for row in latest_clusters
    ]
    metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    posterior_error = _number(latest.get("posterior_bhp_relative_error"), None)
    if posterior_error is None:
        posterior_error = _number(metrics.get("validation_bhp_relative_error_mean"), None)
    within_15 = bool(latest.get("posterior_all_observations_within_15_percent", "false").lower() == "true") if latest else bool(metrics.get("validation_pass", False))
    return {
        "time_s": latest_time,
        "stage": "08",
        "posterior_half_lengths_m": posterior_lengths,
        "bottomhole_pressure_mpa": _number(latest.get("posterior_bottomhole_pressure_mpa"), None),
        "net_pressure_mpa": None,
        "posterior_error": posterior_error,
        "abnormal_probability": None,
        "sand_plug_probability": None,
        "uncertainty": "medium" if posterior_error is None or posterior_error > 0.1 else "low",
        "within_15_percent": within_15,
        "observation_sources": ["cumulative liquid share", "cumulative sand share", "bottom-hole pressure"],
        "unavailable_fields": ["net_pressure_mpa", "abnormal_probability", "sand_plug_probability"],
        "source": {
            "dt_summary": str(summary.get("outputs", {})),
            "history": str(history_path) if history_path else None,
            "cluster_history": str(cluster_history_path) if cluster_history_path else None,
        },
    }


def load_latest_hmi_decision(path: Path | None) -> dict[str, Any]:
    rows = []
    if path and path.exists():
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            rows = value if isinstance(value, list) else [value]
        except (OSError, json.JSONDecodeError):
            rows = []
    if not rows:
        return {
            "risk_level": "unknown",
            "uncertainty": "unknown",
            "recommendation": "暂无可复用的 HMI 决策卡片。",
            "requires_confirmation": True,
            "evidence": {"source_available": False},
        }
    decision = dict(rows[-1])
    decision.setdefault("requires_confirmation", True)
    decision.setdefault("evidence", {})
    decision["source_available"] = True
    return decision


def build_bridge_from_registry(registry_snapshot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    dt = registry_snapshot.get("modules", {}).get("dt", {})
    hmi = registry_snapshot.get("modules", {}).get("hmi", {})
    dt_summary = dt.get("summary", {})
    table_paths = [resolve_project_path(item.get("path")) for item in dt.get("files", {}).get("tables", [])]
    history_path = next((path for path in table_paths if path and path.name == "direct_observation_history.csv"), None)
    cluster_path = next((path for path in table_paths if path and path.name == "cluster_share_history.csv"), None)
    dt_payload = build_dt_hmi_payload(dt_summary, history_path, cluster_path)
    hmi_table_paths = [resolve_project_path(item.get("path")) for item in hmi.get("files", {}).get("tables", [])]
    decision_path = next((path for path in hmi_table_paths if path and path.name == "human_machine_decisions.json"), None)
    return dt_payload, load_latest_hmi_decision(decision_path)
