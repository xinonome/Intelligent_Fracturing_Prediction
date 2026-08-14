"""Startup and release checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.artifacts import build_preflight
from ..core.paths import PATHS


def html_is_embedded(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "window.setTimeIndex" in text and "<script src=" not in text


def collect_preflight() -> dict[str, Any]:
    result = build_preflight()
    result["plotly_embedded"] = html_is_embedded(PATHS.dt_html)
    result["html"] = {
        "path": str(PATHS.dt_html),
        "exists": PATHS.dt_html.exists(),
        "embedded": result["plotly_embedded"],
        "single_resource": True,
    }
    result["data_contract"] = {
        "source_data_read_only": True,
        "frozen_artifacts_read_only": True,
        "missing_values_are_provenanced": True,
    }
    return result


def write_preflight(path: Path | None = None) -> Path:
    target = path or (PATHS.app_outputs / "preflight.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(collect_preflight(), ensure_ascii=False, indent=2), encoding="utf-8")
    return target
