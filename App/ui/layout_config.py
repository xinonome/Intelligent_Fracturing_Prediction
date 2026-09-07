"""Shared saved-layout helpers for the developer and production APPs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.paths import PATHS


DEFAULT_LAYOUT_FILE = PATHS.app_outputs / "layout_test_layout.json"


def load_layout_ratios(path: Path = DEFAULT_LAYOUT_FILE) -> dict[str, list[float]]:
    """Read saved splitter ratios without making layout loading mandatory."""

    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    splitters = payload.get("splitters", {}) if isinstance(payload, dict) else {}
    if not isinstance(splitters, dict):
        return {}
    result: dict[str, list[float]] = {}
    for name, item in splitters.items():
        ratios = item.get("ratios") if isinstance(item, dict) else None
        if not isinstance(ratios, list) or not ratios:
            continue
        try:
            clean = [max(float(value), 0.01) for value in ratios]
        except (TypeError, ValueError):
            continue
        total = sum(clean)
        if total > 0:
            result[str(name)] = [value / total for value in clean]
    return result


def apply_layout_ratios(splitters: dict[str, Any], ratios: dict[str, list[float]]) -> None:
    """Apply saved ratios to already-created QSplitter widgets."""

    for name, splitter in splitters.items():
        values = ratios.get(name)
        if not values or len(values) != splitter.count():
            continue
        available = splitter.height() if splitter.orientation().value == 2 else splitter.width()
        if available <= 0:
            available = sum(splitter.sizes())
        if available <= 0:
            continue
        sizes = [max(20, int(available * value)) for value in values]
        splitter.setSizes(sizes)
