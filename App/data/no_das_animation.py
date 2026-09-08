"""Locate the reviewed no-DAS fracture-evolution GIF for a dataset.

The GIFs are frozen presentation artifacts generated from the same pressure-only
DT caches used by the APP.  Keeping lookup here, rather than in a UI page,
allows the dataset selector and future release packages to use the same mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATHS = (
    ROOT / "outputs" / "app" / "no_das_ppt_delivery_all_single_stages" / "all_stages" / "no_das_animation_manifest.json",
    ROOT / "outputs" / "app" / "no_das_ppt_delivery" / "all_stages" / "no_das_animation_manifest.json",
    ROOT / "outputs" / "app" / "无DAS_裂缝演变_全部单井段_GIF交付_20260826" / "all_stages" / "no_das_animation_manifest.json",
)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return []
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _resolve_manifest_output(manifest: Path, output: str | Path | None) -> Path | None:
    if not output:
        return None
    raw = Path(str(output).replace("\\", "/"))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend((ROOT / raw, manifest.parent / raw.name))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None

def find_no_das_gif(dataset_id: str, *, stage_id: str | None = None) -> Path | None:
    """Return the matching single-stage GIF, or ``None`` when unavailable."""

    dataset_id = str(dataset_id or "")
    stage_id = str(stage_id or "")
    for manifest in MANIFEST_PATHS:
        if not manifest.exists():
            continue
        for item in _read_manifest(manifest):
            if str(item.get("dataset_id", "")) != dataset_id:
                continue
            if str(item.get("status", "ok")) != "ok":
                return None
            found = _resolve_manifest_output(manifest, item.get("output"))
            if found:
                return found

    # A release package may contain the GIFs but omit the optional manifest.
    # The fallback is deliberately limited to the exact dataset/stage name and
    # never guesses from another well section.
    names = [stage_id, dataset_id.removeprefix("raw_")]
    directories = [manifest.parent for manifest in MANIFEST_PATHS]
    for directory in directories:
        for name in names:
            if not name:
                continue
            candidate = directory / f"无DAS_裂缝演变_{name}.gif"
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
    return None
