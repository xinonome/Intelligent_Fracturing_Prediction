from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .exceptions import DataValidationError


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataValidationError(f"YAML root must be a mapping: {path}")
    return payload


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise DataValidationError(f"CSV does not exist: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise DataValidationError(f"Cannot read CSV {path}: {exc}") from exc
    return frame


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_yaml(path: str | Path, value: Any) -> None:
    import yaml

    Path(path).write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")
