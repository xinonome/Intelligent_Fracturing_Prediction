"""QtWebEngine diagnostics and local-file policy."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .preflight import html_is_embedded


def probe() -> dict[str, Any]:
    result: dict[str, Any] = {"python": sys.executable, "qt": "unknown", "available": False, "error": None}
    try:
        from PySide6.QtCore import qVersion

        result["qt"] = qVersion()
        from PySide6 import QtWebEngineWidgets  # noqa: F401

        result["available"] = True
    except Exception as exc:  # pragma: no cover - depends on local Qt installation
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def validate_html(path: Path | None) -> dict[str, Any]:
    return {
        "path": str(path) if path else None,
        "exists": bool(path and path.exists()),
        "embedded": html_is_embedded(path),
        "external_browser": False,
    }
