"""Artifact registration, validation, and reproducible APP run snapshots.

The desktop UI deliberately reads this small registry instead of guessing
where historical experiment files happen to be.  This keeps frozen results,
development results, and unavailable results visibly separate.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = PROJECT_ROOT / "App"
REGISTRY_PATH = APP_ROOT / "config" / "demo_registry.json"
APP_RUNS = PROJECT_ROOT / "outputs" / "app" / "runs"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {"_value": value}
    except (OSError, json.JSONDecodeError) as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}


def resolve_project_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value).replace("/", "\\"))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def relative_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _summary_value(summary: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = summary
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value


def infer_status(module: str, summary: dict[str, Any], required_paths: list[Path]) -> tuple[str, str]:
    if not summary or "_error" in summary:
        return "invalid", "summary.json cannot be read"
    if not all(path.exists() for path in required_paths):
        missing = [relative_path(path) or str(path) for path in required_paths if not path.exists()]
        return "not_available", "missing registered files: " + ", ".join(missing)

    if module == "hmi":
        quality_passed = bool(_summary_value(summary, "quality_gate", "passed", default=False))
        timesteps = int(_summary_value(summary, "total_timesteps", default=0) or 0)
        scientific_status = str(summary.get("scientific_status", ""))
        if scientific_status == "demo_only" or timesteps < 10000 or not quality_passed:
            return "development_only", "HMI quality gate is not passed or result is marked demo_only"
        return "validated", "HMI quality gate passed"

    if module == "dt":
        metrics = summary.get("metrics", {})
        if bool(metrics.get("validation_pass", False)):
            return "validated", "held-out observation-space validation passed"
        return "development_only", "DT result is available but validation_pass is false"

    if module == "fsl":
        return "validated", "registered FSL metrics and representative figure are available"

    return "development_only", "registered artifact is available"


class ArtifactRegistry:
    """Load and validate the small, human-reviewable demo registry."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or REGISTRY_PATH
        self.raw = read_json(self.path) if self.path.exists() else {}
        self.modules = self.raw.get("modules", {}) if isinstance(self.raw, dict) else {}

    def module(self, name: str) -> dict[str, Any]:
        entry = self.modules.get(name, {})
        if not isinstance(entry, dict):
            return {"module": name}

        summary_path = resolve_project_path(entry.get("summary"))
        summary = read_json(summary_path) if summary_path and summary_path.exists() else {}
        registered_files: list[Path] = []
        if summary_path is not None:
            registered_files.append(summary_path)
        for key in ("figures", "tables"):
            for value in entry.get(key, []) or []:
                path = resolve_project_path(value)
                if path is not None:
                    registered_files.append(path)
        html_path = resolve_project_path(entry.get("html"))
        if html_path is not None and entry.get("html_required", False):
            registered_files.append(html_path)

        status, reason = infer_status(name, summary, registered_files)
        result = dict(entry)
        result.update(
            {
                "module": name,
                "summary_path": relative_path(summary_path),
                "summary": summary,
                "status": status,
                "status_reason": reason,
                "files": {
                    "figures": [
                        {
                            "path": relative_path(resolve_project_path(value)),
                            "exists": bool(resolve_project_path(value) and resolve_project_path(value).exists()),
                        }
                        for value in entry.get("figures", []) or []
                    ],
                    "tables": [
                        {
                            "path": relative_path(resolve_project_path(value)),
                            "exists": bool(resolve_project_path(value) and resolve_project_path(value).exists()),
                        }
                        for value in entry.get("tables", []) or []
                    ],
                    "html": {
                        "path": relative_path(html_path),
                        "exists": bool(html_path and html_path.exists()),
                    },
                },
                "supporting": {
                    name: read_json(resolve_project_path(value))
                    for name, value in (entry.get("supporting_summaries", {}) or {}).items()
                    if resolve_project_path(value) is not None and resolve_project_path(value).exists()
                },
            }
        )
        return result

    def snapshot(self) -> dict[str, Any]:
        modules = {name: self.module(name) for name in ("fsl", "dt", "hmi")}
        return {
            "registry_path": relative_path(self.path),
            "registry_version": self.raw.get("version", 1),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "default_demo": self.raw.get("default_demo", {}),
            "modules": modules,
        }


def _command_output(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": command, "returncode": None, "error": f"{type(exc).__name__}: {exc}"}


def build_preflight() -> dict[str, Any]:
    """Collect environment checks without importing Qt in the base process."""

    qt_python = Path(os.environ.get("FRACTURING_QT_PYTHON", r"C:\Users\xinonome\anaconda3\envs\frac_app\python.exe"))
    algorithm_python = Path(
        os.environ.get("FRACTURING_ALGORITHM_PYTHON", r"C:\Users\xinonome\anaconda3\python.exe")
    )
    data_paths = [
        PROJECT_ROOT / "Data" / "3Dfrac" / "光纤本井监测08.txt",
        PROJECT_ROOT / "Data" / "3Dfrac" / "JY84-Z1-stage08-f1.xls",
        PROJECT_ROOT / "Data" / "3Dfrac" / "JY84-Z1HF-1011.csv",
    ]
    qt_probe = _command_output([str(qt_python), "-c", "from PySide6.QtCore import qVersion; print(qVersion())"])
    current_qt_probe = _command_output([sys.executable, "-c", "from PySide6.QtCore import qVersion; print(qVersion())"])
    webengine_probe = _command_output(
        [str(qt_python), "-c", "from PySide6 import QtWebEngineWidgets; print('QtWebEngineWidgets OK')"]
    )
    registry = ArtifactRegistry().snapshot()
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "cwd": str(Path.cwd()),
        "python": {
            "algorithm": str(algorithm_python),
            "algorithm_exists": algorithm_python.exists(),
            "qt": str(qt_python),
            "qt_exists": qt_python.exists(),
        },
        "environment": {
            "HOME": os.environ.get("HOME", ""),
            "PYTHONUTF8": os.environ.get("PYTHONUTF8", ""),
            "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "platform": platform.platform(),
        },
        "qt_probe": qt_probe,
        "current_process_qt_probe": current_qt_probe,
        "qt_webengine_probe": webengine_probe,
        "data": [
            {"path": relative_path(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}
            for path in data_paths
        ],
        "registry": registry,
        "ready_for_gui": qt_probe.get("returncode") == 0,
        "requires_qt_relaunch": current_qt_probe.get("returncode") != 0 and qt_probe.get("returncode") == 0,
        "ready_for_no_gui": True,
    }


def create_app_run(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a lightweight reproducibility snapshot for one APP invocation."""

    run_root = APP_RUNS / now_stamp()
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(exist_ok=True)
    registry_snapshot = ArtifactRegistry().snapshot()
    preflight = build_preflight()
    payload: dict[str, Any] = {
        "app_run_id": run_root.name,
        "run_root": relative_path(run_root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "registry": registry_snapshot,
        "preflight": preflight,
        "commands": [],
    }
    try:
        from .integration import build_bridge_from_registry

        dt_to_hmi, hmi_decision = build_bridge_from_registry(registry_snapshot)
        payload["dt_to_hmi"] = dt_to_hmi
        payload["hmi_decision"] = hmi_decision
        (run_root / "dt_to_hmi.json").write_text(json.dumps(dt_to_hmi, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_root / "hmi_decision.json").write_text(json.dumps(hmi_decision, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        payload["integration_error"] = f"{type(exc).__name__}: {exc}"
    if extra:
        payload.update(extra)

    (run_root / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("fsl", "dt", "hmi"):
        module = registry_snapshot["modules"].get(name, {})
        (run_root / f"{name}_snapshot.json").write_text(
            json.dumps(module, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (run_root / "commands.json").write_text(json.dumps(payload["commands"], ensure_ascii=False, indent=2), encoding="utf-8")
    (run_root / "app_run.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
