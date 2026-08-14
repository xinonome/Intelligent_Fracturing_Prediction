"""Lightweight launcher for the PySide6 acceptance application.

The launcher owns CLI parsing, UTF-8 setup, environment probing and window
startup.  Data, replay, services and widgets live in their own packages.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:  # Works both as ``python App/run_app.py`` and ``python -m App.run_app``.
    from App.core.artifacts import ArtifactRegistry, create_app_run, relative_path, resolve_project_path
    from App.core.paths import PATHS
    from App.core.replay import build_replay_decision
    from App.data.snapshot_builder import build_replay_frames
    from App.services.preflight import collect_preflight
    from App.services.validation_service import run_light_validation
except ImportError:  # pragma: no cover - direct package path fallback
    from App.core.artifacts import ArtifactRegistry, create_app_run, relative_path, resolve_project_path
    from App.core.paths import PATHS
    from App.core.replay import build_replay_decision
    from App.data.snapshot_builder import build_replay_frames
    from App.services.preflight import collect_preflight
    from App.services.validation_service import run_light_validation


ROOT = PATHS.root
OUTPUTS = PATHS.app_outputs
BASE_PYTHON = Path(os.environ.get("FRACTURING_ALGORITHM_PYTHON", sys.executable))
QT_PYTHON = Path(os.environ.get("FRACTURING_QT_PYTHON", sys.executable))


def load_json(path: Path) -> dict:
    if not path.exists():
        return {"status": "not_available", "path": str(path)}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {"value": value}
    except Exception as exc:
        return {"status": "invalid_json", "path": str(path), "error": str(exc)}


def nested(data: dict, *keys, default=None):
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_replay_decision(hmi: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    """Compatibility name retained for existing acceptance tests."""

    return build_replay_decision(hmi, fallback)


def load_playback_frames() -> list[dict[str, Any]]:
    """Compatibility wrapper around the registry-first replay service."""

    return build_replay_frames()


def project_status() -> dict[str, Any]:
    registry = ArtifactRegistry().snapshot()
    modules = registry.get("modules", {})
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "environment": {
            "ui_python": sys.executable,
            "algorithm_python": str(BASE_PYTHON),
            "project_root": str(ROOT),
            "mode": "frozen_replay",
        },
        "registry": registry,
        "fsl": {"artifact": modules.get("fsl", {}), "summary": modules.get("fsl", {}).get("summary", {})},
        "dt": {"artifact": modules.get("dt", {}), "summary": modules.get("dt", {}).get("summary", {})},
        "hmi": {"artifact": modules.get("hmi", {}), "summary": modules.get("hmi", {}).get("summary", {})},
    }


def write_summary() -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    run = create_app_run({"project_status": project_status(), "preflight_enriched": collect_preflight()})
    path = OUTPUTS / "app_summary.json"
    path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def qt_import_error() -> str | None:
    try:
        from PySide6.QtCore import qVersion
        from PySide6 import QtWebEngineWidgets  # noqa: F401

        qVersion()
        return None
    except Exception as exc:
        return str(exc)


def relaunch_with_qt_env(arguments: list[str]) -> int | None:
    if os.environ.get("FRACTURING_APP_RELAUNCHED") == "1" or not QT_PYTHON.exists():
        return None
    probe = subprocess.run(
        [str(QT_PYTHON), "-c", "from PySide6 import QtWebEngineWidgets; from PySide6.QtCore import qVersion; print(qVersion())"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode:
        return None
    env = os.environ.copy()
    env["FRACTURING_APP_RELAUNCHED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.call([str(QT_PYTHON), str(Path(__file__).resolve()), *arguments], cwd=ROOT, env=env)


def run_gui(smoke: bool = False) -> int:
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    from App.data.registry_loader import RegistryLoader
    from App.services.replay_service import ReplayService
    from App.ui.main_window import create_main_window

    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    service = ReplayService(RegistryLoader())
    window = create_main_window(service.timeline, service.registry, html_path=PATHS.dt_html)
    window.show()
    if smoke:
        QTimer.singleShot(1600, app.quit)
    return app.exec()


def main() -> None:
    parser = argparse.ArgumentParser(description="Intelligent fracturing integrated dynamic acceptance APP")
    parser.add_argument("--preflight", action="store_true", help="只运行环境、数据、注册表和 QtWebEngine 预检")
    parser.add_argument("--no-gui", action="store_true", help="生成联合摘要，不启动 GUI")
    parser.add_argument("--demo", action="store_true", help="使用冻结结果动态演示模式启动")
    parser.add_argument("--light-validation", action="store_true", help="执行输出隔离的轻量联调验证")
    parser.add_argument("--no-auto-env", action="store_true", help="不自动切换到 Qt 环境")
    parser.add_argument("--smoke-gui", action="store_true", help="启动 GUI 后自动退出，用于 smoke 验证")
    args = parser.parse_args()

    summary = write_summary()
    report: dict[str, Any] = {
        "summary": str(summary),
        "mode": "frozen_replay",
        "preflight": collect_preflight(),
        "registry": nested(load_json(summary), "registry", "modules", default={}),
    }
    if args.light_validation:
        validation_root = PATHS.app_runs / datetime.now().strftime("%Y%m%d_%H%M%S_cli")
        report["light_validation"] = run_light_validation(validation_root, BASE_PYTHON if BASE_PYTHON.exists() else Path(sys.executable))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.no_gui or args.preflight or args.light_validation:
        return
    qt_error = qt_import_error()
    if qt_error:
        if not args.no_auto_env:
            relaunch_args = ["--no-auto-env"] + (["--demo"] if args.demo else [])
            code = relaunch_with_qt_env(relaunch_args)
            if code is not None:
                raise SystemExit(code)
        raise SystemExit(f"PySide6/QtWebEngine 无法加载：{qt_error}")
    raise SystemExit(run_gui(smoke=args.smoke_gui))


if __name__ == "__main__":
    main()
