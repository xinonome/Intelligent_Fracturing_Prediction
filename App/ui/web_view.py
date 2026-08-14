"""Embedded local HTML views; never opens a system browser."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..core.paths import PATHS
from ..services.webengine_service import probe, validate_html


def create_local_html_view(path: Path | None, *, title: str = "本地 HTML"):
    from PySide6.QtCore import QUrl
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        widget = QWidget()
        layout = QVBoxLayout(widget)
        label = QLabel(_error_text(title, str(exc)))
        label.setObjectName("warning")
        label.setWordWrap(True)
        layout.addWidget(label)
        widget.webengine_available = False
        return widget

    view = QWebEngineView()
    view.setObjectName("embeddedWebView")
    view.webengine_available = True
    if path and path.exists():
        view.setUrl(QUrl.fromLocalFile(str(path.resolve())))
    else:
        view.setHtml(_error_text(title, f"文件不存在：{path}"))
    return view


class Embedded3DView:
    """Factory wrapper for a QWebEngineView with time-sync and camera memory."""

    @staticmethod
    def create(path: Path | None):
        view = create_local_html_view(path, title="3D引擎不可用")
        view._html_path = path

        def set_time_index(time_s: float):
            if getattr(view, "webengine_available", False):
                view.page().runJavaScript(f"window.setTimeIndex({float(time_s)});")

        view.set_time_index = set_time_index
        view._camera_signature = None

        def set_camera(camera):
            if getattr(view, "webengine_available", False) and isinstance(camera, dict):
                payload = json.dumps(camera, ensure_ascii=False, separators=(",", ":"))
                view.page().runJavaScript(f"window.setCamera({payload});")

        view.set_camera = set_camera

        if getattr(view, "webengine_available", False):
            from PySide6.QtCore import QTimer

            saved_camera = _load_saved_camera()
            if saved_camera:
                view._camera_signature = _camera_signature(saved_camera)

            def remember_camera(raw):
                if not raw:
                    return
                try:
                    camera = json.loads(raw) if isinstance(raw, str) else raw
                except (TypeError, ValueError, json.JSONDecodeError):
                    return
                if isinstance(camera, dict):
                    signature = _camera_signature(camera)
                    if signature != view._camera_signature:
                        view._camera_signature = signature
                        _save_camera(camera)

            def poll_camera():
                view.page().runJavaScript(
                    "(() => { const el = document.getElementById('dt3d'); "
                    "return el && el.layout && el.layout.scene ? JSON.stringify(el.layout.scene.camera || null) : ''; })();",
                    remember_camera,
                )

            view._camera_timer = QTimer(view)
            view._camera_timer.setInterval(1200)
            view._camera_timer.timeout.connect(poll_camera)

            def loaded(ok):
                if not ok:
                    return
                if saved_camera:
                    QTimer.singleShot(350, lambda: set_camera(saved_camera))
                QTimer.singleShot(900, poll_camera)
                view._camera_timer.start()

            view.loadFinished.connect(loaded)
        return view


def _error_text(title: str, error: str) -> str:
    details = probe()
    return (
        f"<h3>{title}</h3>"
        f"<p>当前Python：{sys.executable}<br>Qt Python：{details.get('python')}<br>"
        f"PySide6版本：{details.get('qt')}<br>QtWebEngine状态：不可用<br>错误信息：{error}</p>"
        "<p>建议命令：<br><code>conda activate frac_app</code><br>"
        "<code>python -c \"from PySide6 import QtWebEngineWidgets\"</code></p>"
    )


def _load_saved_camera() -> dict | None:
    try:
        value = json.loads(PATHS.ui_config.read_text(encoding="utf-8"))
        camera = value.get("three_d", {}).get("camera") if isinstance(value, dict) else None
        return camera if isinstance(camera, dict) else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _camera_signature(camera: dict) -> str:
    return json.dumps(camera, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _save_camera(camera: dict) -> None:
    try:
        config = json.loads(PATHS.ui_config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            config = {}
        three_d = config.setdefault("three_d", {})
        three_d["camera"] = camera
        PATHS.ui_config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return
