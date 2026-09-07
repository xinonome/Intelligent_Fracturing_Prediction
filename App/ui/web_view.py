"""Embedded local HTML views; never opens a system browser."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from ..core.paths import PATHS
from ..services.webengine_service import probe, validate_html


def configure_webengine_environment() -> None:
    """Select a stable local-rendering path for the embedded Plotly view.

    The 3D page uses WebGL.  On machines whose D3D/GPU context is unstable,
    navigating from the DAS page to the no-DAS page can lose the Chromium GPU
    context and freeze the Qt event loop.  SwiftShader keeps WebGL available
    while avoiding the physical GPU.  Users with a known-good driver can opt
    back into hardware rendering with ``IFP_WEBENGINE_RENDER_MODE=hardware``.
    """

    if str(os.environ.get("IFP_WEBENGINE_RENDER_MODE", "software")).lower() in {
        "hardware",
        "gpu",
    }:
        return
    # A scenario switch destroys the old Plotly/WebGL document and creates a
    # new one.  On some Windows drivers that transition is enough to trigger
    # ``VIDEO_MEMORY_MANAGEMENT_INTERNAL`` even though the initial page loads
    # successfully.  SwiftShader alone is not sufficient here: Chromium may
    # still create a hardware GPU process for compositing.  Disable the GPU
    # path completely and keep SwiftShader as the software WebGL fallback.
    # This affects only the embedded 3D document; the rest of the Qt UI stays
    # on the normal desktop renderer.
    stable_flags = (
        "--disable-gpu "
        "--disable-gpu-compositing "
        "--disable-gpu-rasterization "
        "--disable-gpu-vsync "
        "--use-gl=angle "
        "--use-angle=swiftshader "
        "--enable-unsafe-swiftshader"
    )
    existing = str(os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")).strip()
    # Do not skip the safety flags merely because a launcher supplied an
    # unrelated ``--use-angle=...`` option.  That was the subtle case in
    # which the APP still entered the hardware GPU path on the client PC.
    # Remove conflicting renderer switches first; Chromium uses the last
    # occurrence, but relying on that makes the result launcher-dependent.
    conflicting = {
        "--disable-gpu",
        "--disable-gpu-compositing",
        "--disable-gpu-rasterization",
        "--disable-gpu-vsync",
        "--enable-unsafe-swiftshader",
    }
    kept = [
        token for token in existing.split()
        if token not in conflicting
        and not token.startswith("--use-angle=")
        and not token.startswith("--use-gl=")
    ]
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = " ".join([*kept, *stable_flags.split()]).strip()


# This module is imported before the first QWebEngineView is constructed.
# Configure the Chromium subprocess early enough for the command-line flags
# to take effect, including when the APP is launched outside run_app.py.
configure_webengine_environment()


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
        view._html_loading = bool(getattr(view, "webengine_available", False) and path and path.exists())
        view._pending_time_s = None
        view._html_load_callback = None
        view._html_navigation_id = 0
        view._interaction_enabled = True

        def set_time_index(time_s: float):
            if getattr(view, "webengine_available", False):
                if getattr(view, "_html_loading", False):
                    view._pending_time_s = float(time_s)
                    return
                view.page().runJavaScript(f"window.setTimeIndex({float(time_s)});")

        view.set_time_index = set_time_index

        def set_interaction_enabled(enabled: bool):
            """Enable camera gestures only while the timeline is paused."""

            view._interaction_enabled = bool(enabled)
            if (
                getattr(view, "webengine_available", False)
                and not getattr(view, "_html_loading", False)
            ):
                value = "true" if view._interaction_enabled else "false"
                view.page().runJavaScript(f"window.setInteractionEnabled({value});")

        view.set_interaction_enabled = set_interaction_enabled

        def set_html_path(next_path: Path | None, on_finished=None):
            from PySide6.QtCore import QUrl
            nonlocal saved_camera

            next_path = next_path.resolve() if next_path else None
            current_path = view._html_path.resolve() if view._html_path else None
            if next_path == current_path and getattr(view, "_html_loading", False) is False:
                if callable(on_finished):
                    from PySide6.QtCore import QTimer

                    QTimer.singleShot(0, lambda: on_finished(True))
                return
            view._html_navigation_id += 1
            navigation_id = view._html_navigation_id
            view._html_path = next_path
            view._html_load_callback = on_finished if callable(on_finished) else None
            if not getattr(view, "webengine_available", False):
                callback = view._html_load_callback
                view._html_load_callback = None
                if callback:
                    callback(False)
                return
            if next_path and next_path.exists():
                # Do not keep sending camera/time JavaScript to the old page
                # while Chromium is tearing it down and loading the next
                # 3D document.  On machines with a marginal GPU this queue
                # can make the UI look frozen and can trigger a driver reset.
                view._html_loading = True
                # Pick up the latest camera every time a scenario is entered.
                saved_camera = _load_saved_camera()
                if getattr(view, "_camera_timer", None) is not None:
                    view._camera_timer.stop()
                # Release the previous document's pending navigation before
                # allocating the next Plotly/WebGL document.  ``stop`` is
                # asynchronous but prevents an old local file from keeping a
                # renderer queue alive during the switch.
                view.stop()
                view.setUrl(QUrl.fromLocalFile(str(next_path.resolve())))
                from PySide6.QtCore import QTimer

                def navigation_probe():
                    if navigation_id != view._html_navigation_id or not view._html_loading:
                        return
                    view.page().runJavaScript(
                        "document.readyState === 'complete' && "
                        "typeof window.Plotly !== 'undefined' && !!document.getElementById('dt3d')",
                        lambda ready: finalize_navigation_probe(navigation_id, bool(ready)),
                    )

                # In practice the new page can be usable even when the old
                # page's cancellation signal arrives out of order.  Probe it
                # once before the hard timeout so the APP does not leave the
                # scenario controls disabled while waiting for a stale signal.
                QTimer.singleShot(2500, navigation_probe)

                def navigation_timeout():
                    if navigation_id != view._html_navigation_id or not view._html_loading:
                        return
                    view._html_loading = False
                    callback = view._html_load_callback
                    view._html_load_callback = None
                    if callback:
                        callback(False)

                # A broken Chromium/GPU subprocess must not leave the
                # scenario switch waiting forever.  The data view is already
                # committed; only the 3D document is unavailable.
                QTimer.singleShot(15000, navigation_timeout)
            elif next_path:
                view.setHtml(_error_text("3D引擎不可用", f"文件不存在：{next_path}"))
                view._html_loading = False
                callback = view._html_load_callback
                view._html_load_callback = None
                if callback:
                    callback(False)

        view.set_html_path = set_html_path
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
                nonlocal saved_camera
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
                        saved_camera = camera
                        _save_camera(camera)

            def poll_camera():
                view.page().runJavaScript(
                    "(() => { const el = document.getElementById('dt3d'); "
                    "return el && el.layout && el.layout.scene ? JSON.stringify(el.layout.scene.camera || null) : ''; })();",
                    remember_camera,
                )

            def finalize_navigation_probe(navigation_id: int, ready: bool):
                if not ready or navigation_id != view._html_navigation_id or not view._html_loading:
                    return
                view._html_loading = False
                if view._pending_time_s is not None:
                    pending_time = view._pending_time_s
                    view._pending_time_s = None
                    QTimer.singleShot(0, lambda: set_time_index(pending_time))
                if saved_camera:
                    QTimer.singleShot(350, lambda: set_camera(saved_camera))
                QTimer.singleShot(
                    500,
                    lambda: set_interaction_enabled(view._interaction_enabled),
                )
                QTimer.singleShot(900, poll_camera)
                view._camera_timer.start()
                callback = view._html_load_callback
                view._html_load_callback = None
                if callback:
                    QTimer.singleShot(0, lambda: callback(True))

            view._camera_timer = QTimer(view)
            # Capture a just-finished drag quickly enough that switching
            # scenarios immediately afterwards still restores that view.
            view._camera_timer.setInterval(300)
            view._camera_timer.timeout.connect(poll_camera)

            def loaded(ok):
                if not ok:
                    # ``stop()``/a rapid navigation can emit an intermediate
                    # loadFinished(False) for the document being replaced.
                    # Do not report that transient cancellation as a failed
                    # scenario switch; the navigation timeout below is the
                    # authoritative failure path if the new document never
                    # reaches a successful load.
                    if getattr(view, "_html_loading", False) and getattr(view, "_html_load_callback", None) is not None:
                        return
                    view._html_loading = False
                    callback = view._html_load_callback
                    view._html_load_callback = None
                    if callback:
                        QTimer.singleShot(0, lambda: callback(False))
                    return
                finalize_navigation_probe(view._html_navigation_id, True)

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
