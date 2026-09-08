"""Rapidly switch several real datasets in one fully built application."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --disable-software-rasterizer")
    from App.ui.web_view import configure_webengine_environment

    configure_webengine_environment()
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from App.data.registry_loader import RegistryLoader
    from App.services.replay_service import ReplayService
    from App.ui.main_window import create_main_window

    app = QApplication.instance() or QApplication([])
    registry = RegistryLoader()
    service = ReplayService(registry)
    window = create_main_window(service, registry, theme="light")
    window.resize(1680, 980)
    window.show()
    for index in range(window.navigation.count()):
        window.navigation.setCurrentRow(index)
        app.processEvents()

    requested = sys.argv[1:] or ["raw_fdbh1", "raw_fdbh10", "jy84_z1_stage08", "raw_fdbh1_1"]
    targets = [item for item in requested if window.dataset_selector.findData(item) >= 0]
    output = ROOT / "outputs" / "app" / "ui_smoke_application" / "rapid_dataset_switch_report.json"
    records = []
    state = {"index": 0, "polls": 0}

    def finish(status: str, message: str = "") -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"status": status, "message": message, "switches": records}, ensure_ascii=False, indent=2), encoding="utf-8")
        window.close()
        app.exit(0 if status == "passed" else 1)

    def select_next() -> None:
        if state["index"] >= len(targets):
            finish("passed")
            return
        target = targets[state["index"]]
        selector_index = window.dataset_selector.findData(target)
        if selector_index < 0:
            finish("failed", f"selector does not contain {target}")
            return
        state["polls"] = 0
        window.dataset_selector.setCurrentIndex(selector_index)
        QTimer.singleShot(25, verify)

    def verify() -> None:
        state["polls"] += 1
        if window._dataset_thread is not None:
            if state["polls"] > 600:
                finish("failed", f"switch timeout: {targets[state['index']]}")
                return
            QTimer.singleShot(25, verify)
            return
        target = targets[state["index"]]
        selected = registry.dataset()
        valid = registry.dataset_id == target and bool(service.frames)
        records.append({
            "dataset_id": target,
            "frame_count": len(service.frames),
            "stage_id": selected.get("stage_id"),
            "source": selected.get("pressure_source"),
            "passed": valid,
        })
        if not valid:
            finish("failed", f"context mismatch after switching to {target}")
            return
        state["index"] += 1
        QTimer.singleShot(25, select_next)

    if not targets:
        QTimer.singleShot(0, lambda: finish("failed", "no requested dataset is available"))
    else:
        QTimer.singleShot(300, select_next)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
