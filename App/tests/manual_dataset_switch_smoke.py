"""Verify the global stage selector updates the shared replay context."""

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

    output = ROOT / "outputs" / "app" / "ui_smoke_application" / "application_refactor_20260905"
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    registry = RegistryLoader()
    service = ReplayService(registry)
    window = create_main_window(service, registry, theme="light")
    window.resize(1680, 980)
    window.show()

    # Reproduce the real user path: all lazy pages have been opened before
    # the global selector is changed.  This exercises the WebEngine-backed
    # DT/HMI views during a dataset commit instead of testing only page 0.
    for page_index in range(window.navigation.count()):
        window.navigation.setCurrentRow(page_index)
        app.processEvents()
    window.navigation.setCurrentRow(0)
    app.processEvents()

    candidates = []
    for dataset_id, dataset in (registry.dataset_catalog().get("datasets", {}) or {}).items():
        if dataset.get("data_scope") == "composite":
            continue
        if dataset.get("adapter") == "raw_frac_construction" and registry.dataset_source_ready(dataset_id):
            candidates.append(str(dataset_id))
    requested_target = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    target = requested_target if requested_target in candidates else next((item for item in candidates if item != registry.dataset_id), "")
    state = {"polls": 0}

    def fail(message: str) -> None:
        report = {"status": "failed", "message": message, "target": target}
        (output / "dataset_switch_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        app.exit(1)

    def verify() -> None:
        state["polls"] += 1
        if window._dataset_thread is not None:
            if state["polls"] >= 300:
                fail("global dataset load did not finish within 30 seconds")
                return
            QTimer.singleShot(100, verify)
            return
        frame_stage = str(service.frames[0].get("stage", "")) if service.frames else ""
        selected = registry.dataset()
        source = str(selected.get("pressure_source") or "")
        passed = registry.dataset_id == target and bool(service.frames) and frame_stage == str(selected.get("stage_id"))
        report = {
            "status": "passed" if passed else "failed",
            "target": target,
            "registry_dataset_id": registry.dataset_id,
            "frame_count": len(service.frames),
            "frame_stage_id": frame_stage,
            "selected_stage_id": str(selected.get("stage_id") or ""),
            "pressure_source": source,
            "source_badge": window.source_badge.text(),
        }
        (output / "dataset_switch_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        window.grab().save(str(output / "06_global_stage_switch.png"), "PNG")
        app.exit(0 if passed else 1)

    def start() -> None:
        if not target:
            fail("no second raw single-stage dataset is available")
            return
        index = window.dataset_selector.findData(target)
        if index < 0:
            fail("target dataset is absent from the global selector")
            return
        window.dataset_selector.setCurrentIndex(index)
        QTimer.singleShot(100, verify)

    QTimer.singleShot(500, start)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
