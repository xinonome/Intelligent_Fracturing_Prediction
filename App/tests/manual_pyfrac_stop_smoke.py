"""Verify that a long PyFrac task can be stopped and UI controls recover."""

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
    from PySide6.QtCore import QProcess, QTimer
    from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QPushButton
    from App.data.registry_loader import RegistryLoader
    from App.ui.widgets.pyfrac_workbench import create_pyfrac_workbench

    app = QApplication.instance() or QApplication([])
    widget = create_pyfrac_workbench(dataset=RegistryLoader().dataset())
    widget.findChild(QDoubleSpinBox, "pyfracTargetTime").setValue(4435.0)
    start = widget.findChild(QPushButton, "pyfracStart")
    stop = widget.findChild(QPushButton, "pyfracStop")
    report_path = ROOT / "outputs" / "app" / "ui_smoke_application" / "pyfrac_stop_report.json"
    polls = {"count": 0}

    def finish(ok: bool, message: str = "") -> None:
        runtime = widget._pyfrac_runtime
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({
            "status": "passed" if ok else "failed",
            "message": message,
            "run_root": str(runtime.run_root or ""),
            "retained_point_count": runtime.point_count,
            "start_enabled": start.isEnabled(),
            "stop_enabled": stop.isEnabled(),
            "process_state": widget._runner.process.state().value,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        widget.close()
        app.exit(0 if ok else 1)

    def poll() -> None:
        polls["count"] += 1
        if widget._runner.process.state() != QProcess.NotRunning:
            if polls["count"] > 600:
                finish(False, "process did not stop")
                return
            QTimer.singleShot(25, poll)
            return
        finish(start.isEnabled() and not stop.isEnabled(), "")

    def request_stop() -> None:
        if widget._runner.process.state() == QProcess.NotRunning:
            finish(False, "process exited before stop could be exercised")
            return
        stop.click()
        QTimer.singleShot(25, poll)

    def begin() -> None:
        if not start.isEnabled():
            finish(False, start.toolTip())
            return
        start.click()
        QTimer.singleShot(500, request_stop)

    QTimer.singleShot(0, begin)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
