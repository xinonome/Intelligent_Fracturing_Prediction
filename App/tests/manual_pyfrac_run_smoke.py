"""Run a short real PyFrac task through the workbench controls."""

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
    target = widget.findChild(QDoubleSpinBox, "pyfracTargetTime")
    start = widget.findChild(QPushButton, "pyfracStart")
    stop = widget.findChild(QPushButton, "pyfracStop")
    target.setValue(2.0)
    state = {"polls": 0}
    report_path = ROOT / "outputs" / "app" / "ui_smoke_application" / "pyfrac_short_run_report.json"

    def finish(status: str, message: str = "") -> None:
        runtime = widget._pyfrac_runtime
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({
            "status": status,
            "message": message,
            "run_root": str(runtime.run_root or ""),
            "point_count": runtime.point_count,
            "final_time_s": runtime.frames[-1].time_s if runtime.frames else None,
            "completed": runtime.completed,
            "start_enabled": start.isEnabled(),
            "stop_enabled": stop.isEnabled(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        widget.close()
        app.exit(0 if status == "passed" else 1)

    def poll() -> None:
        state["polls"] += 1
        if widget._runner.process.state() != QProcess.NotRunning:
            if state["polls"] > 600:
                widget.stop_active_task()
                finish("failed", "short native run timed out")
                return
            QTimer.singleShot(25, poll)
            return
        runtime = widget._pyfrac_runtime
        valid = runtime.completed and runtime.point_count >= 1 and runtime.frames[-1].time_s == 2.0
        finish("passed" if valid else "failed", "" if valid else runtime.note)

    def begin() -> None:
        if not start.isEnabled():
            finish("failed", start.toolTip())
            return
        start.click()
        QTimer.singleShot(25, poll)

    QTimer.singleShot(0, begin)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
