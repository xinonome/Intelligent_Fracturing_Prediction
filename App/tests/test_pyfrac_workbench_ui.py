import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QFileDialog, QPushButton, QSlider

from App.data.registry_loader import RegistryLoader
from App.ui.widgets.pyfrac_workbench import create_pyfrac_workbench


def _button(widget, text: str) -> QPushButton:
    return next(button for button in widget.findChildren(QPushButton) if button.text() == text)


def test_pyfrac_workbench_replays_jumps_and_exports_real_frames(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    widget = create_pyfrac_workbench(dataset=RegistryLoader().dataset())
    app.processEvents()

    runtime = widget._pyfrac_runtime
    assert runtime.point_count == 370
    assert runtime.frames[-1].time_s == 4435.0

    slider = widget.findChild(QSlider)
    assert slider.maximum() == 369
    _button(widget, "回到初始点").click()
    assert slider.value() == 0
    _button(widget, "下一个内部点").click()
    assert slider.value() == 1
    _button(widget, "上一个内部点").click()
    assert slider.value() == 0

    scheme_path = tmp_path / "scheme.json"
    frame_path = tmp_path / "frame.json"
    trace_path = tmp_path / "trace.json"
    destinations = iter((scheme_path, frame_path, trace_path))
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(next(destinations)), "JSON (*.json)"),
    )
    _button(widget, "保存参数方案").click()
    _button(widget, "导出当前帧").click()
    _button(widget, "导出完整轨迹").click()

    assert json.loads(scheme_path.read_text(encoding="utf-8"))["parameters"]["target_time_s"] == 4435.0
    assert "frame" in json.loads(frame_path.read_text(encoding="utf-8"))
    assert len(json.loads(trace_path.read_text(encoding="utf-8"))["frames"]) == 370
    widget.close()
