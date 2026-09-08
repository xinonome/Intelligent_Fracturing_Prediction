import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QPushButton

from App.data.registry_loader import RegistryLoader
from App.ui.pages.data_import_page import build_data_import_page
from App.ui.pages.fsl_page import build_fsl_page


def _button(widget, text: str) -> QPushButton:
    return next(button for button in widget.findChildren(QPushButton) if button.text() == text)


def _wait_for(app, predicate, timeout_s=20.0):
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert predicate()


def test_large_file_inspection_runs_off_gui_thread(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "large.csv"
    source.write_text("time,pressure,flow,sand\n" + "1,2,3,4\n" * 9000, encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *args: ([str(source)], ""))
    page = build_data_import_page(RegistryLoader())
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(5)
    _button(page, "选择表格").click()
    assert page.active_import_thread() is not None
    _wait_for(app, lambda: page.active_import_thread() is None)
    timer.stop()
    assert ticks, "GUI event loop did not remain responsive during inspection"
    page.close()


def test_reanalyse_runs_off_gui_thread_and_restores_button():
    app = QApplication.instance() or QApplication([])
    page = build_fsl_page(RegistryLoader(), include_resource_tabs=False)
    button = _button(page, "重新读取与分析")
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(5)
    button.click()
    assert not button.isEnabled()
    _wait_for(app, lambda: page._fsl_refresh_thread() is None, timeout_s=90.0)
    timer.stop()
    assert button.isEnabled()
    assert ticks, "GUI event loop did not remain responsive during reanalysis"
    page.close()
