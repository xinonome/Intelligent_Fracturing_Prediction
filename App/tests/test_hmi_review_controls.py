from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox

from App.data.registry_loader import RegistryLoader
from App.services.replay_service import ReplayService
from App.ui.pages.hmi_page import build_hmi_page


def test_review_flow_up_arrow_is_preserved_during_frame_refresh() -> None:
    app = QApplication.instance() or QApplication([])
    service = ReplayService(RegistryLoader())
    page = build_hmi_page(service, service.registry)
    page.resize(1200, 800)
    page.show()
    app.processEvents()

    flow = page.findChild(QDoubleSpinBox, "hmiReviewFlow")
    assert flow is not None
    flow.setValue(2.95)
    app.processEvents()
    before = flow.value()

    QTest.mouseClick(
        flow,
        Qt.MouseButton.LeftButton,
        pos=QPoint(flow.width() - 8, 5),
    )
    app.processEvents()
    after = flow.value()

    # A replay refresh must not immediately replace the user's arrow edit.
    service.frameChanged.emit(service.current)
    app.processEvents()
    assert after > before
    assert flow.value() == after
    page.close()
