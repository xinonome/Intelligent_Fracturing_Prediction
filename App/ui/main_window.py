"""Main PySide6 window for the integrated industrial-console APP."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from ..core.paths import PATHS
from ..data.registry_loader import RegistryLoader
from .pages.dt_page import build_dt_page
from .pages.fsl_page import build_fsl_page
from .pages.hmi_page import build_hmi_page
from .pages.integrated_page import build_integrated_page
from .theme import PALETTE, stylesheet


def create_main_window(controller, registry: RegistryLoader, *, html_path: Path | None = None):
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            config = _read_config(PATHS.ui_config)
            window = config.get("window", {})
            self.setWindowTitle(window.get("title", "智能压裂预测 · 联合动态联调平台"))
            self.resize(int(window.get("width", 1680)), int(window.get("height", 980)))
            self.setMinimumSize(int(window.get("minimum_width", 1280)), int(window.get("minimum_height", 720)))
            self.setStyleSheet(stylesheet(config.get("font_family", "Microsoft YaHei UI")))
            self._build()
            self._clock = QTimer(self)
            self._clock.timeout.connect(self._update_clock)
            self._clock.start(1000)
            self._update_clock()

        def _build(self):
            root = QWidget()
            outer = QVBoxLayout(root)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)
            header = QFrame()
            header.setObjectName("topbar")
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(18, 12, 18, 12)
            title = QLabel("INTELLIGENT FRACTURING  /  联合动态联调平台")
            title.setStyleSheet(f"font-size:20px;font-weight:800;color:{PALETTE['cyan']};")
            header_layout.addWidget(title)
            header_layout.addWidget(_header_label("井段  <authorized-well> · <authorized-stage>"))
            header_layout.addStretch(1)
            self.clock_label = _header_label("--")
            header_layout.addWidget(self.clock_label)
            outer.addWidget(header)

            body = QHBoxLayout()
            body.setContentsMargins(0, 0, 0, 0)
            self.navigation = QListWidget()
            self.navigation.setObjectName("sidebar")
            self.navigation.setFixedWidth(220)
            self.pages = QStackedWidget()
            body.addWidget(self.navigation)
            body.addWidget(self.pages, 1)
            outer.addLayout(body, 1)
            self.setCentralWidget(root)

            specs = [
                ("联合动态演示", build_integrated_page(controller, registry, html_path or PATHS.dt_html)),
                ("第一部分 · 工况识别", build_fsl_page(registry)),
                ("第二部分 · 数字孪生", build_dt_page(controller, registry)),
                ("第三部分 · 智能决策", build_hmi_page(controller, registry)),
            ]
            for name, page in specs:
                self.navigation.addItem(QListWidgetItem(name))
                self.pages.addWidget(page)
            self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex)
            self.navigation.setCurrentRow(0)

        def _update_clock(self):
            self.clock_label.setText(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

    return MainWindow()


def _read_config(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _header_label(text: str, color: str | None = None):
    from PySide6.QtWidgets import QLabel

    label = QLabel(text)
    label.setStyleSheet(f"color:{color or PALETTE['muted']};padding:5px 10px;")
    return label
