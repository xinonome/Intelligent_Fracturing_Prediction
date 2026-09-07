"""Render every lazy APP workspace and save a compact visual QA report."""

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
    from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QTabWidget
    from App.data.registry_loader import RegistryLoader
    from App.services.replay_service import ReplayService
    from App.ui.main_window import create_main_window

    output = ROOT / "outputs" / "app" / "ui_smoke_application" / "application_architecture_20260906"
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    service = ReplayService(RegistryLoader())
    window = create_main_window(service, service.registry, theme="light")
    window._set_workspace_mode("runtime")
    window.resize(1680, 980)
    window.show()
    pages = []

    def capture(index: int, mode: str = "runtime") -> None:
        if index >= window.navigation.count():
            if mode == "runtime":
                window._set_workspace_mode("research")
                app.processEvents()
                QTimer.singleShot(600, lambda: capture(0, "research"))
                return
            window.theme_button.click()
            app.processEvents()
            window.grab().save(str(output / "05_dark_theme.png"), "PNG")
            report = {
                "status": "passed",
                "dataset_id": service.registry.dataset_id,
                "page_count": window.navigation.count(),
                "pages": pages,
                "workspace_modes": ["runtime", "research"],
                "theme_toggle": window.theme,
            }
            (output / "qa_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            window._set_workspace_mode("runtime")
            app.quit()
            return
        window.navigation.setCurrentRow(index)
        app.processEvents()
        page = window.pages.currentWidget()
        labels = [label.text() for label in page.findChildren(QLabel) if label.text().strip()]
        pages.append({
            "index": index,
            "workspace_mode": mode,
            "navigation": window.navigation.item(index).text(),
            "tabs": [tab.tabText(i) for tab in page.findChildren(QTabWidget) for i in range(tab.count())],
            "enabled_buttons": [button.text() for button in page.findChildren(QPushButton) if button.isEnabled() and button.text().strip()],
            "visible_text_sample": labels[:12],
        })
        prefix = "运行" if mode == "runtime" else "研发"
        window.grab().save(str(output / f"{prefix}_{index + 1:02d}_{window.navigation.item(index).text()}.png"), "PNG")
        QTimer.singleShot(800, lambda: capture(index + 1, mode))

    QTimer.singleShot(800, lambda: capture(0))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
