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
from .pages.research_workbench import build_research_workbench
from .pages.resource_center import build_resource_center
from .theme import set_theme, stylesheet
from .widgets.mode_center import create_mode_center, create_tabbed_center


def create_main_window(
    controller,
    registry: RegistryLoader,
    *,
    html_path: Path | None = None,
    edition: str = "integrated",
    theme: str | None = None,
):
    from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot, QSettings
    from PySide6.QtWidgets import (
        QComboBox,
        QFrame,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QStackedWidget,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )

    class DatasetWorker(QObject):
        ready = Signal(str, str, object)
        failed = Signal(str)

        def __init__(self, dataset_id):
            super().__init__()
            self.dataset_id = dataset_id
            self.agent_model_id = controller.agent_model_id

        @Slot()
        def run(self):
            try:
                from ..core.replay import build_replay_frames
                selected = RegistryLoader(registry=registry.registry,
                    scenario_id=registry.scenario_id, snapshot=registry.snapshot)
                selected.set_dataset(self.dataset_id)
                frames = build_replay_frames(selected, agent_model=self.agent_model_id)
                self.ready.emit(self.dataset_id, selected.scenario_id, frames)
            except Exception as exc:
                self.failed.emit(f"井段加载失败：{exc}")

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.edition = edition if edition in {"integrated", "fsl", "dt_hmi"} else "integrated"
            config = _read_config(PATHS.ui_config)
            window = config.get("window", {})
            edition_meta = _edition_metadata(self.edition)
            self.setWindowTitle(edition_meta["window_title"] if self.edition != "integrated" else window.get("title", edition_meta["window_title"]))
            self.resize(int(window.get("width", 1680)), int(window.get("height", 980)))
            self.setMinimumSize(int(window.get("minimum_width", 1280)), int(window.get("minimum_height", 720)))
            self.font_family = config.get("font_family", "Microsoft YaHei UI")
            self.settings = QSettings("IntelligentFracturing", "DesktopWorkbench")
            self.theme = set_theme(theme or self.settings.value("theme", config.get("theme", "light")))
            saved_mode = str(self.settings.value("workspace_mode", "runtime") or "runtime").lower()
            self.workspace_mode = saved_mode if saved_mode in {"runtime", "research"} else "runtime"
            self._dataset_thread = None
            self._dataset_worker = None
            self.setStyleSheet(stylesheet(self.font_family))
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
            header_layout = QVBoxLayout(header)
            header_layout.setContentsMargins(20, 12, 20, 12)
            header_layout.setSpacing(10)
            brand_row = QHBoxLayout()
            edition_meta = _edition_metadata(self.edition)
            title = QLabel(edition_meta["header_title"])
            title.setObjectName("appHeaderTitle")
            title.setWordWrap(True)
            brand_row.addWidget(title, 1)
            self.clock_label = _header_label("--")
            brand_row.addWidget(self.clock_label)
            self.theme_button = QToolButton()
            self.theme_button.setObjectName("themeToggle")
            self.theme_button.setCheckable(True)
            self.theme_button.setCursor(Qt.PointingHandCursor)
            self.theme_button.clicked.connect(self._toggle_theme)
            brand_row.addWidget(self.theme_button)
            header_layout.addLayout(brand_row)
            context_row = QHBoxLayout()
            self.dataset_context_label = _header_label("当前井段")
            context_row.addWidget(self.dataset_context_label)
            self.dataset_selector = QComboBox()
            self.dataset_selector.setObjectName("globalDatasetSelector")
            self.dataset_selector.setMinimumWidth(280)
            self.dataset_selector.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            self.dataset_selector.setMinimumContentsLength(28)
            self._dataset_selector_guard = False
            self._populate_dataset_selector()
            self.dataset_selector.currentIndexChanged.connect(self._select_global_dataset)
            context_row.addWidget(self.dataset_selector, 2)
            self.source_badge = _header_label("")
            self.source_badge.setObjectName("sourceContext")
            context_row.addWidget(self.source_badge, 1)
            self.sources_button = QToolButton()
            self.sources_button.setObjectName("sourcesButton")
            self.sources_button.setText("资源中心" if self.edition == "integrated" else "数据与运行")
            self.sources_button.clicked.connect(self._show_sources)
            context_row.addWidget(self.sources_button)
            header_layout.addLayout(context_row)
            self._refresh_source_context()
            self._sync_theme_button()
            outer.addWidget(header)

            body = QHBoxLayout()
            body.setContentsMargins(0, 0, 0, 0)
            sidebar = QWidget()
            sidebar.setObjectName("sidebarShell")
            sidebar.setFixedWidth(192)
            sidebar_layout = QVBoxLayout(sidebar)
            sidebar_layout.setContentsMargins(8, 10, 8, 8)
            sidebar_layout.setSpacing(8)
            self.workspace_mode_buttons = {}
            if self.edition == "integrated":
                mode_row = QHBoxLayout()
                mode_row.setSpacing(4)
                for mode, text in (("runtime", "运行应用"), ("research", "模型研发")):
                    button = QToolButton()
                    button.setObjectName("workspaceModeButton")
                    button.setCheckable(True)
                    button.setCursor(Qt.PointingHandCursor)
                    button.setText(text)
                    button.setToolTip("切换当前业务中心的工作状态")
                    button.clicked.connect(lambda _checked=False, value=mode: self._set_workspace_mode(value))
                    self.workspace_mode_buttons[mode] = button
                    mode_row.addWidget(button, 1)
                sidebar_layout.addLayout(mode_row)
            self.navigation = QListWidget()
            self.navigation.setObjectName("sidebar")
            self.navigation.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            sidebar_layout.addWidget(self.navigation, 1)
            self._sync_workspace_mode_buttons()
            self.pages = QStackedWidget()
            body.addWidget(sidebar)
            body.addWidget(self.pages, 1)
            outer.addLayout(body, 1)
            self.setCentralWidget(root)

            # Page construction is intentionally lazy.  The integrated page
            # creates a WebEngine view and a large replay timeline; the FSL
            # page reads a multi-megabyte workbook.  Neither should delay the
            # first paint of the application or make an unused tab expensive.
            standard_factories = {
                "fsl": lambda: build_fsl_page(
                    registry,
                    on_imported=self._handle_data_import,
                    on_dataset_requested=self._select_global_dataset_id,
                ),
                "dt": lambda: build_dt_page(controller, registry),
                "hmi": lambda: build_hmi_page(controller, registry),
                "integrated": lambda: build_integrated_page(
                    controller,
                    registry,
                    html_path or PATHS.dt_html,
                    on_dataset_changed=self._update_dataset_header,
                ),
            }
            if self.edition == "integrated":
                all_factories = {
                    "fsl": lambda: create_mode_center(
                        lambda: build_fsl_page(
                            registry,
                            on_imported=self._handle_data_import,
                            on_dataset_requested=self._select_global_dataset_id,
                            include_resource_tabs=False,
                        ),
                        lambda: build_research_workbench("fsl", registry, controller),
                        initial_mode=self.workspace_mode,
                    ),
                    "dt": lambda: create_mode_center(
                        lambda: build_dt_page(controller, registry),
                        lambda: build_research_workbench("dt", registry, controller),
                        initial_mode=self.workspace_mode,
                    ),
                    "hmi": lambda: create_mode_center(
                        lambda: create_tabbed_center([
                            ("风险与建议", build_hmi_page(controller, registry)),
                            (
                                "全流程联动",
                                build_integrated_page(
                                    controller,
                                    registry,
                                    html_path or PATHS.dt_html,
                                    on_dataset_changed=self._update_dataset_header,
                                ),
                            ),
                        ]),
                        lambda: build_research_workbench("hmi", registry, controller),
                        initial_mode=self.workspace_mode,
                    ),
                    "resources": lambda: create_mode_center(
                        lambda: build_resource_center(
                            registry,
                            controller,
                            on_imported=self._handle_data_import,
                            on_dataset_requested=self._select_global_dataset_id,
                        ),
                        lambda: build_resource_center(
                            registry,
                            controller,
                            on_imported=self._handle_data_import,
                            on_dataset_requested=self._select_global_dataset_id,
                            research_mode=True,
                        ),
                        initial_mode=self.workspace_mode,
                    ),
                }
            else:
                all_factories = standard_factories
            edition_meta = _edition_metadata(self.edition)
            self._page_factories = [all_factories[key] for key, _icon, _name in edition_meta["pages"]]
            self._built_pages = {}
            nav_items = tuple((icon, name) for _key, icon, name in edition_meta["pages"])
            for icon, name in nav_items:
                item_text = f"{icon}  {name}" if icon else name
                self.navigation.addItem(QListWidgetItem(item_text))
                placeholder = QLabel("页面准备中…")
                placeholder.setAlignment(Qt.AlignCenter)
                placeholder.setObjectName("muted")
                self.pages.addWidget(placeholder)
            _enable_text_copy(root)
            self.navigation.currentRowChanged.connect(self._show_page)
            self.navigation.setCurrentRow(0)
            QTimer.singleShot(0, lambda: self._show_page(0))

        def _handle_data_import(self):
            """Refresh lazy data pages after a successful table import.

            The import page calls this while it is still handling its own
            button event.  Defer replacement by one event-loop turn so the
            current page is never deleted from inside its own slot.
            """

            registry.refresh_catalog()
            QTimer.singleShot(0, self._refresh_data_pages)

        def _refresh_data_pages(self):
            self._populate_dataset_selector()
            # Never destroy a page that may own a running training process.
            current_index = self.navigation.currentRow()
            for index, page in self._built_pages.items():
                if index == current_index:
                    continue
                refresh = getattr(page, "refresh_catalog", None)
                if callable(refresh):
                    refresh()
            self._refresh_source_context()

        def _show_page(self, index: int):
            if index < 0 or index >= len(self._page_factories):
                return
            if index not in self._built_pages:
                page = self._page_factories[index]()
                old = self.pages.widget(index)
                self._built_pages[index] = page
                self.pages.removeWidget(old)
                self.pages.insertWidget(index, page)
                old.deleteLater()
                _enable_text_copy(page)
                mode_callback = getattr(page, "set_workspace_mode", None)
                if callable(mode_callback):
                    mode_callback(self.workspace_mode)
            self.pages.setCurrentIndex(index)
            refresh_dataset = getattr(self.pages.currentWidget(), "refresh_dataset_visual", None)
            if callable(refresh_dataset):
                refresh_dataset()
            self._sync_dataset_selection_policy()

        def _sync_dataset_selection_policy(self):
            """Keep the global selector honest about task-specific inputs."""

            current = self.pages.currentWidget() if hasattr(self, "pages") else None
            locked_callback = getattr(current, "dataset_selection_locked", None)
            message_callback = getattr(current, "dataset_selection_message", None)
            locked = bool(locked_callback()) if callable(locked_callback) else False
            message = str(message_callback() or "") if callable(message_callback) else ""
            self.dataset_context_label.setText("当前任务输入" if locked else "当前井段")
            if locked:
                self.dataset_selector.setEnabled(False)
                self.dataset_selector.setToolTip(message)
                self.source_badge.setText("当前任务输入：JY84-Z1 · Stage 08")
                self.source_badge.setToolTip(message)
            else:
                self.dataset_selector.setToolTip(str(registry.dataset().get("pressure_source") or ""))
                if self._dataset_thread is None and not self._dataset_selector_guard:
                    self.dataset_selector.setEnabled(True)
                self._refresh_source_context()

        def _update_dataset_header(self, dataset):
            selected = self.dataset_selector.findData(str(dataset.get("dataset_id", "")))
            if selected >= 0 and self.dataset_selector.currentIndex() != selected:
                self.dataset_selector.blockSignals(True)
                self.dataset_selector.setCurrentIndex(selected)
                self.dataset_selector.blockSignals(False)
            self._sync_dataset_selection_policy()

        def _populate_dataset_selector(self):
            """Build the one dataset selector shared by every application page."""

            self.dataset_selector.blockSignals(True)
            self.dataset_selector.clear()
            catalog = registry.dataset_catalog() if hasattr(registry, "dataset_catalog") else {}
            entries = []
            for dataset_id, dataset in sorted((catalog.get("datasets", {}) or {}).items()):
                if dataset.get("data_scope") == "composite":
                    continue
                ready = bool(registry.dataset_source_ready(dataset_id))
                label = _dataset_identity(dict(dataset, dataset_id=dataset_id))
                if not ready:
                    label += " · 数据未准备"
                self.dataset_selector.addItem(label, str(dataset_id))
                item = self.dataset_selector.model().item(self.dataset_selector.count() - 1)
                if item is not None:
                    item.setEnabled(ready)
                    item.setToolTip(str(dataset.get("pressure_source") or "未登记原始数据"))
                entries.append((str(dataset_id), ready))
            if not entries:
                self.dataset_selector.addItem("暂无可用井段", "")
                item = self.dataset_selector.model().item(0)
                if item is not None:
                    item.setEnabled(False)
            else:
                selected = self.dataset_selector.findData(str(getattr(registry, "dataset_id", "")))
                self.dataset_selector.setCurrentIndex(selected if selected >= 0 else 0)
            self.dataset_selector.blockSignals(False)

        def _select_global_dataset(self, index):
            self._select_global_dataset_id(self.dataset_selector.itemData(index))

        def _select_global_dataset_id(self, dataset_id):
            dataset_id = str(dataset_id or "")
            if not dataset_id or self._dataset_selector_guard:
                return
            if dataset_id == str(getattr(registry, "dataset_id", "")):
                self._notify_dataset_pages(dataset_id)
                return
            selected = registry.dataset(dataset_id)
            if not registry.dataset_source_ready(dataset_id):
                self._populate_dataset_selector()
                return
            self._dataset_selector_guard = True
            for page in self._built_pages.values():
                pause = getattr(page, "pause_playback", None)
                if callable(pause):
                    pause()
                prepare = getattr(page, "prepare_dataset_change", None)
                if callable(prepare):
                    prepare()
            self.dataset_selector.setEnabled(False)
            self.pages.setEnabled(False)
            self.navigation.setEnabled(False)
            for button in self.workspace_mode_buttons.values():
                button.setEnabled(False)
            self.source_badge.setText("正在读取井段数据…")
            self._dataset_thread = QThread(self)
            self._dataset_worker = DatasetWorker(dataset_id)
            self._dataset_worker.moveToThread(self._dataset_thread)
            self._dataset_thread.started.connect(self._dataset_worker.run)
            self._dataset_worker.ready.connect(self._commit_dataset)
            self._dataset_worker.failed.connect(self._dataset_error)
            self._dataset_worker.ready.connect(self._dataset_thread.quit)
            self._dataset_worker.failed.connect(self._dataset_thread.quit)
            self._dataset_thread.finished.connect(self._dataset_worker.deleteLater)
            self._dataset_thread.finished.connect(self._dataset_finished)
            self._dataset_thread.start()

        @Slot(str, str, object)
        def _commit_dataset(self, dataset_id, scenario_id, frames):
            # Commit the new frame list without emitting while multiple
            # WebEngine-backed pages are being notified.  A deferred single
            # frame update prevents simultaneous Plotly document navigation.
            controller.apply_scenario_frames(scenario_id, frames, dataset_id, emit=False)
            self._update_dataset_header(registry.dataset())
            self._notify_dataset_pages(dataset_id)
            QTimer.singleShot(0, lambda: controller.timeline.set_index(0, emit=True))

        @Slot(str)
        def _dataset_error(self, message):
            self._populate_dataset_selector()
            self.source_badge.setText("加载失败 · 可重新选择")
            self.source_badge.setToolTip(message)
            self.statusBar().showMessage(message, 15000)

        @Slot()
        def _dataset_finished(self):
            self._dataset_worker = None
            if self._dataset_thread is not None:
                self._dataset_thread.deleteLater()
            self._dataset_thread = None
            self._dataset_selector_guard = False
            self._sync_dataset_selection_policy()
            self.pages.setEnabled(True)
            self.navigation.setEnabled(True)
            for button in self.workspace_mode_buttons.values():
                button.setEnabled(True)

        def _refresh_source_context(self):
            dataset = registry.dataset()
            scene = "有 DAS" if dataset.get("fiber_source") else "无 DAS · 模型推导"
            cached = bool(registry.frame_source())
            self.source_badge.setText(f"{scene} · {'缓存回放' if cached else '等待推演'}")
            self.source_badge.setToolTip(str(dataset.get("pressure_source") or ""))

        def _show_sources(self):
            if self.edition == "integrated":
                for index, (key, _icon, _name) in enumerate(_edition_metadata(self.edition)["pages"]):
                    if key == "resources":
                        self.navigation.setCurrentRow(index)
                        return
            from .widgets.workspace_sources import show_workspace_sources
            show_workspace_sources(self, registry, controller, edition=self.edition)

        def _set_workspace_mode(self, mode: str):
            mode = str(mode or "runtime").lower()
            if self.edition != "integrated" or mode not in {"runtime", "research"}:
                return
            self.workspace_mode = mode
            for page in self._built_pages.values():
                callback = getattr(page, "set_workspace_mode", None)
                if callable(callback):
                    callback(mode)
            self.settings.setValue("workspace_mode", mode)
            self._sync_workspace_mode_buttons()
            self._sync_dataset_selection_policy()
            self.statusBar().showMessage("已切换到运行应用" if mode == "runtime" else "已切换到模型研发", 4000)

        def _sync_workspace_mode_buttons(self):
            for mode, button in self.workspace_mode_buttons.items():
                button.blockSignals(True)
                button.setChecked(mode == self.workspace_mode)
                button.blockSignals(False)

        def closeEvent(self, event):
            if self._dataset_thread is not None and self._dataset_thread.isRunning():
                self.statusBar().showMessage("正在完成井段读取，请稍后关闭。")
                event.ignore()
                return
            super().closeEvent(event)

        def _notify_dataset_pages(self, dataset_id):
            """Tell already-built pages that the global stage has changed."""

            for page in self._built_pages.values():
                callback = getattr(page, "set_global_dataset", None)
                if callable(callback):
                    callback(str(dataset_id))

        def _update_clock(self):
            self.clock_label.setText(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

        def _toggle_theme(self):
            self.theme = set_theme("dark" if self.theme == "light" else "light")
            self.setStyleSheet(stylesheet(self.font_family))
            self._sync_theme_button()
            self.settings.setValue("theme", self.theme)
            for page in self._built_pages.values():
                update_theme = getattr(page, "refresh_theme", None)
                if callable(update_theme):
                    update_theme()
                page.update()

        def _sync_theme_button(self):
            is_light = self.theme == "light"
            self.theme_button.setChecked(not is_light)
            self.theme_button.setText("☀" if is_light else "☾")
            self.theme_button.setToolTip(
                "当前为浅色主题，点击切换到暗色" if is_light else "当前为暗色主题，点击切换到浅色"
            )

    return MainWindow()


def _edition_metadata(edition: str) -> dict:
    """Return the visible product boundary for each software edition."""

    editions = {
        "integrated": {
            "window_title": "智能压裂精准调控平台",
            "header_title": "智能压裂精准调控平台",
            "pages": (
                ("fsl", "", "工况与风险"),
                ("dt", "", "裂缝与参数"),
                ("hmi", "", "智能调控"),
                ("resources", "", "资源中心"),
            ),
        },
        "fsl": {
            "window_title": "压裂施工工况识别与风险预测软件 V1.0",
            "header_title": "压裂施工工况识别与风险预测软件 V1.0",
            "pages": (
                ("fsl", "▣", "工况识别与风险预测"),
            ),
        },
        "dt_hmi": {
            "window_title": "智能压裂双场景数字孪生与安全建议软件 V1.0",
            "header_title": "智能压裂双场景数字孪生与安全建议软件 V1.0",
            "pages": (
                ("dt", "◇", "双场景数字孪生"),
                ("hmi", "", "智能风险与安全建议"),
                ("integrated", "◉", "全流程联动"),
            ),
        },
    }
    return editions.get(edition, editions["integrated"])


def _read_config(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _header_label(text: str, color: str | None = None):
    from PySide6.QtWidgets import QLabel

    label = QLabel(text)
    label.setObjectName("headerLabel")
    if color:
        label.setStyleSheet(f"color:{color};padding:5px 10px;")
    return label


def _dataset_identity(dataset: dict) -> str:
    """Use one concise well/stage label in the global header."""

    if dataset.get("adapter") == "raw_frac_construction":
        stage = str(dataset.get("stage_id") or dataset.get("dataset_id") or "未命名井段")
        well = str(dataset.get("well_id") or "")
        return f"{well} · {stage}" if well else stage
    well = str(dataset.get("well_id") or dataset.get("display_name") or dataset.get("dataset_id") or "未知井段")
    stage = str(dataset.get("stage_id") or "")
    return f"{well} · Stage {stage}" if stage else well


def _enable_text_copy(root) -> None:
    """Make displayed Qt text selectable without affecting controls."""

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel, QTableWidget

    for label in root.findChildren(QLabel):
        label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        label.setCursor(Qt.IBeamCursor)

    for table in root.findChildren(QTableWidget):
        table.setSelectionMode(QTableWidget.ExtendedSelection)
        table.setSelectionBehavior(QTableWidget.SelectItems)
        table.setFocusPolicy(Qt.StrongFocus)
