"""Developer layout playground for the integrated acceptance APP.

This window intentionally reuses the registered replay data and the same
visual widgets as the acceptance APP, but keeps its layout in nested
``QSplitter`` objects.  Developers can drag the handles, inspect the live
ratios, and save a JSON layout without changing the production page.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import argparse
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from App.core.paths import PATHS
from App.data.dt_loader import DTLoader
from App.data.registry_loader import RegistryLoader
from App.services.replay_service import ReplayService
from App.ui.parameter_format import format_parameter_map
from App.ui.theme import PALETTE, stylesheet
from App.ui.web_view import Embedded3DView, configure_webengine_environment
from App.ui.widgets.chart_panel import build_chart
from App.ui.widgets.cluster_view import create_cluster_share_chart
from App.ui.widgets.decision_card import create_decision_card, update_decision_card
from App.ui.widgets.parameter_panel import create_parameter_panel, update_parameter_panel
from App.ui.widgets.reward_panel import create_reward_panel, update_reward_panel
from App.ui.widgets.status_card import Panel
from App.ui.widgets.timeline_control import create_timeline_control
from App.ui.pages.integrated_page import _chart_series, _control_series


DEFAULT_LAYOUT_FILE = PATHS.app_outputs / "layout_test_layout.json"


def create_layout_test_window(
    controller: ReplayService,
    registry: RegistryLoader,
    *,
    layout_file: Path = DEFAULT_LAYOUT_FILE,
):
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPushButton,
        QScrollArea,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )

    class LayoutTestWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.layout_file = Path(layout_file)
            self.setWindowTitle("智能压裂预测 · 程序员布局测试 APP")
            self.resize(1680, 1120)
            self.setMinimumSize(1280, 720)
            self.setStyleSheet(
                stylesheet("Microsoft YaHei UI")
                + f"""
                QSplitter::handle {{ background:{PALETTE['border']}; }}
                QSplitter::handle:hover {{ background:{PALETTE['cyan']}; }}
                QSplitter::handle:horizontal {{ width:8px; }}
                QSplitter::handle:vertical {{ height:8px; }}
                QPushButton#layoutTool {{ padding:6px 12px; }}
                QLabel#layoutHint {{ color:{PALETTE['muted']}; }}
                """
            )
            self._splitters = {}
            self._text_bindings = {}
            self._text_defaults = {}
            self._text_display_names = {}
            self._custom_texts = {}
            self._custom_text_layout = None
            self._build()
            self._connect_layout_tracking()
            QTimer.singleShot(100, self._load_layout)

        def _build(self):
            root = QWidget()
            outer = QVBoxLayout(root)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)

            header = QFrame()
            header.setObjectName("topbar")
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(18, 12, 18, 12)
            title = QLabel("INTELLIGENT FRACTURING  /  程序员布局测试 APP")
            title.setStyleSheet(f"font-size:20px;font-weight:800;color:{PALETTE['cyan']};")
            self._register_text("header.title", title, "顶部标题")
            header_layout.addWidget(title)
            dataset = registry.dataset()
            stage = QLabel(
                f"测试数据  {dataset.get('well_id') or dataset.get('display_name', '--')}"
                f" · Stage {dataset.get('stage_id') or '--'}"
            )
            stage.setStyleSheet(f"color:{PALETTE['muted']};padding:5px 10px;")
            header_layout.addWidget(stage)
            header_layout.addStretch(1)
            disclaimer = QLabel("仅用于布局调整，不改变正式 APP")
            self._register_text("header.disclaimer", disclaimer, "顶部说明")
            header_layout.addWidget(disclaimer)
            outer.addWidget(header)

            body = QHBoxLayout()
            body.setContentsMargins(0, 0, 0, 0)
            sidebar = QListWidget()
            sidebar.setObjectName("sidebar")
            sidebar.setFixedWidth(220)
            for name in (
                "联合动态演示",
                "第一部分 · 参数预测与风险",
                "第二部分 · 数字孪生",
                "第三部分 · 智能决策",
            ):
                item = QListWidgetItem(name)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                sidebar.addItem(item)
            sidebar.item(0).setSelected(True)
            body.addWidget(sidebar)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            content = QWidget()
            content.setMinimumHeight(1180)
            content_layout = QVBoxLayout(content)
            content_layout.setContentsMargins(18, 14, 18, 18)
            scroll.setWidget(content)
            body.addWidget(scroll, 1)
            outer.addLayout(body, 1)
            self.setCentralWidget(root)

            title_row = QHBoxLayout()
            page_title = QLabel("联合动态演示 · 布局测试")
            page_title.setObjectName("pageTitle")
            self._register_text("page.title", page_title, "页面标题")
            title_row.addWidget(page_title)
            title_row.addStretch(1)
            self.layout_status = QLabel("拖动分隔条调整比例")
            self.layout_status.setObjectName("layoutHint")
            title_row.addWidget(self.layout_status)
            content_layout.addLayout(title_row)

            tools = QHBoxLayout()
            save = QPushButton("保存布局")
            load = QPushButton("加载布局")
            reset = QPushButton("恢复默认")
            edit_text = QPushButton("编辑文字")
            self._register_text("toolbar.save", save, "工具栏·保存布局")
            self._register_text("toolbar.load", load, "工具栏·加载布局")
            self._register_text("toolbar.reset", reset, "工具栏·恢复默认")
            self._register_text("toolbar.edit_text", edit_text, "工具栏·编辑文字")
            for button in (save, load, reset, edit_text):
                button.setObjectName("layoutTool")
            tools.addWidget(save)
            tools.addWidget(load)
            tools.addWidget(reset)
            tools.addWidget(edit_text)
            tools.addWidget(QLabel(f"布局文件：{self.layout_file}"))
            tools.addStretch(1)
            content_layout.addLayout(tools)
            save.clicked.connect(self._save_layout)
            load.clicked.connect(self._load_layout)
            reset.clicked.connect(self._reset_layout)
            edit_text.clicked.connect(self._open_text_editor)

            scenario_label = QLabel(
                "当前场景："
                + ("有 DAS：压力 + 分簇观测校验" if registry.scenario_id == "das_cluster_observation" else "无 DAS：压力在线校正")
                + "　|　拖动以下分隔条即可测试页面比例"
            )
            scenario_label.setObjectName("muted")
            content_layout.addWidget(scenario_label)

            main_splitter = QSplitter(Qt.Orientation.Vertical)
            main_splitter.setChildrenCollapsible(False)
            main_splitter.setHandleWidth(8)
            self._splitters["main_vertical"] = main_splitter

            upper = QSplitter(Qt.Orientation.Horizontal)
            upper.setChildrenCollapsible(False)
            upper.setHandleWidth(8)
            self._splitters["upper_horizontal"] = upper

            charts = QSplitter(Qt.Orientation.Vertical)
            charts.setChildrenCollapsible(False)
            charts.setHandleWidth(8)
            self._splitters["charts_vertical"] = charts

            pressure_chart = build_chart("井底压力 / MPa", 260, y_min=0.0)
            flow_chart = build_chart("排量 / m³/min", 260, y_min=0.0)
            sand_chart = build_chart("砂比 / %", 260, y_min=0.0)
            for chart in (pressure_chart, flow_chart, sand_chart):
                charts.addWidget(chart)
            charts.setSizes([300, 300, 300])

            three_d, three_d_layout = Panel.create("数字孪生三维状态")
            three_d.setMinimumWidth(360)
            model = Embedded3DView.create(registry.html(registry.scenario_id) or PATHS.dt_html)
            model.setMinimumHeight(720)
            three_d_layout.addWidget(model, 1)
            upper.addWidget(charts)
            upper.addWidget(three_d)
            upper.setSizes([1040, 520])
            upper.setStretchFactor(0, 2)
            upper.setStretchFactor(1, 1)

            middle = QSplitter(Qt.Orientation.Horizontal)
            middle.setChildrenCollapsible(False)
            middle.setHandleWidth(8)
            self._splitters["middle_horizontal"] = middle
            cluster_chart = create_cluster_share_chart()
            middle_right = QSplitter(Qt.Orientation.Vertical)
            middle_right.setChildrenCollapsible(False)
            middle_right.setHandleWidth(8)
            self._splitters["middle_right_vertical"] = middle_right
            decision = create_decision_card()
            reward = create_reward_panel()
            middle_right.addWidget(decision)
            middle_right.addWidget(reward)
            middle_right.setSizes([180, 240])
            middle.addWidget(cluster_chart)
            middle.addWidget(middle_right)
            middle.setSizes([760, 800])

            params = create_parameter_panel(
                "EnKF 参数更新",
                columns=[
                    ("先验参数", [("prior", "")]),
                    ("后验参数", [("posterior", "")]),
                    (
                        "状态与簇级结果",
                        [
                            ("cluster_balance", "分簇均衡指数"),
                            ("fracture_length", "裂缝长度(m)"),
                            ("fracture_width", "最大缝宽(mm)"),
                            ("prior_half_lengths", "先验半长(m)"),
                            ("posterior_half_lengths", "后验半长(m)"),
                            ("error", "后验误差"),
                            ("runtime", "计算时间"),
                        ],
                    ),
                ],
                column_stretches=[3, 3, 2],
            )

            custom_panel, custom_panel_layout = Panel.create("自定义文字")
            self._custom_text_layout = QVBoxLayout()
            self._custom_text_layout.setContentsMargins(0, 0, 0, 0)
            custom_panel_layout.addLayout(self._custom_text_layout)
            content_layout.addWidget(custom_panel)

            main_splitter.addWidget(upper)
            main_splitter.addWidget(middle)
            main_splitter.addWidget(params)
            main_splitter.setSizes([720, 220, 260])
            main_splitter.setStretchFactor(0, 3)
            main_splitter.setStretchFactor(1, 1)
            main_splitter.setStretchFactor(2, 1)
            content_layout.addWidget(main_splitter, 1)

            timeline = create_timeline_control(controller)
            content_layout.addWidget(timeline)

            self._register_remaining_text_widgets()

            self._charts = (pressure_chart, flow_chart, sand_chart)
            self._model = model
            self._params = params
            self._cluster_chart = cluster_chart
            self._decision = decision
            self._reward = reward
            self._timeline = timeline
            controller.frameChanged.connect(self._update_frame)
            self._refresh_charts()
            self._update_frame(controller.current or {})

        def _register_text(self, key, widget, display_name=None):
            """Register a visible static widget in the text editor."""

            if key in self._text_bindings:
                return
            self._text_bindings[key] = widget
            self._text_defaults[key] = str(widget.text())
            self._text_display_names[key] = display_name or str(widget.text())
            widget.setProperty("layout_text_key", key)

        def _register_remaining_text_widgets(self):
            """Register static headings/captions without exposing live values."""

            from PySide6.QtWidgets import QLabel

            allowed_names = {"sectionTitle", "key", "caption"}
            index = 1
            for widget in self.findChildren(QLabel):
                if widget.property("layout_text_key") or widget.objectName() not in allowed_names:
                    continue
                text = str(widget.text())
                if not text.strip():
                    continue
                key = f"label.auto.{index:03d}"
                self._register_text(key, widget, text.replace("\n", " / "))
                index += 1

        def _create_custom_text(self, text, custom_id=None, visible=True):
            if self._custom_text_layout is None:
                return None
            text = str(text)
            if custom_id is None:
                index = 1
                while f"custom.{index}" in self._custom_texts:
                    index += 1
                custom_id = f"custom.{index}"
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QLabel

            label = QLabel(text)
            label.setObjectName("layoutCustomText")
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            label.setProperty("layout_text_key", custom_id)
            label.setVisible(bool(visible))
            self._custom_text_layout.addWidget(label)
            self._custom_texts[custom_id] = {"widget": label, "default": text}
            return custom_id

        def _clear_custom_texts(self):
            for item in self._custom_texts.values():
                widget = item.get("widget")
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()
            self._custom_texts = {}

        def _prompt_add_custom_text(self):
            from PySide6.QtWidgets import QInputDialog

            text, accepted = QInputDialog.getMultiLineText(
                self,
                "新增文字",
                "输入文字（支持换行）：",
                "",
            )
            if accepted and text.strip():
                self._create_custom_text(text)

        def _open_text_editor(self):
            from PySide6.QtWidgets import (
                QCheckBox,
                QDialog,
                QFrame,
                QHBoxLayout,
                QLabel,
                QPlainTextEdit,
                QPushButton,
                QScrollArea,
                QVBoxLayout,
                QWidget,
            )

            dialog = QDialog(self)
            dialog.setWindowTitle("编辑测试版文字")
            dialog.resize(720, 760)
            layout = QVBoxLayout(dialog)
            layout.addWidget(QLabel("可编辑页面标题、面板标题和说明文字；文字框支持换行。取消勾选“显示”即可删除该文字。"))

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            editor_body = QWidget()
            editor_layout = QVBoxLayout(editor_body)
            editor_layout.setContentsMargins(8, 8, 8, 8)
            scroll.setWidget(editor_body)
            layout.addWidget(scroll, 1)
            rows = []

            bindings = [(key, widget, self._text_display_names.get(key, key)) for key, widget in self._text_bindings.items()]
            for custom_id, item in self._custom_texts.items():
                bindings.append((custom_id, item["widget"], f"自定义文字 · {custom_id}"))

            for key, widget, name in bindings:
                frame = QFrame()
                frame_layout = QVBoxLayout(frame)
                frame_layout.setContentsMargins(4, 4, 4, 4)
                header = QHBoxLayout()
                header.addWidget(QLabel(name))
                header.addStretch(1)
                visible = QCheckBox("显示")
                visible.setChecked(not widget.isHidden())
                header.addWidget(visible)
                delete = QPushButton("删除")
                delete.setObjectName("layoutTool")
                header.addWidget(delete)
                editor = QPlainTextEdit(str(widget.text()))
                editor.setFixedHeight(62)
                editor.setPlaceholderText("输入文字；可直接回车换行")
                delete.clicked.connect(
                    lambda _checked=False, box=visible, field=editor: (box.setChecked(False), field.clear())
                )
                frame_layout.addLayout(header)
                frame_layout.addWidget(editor)
                editor_layout.addWidget(frame)
                rows.append((key, editor, visible))
            editor_layout.addStretch(1)

            buttons = QHBoxLayout()
            add = QPushButton("新增文字")
            apply = QPushButton("应用")
            cancel = QPushButton("取消")
            for button in (add, apply, cancel):
                button.setObjectName("layoutTool")
            add.clicked.connect(self._prompt_add_custom_text)
            cancel.clicked.connect(dialog.reject)
            apply.clicked.connect(dialog.accept)
            buttons.addWidget(add)
            buttons.addStretch(1)
            buttons.addWidget(apply)
            buttons.addWidget(cancel)
            layout.addLayout(buttons)

            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            for key, editor, visible in rows:
                if key in self._text_bindings:
                    widget = self._text_bindings[key]
                    widget.setText(editor.toPlainText())
                    widget.setVisible(visible.isChecked())
                elif key in self._custom_texts:
                    item = self._custom_texts[key]
                    item["widget"].setText(editor.toPlainText())
                    item["widget"].setVisible(visible.isChecked())
            self.layout_status.setText("文字已应用；点击“保存布局”后写入布局文件")

        def _capture_texts(self):
            widgets = {
                key: {"text": str(widget.text()), "visible": not widget.isHidden()}
                for key, widget in self._text_bindings.items()
            }
            custom = [
                {
                    "id": key,
                    "text": str(item["widget"].text()),
                    "visible": not item["widget"].isHidden(),
                }
                for key, item in self._custom_texts.items()
            ]
            return {"widgets": widgets, "custom": custom}

        def _apply_texts(self, payload):
            if not isinstance(payload, dict):
                return
            widgets = payload.get("widgets", {})
            if isinstance(widgets, dict):
                for key, item in widgets.items():
                    widget = self._text_bindings.get(str(key))
                    if widget is None or not isinstance(item, dict):
                        continue
                    if "text" in item:
                        widget.setText(str(item.get("text", "")))
                    if "visible" in item:
                        widget.setVisible(bool(item.get("visible")))
            self._clear_custom_texts()
            custom = payload.get("custom", [])
            if isinstance(custom, list):
                for item in custom:
                    if not isinstance(item, dict) or not str(item.get("text", "")).strip():
                        continue
                    self._create_custom_text(
                        item.get("text", ""),
                        custom_id=str(item.get("id", "")) or None,
                        visible=bool(item.get("visible", True)),
                    )

        def _reset_texts(self):
            for key, widget in self._text_bindings.items():
                widget.setText(self._text_defaults.get(key, ""))
                widget.setVisible(True)
            self._clear_custom_texts()

        def _connect_layout_tracking(self):
            for splitter in self._splitters.values():
                splitter.splitterMoved.connect(lambda _pos, _index: self._update_layout_status())
            self._update_layout_status()

        def _refresh_charts(self):
            frames = controller.frames
            chart_values, chart_end = _chart_series(registry, frames)
            pressure, flow, sand = self._charts
            pressure.set_series(chart_values)
            if chart_end is not None:
                pressure.set_time_range(1.0, chart_end)
            sample_count = len(chart_values[0][1]) if chart_values else len(frames)
            controls = _control_series(registry, chart_end or float(max(len(frames), 1)), sample_count)
            if len(controls) >= 4:
                flow.set_series(controls[:2])
                sand.set_series(controls[2:])
            for chart in (flow, sand):
                if chart_end is not None:
                    chart.set_time_range(1.0, chart_end)

        def _update_frame(self, frame):
            frame = frame or {}
            if not controller.frames:
                return
            progress = controller.index / max(len(controller.frames) - 1, 1)
            for chart in self._charts:
                chart.set_progress(progress)
            dt = frame.get("dt", {}) or {}
            update_parameter_panel(
                self._params,
                {
                    "prior": format_parameter_map(dt.get("prior_parameters")),
                    "posterior": format_parameter_map(dt.get("posterior_parameters")),
                    "cluster_balance": dt.get("cluster_balance_degree"),
                    "fracture_length": dt.get("fracture_length_m") or _total_half_length(dt.get("posterior_half_lengths_m")),
                    "fracture_width": float(dt.get("fracture_width_m")) * 1000.0 if dt.get("fracture_width_m") is not None else None,
                    "prior_half_lengths": _half_length_text(dt.get("prior_half_lengths_m")),
                    "posterior_half_lengths": _half_length_text(dt.get("posterior_half_lengths_m")),
                    "error": dt.get("posterior_error"),
                    "runtime": dt.get("runtime_ms"),
                },
                {
                    "cluster_balance": "{:.3f}",
                    "fracture_length": "{:.2f} m",
                    "fracture_width": "{:.3f} mm",
                    "error": "{:.3f}",
                    "runtime": "{:.1f} ms",
                },
            )
            self._cluster_chart.set_frame(frame)
            update_decision_card(self._decision, frame)
            update_reward_panel(self._reward, frame)
            self._model.set_time_index(frame.get("time_s", 0))

        def _capture_layout(self):
            payload = {
                "version": 1,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "window_size": [self.width(), self.height()],
                "splitters": {},
                "texts": self._capture_texts(),
            }
            for name, splitter in self._splitters.items():
                sizes = [max(int(value), 0) for value in splitter.sizes()]
                total = max(sum(sizes), 1)
                payload["splitters"][name] = {
                    "orientation": "horizontal" if splitter.orientation() == Qt.Orientation.Horizontal else "vertical",
                    "sizes_px": sizes,
                    "ratios": [round(value / total, 6) for value in sizes],
                }
            return payload

        def _save_layout(self):
            self.layout_file.parent.mkdir(parents=True, exist_ok=True)
            self.layout_file.write_text(json.dumps(self._capture_layout(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.layout_status.setText(f"已保存：{self.layout_file.name}")

        def _load_layout(self):
            try:
                payload = json.loads(self.layout_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError):
                self.layout_status.setText("未找到已保存布局，使用默认比例")
                self._reset_layout(save=False)
                return
            splitters = payload.get("splitters", {}) if isinstance(payload, dict) else {}
            for name, splitter in self._splitters.items():
                item = splitters.get(name, {}) if isinstance(splitters, dict) else {}
                ratios = item.get("ratios") if isinstance(item, dict) else None
                if not isinstance(ratios, list) or len(ratios) != splitter.count():
                    continue
                clean = [max(float(value), 0.01) for value in ratios]
                total = sum(clean)
                available = splitter.height() if splitter.orientation() == Qt.Orientation.Vertical else splitter.width()
                if available <= 0:
                    available = sum(splitter.sizes())
                splitter.setSizes([max(20, int(available * value / total)) for value in clean])
            self._apply_texts(payload.get("texts", {}) if isinstance(payload, dict) else {})
            self.layout_status.setText(f"已加载：{self.layout_file.name}")
            QTimer.singleShot(80, self._update_layout_status)

        def _reset_layout(self, save=False):
            self._splitters["main_vertical"].setSizes([720, 220, 260])
            self._splitters["upper_horizontal"].setSizes([1040, 520])
            self._splitters["charts_vertical"].setSizes([300, 300, 300])
            self._splitters["middle_horizontal"].setSizes([760, 800])
            self._splitters["middle_right_vertical"].setSizes([180, 240])
            self._reset_texts()
            if save:
                self._save_layout()
            else:
                self.layout_status.setText("已恢复默认比例（未覆盖保存文件）")
            QTimer.singleShot(80, self._update_layout_status)

        def _update_layout_status(self):
            def ratios(name):
                sizes = self._splitters[name].sizes()
                total = max(sum(sizes), 1)
                return "/".join(f"{100 * value / total:.0f}%" for value in sizes)

            self.layout_status.setText(
                "主行 " + ratios("main_vertical")
                + " · 上部 " + ratios("upper_horizontal")
                + " · 图表 " + ratios("charts_vertical")
                + " · 风险/安全 " + ratios("middle_horizontal")
                + " · 决策/奖励 " + ratios("middle_right_vertical")
            )

    return LayoutTestWindow()


def _half_length_text(values):
    if not values:
        return "缺失 · 未接入"
    return "<br>".join(f"簇{index + 1} = {float(value):.3f} m" for index, value in enumerate(values))


def _total_half_length(values):
    if not values:
        return None
    try:
        return float(sum(float(value) for value in values))
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="联合动态演示 APP 的可拖动布局测试工具")
    parser.add_argument(
        "--layout-file",
        default=str(DEFAULT_LAYOUT_FILE),
        help="布局 JSON 保存/加载路径，默认 outputs/app/layout_test_layout.json",
    )
    parser.add_argument("--smoke", action="store_true", help="启动并自动退出，用于 GUI smoke 验证")
    args = parser.parse_args()
    # Keep direct launches consistent with the production launcher.  The
    # target environment can be overridden with FRACTURING_QT_PYTHON.
    qt_python = Path(os.environ.get("FRACTURING_QT_PYTHON", r"C:\Users\xinonome\anaconda3\envs\frac_app\python.exe"))
    if os.environ.get("FRACTURING_LAYOUT_TEST_RELAUNCHED") != "1" and qt_python.exists() and Path(sys.executable).resolve() != qt_python.resolve():
        env = os.environ.copy()
        env["FRACTURING_LAYOUT_TEST_RELAUNCHED"] = "1"
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        return subprocess.call([str(qt_python), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=PROJECT_ROOT, env=env)

    configure_webengine_environment()
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    registry = RegistryLoader()
    controller = ReplayService(registry)
    layout_file = Path(args.layout_file)
    if not layout_file.is_absolute():
        layout_file = PROJECT_ROOT / layout_file
    window = create_layout_test_window(controller, registry, layout_file=layout_file)
    window.show()
    if args.smoke:
        QTimer.singleShot(3000, app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
