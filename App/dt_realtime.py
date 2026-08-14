"""Compact dynamic second-part dashboard for the PySide acceptance APP.

The module only reads ``dt_realtime_cache.json``.  Scientific preprocessing is
performed by ``build_dt_realtime_cache.py`` with the algorithm environment,
so the Qt environment remains a small UI-only runtime.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "outputs" / "app" / "dt_realtime_cache.json"
HTML_PATH = ROOT / "outputs" / "app" / "dt_realtime_3d.html"


def _ui_font_family():
    """Load a Windows CJK font when Qt runs without a desktop font registry."""

    from PySide6.QtGui import QFontDatabase

    candidates = [
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id >= 0:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                return families[0]
    return "Microsoft YaHei UI"


def load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {"error": f"同步缓存不存在：{CACHE_PATH}"}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": f"同步缓存读取失败：{type(exc).__name__}: {exc}"}


def _float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_dt_realtime_panel(html_path: Path | None = None):
    """Return a self-contained QWidget implementing the DT dynamic replay."""

    from PySide6.QtCore import QPointF, QRect, Qt, QTimer, QUrl
    from PySide6.QtGui import QColor, QFont, QPainter, QPen
    from PySide6.QtWidgets import (
        QComboBox,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    font_family = _ui_font_family()
    html_path = html_path or HTML_PATH
    data = load_cache()

    def arr(name: str) -> list[float]:
        return [float(value) for value in data.get("arrays", {}).get(name, [])]

    timeline = [int(value) for value in data.get("timeline_s", [])]
    arrays = data.get("arrays", {})
    clusters = data.get("clusters", {})
    meta = data.get("meta", {})

    def value(name: str, index: int, default: float = 0.0) -> float:
        values = arrays.get(name, [])
        if not values:
            return default
        return _float(values[max(0, min(index, len(values) - 1))], default)

    def cluster_max_length(index: int) -> float:
        values = []
        for record in clusters.values():
            series = record.get("posterior_half_length_m", [])
            if series:
                values.append(_float(series[max(0, min(index, len(series) - 1))]))
        return max(values or [0.0])

    def cluster_mean_allocation(index: int) -> float:
        values = []
        for record in clusters.values():
            series = record.get("fiber_liquid_allocation", record.get("posterior_cluster_factor", []))
            if series:
                values.append(_float(series[max(0, min(index, len(series) - 1))]))
        return sum(values) / len(values) if values else 0.0

    def fixed_range(values: list[float], padding=0.05):
        finite = [float(v) for v in values if math.isfinite(float(v))]
        if not finite:
            return 0.0, 1.0
        low, high = min(finite), max(finite)
        if math.isclose(low, high):
            delta = max(abs(low) * 0.05, 1.0)
        else:
            delta = (high - low) * padding
        return low - delta, high + delta

    class Panel(QFrame):
        def __init__(self, title: str):
            super().__init__()
            self.setObjectName("dtPanel")
            self.layout = QVBoxLayout(self)
            self.layout.setContentsMargins(14, 12, 14, 12)
            if title:
                heading = QLabel(title)
                heading.setObjectName("dtSectionTitle")
                self.layout.addWidget(heading)

    class TimelineCanvas(Panel):
        def __init__(self):
            super().__init__("")
            self.index = 1 if len(timeline) > 1 else 0
            self.setMinimumHeight(520)
            self.canvas = self

        def set_index(self, index: int):
            self.index = max(0, min(index, max(len(timeline) - 1, 0)))
            self.update()

        def _x(self, index, left, width):
            return left + width * index / max(len(timeline) - 1, 1)

        def _points(self, values, plot, low, high):
            if not values:
                return []
            points = []
            span = max(high - low, 1e-12)
            for index, raw in enumerate(values):
                current = _float(raw)
                y = plot.bottom() - (current - low) / span * plot.height()
                points.append(QPointF(self._x(index, plot.left(), plot.width()), y))
            return points

        def _line(self, painter, points, color, width=2):
            if len(points) < 2:
                return
            painter.setPen(QPen(QColor(color), width))
            for first, second in zip(points, points[1:]):
                painter.drawLine(first, second)

        def _plot(self, painter, rect, title, series, right_series=None):
            painter.setBrush(QColor("#fbfdff"))
            painter.setPen(QPen(QColor("#d7e1eb"), 1))
            painter.drawRoundedRect(rect, 8, 8)
            painter.setPen(QColor("#16324a"))
            painter.setFont(QFont(font_family, 10, QFont.Bold))
            painter.drawText(rect.left() + 12, rect.top() + 18, title)
            plot = rect.adjusted(58, 48, -48 if right_series else -18, -28)
            all_values = [v for _, values, _ in series for v in values]
            low, high = fixed_range(all_values)
            for tick in range(5):
                y = plot.top() + plot.height() * tick / 4
                painter.setPen(QPen(QColor("#e6edf3"), 1))
                painter.drawLine(plot.left(), y, plot.right(), y)
                painter.setPen(QColor("#66788a"))
                label = f"{high - (high - low) * tick / 4:.1f}"
                painter.drawText(rect.left() + 4, int(y + 4), label)
            for name, values, color in series:
                self._line(painter, self._points(values, plot, low, high), color)

            if right_series:
                right_values = [v for _, values, _ in right_series for v in values]
                right_low, right_high = fixed_range(right_values)
                right_plot = rect.adjusted(58, 30, -18, -28)
                for name, values, color in right_series:
                    self._line(painter, self._points(values, right_plot, right_low, right_high), color)
                painter.setPen(QColor("#66788a"))
                painter.drawText(rect.right() - 42, plot.top() + 4, f"{right_high:.1f}")
                painter.drawText(rect.right() - 42, plot.bottom(), f"{right_low:.1f}")

            marker_x = self._x(self.index, plot.left(), plot.width())
            painter.setPen(QPen(QColor("#dc3545"), 2, Qt.DashLine))
            painter.drawLine(QPointF(marker_x, plot.top()), QPointF(marker_x, plot.bottom()))
            painter.setPen(QColor("#50667a"))
            painter.drawText(rect.left() + 10, rect.bottom() - 8, f"{timeline[0] if timeline else 1} s")
            painter.drawText(rect.right() - 75, rect.bottom() - 8, f"{timeline[-1] if timeline else 0} s")
            legend_x = plot.left() + 8
            legend_y = rect.top() + 37
            painter.setFont(QFont(font_family, 9))
            for name, _, color in series + (right_series or []):
                painter.setPen(QPen(QColor(color), 3))
                painter.drawLine(legend_x, legend_y - 4, legend_x + 18, legend_y - 4)
                painter.setPen(QColor("#334155"))
                painter.drawText(legend_x + 23, legend_y, name)
                legend_x += max(82, len(name) * 10)

        def paintEvent(self, _event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor("#ffffff"))
            if not timeline:
                painter.setPen(QColor("#b42318"))
                painter.drawText(18, 50, data.get("error", "暂无同步时序数据"))
                return
            painter.setPen(QColor("#16324a"))
            painter.setFont(QFont(font_family, 11, QFont.Bold))
            painter.drawText(10, 18, "同步时序：光纤监测与压力换算")
            margin = 8
            gap = 10
            height = (self.height() - 42 - 2 * gap) / 3
            width = self.width() - 2 * margin
            panels = [
                QRect(margin, 28, width, int(height)),
                QRect(margin, 28 + int(height) + gap, width, int(height)),
                QRect(margin, 28 + 2 * (int(height) + gap), width, int(height)),
            ]
            self._plot(
                painter,
                panels[0],
                "光纤累计量（左轴：液量 m³；右轴：砂量 t）",
                [("累计液量", arr("fiber_cumulative_liquid_m3"), "#087f8c")],
                [("累计砂量", arr("fiber_cumulative_sand_t"), "#d97706")],
            )
            self._plot(
                painter,
                panels[1],
                "压力链路（MPa）：施工压力 → 井底压力 → 净压力",
                [
                    ("施工压力", arr("surface_pressure_mpa"), "#2563eb"),
                    ("井底压力", arr("bottomhole_pressure_mpa"), "#dc2626"),
                    ("净压力", arr("net_pressure_mpa"), "#16a34a"),
                ],
            )
            self._plot(
                painter,
                panels[2],
                "后验误差（%）：井底压力",
                [
                    ("压力误差", [v * 100 for v in arr("posterior_bhp_error")], "#dc2626"),
                ],
            )

    class Fracture3DView(Panel):
        def __init__(self, path: Path | None):
            super().__init__("真实三维井轨迹与分簇裂缝模型")
            self.web = None
            self.setMinimumWidth(520)
            # The global APP timeline controls playback. The embedded 3D
            # canvas deliberately has no second timeline or playback toolbar.
            self.setMinimumHeight(460)
            if path and path.exists() and os.environ.get("QT_QPA_PLATFORM") not in {"offscreen", "minimal"}:
                try:
                    from PySide6.QtWebEngineWidgets import QWebEngineView

                    self.web = QWebEngineView(self)
                    self.web.setUrl(QUrl.fromLocalFile(str(path.resolve())))
                    self.layout.addWidget(self.web, 1)
                    return
                except Exception as exc:
                    message = f"三维 WebEngine 加载失败：{type(exc).__name__}: {exc}"
            elif path and path.exists():
                message = "无界面测试模式不启动 WebEngine；桌面启动后显示真实三维模型。"
            else:
                message = "三维模型页面尚未生成，请检查 outputs/app/dt_realtime_3d.html。"
            label = QLabel(message + "\n东向与垂深使用井轨迹坐标；北向为垂深3000 m后的展示坐标（77–1650 m），原始北向保留在缓存中；簇面由PKN后验半缝长生成。")
            label.setObjectName("notice")
            label.setWordWrap(True)
            self.layout.addWidget(label, 1)

        def set_time(self, seconds: int):
            if self.web:
                self.web.page().runJavaScript(f"window.setTimeIndex && window.setTimeIndex({int(seconds)});")

    class ValuePanel(Panel):
        def __init__(self, title: str, rows: list[tuple[str, str]]):
            super().__init__(title)
            self.labels = {}
            grid = QGridLayout()
            grid.setHorizontalSpacing(14)
            grid.setVerticalSpacing(5)
            for row, (key, label) in enumerate(rows):
                key_label = QLabel(label)
                key_label.setObjectName("dtKey")
                value_label = QLabel("--")
                value_label.setObjectName("dtValue")
                value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                grid.addWidget(key_label, row, 0)
                grid.addWidget(value_label, row, 1)
                self.labels[key] = value_label
            self.layout.addLayout(grid)

        def update_values(self, values: dict[str, str]):
            for key, value_label in self.labels.items():
                value_label.setText(values.get(key, "--"))

    class ErrorPanel(Panel):
        def __init__(self):
            super().__init__("误差与计算性能")
            self.labels = {}
            grid = QGridLayout()
            for col, (key, title) in enumerate([
                ("pressure", "井底压力误差"),
                ("length", "缝长误差"),
                ("runtime", "平均计算时间"),
            ]):
                card = QFrame()
                card.setObjectName("dtMetric")
                card_layout = QVBoxLayout(card)
                title_label = QLabel(title)
                title_label.setObjectName("dtMetricTitle")
                value_label = QLabel("--")
                value_label.setObjectName("dtMetricValue")
                note_label = QLabel("")
                note_label.setObjectName("dtMetricNote")
                note_label.setWordWrap(True)
                card_layout.addWidget(title_label)
                card_layout.addWidget(value_label)
                card_layout.addWidget(note_label)
                grid.addWidget(card, 0, col)
                self.labels[key] = (value_label, note_label)
            self.layout.addLayout(grid)

        def update_values(self, values):
            for key, (value_label, note_label) in self.labels.items():
                value_label.setText(values.get(key, ("--", ""))[0])
                note_label.setText(values.get(key, ("", ""))[1])

    class RealtimePanel(QFrame):
        def __init__(self):
            super().__init__()
            self.setObjectName("dtRealtime")
            self.index = 0
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.next_frame)
            root = QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(10)

            if data.get("error"):
                warning = QLabel(data["error"] + "\n请用算法环境运行：python App\\build_dt_realtime_cache.py")
                warning.setObjectName("warning")
                root.addWidget(warning)
                return

            controls_panel = QFrame()
            controls_panel.setObjectName("dtControlPanel")
            controls = QHBoxLayout(controls_panel)
            self.play = QPushButton("▶ 播放")
            self.play.clicked.connect(self.toggle_play)
            self.speed = QComboBox()
            for label, interval in [("0.5×", 200), ("1×", 100), ("2×", 50), ("4×", 20), ("8×", 10)]:
                self.speed.addItem(label, interval)
            self.speed.setCurrentIndex(1)
            self.speed.currentIndexChanged.connect(self.change_speed)
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(0, max(len(timeline) - 1, 0))
            self.slider.valueChanged.connect(self.set_index)
            self.time_spin = QSpinBox()
            self.time_spin.setRange(int(timeline[0]) if timeline else 0, max(timeline or [0]))
            self.time_spin.setSuffix(" s")
            self.time_spin.valueChanged.connect(self.set_time)
            self.total_time_label = QLabel()
            self.total_time_label.setObjectName("dtTotalTime")
            controls.addWidget(self.play)
            controls.addWidget(QLabel("演示倍速"))
            controls.addWidget(self.speed)
            controls.addWidget(QLabel("时间轴"))
            controls.addWidget(self.slider, 1)
            controls.addWidget(self.time_spin)
            controls.addWidget(self.total_time_label)
            root.addWidget(controls_panel)

            upper = QHBoxLayout()
            self.chart = TimelineCanvas()
            self.model3d = Fracture3DView(html_path)
            upper.addWidget(self.chart, 3)
            upper.addWidget(self.model3d, 3)
            root.addLayout(upper)

            middle = QHBoxLayout()
            self.pkn_input = ValuePanel(
                "PKN 当前输入",
                [("eprime", "E′ 平面应变模量"), ("leakoff", "Cₗ 滤失系数"), ("viscosity", "μ 流体黏度"), ("stress", "σmin 最小水平应力"), ("rate", "当前总排量"), ("sand", "当前砂比"), ("height", "H 裂缝高度")],
            )
            self.pkn_output = ValuePanel(
                "PKN 当前输出",
                [
                    ("length", "最大分簇半缝长"),
                    ("pressure", "PKN 井底压力"),
                    ("net", "净压力"),
                    ("factor", "平均光纤液量分配"),
                    ("step", "当前计算步"),
                ],
            )
            self.enkf = ValuePanel(
                "EnKF 参数更新过程",
                [("sequence", "流程"), ("observation", "当前观测"), ("residual", "先验误差"), ("gain", "平均 Kalman Gain"), ("update", "参数更新"), ("rerun", "更新后动作")],
            )
            middle.addWidget(self.pkn_input, 1)
            middle.addWidget(self.pkn_output, 1)
            middle.addWidget(self.enkf, 1)
            root.addLayout(middle)

            self.errors = ErrorPanel()
            root.addWidget(self.errors)
            self.set_index(self.index)

        def record_values(self, index):
            idx = max(0, min(index, max(len(timeline) - 1, 0)))
            prior_eprime = value("prior_eprime_gpa", idx)
            post_eprime = value("posterior_eprime_gpa", idx)
            prior_cl = value("prior_leakoff_m_sqrt_s", idx)
            post_cl = value("posterior_leakoff_m_sqrt_s", idx)
            prior_mu = value("prior_viscosity_pa_s", idx)
            post_mu = value("posterior_viscosity_pa_s", idx)
            prior_stress = value("prior_min_stress_mpa", idx)
            post_stress = value("posterior_min_stress_mpa", idx)
            return {
                "pkn_input": {
                    "eprime": f"{prior_eprime:.3f} GPa",
                    "leakoff": f"{prior_cl:.3e} m/√s",
                    "viscosity": f"{prior_mu:.4f} Pa·s",
                    "stress": f"{prior_stress:.2f} MPa",
                    "rate": f"{value('flow_rate_m3_min', idx):.2f} m³/min",
                    "sand": f"{value('sand_ratio_percent', idx):.2f}%",
                    "height": "30.00 m（固定输入）",
                },
                "pkn_output": {
                    "length": f"{cluster_max_length(idx):.2f} m",
                    "pressure": f"{value('posterior_pkn_bhp_mpa', idx):.2f} MPa",
                    "net": f"{value('net_pressure_mpa', idx):.2f} MPa",
                    "factor": f"{cluster_mean_allocation(idx):.3f}",
                    "step": f"{value('step_compute_ms', idx):.1f} ms",
                },
                "enkf": {
                    "sequence": "预测 → 观测 → 残差 → 增益 → 更新参数 → 重跑 PKN",
                    "observation": f"液量 {value('fiber_cumulative_liquid_m3', idx):.2f} m³；砂量 {value('fiber_cumulative_sand_t', idx):.3f} t；压力 {value('bottomhole_pressure_mpa', idx):.2f} MPa",
                    "residual": f"压力 {value('prior_bhp_error', idx)*100:.2f}%；缝长 {value('posterior_length_error', idx)*100:.2f}%",
                    "gain": f"{value('kalman_gain', idx):.4f}",
                    "update": f"ΔE′ {post_eprime-prior_eprime:+.3f} GPa；ΔCₗ {(post_cl-prior_cl):+.2e}；Δμ {post_mu-prior_mu:+.4f}",
                    "rerun": f"后验 E′ {post_eprime:.3f} GPa；σmin {post_stress:.2f} MPa；缝长 {cluster_max_length(idx):.2f} m",
                },
                "errors": {
                    "pressure": (f"{value('posterior_bhp_error', idx)*100:.2f}%", "井底压力相对误差"),
                    "length": (f"{value('posterior_length_error', idx)*100:.2f}%", "PKN后验半缝长 vs 液量^0.6等效估计"),
                    "runtime": (f"{value('step_compute_ms', idx):.1f} ms", f"全程平均 {sum(arr('step_compute_ms'))/max(len(arr('step_compute_ms')),1):.1f} ms"),
                },
            }

        def set_index(self, index):
            if not timeline:
                return
            self.index = max(0, min(int(index), len(timeline) - 1))
            self.slider.blockSignals(True)
            self.time_spin.blockSignals(True)
            self.slider.setValue(self.index)
            self.time_spin.setValue(timeline[self.index])
            self.slider.blockSignals(False)
            self.time_spin.blockSignals(False)
            self.chart.set_index(self.index)
            self.model3d.set_time(timeline[self.index])
            current_time = timeline[self.index]
            self.total_time_label.setText(f"总时间 {current_time}s / {timeline[-1]}s")
            values = self.record_values(self.index)
            self.pkn_input.update_values(values["pkn_input"])
            self.pkn_output.update_values(values["pkn_output"])
            self.enkf.update_values(values["enkf"])
            self.errors.update_values(values["errors"])

        def set_time(self, seconds):
            if not timeline:
                return
            closest = min(range(len(timeline)), key=lambda idx: abs(timeline[idx] - int(seconds)))
            self.set_index(closest)

        def toggle_play(self):
            if self.timer.isActive():
                self.timer.stop()
                self.play.setText("▶ 播放")
            else:
                if self.index >= len(timeline) - 1:
                    self.set_index(0)
                self.timer.start(int(self.speed.currentData()))
                self.play.setText("⏸ 暂停")

        def change_speed(self):
            if self.timer.isActive():
                self.timer.start(int(self.speed.currentData()))

        def next_frame(self):
            if self.index >= len(timeline) - 1:
                self.timer.stop()
                self.play.setText("▶ 播放")
                return
            self.set_index(self.index + 1)

    panel = RealtimePanel()
    return panel
