"""First-part construction timeline and stage result table widgets."""

from __future__ import annotations

import math
import json
from statistics import median

from ..theme import PALETTE
from ..display_text import display_text


def _finite(value) -> bool:
    return value is not None and math.isfinite(float(value))


def _fmt(value, suffix="", digits=1):
    return "—" if not _finite(value) else f"{float(value):.{digits}f}{suffix}"


def _label_color(label: str) -> str:
    return {
        "主缝延伸": "#48A868",
        "缝内暂堵": "#F2A93B",
        "缝口暂堵": "#E05252",
        "缝高延伸": "#4D9DE0",
        "延伸受阻": "#B44F9E",
        "滤失过大": "#D97835",
        "压力异常": "#D1495B",
        "砂堵": "#C0392B",
        "其他": "#78909C",
    }.get(label, PALETTE["muted"])


def _nice_axis_max(values) -> float:
    """Round an axis maximum to a readable 1/2/5 × 10ⁿ value."""

    valid = [float(value) for value in values if _finite(value) and float(value) > 0.0]
    if not valid:
        return 1.0
    maximum = max(valid)
    magnitude = 10.0 ** math.floor(math.log10(maximum))
    normalized = maximum / magnitude
    for candidate in (1.0, 2.0, 5.0, 10.0):
        if normalized <= candidate:
            return candidate * magnitude
    return 10.0 * magnitude


class FSLTimelineChart:
    """QFrame painter for a pressure/flow/sand timeline with event bands."""

    def __new__(cls, *args, **kwargs):
        from PySide6.QtCore import QPointF, Qt, Signal
        from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
        from PySide6.QtWidgets import QFrame, QMenu, QSizePolicy, QApplication

        class Timeline(QFrame):
            intervalSelected = Signal(object)

            def __init__(self):
                super().__init__()
                self.setObjectName("chartPanel")
                self.setMinimumHeight(540)
                self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.stage_data = {}
                self.current_index = 0
                self.selected_interval = None
                self._time_axis_cache = None
                self.setContextMenuPolicy(Qt.CustomContextMenu)
                self.customContextMenuRequested.connect(self._show_copy_menu)

            def set_stage(self, stage):
                self.stage_data = dict(stage or {})
                self.current_index = 0
                self.selected_interval = None
                self._time_axis_cache = None
                self.update()

            def set_index(self, index):
                times = self.stage_data.get("time_s", [])
                self.current_index = max(0, min(int(index), max(len(times) - 1, 0)))
                self.update()

            def mousePressEvent(self, event):
                plot = self._plot_rect()
                if self.stage_data and plot.contains(event.position().toPoint()):
                    duration = max(float(self.stage_data.get("duration_s") or 0.0), 1.0)
                    seconds = self._time_from_x(event.position().x(), plot, duration)
                    all_intervals = (
                        list(self.stage_data.get("predicted_condition_intervals", []))
                        + list(self.stage_data.get("rule_condition_intervals", []))
                        + list(self.stage_data.get("actual_condition_intervals", self.stage_data.get("intervals", [])))
                    )
                    selected = next(
                        (
                            item for item in all_intervals
                            if float(item.get("start_s") or 0.0) <= seconds <= float(item.get("end_s") or 0.0)
                        ),
                        None,
                    )
                    self.selected_interval = selected
                    self.intervalSelected.emit(selected or {})
                    self.update()
                super().mousePressEvent(event)

            def _show_copy_menu(self, position):
                menu = QMenu(self)
                action = menu.addAction("复制当前井段时序数据")
                action.triggered.connect(self._copy_data)
                menu.exec(self.mapToGlobal(position))

            def _copy_data(self):
                if not self.stage_data:
                    return
                rows = [["井段", self.stage_data.get("stage_id", "--")], ["时间(s)", *self.stage_data.get("time_s", [])]]
                for key, title in (
                    ("pressure_mpa", "施工泵压实测(MPa)"),
                    ("predicted_pressure_mpa", "施工泵压趋势基线(MPa)"),
                    ("flow_m3_min", "排量实测(m³/min)"),
                    ("predicted_flow_m3_min", "排量趋势基线(m³/min)"),
                    ("sand_ratio_pct", "砂比(%)"),
                    ("predicted_sand_ratio_pct", "砂比趋势基线(%)"),
                ):
                    rows.append([title, *["" if value is None else f"{float(value):.6g}" for value in self.stage_data.get(key, [])]])
                rows.append(["实际工况区间", json.dumps(self.stage_data.get("actual_condition_intervals", []), ensure_ascii=False)])
                rows.append(["预测工况区间", json.dumps(self.stage_data.get("predicted_condition_intervals", []), ensure_ascii=False)])
                rows.append(["规则检测区间", json.dumps(self.stage_data.get("rule_condition_intervals", []), ensure_ascii=False)])
                rows.append(["预测方法", self.stage_data.get("point_prediction_source", "未接入")])
                QApplication.clipboard().setText("\n".join("\t".join(str(item) for item in row) for row in rows))

            def paintEvent(self, _event):
                painter = QPainter(self)
                painter.setRenderHint(QPainter.Antialiasing)
                painter.fillRect(self.rect(), QColor(PALETTE["panel"]))
                if not self.stage_data:
                    painter.setPen(QColor(PALETTE["muted"]))
                    painter.drawText(18, 28, "暂无工况时序数据")
                    return

                stage_id = self.stage_data.get("stage_id", "--")
                duration = max(float(self.stage_data.get("duration_s") or 0.0), 1.0)
                time_axis = self._time_axis()
                interruptions = time_axis["interruptions"]
                painter.setPen(QColor(PALETTE["text"]))
                chart_title = self.stage_data.get("chart_title") or "施工工况与逐点预测"
                painter.drawText(16, 26, display_text(f"{chart_title} · 井段 {stage_id}"))
                painter.setPen(QColor(PALETTE["muted"]))
                interruption_note = f"   |   已压缩 {len(interruptions)} 段施工中断" if interruptions else ""
                painter.drawText(16, 47, f"{self.stage_data.get('start_time', '--')} — {self.stage_data.get('end_time', '--')}   |   {int(duration)} s / {self.stage_data.get('sample_count', 0)} 点{interruption_note}")

                # Solid lines are measured values; dashed lines are only
                # listed when a real point-prediction series exists.  This
                # keeps a pressure-only replay honest instead of showing a
                # legend entry for a curve that is not drawn.
                pressure = self.stage_data.get("pressure_mpa", [])
                flow = self.stage_data.get("flow_m3_min", [])
                sand = self.stage_data.get("sand_ratio_pct", [])
                predicted_pressure = self.stage_data.get("predicted_pressure_mpa", [])
                predicted_flow = self.stage_data.get("predicted_flow_m3_min", [])
                predicted_sand = self.stage_data.get("predicted_sand_ratio_pct", [])

                def has_values(values):
                    return any(_finite(value) for value in values)

                legend = []
                if has_values(pressure):
                    legend.append(("压力实测", PALETTE["blue"], Qt.SolidLine))
                if has_values(predicted_pressure):
                    legend.append(("压力基线", "#C084FC", Qt.DashLine))
                if has_values(flow):
                    legend.append(("排量实测", PALETTE["cyan"], Qt.SolidLine))
                if has_values(predicted_flow):
                    legend.append(("排量基线", "#76E6A7", Qt.DashLine))
                if has_values(sand):
                    legend.append(("砂比实测", PALETTE["orange"], Qt.SolidLine))
                if has_values(predicted_sand):
                    legend.append(("砂比基线", "#E8B87B", Qt.DashLine))
                x_legend = 16
                for name, color, style in legend:
                    painter.setPen(QPen(QColor(color), 3, style))
                    painter.drawLine(x_legend, 64, x_legend + 16, 64)
                    painter.setPen(QColor(color))
                    painter.drawText(x_legend + 22, 68, name)
                    x_legend += 92
                plot = self._plot_rect()
                p_max = _nice_axis_max(pressure + predicted_pressure)
                q_max = _nice_axis_max(flow + predicted_flow)
                s_max = _nice_axis_max(sand + predicted_sand)

                self._draw_intervals(painter, plot, duration)
                painter.setPen(QPen(QColor(PALETTE["border"]), 1))
                for tick in range(6):
                    ratio = tick / 5.0
                    y = plot.bottom() - plot.height() * ratio
                    painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
                    painter.setPen(QColor(PALETTE["blue"]))
                    painter.drawText(10, int(y + 4), f"{p_max * ratio:.0f}")
                    painter.setPen(QColor(PALETTE["cyan"]))
                    painter.drawText(int(plot.right() + 10), int(y + 4), f"{q_max * ratio:.1f}")
                    painter.setPen(QColor(PALETTE["orange"]))
                    painter.drawText(int(plot.right() + 92), int(y + 4), f"{s_max * ratio:.1f}")
                    painter.setPen(QPen(QColor(PALETTE["border"]), 1))
                axis_title_y = int(plot.top() - 24)
                painter.setPen(QColor(PALETTE["blue"]))
                painter.drawText(10, axis_title_y, "压力 / MPa")
                painter.setPen(QColor(PALETTE["cyan"]))
                painter.drawText(int(plot.right() + 10), axis_title_y, "排量")
                painter.drawText(int(plot.right() + 10), axis_title_y + 16, "m³/min")
                painter.setPen(QColor(PALETTE["orange"]))
                painter.drawText(int(plot.right() + 92), axis_title_y, "砂比")
                painter.drawText(int(plot.right() + 92), axis_title_y + 16, "%")

                self._draw_series(painter, plot, pressure, p_max, PALETTE["blue"])
                self._draw_series(painter, plot, predicted_pressure, p_max, "#C084FC", Qt.DashLine)
                self._draw_series(painter, plot, flow, q_max, PALETTE["cyan"])
                self._draw_series(painter, plot, predicted_flow, q_max, "#76E6A7", Qt.DashLine)
                self._draw_series(painter, plot, sand, s_max, PALETTE["orange"])
                self._draw_series(painter, plot, predicted_sand, s_max, "#E8B87B", Qt.DashLine)
                self._draw_interruptions(painter, plot, interruptions)

                times = self.stage_data.get("time_s", [])
                if times:
                    marker_x = self._x(times[self.current_index], plot, duration)
                    painter.setPen(QPen(QColor(PALETTE["text"]), 2, Qt.DashLine))
                    painter.drawLine(QPointF(marker_x, plot.top()), QPointF(marker_x, plot.bottom()))

                painter.setPen(QColor(PALETTE["muted"]))
                painter.drawText(int(plot.left()), self.height() - 16, "t=0 s")
                end_text = f"t={int(duration)} s"
                painter.drawText(int(plot.right() - painter.fontMetrics().horizontalAdvance(end_text)), self.height() - 16, end_text)

            def _plot_rect(self):
                return self.rect().adjusted(78, 108, -172, -42)

            def _x(self, value, plot, duration):
                display_duration = max(float(self._time_axis()["display_duration"]), 1.0)
                display_time = self._display_time(float(value))
                return plot.left() + plot.width() * min(max(display_time / display_duration, 0.0), 1.0)

            def _time_axis(self):
                """Build a display axis that compresses long no-sample gaps.

                The source timestamps remain unchanged.  Only the visual x
                coordinate is remapped so a long shutdown/pause does not
                consume most of the chart width.
                """

                if self._time_axis_cache is not None:
                    return self._time_axis_cache
                raw_times = []
                for value in self.stage_data.get("time_s", []):
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        continue
                    if not raw_times or number >= raw_times[-1]:
                        raw_times.append(number)
                if len(raw_times) < 2:
                    self._time_axis_cache = {
                        "times": raw_times,
                        "interruptions": [],
                        "display_duration": max(raw_times[-1] if raw_times else 1.0, 1.0),
                    }
                    return self._time_axis_cache

                gaps = [right - left for left, right in zip(raw_times, raw_times[1:]) if right > left]
                typical_gap = median(gaps) if gaps else 1.0
                # Normal sampled data may be tens of seconds apart.  Only a
                # clearly abnormal gap is collapsed, avoiding distortion of
                # ordinary stage dynamics.
                threshold = max(300.0, typical_gap * 10.0)
                marker_duration = max(60.0, min(180.0, typical_gap * 2.0))
                interruptions = []
                removed = 0.0
                mapped_times = [raw_times[0]]
                for index, (left, right) in enumerate(zip(raw_times, raw_times[1:])):
                    gap = right - left
                    if gap > threshold:
                        interruptions.append(
                            {
                                "start_s": left,
                                "end_s": right,
                                "duration_s": gap,
                                "display_start_s": left - removed,
                                "display_end_s": left - removed + marker_duration,
                            }
                        )
                        removed += gap - marker_duration
                    mapped_times.append(right - removed)
                self._time_axis_cache = {
                    "times": mapped_times,
                    "interruptions": interruptions,
                    "display_duration": max(mapped_times[-1], 1.0),
                }
                return self._time_axis_cache

            def _display_time(self, value: float) -> float:
                """Map a real timestamp to the compressed display axis."""

                removed = 0.0
                for gap in self._time_axis()["interruptions"]:
                    if value <= gap["start_s"]:
                        break
                    if value < gap["end_s"]:
                        ratio = (value - gap["start_s"]) / max(gap["duration_s"], 1e-9)
                        return gap["display_start_s"] + ratio * (gap["display_end_s"] - gap["display_start_s"])
                    removed += gap["duration_s"] - (gap["display_end_s"] - gap["display_start_s"])
                return value - removed

            def _time_from_x(self, x, plot, duration: float) -> float:
                """Convert a display coordinate back to a real timestamp."""

                axis = self._time_axis()
                display_duration = max(float(axis["display_duration"]), 1.0)
                display_time = display_duration * (float(x) - plot.left()) / max(plot.width(), 1.0)
                removed = 0.0
                for gap in axis["interruptions"]:
                    start = gap["display_start_s"]
                    end = gap["display_end_s"]
                    if display_time < start:
                        break
                    if display_time <= end:
                        return float(gap["start_s"])
                    removed += gap["duration_s"] - (end - start)
                return min(max(display_time + removed, 0.0), duration)

            def _draw_interruptions(self, painter, plot, interruptions):
                """Draw interruption markers in a dedicated strip above the plot.

                Event labels occupy the upper part of the plot itself.  Keep
                interruption text in the margin between the legend and the
                plotting area so the two annotation types never compete for
                the same pixels.
                """

                if not interruptions:
                    return

                # Two compact lanes make the annotation readable even when
                # more than one shutdown gap is present.  The strip is kept
                # below the legend and above the first event label.
                label_y = [plot.top() - 25, plot.top() - 9]
                occupied = [[], []]
                metrics = painter.fontMetrics()
                for gap in sorted(interruptions, key=lambda item: float(item.get("start_s") or 0.0)):
                    left = self._x(gap["start_s"], plot, 1.0)
                    right = self._x(gap["end_s"], plot, 1.0)
                    marker_left = min(left, right)
                    marker_right = max(right, left)
                    marker_width = max(10, int(marker_right - marker_left))
                    painter.fillRect(
                        int(marker_left),
                        plot.top(),
                        marker_width,
                        plot.height(),
                        QColor(PALETTE["muted"] + "35"),
                    )
                    painter.setPen(QPen(QColor(PALETTE["muted"]), 1, Qt.DashLine))
                    painter.drawLine(QPointF(marker_left, plot.top()), QPointF(marker_left, plot.bottom()))
                    painter.drawLine(QPointF(marker_right, plot.top()), QPointF(marker_right, plot.bottom()))
                    center = (marker_left + marker_right) / 2.0
                    painter.setPen(QPen(QColor(PALETTE["orange"]), 2))
                    painter.drawLine(QPointF(center - 5, plot.top() + 8), QPointF(center, plot.top() + 2))
                    painter.drawLine(QPointF(center, plot.top() + 2), QPointF(center + 5, plot.top() + 8))
                    label = f"施工中断 {int(round(gap['duration_s']))} s"
                    label_width = metrics.horizontalAdvance(label) + 10
                    preferred_x = center - label_width / 2.0
                    label_x = max(float(plot.left()), min(preferred_x, float(plot.right() - label_width)))

                    # Pick a free lane.  If both lanes are occupied, keep the
                    # label anchored to its marker and shorten it instead of
                    # allowing it to overlap another annotation.
                    lane = None
                    for lane_index, items in enumerate(occupied):
                        if all(label_x + label_width + 4 < item_left or label_x > item_right + 4 for item_left, item_right in items):
                            lane = lane_index
                            break
                    if lane is None:
                        label = f"中断 {int(round(gap['duration_s']))}s"
                        label_width = metrics.horizontalAdvance(label) + 8
                        label_x = max(float(plot.left()), min(preferred_x, float(plot.right() - label_width)))
                        lane = 0

                    occupied[lane].append((label_x, label_x + label_width))
                    baseline = label_y[lane]
                    label_rect_top = baseline - metrics.ascent() - 2
                    painter.fillRect(
                        int(label_x),
                        int(label_rect_top),
                        int(label_width),
                        int(metrics.height() + 4),
                        QColor(PALETTE["panel"] + "E6"),
                    )
                    painter.setPen(QColor(PALETTE["muted"]))
                    painter.drawText(int(label_x + 5), int(baseline), label)
                    # Short connector from the dedicated annotation strip to
                    # the interruption marker; it does not enter the event
                    # label area.
                    painter.setPen(QPen(QColor(PALETTE["orange"]), 1, Qt.DotLine))
                    painter.drawLine(QPointF(center, baseline + 3), QPointF(center, plot.top() - 3))

            def _draw_intervals(self, painter, plot, duration):
                actual = self.stage_data.get("actual_condition_intervals", self.stage_data.get("intervals", []))
                predicted = self.stage_data.get("predicted_condition_intervals", [])
                rules = self.stage_data.get("rule_condition_intervals", [])
                intervals = [(item, "actual") for item in actual] + [(item, "predicted") for item in predicted] + [(item, "rule") for item in rules]
                for index, (interval, kind) in enumerate(intervals):
                    start = self._x(interval.get("start_s", 0.0), plot, duration)
                    end = self._x(interval.get("end_s", interval.get("start_s", 0.0)), plot, duration)
                    if end < start:
                        start, end = end, start
                    color = QColor(_label_color(str(interval.get("label", ""))))
                    selected = interval is self.selected_interval
                    color.setAlpha(110 if selected else (45 if kind == "actual" else 95))
                    if kind == "actual":
                        band_top, band_height = plot.top(), plot.height()
                    else:
                        band_top, band_height = plot.bottom() - 22, 22
                    painter.fillRect(int(start), int(band_top), max(2, int(end - start)), int(band_height), color)
                    painter.setPen(QPen(QColor(_label_color(str(interval.get("label", "")))), 2 if selected else 1, Qt.DashLine))
                    painter.drawLine(QPointF(start, plot.top()), QPointF(start, plot.bottom()))
                    if end - start > 26:
                        painter.setPen(QColor(_label_color(str(interval.get("label", "")))))
                        prefix = {"predicted": "模型:", "rule": "规则:", "actual": "标签:"}[kind]
                        text = prefix + str(interval.get("label", ""))
                        painter.drawText(int(start + 4), int(plot.top() + 18 + (index % 3) * 16), display_text(text))

            def _draw_series(self, painter, plot, values, maximum, color, line_style=None):
                points = []
                times = self.stage_data.get("time_s", [])
                for index, value in enumerate(values):
                    if not _finite(value):
                        if len(points) > 1:
                            self._paint_path(painter, points, color, line_style)
                        points = []
                        continue
                    time_value = float(times[index] if index < len(times) else index)
                    x = self._x(time_value, plot, float(self.stage_data.get("duration_s") or 1.0))
                    y = plot.bottom() - plot.height() * float(value) / max(maximum, 1e-9)
                    points.append((x, y))
                if len(points) > 1:
                    self._paint_path(painter, points, color, line_style)

            @staticmethod
            def _paint_path(painter, points, color, line_style=None):
                path = QPainterPath()
                path.moveTo(points[0][0], points[0][1])
                for x, y in points[1:]:
                    path.lineTo(x, y)
                dashed = line_style == Qt.DashLine
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(color), 3 if dashed else 2, line_style or Qt.SolidLine))
                painter.drawPath(path)
                if dashed:
                    painter.setPen(QPen(QColor(color), 1))
                    painter.setBrush(QColor(color))
                    marker_step = max(1, len(points) // 20)
                    for x, y in points[::marker_step]:
                        painter.drawEllipse(QPointF(x, y), 2.5, 2.5)
                    painter.setBrush(Qt.NoBrush)

        return Timeline()


def build_stage_table(rows):
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem

    class StageTable(QTableWidget):
        stageSelected = Signal(str)

        def __init__(self):
            super().__init__(0, 6)
            self.setObjectName("fslStageTable")
            self.setHorizontalHeaderLabels(["井段", "源表工况", "源表提示", "源表概率/置信度", "时间", "峰值 P / Q / S"])
            self.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.setSelectionMode(QAbstractItemView.SingleSelection)
            self.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.setAlternatingRowColors(True)
            self.setWordWrap(True)
            self.setTextElideMode(Qt.ElideNone)
            self.verticalHeader().setVisible(False)
            self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            self.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
            self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
            self.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
            self.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
            self.itemSelectionChanged.connect(self._emit_selection)
            self.setMinimumHeight(540)

        def set_rows(self, stage_rows):
            was_blocked = self.blockSignals(True)
            self.setRowCount(0)
            for row_index, row in enumerate(stage_rows):
                self.insertRow(row_index)
                risk = _fmt(row.get("risk_max_pct"), "%", 0) if row.get("risk_max_pct") is not None else "未接入"
                condition_labels = row.get("source_condition_labels") or []
                condition_text = "、".join(
                    display_text(label) for label in condition_labels if str(label).strip()
                ) or "\\"
                suggestion_text = str(row.get("suggestion") or "").strip() or "\\"
                risk_text = risk if risk != "未接入" else "\\"
                values = [
                    f"{row.get('stage_id', '--')}",
                    condition_text,
                    suggestion_text,
                    risk_text,
                    f"{int(row.get('duration_s') or 0)} s" if row.get("duration_s") is not None else "\\",
                    f"{_fmt(row.get('pressure_max'), '', 1)} / {_fmt(row.get('flow_max'), '', 1)} / {_fmt(row.get('sand_max'), '', 1)}".replace("—", "\\"),
                ]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setData(Qt.UserRole, str(row.get("stage_id", "")))
                    item.setToolTip(value)
                    self.setItem(row_index, column, item)
            self.resizeColumnsToContents()
            # Keep the prompt column readable without forcing the table wider
            # than its viewport; the other columns use their content width.
            prompt_column = self.horizontalHeader().sectionSize(2)
            if prompt_column < 180:
                self.horizontalHeader().resizeSection(2, 180)
            self.resizeRowsToContents()
            self.blockSignals(was_blocked)

        def resizeEvent(self, event):
            super().resizeEvent(event)
            # Stretch-column geometry is finalized after the table resize.
            # Recalculate wrapped row heights on the next event turn.
            from PySide6.QtCore import QTimer

            QTimer.singleShot(0, self.resizeRowsToContents)

        def select_stage(self, stage_id):
            """Select and reveal a stage when the external selector changes."""

            target = str(stage_id or "")
            for row_index in range(self.rowCount()):
                item = self.item(row_index, 0)
                if item and str(item.data(Qt.UserRole) or item.text()) == target:
                    was_blocked = self.blockSignals(True)
                    self.selectRow(row_index)
                    self.blockSignals(was_blocked)
                    self.scrollToItem(item, QAbstractItemView.PositionAtCenter)
                    return

        def _emit_selection(self):
            rows = self.selectionModel().selectedRows()
            if rows:
                item = self.item(rows[0].row(), 0)
                if item:
                    self.stageSelected.emit(str(item.data(Qt.UserRole) or item.text()))

    table = StageTable()
    table.set_rows(rows)
    return table
