from __future__ import annotations

import math


def _nice_step(raw_step: float) -> float:
    """Return a readable 1/2/2.5/5/10 engineering tick interval."""

    if not math.isfinite(raw_step) or raw_step <= 0:
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = multiple * magnitude
        if step >= raw_step * (1.0 - 1e-12):
            return step
    return 10.0 * magnitude


def _nice_axis_ticks(
    values,
    y_min: float | None = None,
    target_intervals: int = 5,
) -> tuple[float, float, list[float], float]:
    """Build an automatic, readable numeric axis.

    The integrated dashboard passes ``y_min=0`` for pressure, flow and sand
    ratio.  In that mode the upper bound is rounded up to a readable step, so
    values such as 137 become 0/50/100/150 rather than five divisions of a
    padded data maximum.  The helper also supports charts without an explicit
    floor by rounding both ends to the same readable step.
    """

    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    intervals = max(int(target_intervals), 1)
    if not finite:
        step = 1.0
        low = float(y_min) if y_min is not None and math.isfinite(float(y_min)) else 0.0
        high = low + intervals * step
        return low, high, [low + step * index for index in range(intervals + 1)], step

    data_low, data_high = min(finite), max(finite)
    explicit_floor = y_min is not None and math.isfinite(float(y_min))
    low = float(y_min) if explicit_floor else data_low
    if not explicit_floor:
        span = max(data_high - data_low, abs(data_high) * 0.05, 1.0)
    else:
        span = max(data_high - low, 0.0)

    if span <= 1e-12:
        # Keep a flat/all-zero series visible without inventing a negative
        # floor.  The exact scale is still chosen from a readable step.
        step = _nice_step(max(abs(data_high), 1.0) / intervals)
        if explicit_floor:
            high = low + intervals * step
        else:
            low = math.floor(data_low / step) * step
            high = low + intervals * step
    else:
        step = _nice_step(span / intervals)
        if explicit_floor:
            high = max(low + step, math.ceil(data_high / step - 1e-12) * step)
        else:
            low = math.floor(data_low / step) * step
            high = max(low + step, math.ceil(data_high / step - 1e-12) * step)

    count = max(int(round((high - low) / step)), 1)
    high = low + count * step
    ticks = [low + step * index for index in range(count + 1)]
    return low, high, ticks, step


def build_chart(title: str, height: int = 240, y_min: float | None = None):
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QPainter, QPen
    from PySide6.QtWidgets import QFrame, QMenu
    from ..theme import PALETTE

    class LineChart(QFrame):
        def __init__(self):
            super().__init__()
            self.setObjectName("chartPanel")
            self.setMinimumHeight(height)
            self.title = title
            self.y_min = y_min
            self.series: list[tuple[str, list[float], str]] = []
            self.index = 0
            self.progress: float | None = None
            self.time_range: tuple[float, float] | None = None
            self.setMouseTracking(True)
            self.setContextMenuPolicy(Qt.CustomContextMenu)
            self.customContextMenuRequested.connect(self._show_copy_menu)

        def _show_copy_menu(self, position):
            menu = QMenu(self)
            copy_action = menu.addAction("复制图表数据")
            copy_action.triggered.connect(self._copy_data)
            menu.exec(self.mapToGlobal(position))

        def _copy_data(self):
            from PySide6.QtWidgets import QApplication

            lines = [["图表", self.title], ["时间(s)", *self._times()]]
            for name, values, _color in self.series:
                lines.append([name, *["" if not math.isfinite(value) else f"{value:.6g}" for value in values]])
            QApplication.clipboard().setText("\n".join("\t".join(str(item) for item in row) for row in lines))

        def _times(self):
            count = max((len(line) for _name, line, _color in self.series), default=0)
            if count <= 1:
                return [f"{self.time_range[0]:.6g}" if self.time_range else "0"]
            start, end = self.time_range or (0.0, float(count - 1))
            return [f"{start + (end - start) * index / (count - 1):.6g}" for index in range(count)]

        def set_series(self, series):
            # Do not advertise a curve that has no finite samples.  This is
            # important for partial/no-DAS replays: an unavailable
            # recommendation must not leave a coloured legend entry beside a
            # chart that only contains the measured series.
            prepared = []
            for name, values, color in series:
                numeric = [float(v) if v is not None else float("nan") for v in values]
                if not any(math.isfinite(value) for value in numeric):
                    continue
                prepared.append((str(name), numeric, color))
            self.series = prepared
            self.update()

        def set_index(self, index: int):
            self.index = max(0, int(index))
            self.progress = None
            self.update()

        def set_progress(self, progress: float):
            self.progress = min(max(float(progress), 0.0), 1.0)
            self.update()

        def set_time_range(self, start: float, end: float):
            self.time_range = (float(start), float(end))
            self.update()

        def mouseMoveEvent(self, event):
            """Show a lightweight readout instead of forcing users to read a table."""

            from PySide6.QtWidgets import QToolTip

            if not self.series:
                return
            plot_left = 70
            plot_right = max(self.width() - 12, plot_left + 1)
            if event.position().x() < plot_left or event.position().x() > plot_right:
                QToolTip.hideText()
                return
            count = max((len(line) for _name, line, _color in self.series), default=0)
            if count <= 0:
                return
            ratio = (event.position().x() - plot_left) / max(plot_right - plot_left, 1)
            index = min(max(round(ratio * max(count - 1, 0)), 0), count - 1)
            start, end = self.time_range or (0.0, float(max(count - 1, 1)))
            time_s = start + (end - start) * index / max(count - 1, 1)
            lines = [f"t = {time_s:.1f} s"]
            for name, values, _color in self.series:
                if index < len(values) and math.isfinite(values[index]):
                    lines.append(f"{name}：{values[index]:.3f}")
            QToolTip.showText(event.globalPosition().toPoint(), "\n".join(lines), self)
            super().mouseMoveEvent(event)

        def paintEvent(self, _event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor(PALETTE["chart_bg"]))
            painter.setPen(QColor(PALETTE["text"]))
            painter.drawText(14, 22, self.title)
            if not self.series:
                painter.setPen(QColor(PALETTE["chart_axis"]))
                painter.drawText(14, 52, "暂无有效数据")
                return
            values = [v for _, line, _ in self.series for v in line if math.isfinite(v)]
            if not values:
                painter.drawText(14, 52, "数据缺失（未用 0 静默填充）")
                return
            low, high, ticks, tick_step = _nice_axis_ticks(values, self.y_min)

            legend_y = 42
            legend_x = 14
            legend_rows = 1
            row_height = 18
            available_width = max(self.width() - 28, 80)
            for name, _line, color in self.series:
                item_width = 30 + painter.fontMetrics().horizontalAdvance(name)
                if legend_x > 14 and legend_x + item_width > available_width:
                    legend_rows += 1
                    legend_x = 14
                    legend_y += row_height
                painter.setPen(QPen(QColor(color), 3))
                painter.drawLine(QPointF(legend_x, legend_y - 4), QPointF(legend_x + 16, legend_y - 4))
                painter.setPen(QColor(color))
                painter.drawText(legend_x + 22, legend_y, name)
                legend_x += item_width

            plot = QRectF(70, 52 + row_height * legend_rows, max(self.width() - 82, 30), max(self.height() - 94 - row_height * (legend_rows - 1), 30))
            painter.setPen(QPen(QColor(PALETTE["chart_grid"]), 1))
            for value in reversed(ticks):
                ratio = (value - low) / max(high - low, 1e-9)
                y = plot.bottom() - plot.height() * ratio
                painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
                painter.setPen(QColor(PALETTE["chart_axis"]))
                label = _format_tick(value, tick_step)
                painter.drawText(8, int(y + painter.fontMetrics().height() / 3), label)
                painter.setPen(QPen(QColor(PALETTE["chart_grid"]), 1))
            for name, line, color in self.series:
                points = []
                for i, value in enumerate(line):
                    if not math.isfinite(value):
                        continue
                    x = plot.left() + plot.width() * i / max(len(line) - 1, 1)
                    ratio = (value - low) / max(high - low, 1e-9)
                    ratio = min(max(ratio, 0.0), 1.0)
                    y = plot.bottom() - ratio * plot.height()
                    points.append(QPointF(x, y))
                painter.setPen(QPen(QColor(color), 2))
                for first, second in zip(points, points[1:]):
                    painter.drawLine(first, second)
            marker_progress = self.progress if self.progress is not None else self.index / max(max(len(self.series[0][1]) - 1, 1), 1)
            marker_x = plot.left() + plot.width() * min(max(marker_progress, 0.0), 1.0)
            painter.setPen(QPen(QColor(PALETTE["orange"]), 2, Qt.DashLine))
            painter.drawLine(QPointF(marker_x, plot.top()), QPointF(marker_x, plot.bottom()))
            current_index = round(min(max(marker_progress, 0.0), 1.0) * max(len(self.series[0][1]) - 1, 0))
            for _name, line, color in self.series:
                if current_index >= len(line) or not math.isfinite(line[current_index]):
                    continue
                ratio = (line[current_index] - low) / max(high - low, 1e-9)
                point_y = plot.bottom() - min(max(ratio, 0.0), 1.0) * plot.height()
                painter.setPen(QPen(QColor(color), 2))
                painter.setBrush(QColor(color))
                painter.drawEllipse(QPointF(marker_x, point_y), 3.5, 3.5)
            if self.time_range:
                painter.setPen(QColor(PALETTE["chart_axis"]))
                painter.drawText(int(plot.left()), self.height() - 10, f"t={self.time_range[0]:.0f}s")
                end_text = f"t={self.time_range[1]:.0f}s"
                painter.drawText(int(plot.right() - painter.fontMetrics().horizontalAdvance(end_text)), self.height() - 10, end_text)

    return LineChart()


def _format_tick(value: float, step: float) -> str:
    """Format a tick from its interval, avoiding arbitrary decimal noise."""

    if abs(value) < max(abs(step), 1.0) * 1e-10:
        value = 0.0
    if not math.isfinite(step) or step <= 0:
        return f"{value:.0f}"
    decimals = 0
    for candidate in range(0, 7):
        scaled = step * (10 ** candidate)
        if math.isclose(scaled, round(scaled), rel_tol=1e-9, abs_tol=1e-9):
            decimals = candidate
            break
    return f"{value:.{decimals}f}"
