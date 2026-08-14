from __future__ import annotations

import math


def build_chart(title: str, height: int = 240, y_min: float | None = None):
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QPainter, QPen
    from PySide6.QtWidgets import QFrame

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

        def set_series(self, series):
            self.series = [(str(name), [float(v) if v is not None else float("nan") for v in values], color) for name, values, color in series]
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

        def paintEvent(self, _event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor("#1B2A36"))
            painter.setPen(QColor("#E8F0F5"))
            painter.drawText(14, 22, self.title)
            if not self.series:
                painter.setPen(QColor("#9EB2C1"))
                painter.drawText(14, 52, "暂无有效数据")
                return
            values = [v for _, line, _ in self.series for v in line if math.isfinite(v)]
            if not values:
                painter.drawText(14, 52, "数据缺失（未用 0 静默填充）")
                return
            low, high = min(values), max(values)
            if self.y_min is not None:
                # y_min is an axis floor, not a candidate value to pad below.
                # Keeping the padding step below this assignment used to turn
                # an explicit 0 floor into negative tick labels.
                low = float(self.y_min)
            if math.isclose(low, high):
                margin = max(abs(low) * 0.05, 1.0)
            else:
                margin = max((high - low) * 0.05, 1e-6)
            if self.y_min is None:
                low -= margin
            high += margin

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
            painter.setPen(QPen(QColor("#344B5A"), 1))
            for tick in range(5):
                y = plot.top() + plot.height() * tick / 4
                painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
                value = high - (high - low) * tick / 4
                painter.setPen(QColor("#9EB2C1"))
                label = _format_tick(value, high - low)
                painter.drawText(8, int(y + painter.fontMetrics().height() / 3), label)
                painter.setPen(QPen(QColor("#344B5A"), 1))
            for name, line, color in self.series:
                points = []
                for i, value in enumerate(line):
                    if not math.isfinite(value):
                        continue
                    x = plot.left() + plot.width() * i / max(len(line) - 1, 1)
                    y = plot.bottom() - (value - low) / max(high - low, 1e-9) * plot.height()
                    points.append(QPointF(x, y))
                painter.setPen(QPen(QColor(color), 2))
                for first, second in zip(points, points[1:]):
                    painter.drawLine(first, second)
            marker_progress = self.progress if self.progress is not None else self.index / max(max(len(self.series[0][1]) - 1, 1), 1)
            marker_x = plot.left() + plot.width() * min(max(marker_progress, 0.0), 1.0)
            painter.setPen(QPen(QColor("#F2A93B"), 2, Qt.DashLine))
            painter.drawLine(QPointF(marker_x, plot.top()), QPointF(marker_x, plot.bottom()))
            if self.time_range:
                painter.setPen(QColor("#9EB2C1"))
                painter.drawText(int(plot.left()), self.height() - 10, f"t={self.time_range[0]:.0f}s")
                end_text = f"t={self.time_range[1]:.0f}s"
                painter.drawText(int(plot.right() - painter.fontMetrics().horizontalAdvance(end_text)), self.height() - 10, end_text)

    return LineChart()


def _format_tick(value: float, span: float) -> str:
    """Keep numeric y-axis labels readable across pressure, flow and sand charts."""

    if span >= 100:
        return f"{value:.0f}"
    if span >= 10:
        return f"{value:.1f}"
    if span >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"
