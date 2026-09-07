"""Native solver plots with actual, non-uniform timestamps (no resampling)."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QFrame, QMenu, QToolTip

from .chart_panel import _format_tick, _nice_axis_ticks
from ..theme import PALETTE


class NativeTimeChart(QFrame):
    def __init__(self, title, height=180, y_min=None):
        super().__init__()
        self.title, self.y_min = title, y_min
        self.series = []
        self.times = []
        self.index = 0
        self.setObjectName("chartPanel")
        self.setMinimumHeight(height)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

    def set_time_points(self, times):
        self.times = list(times)
        self.update()

    def set_series(self, series):
        # Optional fourth tuple element is the comparison run's own time axis.
        self.series = list(series)
        self.update()

    def set_index(self, index):
        self.index = max(0, int(index))
        self.update()

    def _points(self, series):
        times = series[3] if len(series) > 3 else self.times
        return [(t, v) for t, v in zip(times, series[1])
                if v is not None and math.isfinite(float(v)) and math.isfinite(float(t))]

    def _range(self):
        times = [t for series in self.series for t, _ in self._points(series)]
        return (min(times), max(times)) if times else (0.0, 1.0)

    def _menu(self, position):
        menu = QMenu(self)
        menu.addAction("复制真实时间点与数值", self._copy)
        menu.exec(self.mapToGlobal(position))

    def _copy(self):
        rows = ["曲线\t真实时间(s)\t值"]
        for series in self.series:
            rows.extend(f"{series[0]}\t{t:.9g}\t{v:.9g}" for t, v in self._points(series))
        QApplication.clipboard().setText("\n".join(rows))

    def mouseMoveEvent(self, event):
        low, high = self._range()
        time = low + (high - low) * (event.position().x() - 64) / max(self.width() - 80, 1)
        rows = []
        for series in self.series:
            points = self._points(series)
            if points:
                t, v = min(points, key=lambda point: abs(point[0] - time))
                rows.append(f"{series[0]} · t={t:.3f} s：{v:.6g}")
        if rows:
            QToolTip.showText(event.globalPosition().toPoint(), "\n".join(rows), self)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(PALETTE["chart_bg"]))
        painter.setPen(QColor(PALETTE["text"]))
        painter.drawText(12, 22, self.title)
        values = [v for series in self.series for _, v in self._points(series)]
        if not values:
            painter.drawText(12, 52, "无已保存的真实数值")
            return
        x, y = 12, 43
        for series in self.series:
            if not self._points(series):
                continue
            width = painter.fontMetrics().horizontalAdvance(series[0]) + 24
            if x + width > self.width() - 12 and x > 12:
                x, y = 12, y + 18
            painter.setPen(QColor(series[2]))
            painter.drawText(x, y, series[0])
            x += width
        plot = QRectF(64, y + 14, max(self.width() - 80, 1), max(self.height() - y - 44, 1))
        low, high, ticks, step = _nice_axis_ticks(values, self.y_min)
        t0, t1 = self._range()

        def xy(t, v):
            return QPointF(plot.left() + plot.width() * (t - t0) / max(t1 - t0, 1e-9),
                           plot.bottom() - plot.height() * (v - low) / max(high - low, 1e-9))

        for tick in ticks:
            yy = xy(t0, tick).y()
            painter.setPen(QColor(PALETTE["chart_grid"]))
            painter.drawLine(QPointF(plot.left(), yy), QPointF(plot.right(), yy))
            painter.setPen(QColor(PALETTE["muted"]))
            painter.drawText(5, int(yy + 4), _format_tick(tick, step))
        for series in self.series:
            painter.setPen(QPen(QColor(series[2]), 2))
            previous = None
            times = series[3] if len(series) > 3 else self.times
            for t, v in zip(times, series[1]):
                if v is None or not math.isfinite(float(v)):
                    previous = None
                    continue
                point = xy(t, v)
                if previous is not None:
                    painter.drawLine(previous, point)
                else:
                    painter.drawEllipse(point, 2, 2)
                previous = point
        if self.times:
            marker = xy(self.times[min(self.index, len(self.times) - 1)], low).x()
            painter.setPen(QPen(QColor(PALETTE["orange"]), 1, Qt.DashLine))
            painter.drawLine(QPointF(marker, plot.top()), QPointF(marker, plot.bottom()))
        painter.setPen(QColor(PALETTE["muted"]))
        painter.drawText(int(plot.left()), self.height() - 8, f"{t0:.3f} s")
        label = f"{t1:.3f} s"
        painter.drawText(int(plot.right() - painter.fontMetrics().horizontalAdvance(label)), self.height() - 8, label)
