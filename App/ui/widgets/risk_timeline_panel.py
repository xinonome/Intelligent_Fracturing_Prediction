"""Clickable real-time risk timeline built from replay decision fields."""

from __future__ import annotations

import math
from typing import Any

from ..theme import PALETTE


def risk_intervals(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for frame in frames:
        decision = frame.get("decision") if isinstance(frame.get("decision"), dict) else {}
        hmi = frame.get("hmi") if isinstance(frame.get("hmi"), dict) else {}
        fsl = frame.get("fsl") if isinstance(frame.get("fsl"), dict) else {}
        raw_level = decision.get("risk_level") or hmi.get("risk_level")
        level = _level(raw_level)
        time_s = _number(frame.get("time_s"))
        if level is None or time_s is None:
            continue
        points.append(
            {
                "time_s": time_s,
                "level": level,
                "condition": fsl.get("working_type") or frame.get("phase") or "未标注",
                "reason": decision.get("main_risk") or decision.get("recommendation") or "已有风险判断结果",
                "recommendation": decision.get("recommendation") or "当前建议未提供影响说明",
            }
        )
    if not points:
        return []
    output: list[dict[str, Any]] = []
    begin = points[0]
    previous = begin
    for point in points[1:] + [None]:
        if point is not None and point["level"] == begin["level"]:
            previous = point
            continue
        output.append(
            {
                "level": begin["level"],
                "start_s": begin["time_s"],
                "end_s": previous["time_s"],
                "condition": previous["condition"],
                "reason": previous["reason"],
                "recommendation": previous["recommendation"],
            }
        )
        if point is not None:
            begin = previous = point
    return output


def create_risk_timeline(frames: list[dict[str, Any]]):
    from PySide6.QtCore import QPointF, QRectF, Qt, Signal
    from PySide6.QtGui import QColor, QPainter, QPen
    from PySide6.QtWidgets import QFrame, QSizePolicy

    class RiskTimeline(QFrame):
        intervalSelected = Signal(object)

        def __init__(self):
            super().__init__()
            self.setObjectName("chartPanel")
            self.setMinimumHeight(190)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.frames = []
            self.intervals = []
            self.index = 0
            self.selected = None
            self.set_frames(frames)

        def set_frames(self, values):
            self.frames = list(values or [])
            self.intervals = risk_intervals(self.frames)
            self.index = 0
            self.selected = None
            self.update()

        def set_index(self, index):
            self.index = max(0, min(int(index), max(len(self.frames) - 1, 0)))
            self.update()

        def mousePressEvent(self, event):
            plot = self._plot_rect()
            if self.intervals and plot.contains(event.position().toPoint()):
                start, end = self._time_range()
                seconds = start + (end - start) * (event.position().x() - plot.left()) / max(plot.width(), 1.0)
                self.selected = next(
                    (item for item in self.intervals if item["start_s"] <= seconds <= item["end_s"]),
                    None,
                )
                self.intervalSelected.emit(self.selected or {})
                self.update()
            super().mousePressEvent(event)

        def paintEvent(self, _event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor(PALETTE["chart_bg"]))
            painter.setPen(QColor(PALETTE["text"]))
            painter.drawText(14, 24, "实时风险")
            if not self.intervals:
                painter.setPen(QColor(PALETTE["muted"]))
                painter.drawText(14, 54, "当前暂无可用风险判断")
                return
            plot = self._plot_rect()
            levels = (("高风险", "high", PALETTE["red"]), ("关注", "attention", PALETTE["orange"]), ("正常", "normal", "#48A868"))
            row_height = plot.height() / 3.0
            start, end = self._time_range()
            for row, (label, level, color) in enumerate(levels):
                top = plot.top() + row * row_height
                painter.setPen(QColor(PALETTE["muted"]))
                painter.drawText(12, int(top + row_height * 0.65), label)
                painter.setPen(QPen(QColor(PALETTE["chart_grid"]), 1))
                painter.drawLine(QPointF(plot.left(), top + row_height), QPointF(plot.right(), top + row_height))
                for interval in self.intervals:
                    if interval["level"] != level:
                        continue
                    left = plot.left() + plot.width() * (interval["start_s"] - start) / max(end - start, 1.0)
                    right = plot.left() + plot.width() * (interval["end_s"] - start) / max(end - start, 1.0)
                    fill = QColor(color)
                    fill.setAlpha(220 if interval is self.selected else 150)
                    painter.fillRect(QRectF(left, top + 5, max(right - left, 3.0), row_height - 10), fill)
            if self.frames:
                time_s = _number(self.frames[self.index].get("time_s"))
                if time_s is not None:
                    marker_x = plot.left() + plot.width() * (time_s - start) / max(end - start, 1.0)
                    painter.setPen(QPen(QColor(PALETTE["blue"]), 2, Qt.DashLine))
                    painter.drawLine(QPointF(marker_x, plot.top()), QPointF(marker_x, plot.bottom()))
            painter.setPen(QColor(PALETTE["muted"]))
            painter.drawText(int(plot.left()), self.height() - 8, f"t={start:.0f}s")
            end_text = f"t={end:.0f}s"
            painter.drawText(int(plot.right() - painter.fontMetrics().horizontalAdvance(end_text)), self.height() - 8, end_text)

        def _plot_rect(self):
            return self.rect().adjusted(74, 36, -14, -24)

        def _time_range(self):
            times = [_number(frame.get("time_s")) for frame in self.frames]
            times = [value for value in times if value is not None]
            return (min(times), max(times)) if times else (0.0, 1.0)

    return RiskTimeline()


def _level(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"low", "normal", "正常", "常规", "低风险"}:
        return "normal"
    if text in {"medium", "attention", "关注", "中风险"}:
        return "attention"
    if text in {"high", "critical", "高风险", "严重"}:
        return "high"
    return None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


__all__ = ["create_risk_timeline", "risk_intervals"]
