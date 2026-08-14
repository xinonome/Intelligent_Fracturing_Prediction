from __future__ import annotations

from ..theme import PALETTE


def _widgets():
    from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

    return QFrame, QLabel, QVBoxLayout


class Panel:
    """Factory namespace to keep imports lazy in headless mode."""

    @staticmethod
    def create(title: str = ""):
        QFrame, QLabel, QVBoxLayout = _widgets()
        frame = QFrame()
        frame.setObjectName("panel")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        if title:
            heading = QLabel(title)
            heading.setObjectName("sectionTitle")
            layout.addWidget(heading)
        return frame, layout


class MetricCardMixin:
    @staticmethod
    def create(title: str, value: str, caption: str, accent: str | None = None):
        QFrame, QLabel, QVBoxLayout = _widgets()
        card = QFrame()
        card.setObjectName("metricCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(13, 11, 13, 11)
        label = QLabel(title)
        label.setObjectName("caption")
        value_label = QLabel(value)
        value_label.setObjectName("metricValue")
        if accent:
            value_label.setStyleSheet(f"color:{accent};")
        caption_label = QLabel(caption)
        caption_label.setObjectName("metricCaption")
        caption_label.setWordWrap(True)
        layout.addWidget(label)
        layout.addWidget(value_label)
        layout.addWidget(caption_label)
        return card


class StatusCardMixin:
    @staticmethod
    def create(title: str, value: str, reason: str, status: str):
        QFrame, QLabel, QVBoxLayout = _widgets()
        card = QFrame()
        card.setObjectName("statusCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        head = QLabel(title)
        head.setObjectName("caption")
        badge = QLabel(f"{status.upper()}  ·  {value}")
        color = {"validated": PALETTE["cyan"], "development_only": PALETTE["orange"], "invalid": PALETTE["red"], "not_available": PALETTE["muted"]}.get(status, PALETTE["muted"])
        badge.setStyleSheet(f"color:{color};font-weight:800;")
        detail = QLabel(reason)
        detail.setObjectName("muted")
        detail.setWordWrap(True)
        layout.addWidget(head)
        layout.addWidget(badge)
        layout.addWidget(detail)
        return card
