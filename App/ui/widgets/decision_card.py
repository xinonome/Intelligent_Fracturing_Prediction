from __future__ import annotations

from ..theme import PALETTE
from ..display_text import display_text


def create_decision_card():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QSizePolicy, QVBoxLayout

    card = QFrame()
    card.setObjectName("decisionCard")
    card.setMinimumHeight(220)
    card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(14, 12, 14, 12)
    title = QLabel("当前控制建议")
    title.setObjectName("sectionTitle")
    status = QLabel("暂无可执行建议")
    status.setObjectName("value")
    status.setWordWrap(True)
    status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    layout.addWidget(title)
    layout.addWidget(status)
    compare = QGridLayout()
    compare.setHorizontalSpacing(12)
    compare.setVerticalSpacing(5)
    for column, text in enumerate(("控制量", "当前值", "推荐值", "变化")):
        label = QLabel(text)
        label.setObjectName("key")
        compare.addWidget(label, 0, column)
    value_labels = {}
    for row, name in enumerate(("排量", "砂比"), start=1):
        label = QLabel(name)
        label.setObjectName("caption")
        compare.addWidget(label, row, 0)
        for column, key in enumerate((f"{name}_current", f"{name}_recommended", f"{name}_delta"), start=1):
            value = QLabel("--")
            value.setObjectName("value")
            value.setWordWrap(True)
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            compare.addWidget(value, row, column)
            value_labels[key] = value
    compare.setColumnStretch(0, 1)
    compare.setColumnStretch(1, 1)
    compare.setColumnStretch(2, 1)
    compare.setColumnStretch(3, 1)
    layout.addLayout(compare)
    reason = QLabel("等待当前帧")
    reason.setObjectName("muted")
    reason.setWordWrap(True)
    layout.addWidget(reason)
    confirm = QLabel("人工确认：--")
    confirm.setObjectName("caption")
    confirm.setWordWrap(True)
    layout.addWidget(confirm)
    card._status_label = status
    card._value_labels = value_labels
    card._reason_label = reason
    card._confirm_label = confirm
    return card


def update_decision_card(card, frame: dict):
    hmi = frame.get("hmi", {}) or {}
    legacy = frame.get("decision", {}) or {}
    action = hmi.get("high_level_action", frame.get("hmi_option", "--"))
    risk = hmi.get("risk_level", legacy.get("risk_level", ""))
    color = {"low": PALETTE["cyan"], "medium": PALETTE["yellow"], "high": PALETTE["red"]}.get(risk, PALETTE["muted"])
    current_flow = hmi.get("current_flow_m3_min", frame.get("current_flow"))
    recommended_flow = hmi.get("recommended_flow_m3_min", frame.get("action_flow"))
    current_sand = hmi.get("current_sand_ratio_percent", frame.get("current_sand"))
    recommended_sand = hmi.get("recommended_sand_ratio_percent", frame.get("action_sand"))
    has_recommendation = _numeric(recommended_flow) is not None or _numeric(recommended_sand) is not None
    card._status_label.setText(
        f"建议动作：{display_text(action)}" if has_recommendation else "暂无可执行建议"
    )
    card._status_label.setStyleSheet(f"color:{color};font-weight:800;")
    _set_value(card._value_labels["排量_current"], current_flow, " m³/min")
    _set_value(card._value_labels["排量_recommended"], recommended_flow, " m³/min")
    _set_delta(card._value_labels["排量_delta"], current_flow, recommended_flow, " m³/min")
    _set_value(card._value_labels["砂比_current"], current_sand, "%")
    _set_value(card._value_labels["砂比_recommended"], recommended_sand, "%")
    _set_delta(card._value_labels["砂比_delta"], current_sand, recommended_sand, "%")
    card._reason_label.setText(display_text(legacy.get("recommendation", "请结合当前压力和控制曲线审核。")))
    confirmed = hmi.get("requires_confirmation", legacy.get("requires_confirmation", True))
    card._confirm_label.setText(f"人工确认：{'需要' if confirmed else '不需要'}")


def _num(value):
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "--"


def _numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _set_value(label, value, suffix):
    try:
        label.setText(f"{float(value):.2f}{suffix}")
    except (TypeError, ValueError):
        label.setText("--")


def _set_delta(label, current, recommended, suffix):
    try:
        delta = float(recommended) - float(current)
        sign = "+" if delta > 0 else ""
        label.setText(f"{sign}{delta:.2f}{suffix}")
    except (TypeError, ValueError):
        label.setText("--")
