from __future__ import annotations

from ..theme import PALETTE


def create_decision_card():
    from PySide6.QtWidgets import QLabel, QFrame, QVBoxLayout

    card = QFrame()
    card.setObjectName("decisionCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(14, 12, 14, 12)
    title = QLabel("智能体风险判断与未来动作")
    title.setObjectName("sectionTitle")
    body = QLabel("等待时间轴")
    body.setWordWrap(True)
    body.setTextFormat(body.textFormat())
    layout.addWidget(title)
    layout.addWidget(body)
    card._body_label = body
    return card


def update_decision_card(card, frame: dict):
    hmi = frame.get("hmi", {}) or {}
    legacy = frame.get("decision", {}) or {}
    evidence = legacy.get("evidence", {}) or {}
    action = hmi.get("high_level_action", frame.get("hmi_option", "--"))
    risk = hmi.get("risk_level", legacy.get("risk_level", "unknown"))
    color = {"low": PALETTE["cyan"], "medium": PALETTE["yellow"], "high": PALETTE["red"]}.get(risk, PALETTE["muted"])
    body = (
        f"动作：<b>{action}</b>　风险：<span style='color:{color}'><b>{risk}</b></span>　"
        f"不确定性：{hmi.get('uncertainty', legacy.get('uncertainty', '--'))}<br>"
        f"未来60秒：排量 {_num(hmi.get('recommended_flow_m3_min', frame.get('action_flow')))} m³/min，"
        f"砂比 {_num(hmi.get('recommended_sand_ratio_percent', frame.get('action_sand')))}%<br>"
        f"人工确认：{'是' if hmi.get('requires_confirmation', legacy.get('requires_confirmation', True)) else '否'}　"
        f"异常概率 {_num(evidence.get('max_abnormal_probability', frame.get('hmi_abnormal')))}　"
        f"砂堵概率 {_num(evidence.get('max_sand_plug_probability', frame.get('hmi_sand_plug')))}<br>"
        f"建议：{legacy.get('recommendation', '暂无建议')}"
    )
    card._body_label.setText(body)


def _num(value):
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "--"
