from __future__ import annotations


def create_reward_panel():
    from PySide6.QtWidgets import QLabel, QFrame, QVBoxLayout

    panel = QFrame()
    panel.setObjectName("panel")
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(12, 10, 12, 10)
    title = QLabel("安全验证")
    title.setObjectName("sectionTitle")
    body = QLabel("等待当前帧")
    body.setWordWrap(True)
    layout.addWidget(title)
    layout.addWidget(body)
    panel._body_label = body
    return panel


def update_reward_panel(panel, frame: dict):
    hmi = frame.get("hmi", {}) or {}
    components = hmi.get("reward_components", {}) or {}
    validation = hmi.get("validation_180s", {}) or {}
    rl = validation.get("rl_policy", validation) if isinstance(validation, dict) else {}
    panel._body_label.setText(
        f"总奖励：{_fmt(components.get('integrated_reward'))}　"
        f"改造效果：{_fmt(components.get('effectiveness'))}<br>"
        f"压力安全：{_fmt(components.get('pressure_safety'))}　"
        f"异常风险：{_fmt(components.get('abnormal_risk'))}　"
        f"施工成本：{_fmt(components.get('construction_cost'))}<br>"
        f"验证时间：{_duration(rl.get('validation_seconds')) if rl else '--'} s　"
        f"安全率：{_rate(rl.get('safe_within_180s_rate')) if rl else '--'}"
    )


def _fmt(value):
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "--"


def _duration(value):
    try:
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return "--"


def _rate(value):
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "--"
