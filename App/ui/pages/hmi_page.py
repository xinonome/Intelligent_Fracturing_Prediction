from __future__ import annotations

from ..theme import PALETTE
from ..widgets.decision_card import create_decision_card, update_decision_card
from ..widgets.reward_panel import create_reward_panel, update_reward_panel
from ..widgets.status_card import MetricCardMixin, Panel


def build_hmi_page(controller, registry):
    from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(20, 16, 20, 16)
    title = QLabel("第三部分 · 智能体风险判断与人机协同")
    title.setObjectName("pageTitle")
    layout.addWidget(title)
    layout.addWidget(_label("状态输入：当前300秒窗口、工况、异常风险、EnKF残差和不确定性；动作输出：未来60秒排量/砂比与人工确认。", "subtitle"))
    module = registry.module("hmi")
    summary = registry.summary("hmi")
    rl = summary.get("rl_policy", {})
    validation = summary.get("validation_180s", {}).get("rl_policy", {})
    cards = QGridLayout()
    cards.addWidget(MetricCardMixin.create("科学状态", str(summary.get("scientific_status", "unknown")), "不是现场在线闭环", PALETTE["orange"]), 0, 0)
    cards.addWidget(MetricCardMixin.create("动作输出", "grow / hold / divert / safe", "逐帧回放", PALETTE["cyan"]), 0, 1)
    cards.addWidget(MetricCardMixin.create("180秒安全率", _pct(validation.get("safe_within_180s_rate")), "砂堵场景需持续改进", PALETTE["red"]), 0, 2)
    cards.addWidget(MetricCardMixin.create("不确定性", "显式显示", "缺失字段不造数", PALETTE["yellow"]), 0, 3)
    layout.addLayout(cards)
    warning = QLabel(f"产物状态：{module.get('status', 'not_available')} · {module.get('status_reason', '')}\n{'; '.join(module.get('limitations', []))}")
    warning.setObjectName("warning" if module.get("status") != "validated" else "notice")
    warning.setWordWrap(True)
    layout.addWidget(warning)
    bottom = QHBoxLayout()
    decision = create_decision_card()
    reward = create_reward_panel()
    bottom.addWidget(decision, 1)
    bottom.addWidget(reward, 1)
    layout.addLayout(bottom)
    panel, panel_layout = Panel.create("合同指标与安全门禁")
    panel_layout.addWidget(_label(f"五分钟预警：异常阈值由风险规则定义；计算时间门槛：15秒；安全验证窗口：180秒；当前 HMI scientific_status={summary.get('scientific_status', '--')}，因此不得显示为最终验收通过。", "notice"))
    layout.addWidget(panel)

    def update(frame):
        update_decision_card(decision, frame)
        update_reward_panel(reward, frame)

    controller.frameChanged.connect(update)
    update(controller.current or {})
    return page


def _label(text, name=None):
    from PySide6.QtWidgets import QLabel

    label = QLabel(text)
    if name:
        label.setObjectName(name)
    label.setWordWrap(True)
    return label


def _pct(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "--"
