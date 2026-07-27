from __future__ import annotations


RECOMMENDATIONS = {
    "砂堵": "降低砂比或排量，观察泵压是否继续快速升高。",
    "缝口暂堵": "保持监测，关注泵压波动和排量响应，必要时调整暂堵剂节奏。",
    "主缝延伸": "维持当前施工参数，持续跟踪泵压与砂比趋势。",
    "延伸受阻": "评估近井摩阻和泵压异常，必要时降低施工强度。",
    "新缝开启": "关注压力突降和排量变化，核实施工阶段是否符合设计。",
    "滤失过大": "检查液量、压力保持能力和地层吸液情况。",
    "异常": "进入人工复核流程，结合现场参数确认异常类型。",
    "正常": "维持当前施工参数，继续监测关键指标。",
}


def recommend(next_state: str, risk: str) -> str:
    base = RECOMMENDATIONS.get(next_state, "进入人工复核流程，结合现场参数确认。")
    if risk == "high":
        return f"高风险：{base}"
    if risk == "medium":
        return f"中风险：{base}"
    return base
