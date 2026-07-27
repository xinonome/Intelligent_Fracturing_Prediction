from __future__ import annotations


NORMAL_STATE = "正常"
ABNORMAL_STATES = {
    "砂堵",
    "缝口暂堵",
    "主缝延伸",
    "延伸受阻",
    "新缝开启",
    "滤失过大",
    "异常",
}


def risk_level(next_state: str, transition_probability: float, uncertainty: str) -> str:
    abnormal = next_state in ABNORMAL_STATES
    if abnormal and (transition_probability >= 0.2 or uncertainty == "high"):
        return "high"
    if abnormal or uncertainty == "medium":
        return "medium"
    if uncertainty == "high":
        return "medium"
    return "low"


def requires_confirmation(risk: str, action_type: str = "advisory") -> bool:
    return risk == "high" or action_type in {"control", "shutdown", "parameter_change"}
