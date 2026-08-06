from __future__ import annotations


NORMAL_STATE = "正常"
ABNORMAL_STATES = {
    "砂堵",
    "缝口暂堵",
    "缝内暂堵",
    "延伸受阻",
    "滤失过大",
    "缝高延伸",
    "新缝开启",
    "其他",
    "异常",
}


def is_abnormal_state(state: str) -> bool:
    text = str(state).strip()
    return text != NORMAL_STATE and any(label in text for label in ABNORMAL_STATES)


def risk_level(next_state: str, transition_probability: float, uncertainty: str) -> str:
    abnormal = is_abnormal_state(next_state)
    if abnormal and (transition_probability >= 0.2 or uncertainty == "high"):
        return "high"
    if abnormal or uncertainty == "medium":
        return "medium"
    if uncertainty == "high":
        return "medium"
    return "low"


def requires_confirmation(risk: str, action_type: str = "advisory") -> bool:
    return risk == "high" or action_type in {"control", "shutdown", "parameter_change"}
