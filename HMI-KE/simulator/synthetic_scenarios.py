from __future__ import annotations


def default_scenarios() -> list[dict]:
    return [
        {
            "name": "正常施工",
            "current_state": "正常",
            "transition_probabilities": {"正常": 0.82, "砂堵": 0.08, "缝口暂堵": 0.06, "主缝延伸": 0.04},
            "enkf_abs_residual": 1.5,
        },
        {
            "name": "砂堵预警",
            "current_state": "正常",
            "transition_probabilities": {"砂堵": 0.36, "正常": 0.32, "缝口暂堵": 0.20, "滤失过大": 0.12},
            "enkf_abs_residual": 9.2,
        },
        {
            "name": "裂缝延伸",
            "current_state": "缝口暂堵",
            "transition_probabilities": {"主缝延伸": 0.51, "正常": 0.31, "新缝开启": 0.18},
            "enkf_abs_residual": 4.0,
        },
    ]
