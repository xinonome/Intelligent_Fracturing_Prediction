from App.ui.widgets.risk_timeline_panel import risk_intervals


def test_risk_intervals_group_real_decision_states() -> None:
    frames = [
        {"time_s": 1, "decision": {"risk_level": "low", "main_risk": "正常"}, "fsl": {"working_type": "主缝延伸"}},
        {"time_s": 2, "decision": {"risk_level": "low", "main_risk": "正常"}, "fsl": {"working_type": "主缝延伸"}},
        {"time_s": 3, "decision": {"risk_level": "high", "main_risk": "砂堵"}, "fsl": {"working_type": "砂堵"}},
    ]
    intervals = risk_intervals(frames)
    assert [(item["level"], item["start_s"], item["end_s"]) for item in intervals] == [
        ("normal", 1.0, 2.0),
        ("high", 3.0, 3.0),
    ]


def test_unknown_risk_is_not_fabricated() -> None:
    assert risk_intervals([{"time_s": 1, "decision": {"risk_level": "unknown"}}]) == []
