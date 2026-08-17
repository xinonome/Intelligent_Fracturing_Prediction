from __future__ import annotations

from bisect import bisect_left
from pathlib import Path

from ..theme import PALETTE
from ...data.dt_loader import DTLoader, number
from ...data.hmi_loader import HMILoader
from ..web_view import Embedded3DView
from ..widgets.chart_panel import build_chart
from ..widgets.decision_card import create_decision_card, update_decision_card
from ..widgets.parameter_panel import create_parameter_panel, update_parameter_panel
from ..widgets.reward_panel import create_reward_panel, update_reward_panel
from ..widgets.status_card import Panel
from ..widgets.timeline_control import create_timeline_control


def build_integrated_page(controller, registry, html_path: Path | None):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QComboBox, QGridLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

    page = QWidget()
    page_root = QVBoxLayout(page)
    page_root.setContentsMargins(0, 0, 0, 0)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll_content = QWidget()
    scroll.setWidget(scroll_content)
    page_root.addWidget(scroll)

    root = QVBoxLayout(scroll_content)
    root.setContentsMargins(20, 16, 20, 16)
    title = QLabel("联合动态演示")
    title.setObjectName("pageTitle")
    root.addWidget(title)
    scenario_row = QVBoxLayout()
    scenario_box = QComboBox()
    scenario_box.addItem("无 DAS：压力在线校正", "no_das_pressure_only")
    scenario_box.addItem("有 DAS：压力 + 分簇观测校验", "das_cluster_observation")
    selected = getattr(registry, "scenario_id", "das_cluster_observation")
    scenario_box.setCurrentIndex(max(0, scenario_box.findData(selected)))
    source_label = QLabel()
    source_label.setObjectName("muted")
    source_label.setWordWrap(True)
    scenario_row.addWidget(scenario_box)
    scenario_row.addWidget(source_label)
    root.addLayout(scenario_row)

    content = QGridLayout()
    content.setContentsMargins(0, 0, 0, 0)
    content.setHorizontalSpacing(8)
    content.setVerticalSpacing(8)

    # Keep all three charts the same height and give the upper area enough
    # vertical room for readable legends, axes and time labels. The scroll
    # container lets this larger layout remain usable on smaller screens.
    chart_height = 280
    pressure_chart = build_chart("井底压力 / MPa", chart_height, y_min=0.0)
    flow_chart = build_chart("排量 / m³/min", chart_height, y_min=0.0)
    sand_chart = build_chart("砂比 / %", chart_height, y_min=0.0)
    content.addWidget(pressure_chart, 0, 0)
    content.addWidget(flow_chart, 1, 0)
    content.addWidget(sand_chart, 2, 0)

    center, center_layout = Panel.create("数字孪生三维状态")
    model = Embedded3DView.create(html_path)
    model.setMinimumHeight(chart_height)
    center_layout.addWidget(model, 1)
    content.addWidget(center, 0, 1, 3, 1)

    decision = create_decision_card()
    reward = create_reward_panel()
    content.addWidget(decision, 3, 0)
    content.addWidget(reward, 3, 1)

    params = create_parameter_panel("EnKF 参数更新", [
        ("prior", "先验参数"),
        ("posterior", "后验参数"),
        ("prior_half_lengths", "先验半长(m)"),
        ("posterior_half_lengths", "后验半长(m)"),
        ("error", "后验误差"),
        ("runtime", "计算时间"),
    ])
    content.addWidget(params, 4, 0, 1, 2)
    for column, stretch in enumerate((2, 1)):
        content.setColumnStretch(column, stretch)
    # Equal chart-row stretches remove the previous short sand-ratio panel.
    # Each chart row has the same minimum height; the combined upper area is
    # about 1.5x the previous compact layout and can be reached by scrolling.
    content.setRowStretch(0, 1)
    content.setRowStretch(1, 1)
    content.setRowStretch(2, 1)
    content.setRowStretch(3, 2)
    content.setRowStretch(4, 3)
    root.addLayout(content, 1)

    timeline = create_timeline_control(controller)
    root.addWidget(timeline)
    events = QLabel("关键事件：" + " · ".join(f"{name} 第{index + 1}帧" for name, index in _events(controller.frames).items()))
    events.setObjectName("muted")
    events.setWordWrap(True)
    root.addWidget(events)

    def refresh_scenario_view():
        chart_values, chart_end_s = _chart_series(registry, controller.frames)
        pressure_chart.set_series(chart_values)
        if chart_end_s is not None:
            pressure_chart.set_time_range(1.0, chart_end_s)
        sample_count = len(chart_values[0][1]) if chart_values else len(controller.frames)
        control_values = _control_series(registry, chart_end_s or float(max(len(controller.frames), 1)), sample_count)
        flow_chart.set_series(control_values[:2])
        sand_chart.set_series(control_values[2:])
        for control in (flow_chart, sand_chart):
            if chart_end_s is not None:
                control.set_time_range(1.0, chart_end_s)
        scenario = registry.scenario()
        source_label.setText(
            f"{scenario.get('display_name', '')}  · 观测路径："
            + ("施工曲线 → 井口—井底压力换算；簇级观测：未接入；状态：待校准" if scenario.get("observation_mode") == "pressure_only" else "施工曲线 + DAS/FracMonitor 分簇观测；覆盖：1–4435 s；状态：待校准")
        )

    refresh_scenario_view()

    def change_scenario(index):
        scenario_id = scenario_box.itemData(index)
        if not scenario_id or scenario_id == getattr(registry, "scenario_id", None):
            return
        if hasattr(controller, "set_scenario"):
            controller.set_scenario(str(scenario_id))
            refresh_scenario_view()

    scenario_box.currentIndexChanged.connect(change_scenario)

    def update(frame):
        frame = frame or {}
        progress = controller.index / max(len(controller.frames) - 1, 1)
        pressure_chart.set_progress(progress)
        flow_chart.set_progress(progress)
        sand_chart.set_progress(progress)
        dt = frame.get("dt", {}) or {}
        update_parameter_panel(params, {
            "prior": _parameter_text(dt.get("prior_parameters")),
            "posterior": _parameter_text(dt.get("posterior_parameters")),
            "prior_half_lengths": _half_length_text(dt.get("prior_half_lengths_m")),
            "posterior_half_lengths": _half_length_text(dt.get("posterior_half_lengths_m")),
            "error": dt.get("posterior_error"),
            "runtime": dt.get("runtime_ms"),
        }, {"error": "{:.3f}", "runtime": "{:.1f} ms"})
        update_decision_card(decision, frame)
        update_reward_panel(reward, frame)
        if getattr(model, "set_time_index", None):
            model.set_time_index(frame.get("time_s", 0))

    controller.frameChanged.connect(update)
    update(controller.current or {})
    page._model_view = model
    page._controller = controller
    return page


def _series(frames):
    if not frames:
        return []
    def values(section, key, legacy):
        return [((frame.get(section, {}) or {}).get(key, frame.get(legacy))) for frame in frames]
    return [
        ("PKN先验", values("dt", "prior_bottomhole_pressure_mpa", "prior_bhp"), PALETTE["orange"]),
        ("观测压力", values("dt", "observed_bottomhole_pressure_mpa", "observed_bhp"), PALETTE["blue"]),
        ("EnKF后验", values("dt", "bottomhole_pressure_mpa", "posterior_bhp"), PALETTE["cyan"]),
    ]


def _chart_series(registry, frames):
    """Use the selected scenario cache without extending observation coverage."""

    loader = DTLoader(registry)
    cache = loader.cache
    timeline = [float(value) for value in cache.get("timeline_s", [])]
    arrays = cache.get("arrays", {}) if isinstance(cache, dict) else {}
    if not timeline or not arrays:
        return _series(frames), None
    end_s = timeline[-1]
    indices = [index for index, value in enumerate(timeline) if value <= end_s]
    if not indices:
        return _series(frames), None
    times = [timeline[index] for index in indices]
    prior = [_number_at(arrays.get("prior_bhp_mpa", []), index) for index in indices]
    posterior = [_number_at(arrays.get("posterior_bhp_mpa", []), index) for index in indices]
    observed_source = arrays.get("observed_bhp_mpa") or arrays.get("bottomhole_pressure_mpa", [])
    observed = [_number_at(observed_source, index) for index in indices]
    return [
        ("PKN先验", prior, PALETTE["orange"]),
        ("观测压力", observed, PALETTE["blue"]),
        ("EnKF后验", posterior, PALETTE["cyan"]),
    ], end_s


def _control_series(registry, end_s, sample_count):
    loader = HMILoader(registry)
    rows = loader.rows
    if not rows or sample_count <= 0:
        return []
    fields = [
        ("当前排量", "current_flow_m3_min", PALETTE["cyan"]),
        ("推荐排量", "flow_m3_min", PALETTE["blue"]),
        ("当前砂比", "current_sand_ratio_percent", PALETTE["orange"]),
        ("推荐砂比", "sand_ratio_percent", PALETTE["red"]),
    ]
    result = []
    for name, key, color in fields:
        source = [number(row.get(key), float("nan")) for row in rows]
        result.append((name, _resample_sequence(source, sample_count), color))
    return result


def _resample_sequence(values, count):
    if not values:
        return [float("nan")] * count
    if len(values) == 1 or count == 1:
        return [values[0]] * count
    result = []
    for index in range(count):
        position = index * (len(values) - 1) / (count - 1)
        left = int(position)
        right = min(left + 1, len(values) - 1)
        ratio = position - left
        left_value = values[left]
        right_value = values[right]
        if left_value != left_value:
            result.append(right_value)
        elif right_value != right_value:
            result.append(left_value)
        else:
            result.append(left_value + (right_value - left_value) * ratio)
    return result


def _number_at(values, index):
    try:
        return float(values[index])
    except (IndexError, TypeError, ValueError):
        return float("nan")


def _resample_history(rows, times, key):
    points = []
    for row in rows:
        time_s = number(row.get("time_s"))
        value = number(row.get(key))
        if time_s is not None and value is not None:
            points.append((time_s, value))
    if not points:
        return [float("nan")] * len(times)
    points.sort()
    source_times = [item[0] for item in points]
    source_values = [item[1] for item in points]
    result = []
    for time_s in times:
        right = bisect_left(source_times, time_s)
        if right <= 0:
            result.append(source_values[0])
        elif right >= len(source_times):
            result.append(source_values[-1])
        else:
            left = right - 1
            span = source_times[right] - source_times[left]
            ratio = (time_s - source_times[left]) / span if span else 0.0
            result.append(source_values[left] + ratio * (source_values[right] - source_values[left]))
    return result


def _parameter_text(value):
    if not value:
        return "缺失 · 未接入"
    return "；".join(f"{key}={_fmt(item)}" for key, item in value.items())


def _half_length_text(values):
    if not values:
        return "缺失 · 未接入"
    return "；".join(f"簇{index + 1}={_fmt(value)} m" for index, value in enumerate(values))


def _fmt(value):
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "--"


def _events(frames):
    result = {}
    for index, frame in enumerate(frames):
        option = frame.get("hmi_option")
        if option and option not in result:
            result[option] = index
    return result
