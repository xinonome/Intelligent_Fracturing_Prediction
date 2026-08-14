from __future__ import annotations

from ..theme import PALETTE
from ..widgets.chart_panel import build_chart
from ..widgets.cluster_view import create_cluster_view, update_cluster_view
from ..widgets.parameter_panel import create_parameter_panel, update_parameter_panel
from ..widgets.status_card import MetricCardMixin, Panel


def build_dt_page(controller, registry):
    from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(20, 16, 20, 16)
    title = QLabel("第二部分 · 光纤驱动裂缝数字孪生")
    title.setObjectName("pageTitle")
    layout.addWidget(title)
    layout.addWidget(_label("光纤数据 → 时间同步 → 施工压力换算 → PKN正演 → EnKF物理参数更新 → 更新后重新正演 → 三维状态", "subtitle"))
    summary = registry.summary("dt")
    metrics = summary.get("metrics", {})
    cards = QGridLayout()
    cards.addWidget(MetricCardMixin.create("观测验证", _pct(metrics.get("validation_all_observations_within_15_percent_rate")), "全观测15%目标", PALETTE["cyan"]), 0, 0)
    cards.addWidget(MetricCardMixin.create("计算 P95", _ms(metrics.get("all_steps_compute_p95_ms")), "不含3D渲染", PALETTE["blue"]), 0, 1)
    cards.addWidget(MetricCardMixin.create("EnKF更新对象", "物理参数", "不是直接改写缝长", PALETTE["orange"]), 0, 2)
    cards.addWidget(MetricCardMixin.create("数据对齐", "1秒轴", "插值来源显式记录", PALETTE["yellow"]), 0, 3)
    layout.addLayout(cards)
    chart = build_chart("DT状态 · 井底压力与后验误差", 270)
    chart.set_series([
        ("井底压力", [frame.get("posterior_bhp") for frame in controller.frames], PALETTE["cyan"]),
        ("观测压力", [frame.get("observed_bhp") for frame in controller.frames], PALETTE["blue"]),
        ("压力相对误差", [frame.get("posterior_pressure_error") for frame in controller.frames], PALETTE["orange"]),
    ])
    layout.addWidget(chart)
    lower = QHBoxLayout()
    params = create_parameter_panel("PKN / EnKF 参数", [("prior", "先验"), ("posterior", "后验"), ("error", "压力相对误差"), ("runtime", "运行时间")])
    clusters = create_cluster_view()
    lower.addWidget(params, 1)
    lower.addWidget(clusters, 2)
    layout.addLayout(lower)

    def update(frame):
        chart.set_index(controller.index)
        dt = frame.get("dt", {}) or {}
        update_parameter_panel(params, {"prior": _text(dt.get("prior_parameters")), "posterior": _text(dt.get("posterior_parameters")), "error": dt.get("posterior_pressure_error", dt.get("posterior_error")), "runtime": dt.get("runtime_ms")}, {"error": "{:.3f}"})
        update_cluster_view(clusters, frame)

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


def _text(value):
    return "；".join(f"{key}={_fmt(item)}" for key, item in (value or {}).items()) or "缺失 · 未接入"


def _fmt(value):
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "--"


def _pct(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "--"


def _ms(value):
    try:
        return f"{float(value):.1f} ms"
    except (TypeError, ValueError):
        return "--"
