from __future__ import annotations

from ...data.pyfrac_loader import PyFracComparison
from ..theme import PALETTE
from .chart_panel import build_chart
from .status_card import Panel


def create_pyfrac_comparison_panel(comparison: PyFracComparison):
    """Create an honest offline PKN/PyFrac comparison panel for the APP."""

    from PySide6.QtWidgets import QGridLayout, QLabel

    panel, layout = Panel.create("PKN / PyFrac 离线对照")
    status = QLabel()
    status.setObjectName("muted")
    status.setWordWrap(True)
    layout.addWidget(status)

    metric = QLabel()
    metric.setObjectName("value")
    metric.setWordWrap(True)
    layout.addWidget(metric)

    charts = QGridLayout()
    charts.setContentsMargins(0, 4, 0, 0)
    charts.setHorizontalSpacing(8)
    length_chart = build_chart("半缝长 / m", 230, y_min=0.0)
    aperture_chart = build_chart("最大开度 / mm", 230, y_min=0.0)
    pressure_chart = build_chart("净压力 / MPa", 230, y_min=0.0)
    charts.addWidget(length_chart, 0, 0)
    charts.addWidget(aperture_chart, 0, 1)
    charts.addWidget(pressure_chart, 1, 0, 1, 2)
    layout.addLayout(charts)
    layout.setStretch(2, 1)

    def update_view(value: PyFracComparison) -> None:
        if not value.available or not value.points:
            status.setText(value.note or "暂无 PyFrac 对照结果")
            metric.clear()
            length_chart.set_series([])
            aperture_chart.set_series([])
            pressure_chart.set_series([])
            return

        times = [item.time_s for item in value.points]
        length_chart.set_series([
            ("PKN", [_or_nan(item.pkn_half_length_m) for item in value.points], PALETTE["orange"]),
            ("PyFrac 原生动态", [_or_nan(item.pyfrac_half_length_m) for item in value.points], PALETTE["cyan"]),
        ])
        aperture_chart.set_series([
            ("PKN", [_or_nan(item.pkn_max_aperture_mm) for item in value.points], PALETTE["orange"]),
            ("PyFrac 原生动态", [_or_nan(item.pyfrac_max_aperture_mm) for item in value.points], PALETTE["cyan"]),
        ])
        pressure_chart.set_series([
            ("PKN", [_or_nan(item.pkn_net_pressure_mpa) for item in value.points], PALETTE["orange"]),
            ("PyFrac 原生动态", [_or_nan(item.pyfrac_net_pressure_mpa) for item in value.points], PALETTE["cyan"]),
        ])
        for chart in (length_chart, aperture_chart, pressure_chart):
            chart.set_time_range(times[0], times[-1] if len(times) > 1 else times[0])
        latest = value.latest
        if latest is None:
            return
        status.setText(
            f"已接入 · {value.engine_mode} · 第 {value.cluster_id} 簇 · "
            f"{len(value.points)} 个成功检查点 · {value.note}"
        )
        metric.setText(
            f"末检查点 t={latest.time_s:.0f}s："
            f"半缝长 PKN {_fmt(latest.pkn_half_length_m)} m / PyFrac {_fmt(latest.pyfrac_half_length_m)} m；"
            f"最大开度 PKN {_fmt(latest.pkn_max_aperture_mm)} mm / PyFrac {_fmt(latest.pyfrac_max_aperture_mm)} mm；"
            f"PyFrac耗时 {_fmt(latest.pyfrac_runtime_s)} s，成功步数 {latest.successful_time_steps or ''}。"
        )

    update_view(comparison)
    panel._update_pyfrac_comparison = update_view
    panel._pyfrac_length_chart = length_chart
    panel._pyfrac_aperture_chart = aperture_chart
    panel._pyfrac_pressure_chart = pressure_chart
    return panel


def _or_nan(value):
    return float(value) if value is not None else float("nan")


def _fmt(value):
    return "" if value is None else f"{float(value):.3f}"
