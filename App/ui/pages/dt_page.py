"""Second contract page: data-driven digital-twin state and assimilation view."""

from __future__ import annotations

import math
from pathlib import Path
import json
import sys

from ..parameter_format import format_parameter_map
from ..theme import PALETTE
from ..widgets.chart_panel import build_chart
from ..widgets.cluster_view import create_cluster_share_chart, create_cluster_view, update_cluster_view
from ..widgets.parameter_panel import create_parameter_panel, update_parameter_panel
from ..widgets.status_card import Panel
from ..widgets.timeline_control import create_timeline_control
from ..widgets.pyfrac_workbench import create_pyfrac_workbench
from ..web_view import Embedded3DView
from ...services.task_runner import TaskRunner


_PRESSURE_SERIES = (
    ("PKN先验", "prior_bottomhole_pressure_mpa", "prior_bhp", "orange"),
    ("观测压力", "observed_bottomhole_pressure_mpa", "observed_bhp", "blue"),
    ("EnKF后验", "bottomhole_pressure_mpa", "posterior_bhp", "cyan"),
)

_PRESSURE_PARAMETER_KEYS = (
    "E_prime_gpa",
    "C_L_m_sqrt_s",
    "mu_pa_s",
    "sigma_min_mpa",
    "K_IC_pa_sqrt_m",
)
_CLUSTER_PARAMETER_KEYS = tuple(f"intake_capacity_C{index}" for index in range(1, 7))
_INTERACTION_PARAMETER_KEYS = (
    "stress_shadow_scale",
    "boundary_relief_scale",
    "allocation_exponent",
)


def build_dt_page(controller, registry):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

    page = QScrollArea()
    page.setObjectName("dtPageScroll")
    page.setWidgetResizable(True)
    content = QWidget()
    page.setWidget(content)
    layout = QVBoxLayout(content)
    layout.setContentsMargins(20, 16, 20, 16)
    layout.setSpacing(10)

    title = QLabel("裂缝数字孪生")
    title.setObjectName("pageTitle")
    layout.addWidget(title)
    timeline = create_timeline_control(controller)
    layout.addWidget(timeline)

    operations, operations_layout = Panel.create("模型、场景与参数")
    operation_row = QHBoxLayout()
    model_box = QComboBox()
    model_box.addItem("PKN + KG-EnKF", "online")
    model_box.addItem("PyFrac原生模型", "pyfrac")
    model_box.addItem("PyFrac代理模型（不可用）", "surrogate")
    surrogate_item = model_box.model().item(2)
    if surrogate_item is not None:
        surrogate_item.setEnabled(False)
    scenario_box = QComboBox()
    scenario_box.addItem("有 DAS", "das_cluster_observation")
    scenario_box.addItem("无 DAS", "no_das_pressure_only")
    scenario_box.setCurrentIndex(max(0, scenario_box.findData(getattr(registry, "scenario_id", "das_cluster_observation"))))
    dataset_label = _label("", "key")
    dataset_label.setMinimumWidth(280)
    dataset_label.setVisible(False)
    dataset_label.setToolTip("井段由窗口顶部的统一数据选择器控制")
    recompute_button = QPushButton("重新计算")
    stop_compute_button = QPushButton("停止计算")
    stop_compute_button.setEnabled(False)
    apply_button = QPushButton("应用当前后验")
    apply_button.setEnabled(False)
    apply_button.setToolTip("现场参数写入接口尚未接入；当前页面只读回放")
    rollback_button = QPushButton("回退到上一步")
    export_button = QPushButton("导出当前帧")
    for widget in (QLabel("模型"), model_box, QLabel("场景"), scenario_box):
        operation_row.addWidget(widget)
    operation_row.addStretch(1)
    operations_layout.addLayout(operation_row)
    operation_actions = QHBoxLayout()
    for widget in (recompute_button, stop_compute_button, apply_button, rollback_button, export_button):
        operation_actions.addWidget(widget)
    operation_actions.addStretch(1)
    operations_layout.addLayout(operation_actions)
    operation_status = _label("", "muted")
    operations_layout.addWidget(operation_status)
    layout.addWidget(operations)

    pyfrac_workbench = create_pyfrac_workbench(page)
    pyfrac_workbench.setVisible(False)
    layout.addWidget(pyfrac_workbench)

    pressure_chart = build_chart("井底压力 / MPa · 观测 / PKN先验 / EnKF后验", 285, y_min=0.0)
    error_chart = build_chart("压力残差 / % · EnKF后验相对误差", 285, y_min=0.0)
    chart_row = QGridLayout()
    chart_row.setSpacing(10)
    chart_row.addWidget(pressure_chart, 0, 0)
    chart_row.addWidget(error_chart, 0, 1)
    chart_row.setColumnStretch(0, 2)
    chart_row.setColumnStretch(1, 1)
    layout.addLayout(chart_row)

    parameter_trend = build_chart("EnKF参数分组变化 / % · 相对先验", 250)
    cluster_chart = create_cluster_share_chart()
    state_row = QGridLayout()
    state_row.setSpacing(10)
    state_row.addWidget(parameter_trend, 0, 0)
    state_row.addWidget(cluster_chart, 0, 1)
    state_row.setColumnStretch(0, 2)
    state_row.setColumnStretch(1, 1)
    layout.addLayout(state_row)

    model_panel, model_layout = Panel.create("六簇裂缝演化")
    model = Embedded3DView.create(registry.html(getattr(registry, "scenario_id", None)))
    model.setMinimumHeight(430)
    if getattr(timeline, "set_playback_callback", None):
        timeline.set_playback_callback(lambda playing: model.set_interaction_enabled(not playing))
    model_layout.addWidget(model)
    layout.addWidget(model_panel)

    params = create_parameter_panel(
        "EnKF参数更新 · 当前帧",
        columns=[
            (
                "先验参数",
                [
                    ("prior_pressure", "压力 / 裂缝"),
                    ("prior_cluster", "簇进液能力"),
                    ("prior_interaction", "应力 / 边界 / 分配"),
                ],
            ),
            (
                "后验参数",
                [
                    ("posterior_pressure", "压力 / 裂缝"),
                    ("posterior_cluster", "簇进液能力"),
                    ("posterior_interaction", "应力 / 边界 / 分配"),
                ],
            ),
            (
                "当前状态",
                [
                    ("balance", "分簇均衡度"),
                    ("net_pressure", "净压力"),
                    ("fracture_length", "总半缝长"),
                    ("fracture_width", "最大缝宽"),
                    ("runtime", "当前帧耗时"),
                ],
            ),
        ],
        column_stretches=[3, 3, 2],
    )
    clusters = create_cluster_view()
    cluster_title = QLabel("六簇裂缝与分配结果 · 当前帧")
    cluster_title.setObjectName("key")
    params.layout().addWidget(cluster_title)
    params.layout().addWidget(clusters)
    layout.addWidget(params)

    source_label = _label("", "muted")
    layout.addWidget(source_label)

    runner = TaskRunner(page)

    def sync_dataset_context():
        """Refresh only the read-only page context after global selection."""

        dataset = registry.dataset()
        dataset_label.setText(str(dataset.get("display_name") or dataset.get("stage_id") or registry.dataset_id))
        current_scenario = str(getattr(registry, "scenario_id", ""))
        for index in range(scenario_box.count()):
            scenario_id = str(scenario_box.itemData(index) or "")
            item = scenario_box.model().item(index)
            if item is not None:
                item.setEnabled(bool(registry.dataset_ready_for_scenario(registry.dataset_id, scenario_id)))
        scenario_index = scenario_box.findData(current_scenario)
        if scenario_index >= 0:
            scenario_box.blockSignals(True)
            scenario_box.setCurrentIndex(scenario_index)
            scenario_box.blockSignals(False)
        operation_status.setText("")
        pyfrac_workbench.set_dataset(dict(dataset, dataset_id=str(getattr(registry, "dataset_id", ""))))
        recompute_button.setEnabled(dataset.get("adapter") == "raw_frac_construction")
        recompute_button.setToolTip(
            "根据当前施工表重新生成无 DAS 压力校正、PKN估计和三维回放。"
            if dataset.get("adapter") == "raw_frac_construction"
            else "有 DAS 参考井段使用已登记同化结果；此处不覆盖研究运行。"
        )

    def load_selection():
        scenario_id = str(scenario_box.currentData() or "")
        dataset_id = str(getattr(registry, "dataset_id", "") or "")
        if not dataset_id or not registry.dataset_ready_for_scenario(dataset_id, scenario_id):
            operation_status.setText("当前井段没有该场景的可用数据。请切换顶部数据井段或场景。")
            return
        operation_status.setText("正在读取井段并构建回放…")
        try:
            frames = controller.prepare_scenario_frames(scenario_id, dataset_id=dataset_id)
            controller.apply_scenario_frames(scenario_id, frames, dataset_id=dataset_id)
            model.set_html_path(registry.html(scenario_id))
            operation_status.setText(f"已加载当前井段 · {len(frames)} 个回放节点")
        except Exception as exc:
            operation_status.setText(f"井段加载失败：{exc}")

    def change_scenario(index):
        scenario_id = str(scenario_box.itemData(index) or "")
        if not scenario_id or scenario_id == str(getattr(registry, "scenario_id", "")):
            return
        if not registry.dataset_ready_for_scenario(registry.dataset_id, scenario_id):
            operation_status.setText("当前井段不支持该场景，已保留原场景。")
            sync_dataset_context()
            return
        load_selection()

    def recompute():
        dataset = registry.dataset()
        if dataset.get("adapter") != "raw_frac_construction":
            operation_status.setText("当前有 DAS 井段使用已登记结果，不从此入口覆盖研究运行。")
            return
        command = [
            str(Path(sys.executable)),
            str(Path(__file__).resolve().parents[2] / "build_selected_dt_view.py"),
            "--dataset-id", str(registry.dataset_id),
        ]
        recompute_button.setEnabled(False)
        stop_compute_button.setEnabled(True)
        model_box.setEnabled(False)
        scenario_box.setEnabled(False)
        operation_status.setText("正在从当前井段原始数据重新计算无 DAS 数字孪生…")
        runner.start(command[0], command[1:], Path(__file__).resolve().parents[3])

    def recompute_finished(exit_code, _status):
        stop_compute_button.setEnabled(False)
        model_box.setEnabled(True)
        scenario_box.setEnabled(True)
        recompute_button.setEnabled(registry.dataset().get("adapter") == "raw_frac_construction")
        if exit_code != 0:
            operation_status.setText(f"重新计算失败或已停止（返回码 {exit_code}）。")
            return
        registry.refresh_catalog()
        try:
            controller.set_dataset(str(registry.dataset_id))
            model.set_html_path(registry.html(getattr(registry, "scenario_id", None)))
            refresh_visuals()
            operation_status.setText("当前井段计算完成，压力、裂缝与六簇估计已重新载入。")
        except Exception as exc:
            operation_status.setText(f"计算完成，但回放载入失败：{exc}")

    def export_frame():
        destination, _ = QFileDialog.getSaveFileName(page, "导出当前帧", "数字孪生当前帧.json", "JSON (*.json)")
        if destination:
            Path(destination).write_text(json.dumps(controller.current or {}, ensure_ascii=False, indent=2), encoding="utf-8")
            operation_status.setText(f"当前帧已导出：{destination}")

    scenario_box.currentIndexChanged.connect(change_scenario)
    recompute_button.clicked.connect(recompute)
    stop_compute_button.clicked.connect(runner.cancel)
    rollback_button.clicked.connect(lambda: controller.step(-1))
    export_button.clicked.connect(export_frame)
    runner.process.finished.connect(recompute_finished)
    runner.process.errorOccurred.connect(
        lambda _error: (
            stop_compute_button.setEnabled(False),
            model_box.setEnabled(True),
            scenario_box.setEnabled(True),
            operation_status.setText(f"计算进程未能启动：{runner.process.errorString()}"),
        )
    )
    standard_widgets = (
        pressure_chart,
        error_chart,
        parameter_trend,
        cluster_chart,
        model_panel,
        params,
        clusters,
        source_label,
        timeline,
    )

    def set_model_mode():
        mode = str(model_box.currentData() or "online")
        pyfrac_mode = mode == "pyfrac"
        pyfrac_workbench.setVisible(pyfrac_mode)
        for widget in standard_widgets:
            widget.setVisible(not pyfrac_mode)
        for widget in (scenario_box, recompute_button, rollback_button, export_button):
            widget.setEnabled(not pyfrac_mode)
        if pyfrac_mode:
            stop_compute_button.setEnabled(False)
        apply_button.setEnabled(False)
        if pyfrac_mode:
            operation_status.setText("PyFrac原生推演 · 当前井段能力见参数页状态")
        else:
            sync_dataset_context()

    model_box.currentIndexChanged.connect(set_model_mode)
    sync_dataset_context()
    set_model_mode()

    last_frames_token = id(controller.frames)

    def refresh_visuals():
        nonlocal last_frames_token
        frames = list(controller.frames)
        pressure_series = []
        for name, dt_key, flat_key, color_name in _PRESSURE_SERIES:
            values = [_field(frame, dt_key, flat_key) for frame in frames]
            if _has_finite(values):
                pressure_series.append((name, values, PALETTE[color_name]))
        pressure_chart.set_series(pressure_series)

        error_values = [
            _percent_value(_field(frame, "posterior_pressure_error", "posterior_pressure_error"))
            for frame in frames
        ]
        error_chart.set_series(
            [("后验相对误差", error_values, PALETTE["red"])]
            if _has_finite(error_values)
            else []
        )

        parameter_series = []
        for label, keys, color_name in (
            ("压力 / 裂缝参数", _PRESSURE_PARAMETER_KEYS, "orange"),
            ("簇进液能力参数", _CLUSTER_PARAMETER_KEYS, "blue"),
            ("应力-边界-分配参数", _INTERACTION_PARAMETER_KEYS, "cyan"),
        ):
            values = [_parameter_group_change(frame, keys) for frame in frames]
            if _has_finite(values):
                parameter_series.append((label, values, PALETTE[color_name]))
        parameter_trend.set_series(parameter_series)

        if frames:
            start = _field(frames[0], "time_s", "time_s")
            end = _field(frames[-1], "time_s", "time_s")
            if start is not None and end is not None:
                pressure_chart.set_time_range(start, end)
                error_chart.set_time_range(start, end)
                parameter_trend.set_time_range(start, end)
            source = _source_name(frames[-1])
            source_label.setText(f"数据源：{source} · 时间覆盖 {_fmt_seconds(start)}–{_fmt_seconds(end)} s")
        else:
            source_label.setText("暂无可回放数据")
        last_frames_token = id(controller.frames)

    def update(frame):
        nonlocal last_frames_token
        if id(controller.frames) != last_frames_token:
            refresh_visuals()
        pressure_chart.set_index(controller.index)
        error_chart.set_index(controller.index)
        parameter_trend.set_index(controller.index)
        cluster_chart.set_frame(frame)
        update_cluster_view(clusters, frame)
        time_s = _field(frame, "time_s", "time_s")
        if time_s is not None:
            model.set_time_index(time_s)

        dt = frame.get("dt", {}) or {}
        prior = dt.get("prior_parameters", {}) or {}
        posterior = dt.get("posterior_parameters", {}) or {}
        update_parameter_panel(
            params,
            {
                "prior_pressure": _text(_select_parameters(prior, _PRESSURE_PARAMETER_KEYS)),
                "prior_cluster": _text(_select_parameters(prior, _CLUSTER_PARAMETER_KEYS)),
                "prior_interaction": _text(_select_parameters(prior, _INTERACTION_PARAMETER_KEYS)),
                "posterior_pressure": _text(_select_parameters(posterior, _PRESSURE_PARAMETER_KEYS)),
                "posterior_cluster": _text(_select_parameters(posterior, _CLUSTER_PARAMETER_KEYS)),
                "posterior_interaction": _text(_select_parameters(posterior, _INTERACTION_PARAMETER_KEYS)),
                "balance": _field(frame, "cluster_balance_degree", "cluster_balance_degree"),
                "net_pressure": _field(frame, "net_pressure_mpa", "net_pressure_mpa"),
                "fracture_length": _field(frame, "fracture_length_m", "fracture_length_m"),
                "fracture_width": _scale(_field(frame, "fracture_width_m", "fracture_width_m"), 1000.0),
                "runtime": _field(frame, "runtime_ms", "runtime_ms"),
            },
            {
                "balance": "{:.3f}",
                "net_pressure": "{:.2f} MPa",
                "fracture_length": "{:.2f} m",
                "fracture_width": "{:.3f} mm",
                "runtime": "{:.1f} ms",
            },
        )

    refresh_visuals()
    controller.frameChanged.connect(update)
    update(controller.current or {})
    pending_dataset_visual = {"value": False}

    def refresh_dataset_visual():
        """Load the selected stage's 3D document only when this page is visible."""

        if not page.isVisible():
            pending_dataset_visual["value"] = True
            return
        pending_dataset_visual["value"] = False
        scenario_html = registry.html(getattr(registry, "scenario_id", None))
        if scenario_html and getattr(model, "set_html_path", None):
            model.set_html_path(scenario_html)

    def set_global_dataset(_dataset_id):
        sync_dataset_context()
        refresh_dataset_visual()

    page.set_global_dataset = set_global_dataset
    page.refresh_dataset_visual = refresh_dataset_visual
    page.pause_playback = timeline.pause
    page.prepare_dataset_change = lambda: (
        runner.cancel(),
        pyfrac_workbench.stop_active_task(),
    )
    return page


def _label(text, name=None):
    from PySide6.QtWidgets import QLabel

    label = QLabel(text)
    if name:
        label.setObjectName(name)
    label.setWordWrap(True)
    return label


def _set_card_value(card, value: str, caption: str | None = None) -> None:
    from PySide6.QtWidgets import QLabel

    value_label = card.findChild(QLabel, "metricValue")
    if value_label is not None:
        value_label.setText(value)
    if caption is not None:
        caption_label = card.findChild(QLabel, "metricCaption")
        if caption_label is not None:
            caption_label.setText(caption)


def _field(frame: dict, dt_key: str, flat_key: str | None = None) -> float | None:
    """Read typed DT fields first and legacy flat fields only as fallback."""

    dt = frame.get("dt", {}) or {}
    if dt_key in dt:
        return _number(dt.get(dt_key))
    return _number(frame.get(flat_key or dt_key))


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _has_finite(values) -> bool:
    return any(_number(value) is not None for value in values)


def _scale(value, factor: float):
    value = _number(value)
    return None if value is None else value * factor


def _percent_value(value):
    value = _number(value)
    return None if value is None else value * 100.0


def _text(value):
    return format_parameter_map(value)


def _select_parameters(parameters: dict, keys: tuple[str, ...]) -> dict:
    return {key: parameters[key] for key in keys if _number(parameters.get(key)) is not None}


def _parameter_group_change(frame: dict, keys: tuple[str, ...]):
    dt = frame.get("dt", {}) or {}
    prior = dt.get("prior_parameters", {}) or {}
    posterior = dt.get("posterior_parameters", {}) or {}
    changes = []
    for key in keys:
        prior_value = _number(prior.get(key))
        posterior_value = _number(posterior.get(key))
        if prior_value is None or posterior_value is None or abs(prior_value) < 1.0e-15:
            continue
        changes.append((posterior_value - prior_value) / abs(prior_value) * 100.0)
    if not changes:
        return None
    return sum(changes) / len(changes)


def _source_name(frame: dict) -> str:
    quality = (frame.get("dt", {}) or {}).get("quality", {}) or {}
    source = str(quality.get("source", "当前回放缓存"))
    if source in {"", "missing"}:
        return "当前回放缓存"
    return Path(source).name or "当前回放缓存"


def _fmt_seconds(value) -> str:
    number = _number(value)
    return "--" if number is None else f"{number:.0f}"


def _pct(value):
    value = _number(value)
    return "--" if value is None else f"{value * 100:.1f}%"


def _ms(value):
    value = _number(value)
    return "--" if value is None else f"{value:.1f} ms"


__all__ = ["build_dt_page"]
