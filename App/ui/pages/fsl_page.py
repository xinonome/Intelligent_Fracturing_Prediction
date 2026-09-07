from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..display_text import display_text
from ...data.fsl_timeline_loader import FSLTimelineLoader
from ..widgets.status_card import Panel
from ..widgets.fsl_timeline_panel import build_stage_table, FSLTimelineChart
from ..widgets.fsl_transfer_workbench import create_fsl_transfer_workbench
from ..widgets.knowledge_graph_panel import build_knowledge_graph_panel
from .data_import_page import build_data_import_page


def build_fsl_page(
    registry,
    on_imported=None,
    on_dataset_requested=None,
    *,
    include_resource_tabs: bool = True,
):
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtWidgets import (
        QComboBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QScrollArea,
        QSlider,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    page = QScrollArea()
    page.setFrameShape(QFrame.NoFrame)
    page.setWidgetResizable(True)
    content = QWidget()
    page_layout = QVBoxLayout(content)
    page_layout.setContentsMargins(20, 16, 20, 20)
    page_layout.setSpacing(10)
    page.setWidget(content)

    tabs = QTabWidget()
    prediction_tab = QWidget()
    layout = QVBoxLayout(prediction_tab)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.setSpacing(10)
    refresh_imported_callback = lambda: None

    def on_data_imported():
        if hasattr(registry, "refresh_catalog"):
            registry.refresh_catalog()
        refresh_imported_callback()
        if on_imported:
            on_imported()

    tabs.addTab(prediction_tab, "实时推理与事件")
    if include_resource_tabs:
        tabs.addTab(build_knowledge_graph_panel(registry), "知识查询")
        tabs.addTab(build_data_import_page(registry, on_imported=on_data_imported), "数据导入")
    page_layout.addWidget(tabs)

    title = QLabel("工况识别与风险预测")
    title.setObjectName("pageTitle")
    layout.addWidget(title)

    module = registry.module("fsl")
    timeline_loader = FSLTimelineLoader(registry)
    timeline_rows = timeline_loader.summaries()
    transfer_workbench = create_fsl_transfer_workbench(registry, timeline_rows)
    tabs.insertTab(1, transfer_workbench, "跨井应用")
    timeline_panel, timeline_layout = Panel.create("施工数据工作台")
    timeline_layout.setContentsMargins(8, 8, 8, 8)
    timeline_chart = FSLTimelineChart()
    timeline_table = build_stage_table(timeline_rows)
    timeline_table.setToolTip("点击井段查看对应施工曲线、工况区间和时序结果")
    current_stage: dict[str, Any] = {}
    selected_dataset_id = str(getattr(registry, "dataset_id", ""))

    selector_row = QHBoxLayout()
    selector_row.setContentsMargins(0, 0, 0, 4)
    stage_identity = _label("", "key")
    stage_identity.setVisible(False)
    stage_identity.setToolTip("井段由窗口顶部的统一数据选择器控制")
    data_status = _label("", "muted")
    selector_row.addWidget(data_status, 1)
    timeline_layout.addLayout(selector_row)

    action_row = QHBoxLayout()
    analyse_button = QPushButton("重新读取与分析")
    analyse_button.setToolTip("重新读取独立井段文件，执行数据清洗、工况区间提取和逐点预测")
    export_chart_button = QPushButton("导出当前图表")
    export_event_button = QPushButton("导出事件报告")
    for button in (analyse_button, export_chart_button, export_event_button):
        action_row.addWidget(button)
    action_row.addStretch(1)
    timeline_layout.addLayout(action_row)

    playback = QWidget()
    playback_layout = QHBoxLayout(playback)
    playback_layout.setContentsMargins(0, 0, 0, 0)
    play_button = QPushButton("▶ 播放")
    reset_button = QPushButton("↺ 重置")
    back_button = QPushButton("‹ 上一步")
    next_button = QPushButton("下一步 ›")
    speed_box = QComboBox()
    for value in (0.5, 1.0, 2.0, 4.0):
        speed_box.addItem(f"{value:g}×", value)
    speed_box.setCurrentIndex(1)
    slider = QSlider(Qt.Horizontal)
    time_label = QLabel("t=--")
    for widget in (play_button, reset_button, back_button, next_button, QLabel("速度"), speed_box):
        playback_layout.addWidget(widget)
    playback_layout.addWidget(slider, 1)
    playback_layout.addWidget(time_label)
    timeline_layout.addWidget(playback)
    timer = QTimer(playback)

    condition_status = _label("", "warning")
    timeline_layout.addWidget(condition_status)
    condition_button = QPushButton("工况模型未接入")
    condition_button.setEnabled(False)
    condition_button.setToolTip("正式迁移权重缺配套预处理包；两阶段模型仅有评估输出。smoke 测试模型不作为正式工况预测。")
    action_row.insertWidget(1, condition_button)
    event_detail = _label("请选择事件区间：点击左侧曲线中的色带或事件边界查看详情；右侧表格用于切换井段。", "notice")
    timeline_layout.addWidget(event_detail)

    def set_playback_index(index):
        if not current_stage:
            return
        times = current_stage.get("time_s", [])
        index = max(0, min(int(index), max(len(times) - 1, 0)))
        slider.blockSignals(True)
        slider.setValue(index)
        slider.blockSignals(False)
        timeline_chart.set_index(index)
        seconds = float(times[index]) if times else 0.0
        duration = float(current_stage.get("duration_s") or seconds)
        time_label.setText(f"t={seconds:.0f}s / {duration:.0f}s")
        active = next(
            (
                item
                for item in current_stage.get("intervals", [])
                if float(item.get("start_s") or 0.0) <= seconds <= float(item.get("end_s") or 0.0)
            ),
            None,
        )
        show_interval(active)

    def clear_stage():
        nonlocal current_stage
        timer.stop()
        play_button.setText("▶ 播放")
        current_stage = {}
        timeline_chart.set_stage({})
        timeline_table.blockSignals(True)
        timeline_table.clearSelection()
        timeline_table.blockSignals(False)
        slider.blockSignals(True)
        slider.setRange(0, 0)
        slider.setValue(0)
        slider.blockSignals(False)
        time_label.setText("t=--")
        for widget in (play_button, reset_button, back_button, next_button, slider, export_chart_button, export_event_button):
            widget.setEnabled(False)
        data_status.setText("当前井段没有可用的施工工况识别数据；可在数据导入后刷新")
        condition_status.setText("工况模型未接入 · 源表标签与规则检测分开显示")
        show_interval(None)

    def set_playback_stage(stage):
        count = max(len(stage.get("time_s", [])) - 1, 0)
        slider.setRange(0, count)
        set_playback_index(0)

    def toggle_playback():
        if not current_stage.get("time_s"):
            return
        if timer.isActive():
            timer.stop()
            play_button.setText("▶ 播放")
        else:
            timer.start(int(400 / max(float(speed_box.currentData() or 1.0), 0.5)))
            play_button.setText("⏸ 暂停")

    def advance():
        if slider.value() >= slider.maximum():
            timer.stop()
            play_button.setText("▶ 播放")
            return
        set_playback_index(slider.value() + 1)

    def show_interval(interval):
        if not interval:
            available = any(
                current_stage.get(key)
                for key in ("intervals", "actual_condition_intervals", "rule_condition_intervals", "predicted_condition_intervals")
            )
            if available:
                event_detail.setText("请选择事件区间：点击左侧曲线中的色带或事件边界查看详情；右侧表格用于切换井段。")
            else:
                event_detail.setText("当前井段没有可选事件区间；可继续回放压力、排量和砂比，或切换到有工况标签的井段。")
            return
        event_detail.setText(
            f"类型：{ {'predicted': '模型识别', 'rule': '规则检测', 'actual': '源表实际标签'}.get(interval.get('kind'), '未知来源事件') }　"
            f"类别：{display_text(interval.get('label', '未标注'))}　"
            f"边界：{float(interval.get('start_s') or 0):.0f}–{float(interval.get('end_s') or 0):.0f} s　"
            f"依据：{display_text(interval.get('trigger_reason') or current_stage.get('source_file', '--'))}"
        )

    def select_stage(stage_id):
        nonlocal current_stage
        clear_stage()
        stage_id = str(stage_id or "")
        stage = timeline_loader.stage(stage_id)
        if not stage:
            return
        current_stage = stage
        timeline_chart.set_stage(stage)
        timeline_table.select_stage(stage_id)
        for widget in (play_button, reset_button, back_button, next_button, slider, export_chart_button, export_event_button):
            widget.setEnabled(bool(stage.get("time_s")))
        set_playback_stage(stage)
        data_status.setText(
            f"{stage.get('sample_count', 0)} 点 · {len(stage.get('intervals', []))} 个事件 · {stage.get('source_file', '--')}"
        )
        if stage.get("condition_prediction_status") == "ready":
            condition_status.setText(f"工况预测 · {len(stage.get('predicted_condition_intervals', []))} 个区间")
        elif stage.get("condition_prediction_status") == "no_labels":
            condition_status.setText("当前井段没有工况标签 · 只能回放压力/排量/砂比，不能选择工况区间")
        else:
            condition_status.setText("工况预测模型未接入 · 当前色带来自源表标签，点击色带可查看事件区间")
        timeline_chart.setToolTip(_point_prediction_summary(stage))
        show_interval({})

    def set_global_dataset(dataset_id):
        """Follow the application-wide dataset selection."""
        nonlocal selected_dataset_id
        selected_dataset_id = str(dataset_id or "")
        clear_stage()
        dataset = registry.dataset(dataset_id)
        stage_id = str(dataset.get("stage_id") or "")
        stage_identity.setText(str(dataset.get("display_name") or stage_id or dataset_id))
        stage = timeline_loader.stage(stage_id) if stage_id else None
        if not stage and stage_id:
            wanted = re.sub(r"[^0-9a-z]+", "", stage_id.lower()).replace("fdbh", "")
            well = str(dataset.get("well_id") or "").strip().lower()
            source_name = Path(str(dataset.get("pressure_source") or "")).name.lower()
            candidates = []
            for value in timeline_loader.stage_ids():
                normalized = re.sub(r"[^0-9a-z]+", "", value.lower()).replace("fdbh", "")
                candidate = timeline_loader.stage(value) or {}
                identity_matches = not well or str(candidate.get("well_id") or "").lower() == well
                source_matches = not source_name or str(candidate.get("source_file") or "").lower() == source_name
                if normalized.startswith(wanted) and identity_matches and source_matches:
                    candidates.append(candidate)
            stage = candidates[0] if len(candidates) == 1 else None
        if stage:
            select_stage(str(stage.get("stage_id") or stage_id))
            return
        # The default unified dataset (for example JY84-Z1 Stage 08) is a
        # registered DT source rather than an independent FDBH label table.
        # It still has valid construction curves in the DT cache, so keep the
        # pressure replay visible instead of showing an empty FSL panel.
        replay_stage = timeline_loader.pressure_replay_stage(dataset)
        if replay_stage:
            current_stage = replay_stage
            timeline_chart.set_stage(replay_stage)
            timeline_table.clearSelection()
            for widget in (play_button, reset_button, back_button, next_button, slider, export_chart_button, export_event_button):
                widget.setEnabled(bool(replay_stage.get("time_s")))
            set_playback_stage(replay_stage)
            data_status.setText(
                f"{replay_stage.get('sample_count', 0)} 点 · 无工况标签 · {replay_stage.get('data_source', '--')}"
            )
            condition_status.setText("当前井段没有工况标签 · 只能回放压力/排量/砂比，不能选择工况区间")
            timeline_chart.setToolTip("当前仅回放已登记的施工压力、排量和砂比数据")
            show_interval({})

    def request_stage(stage_id):
        """Keep table clicks consistent with the global selector."""

        target = None
        for dataset_id, dataset in (registry.dataset_catalog().get("datasets", {}) or {}).items():
            if str(dataset.get("stage_id") or "") == str(stage_id):
                target = str(dataset_id)
                break
        if target and on_dataset_requested:
            on_dataset_requested(target)
        else:
            select_stage(stage_id)

    timeline_table.stageSelected.connect(request_stage)
    timeline_chart.intervalSelected.connect(show_interval)
    slider.valueChanged.connect(set_playback_index)
    play_button.clicked.connect(toggle_playback)
    reset_button.clicked.connect(lambda: set_playback_index(0))
    back_button.clicked.connect(lambda: set_playback_index(slider.value() - 1))
    next_button.clicked.connect(lambda: set_playback_index(slider.value() + 1))
    speed_box.currentIndexChanged.connect(
        lambda: timer.start(int(400 / max(float(speed_box.currentData() or 1.0), 0.5))) if timer.isActive() else None
    )
    timer.timeout.connect(advance)
    set_global_dataset(str(getattr(registry, "dataset_id", "")))
    timeline_grid = QGridLayout()
    timeline_grid.setContentsMargins(0, 0, 0, 0)
    timeline_grid.setHorizontalSpacing(8)
    timeline_grid.addWidget(timeline_chart, 0, 0)
    timeline_grid.addWidget(timeline_table, 0, 1)
    timeline_grid.setColumnStretch(0, 2)
    timeline_grid.setColumnStretch(1, 1)
    timeline_layout.addLayout(timeline_grid)
    layout.addWidget(timeline_panel)

    def reanalyse():
        nonlocal timeline_rows
        clear_stage()
        analyse_button.setEnabled(False)
        data_status.setText("正在重新读取、清洗并分析独立井段数据…")
        try:
            timeline_rows = timeline_loader.refresh()
            timeline_table.set_rows(timeline_rows)
            transfer_workbench.refresh_rows(timeline_rows)
            set_global_dataset(selected_dataset_id)
            if not timeline_rows:
                data_status.setText(timeline_loader.status_reason + "；可继续导入后刷新")
        except Exception as exc:
            data_status.setText(f"分析失败：{exc}")
        finally:
            analyse_button.setEnabled(True)

    def export_chart():
        destination, _ = QFileDialog.getSaveFileName(page, "导出当前图表", "施工工况与逐点预测.png", "PNG 图片 (*.png)")
        if destination and timeline_chart.grab().save(destination, "PNG"):
            data_status.setText(f"图表已导出：{destination}")

    def export_events():
        if not current_stage:
            return
        destination, _ = QFileDialog.getSaveFileName(page, "导出事件报告", "施工事件报告.json", "JSON (*.json)")
        if destination:
            payload = {
                "stage_id": current_stage.get("stage_id"),
                "well_id": current_stage.get("well_id"),
                "source_file": current_stage.get("source_file"),
                "start_time": current_stage.get("start_time"),
                "end_time": current_stage.get("end_time"),
                "intervals": current_stage.get("intervals", []),
                "actual_condition_intervals": current_stage.get("actual_condition_intervals", []),
                "rule_condition_intervals": current_stage.get("rule_condition_intervals", []),
                "predicted_condition_intervals": current_stage.get("predicted_condition_intervals", []),
                "condition_prediction_status": current_stage.get("condition_prediction_status"),
                "point_prediction_source": current_stage.get("point_prediction_source"),
                "point_prediction_semantics": current_stage.get("point_prediction_semantics"),
                "point_prediction_metrics": current_stage.get("point_prediction_metrics", {}),
            }
            Path(destination).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            data_status.setText(f"事件报告已导出：{destination}")

    analyse_button.clicked.connect(reanalyse)
    export_chart_button.clicked.connect(export_chart)
    export_event_button.clicked.connect(export_events)
    refresh_imported_callback = reanalyse
    page.set_global_dataset = set_global_dataset
    page.refresh_data = reanalyse
    page.refresh_catalog = reanalyse
    page._fsl_chart = timeline_chart
    page._fsl_timer = timer
    page._fsl_slider = slider
    page._fsl_transfer = transfer_workbench
    page.pause_playback = lambda: timer.stop()

    return page


def _risk_summary(module: dict[str, Any]) -> dict[str, Any]:
    supporting = module.get("supporting", {}) or {}
    value = supporting.get("risk_prediction", {})
    return value if isinstance(value, dict) else {}


def _get(data: dict[str, Any], *keys: str, default: Any = "--") -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _layout_widget(layout):
    from PySide6.QtWidgets import QWidget

    widget = QWidget()
    widget.setLayout(layout)
    return widget


def _label(text, name=None):
    from PySide6.QtWidgets import QLabel

    label = QLabel(display_text(text))
    if name:
        label.setObjectName(name)
    label.setWordWrap(True)
    return label


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


def _point_prediction_summary(stage: dict[str, Any]) -> str:
    """Describe the point curves without conflating them with frozen metrics."""

    source = display_text(stage.get("point_prediction_source", "未接入逐点预测"))
    semantics = display_text(stage.get("point_prediction_semantics", ""))
    metrics = stage.get("point_prediction_metrics", {})
    if not isinstance(metrics, dict):
        return f"逐点预测：{source}。{semantics}"

    def metric_text(key: str, label: str, unit: str, tolerance: str) -> str:
        value = metrics.get(key, {})
        if not isinstance(value, dict) or not value.get("sample_count"):
            return f"{label}暂无有效对比"
        mae = value.get("mae")
        hit = value.get("within_tolerance_rate")
        try:
            mae_text = f"{float(mae):.2f}{unit}"
        except (TypeError, ValueError):
            mae_text = "--"
        return f"{label} MAE {mae_text}，{tolerance}内 {_pct(hit)}"

    pressure = metric_text("pressure", "压力", " MPa", "±2.0 MPa")
    flow = metric_text("flow", "排量", " m³/min", "±0.5 m³/min")
    sand = metric_text("sand", "砂比", " 个百分点", "±1.0 个百分点")
    count = stage.get("point_prediction_count", 0)
    try:
        count_text = str(int(count))
    except (TypeError, ValueError):
        count_text = "0"
    return f"逐点预测：{source}；{pressure}；{flow}；{sand}；有效对比最多 {count_text} 点。{semantics}"
