from __future__ import annotations

import math
from pathlib import Path

from ...services.operator_decisions import DEFAULT_LOG, append_decision, load_decisions, rollback_decision
from ...data.hmi_loader import dataset_agent_evaluation_path, discover_agent_models
from ...core.paths import PATHS
from ...services.task_runner import TaskRunner
from ...services.ml_runtime import resolve_ml_python
from ..theme import PALETTE
from ..widgets.chart_panel import build_chart
from ..widgets.status_card import Panel
from ..widgets.timeline_control import create_timeline_control
from ..widgets.risk_timeline_panel import create_risk_timeline


def build_hmi_page(controller, registry):
    """Operator-facing advisory review and confirmation workbench."""
    from PySide6.QtCore import QProcess, Qt
    from PySide6.QtWidgets import (
        QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton,
        QScrollArea, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
    )

    page = QScrollArea()
    page.setObjectName("hmiPageScroll")
    page.setWidgetResizable(True)
    page.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    content = QWidget()
    page.setWidget(content)
    layout = QVBoxLayout(content)
    layout.setContentsMargins(20, 16, 20, 18)
    layout.setSpacing(10)
    title = QLabel("智能决策工作台")
    title.setObjectName("pageTitle")
    layout.addWidget(title)

    data_row = QHBoxLayout()
    data_row.setSpacing(8)
    data_label = _label("", "key")
    data_label.setObjectName("hmiDatasetIdentity")
    data_label.setToolTip("井段由窗口顶部的统一数据选择器控制")
    predict_data = QPushButton("计算并缓存")
    predict_data.setObjectName("hmiPredict")
    stop_prediction = QPushButton("停止计算")
    stop_prediction.setObjectName("hmiStopPrediction")
    stop_prediction.setEnabled(False)
    data_status = _label("", "muted")
    data_status.setObjectName("hmiCacheStatus")
    data_status.setWordWrap(True)
    data_row.addWidget(data_label)
    data_row.addWidget(predict_data)
    data_row.addWidget(stop_prediction)
    data_row.addWidget(data_status, 1)
    layout.addLayout(data_row)

    model_row = QHBoxLayout()
    model_row.setSpacing(8)
    model_row.addWidget(QLabel("智能体模型"))
    model_selector = QComboBox()
    model_selector.setObjectName("agentModelSelector")
    model_selector.setMinimumWidth(260)
    model_status = _label("正在读取真实模型回放…", "muted")
    model_status.setToolTip("切换当前智能体回放数据")
    agent_models = discover_agent_models(registry)
    for item in agent_models:
        model_selector.addItem(str(item.get("label", item.get("model_id", "未知模型"))), item.get("model_id"))
        index = model_selector.count() - 1
        model_selector.setItemData(index, str(item.get("evaluation_path") or ""), Qt.ToolTipRole)
        if not item.get("ready"):
            model_selector.model().item(index).setEnabled(False)
    if not agent_models:
        model_selector.addItem("暂无可用智能体回放", "")
        model_selector.model().item(0).setEnabled(False)
        model_status.setText("暂无可用的智能体模型回放")
    else:
        current_model = str(getattr(controller, "agent_model_id", "") or "").lower()
        selected_index = model_selector.findData(current_model)
        model_selector.setCurrentIndex(selected_index if selected_index >= 0 else 0)
        selected = agent_models[model_selector.currentIndex()]
        model_status.setText(str(selected.get("status", "可用")))
    model_row.addWidget(model_selector)
    model_row.addWidget(model_status, 1)
    layout.addLayout(model_row)
    timeline = create_timeline_control(controller)
    layout.addWidget(timeline)

    prediction_runner = TaskRunner(page)
    prediction_dataset_id = None
    prediction_model_id = None
    prediction_cancelled = False
    task_status = _label("本次任务：未启动。模型回放/已有缓存不代表新计算。", "muted")
    task_status.setObjectName("hmiTaskStatus")
    layout.addWidget(task_status)
    layout.addWidget(_label("人工审核仅写入本地记录，不下发现场；撤销仅针对选中确认记录。", "muted"))

    tabs = QTabWidget()
    tabs.setObjectName("hmiWorkbenchTabs")
    layout.addWidget(tabs, 1)
    review_page, curves_page, records_page = QWidget(), QWidget(), QWidget()
    review_layout, curves_layout, records_layout = QVBoxLayout(review_page), QVBoxLayout(curves_page), QVBoxLayout(records_page)
    tabs.addTab(review_page, "1 · 风险与审核")
    tabs.addTab(curves_page, "2 · 响应曲线")
    tabs.addTab(records_page, "3 · 审核记录")

    status = _label("", "notice")
    status.setObjectName("hmiReviewStatus")
    review_layout.addWidget(status)
    feedback = _label("", "muted")
    feedback.setObjectName("hmiAuditFeedback")
    layout.addWidget(feedback)
    risk_timeline = create_risk_timeline(list(controller.frames))
    risk_detail = _label("未选择风险区间", "notice")
    risk_timeline.setMinimumHeight(150)
    review_layout.addWidget(risk_timeline)
    review_layout.addWidget(risk_detail)
    pressure_chart = build_chart("压力响应 · 观测 / PKN先验 / EnKF后验", 230, y_min=0.0)
    flow_chart = build_chart("排量 · 当前 / 建议", 230, y_min=0.0)
    sand_chart = build_chart("砂比 · 当前 / 建议", 230, y_min=0.0)
    curve_tabs = QTabWidget()
    for chart, name in ((pressure_chart, "压力"), (flow_chart, "排量"), (sand_chart, "砂比")):
        curve_tabs.addTab(chart, name)
    curves_layout.addWidget(curve_tabs)

    panel, panel_layout = Panel.create("建议审核")
    reason = _label("暂无建议", "muted")
    panel_layout.addWidget(reason)
    values = QGridLayout()
    flow_value = QDoubleSpinBox()
    flow_value.setObjectName("hmiReviewFlow")
    flow_value.setRange(0.0, 100.0)
    flow_value.setDecimals(2)
    flow_value.setSuffix(" m³/min")
    sand_value = QDoubleSpinBox()
    sand_value.setObjectName("hmiReviewSand")
    sand_value.setRange(0.0, 100.0)
    sand_value.setDecimals(2)
    sand_value.setSuffix(" %")
    values.addWidget(QLabel("审核排量"), 0, 0)
    values.addWidget(flow_value, 0, 1)
    values.addWidget(QLabel("审核砂比"), 0, 2)
    values.addWidget(sand_value, 0, 3)
    panel_layout.addLayout(values)
    button_row = QHBoxLayout()
    accept = QPushButton("记录采用")
    accept.setObjectName("hmiAccept")
    reject = QPushButton("暂不采用")
    modify = QPushButton("记录修改后采用")
    modify.setObjectName("hmiModify")
    rollback = QPushButton("撤销选中确认记录")
    rollback.setObjectName("hmiRollback")
    rollback.setEnabled(False)
    export = QPushButton("导出全部审核记录")
    for button in (accept, reject, modify):
        button_row.addWidget(button)
    button_row.addStretch(1)
    panel_layout.addLayout(button_row)
    history = QTableWidget(0, 7)
    history.setObjectName("hmiDecisionHistory")
    history.setHorizontalHeaderLabels(["记录编号", "记录时间", "施工时刻", "决定", "排量", "砂比", "状态"])
    history.horizontalHeader().setStretchLastSection(True)
    history.setEditTriggers(QTableWidget.NoEditTriggers)
    history.setSelectionBehavior(QTableWidget.SelectRows)
    history.setSelectionMode(QTableWidget.SingleSelection)
    history.setMinimumHeight(190)
    review_layout.addWidget(panel)
    review_layout.addStretch(1)
    records_layout.addWidget(_label("仅显示当前井段 / 模型的最近 50 条记录；导出包含全部井段与旧版记录。", "muted"))
    history_buttons = QHBoxLayout()
    history_buttons.addWidget(rollback)
    history_buttons.addWidget(export)
    history_buttons.addStretch(1)
    records_layout.addLayout(history_buttons)
    records_layout.addWidget(history, 1)
    history_rows = []
    review_snapshot = None

    def current_context():
        return (str(getattr(registry, "dataset_id", "") or ""),
                str(getattr(controller, "agent_model_id", "") or "").lower())

    def selected_context_matches(context):
        return bool(context and context == current_context()
                    and context[1] == str(model_selector.currentData() or "").lower())

    def refresh_history():
        nonlocal history_rows
        dataset_id, model_id = current_context()
        try:
            all_rows = load_decisions(limit=None, dataset_id=dataset_id, model_id=model_id)
        except OSError as exc:
            feedback.setText(f"审核记录读取失败：{exc}")
            all_rows = []
        reversed_ids = {row.get("rollback_of") for row in all_rows if row.get("rollback_of")}
        rows = list(reversed(all_rows[-50:]))
        history_rows = rows
        history.clearSelection()
        rollback.setEnabled(False)
        history.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            row["_reversed"] = row.get("record_id") in reversed_ids
            row_values = (
                str(row.get("record_id", "旧版无编号"))[:10],
                row.get("recorded_at", "--"), _seconds(row.get("time_s")), row.get("decision", "--"),
                _value(row.get("flow_m3_min")), _value(row.get("sand_ratio_pct")),
                "已撤销 · 未下发现场" if row["_reversed"] else row.get("execution_state", "仅记录"),
            )
            for column, value in enumerate(row_values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(f"井段：{dataset_id} / 模型：{model_id}\n记录：{row.get('record_id', '--')}\n关联确认：{row.get('rollback_of', '--')}")
                history.setItem(row_index, column, item)

    def recommendation(frame):
        hmi = (frame or {}).get("hmi", {}) or {}
        return (
            _number((frame or {}).get("action_flow")),
            _number((frame or {}).get("action_sand")),
            hmi.get("high_level_action") or (frame or {}).get("hmi_option"),
        )

    def record(decision):
        snapshot = review_snapshot
        if (not snapshot or not selected_context_matches(snapshot["context"])
                or controller.current is not snapshot["original"]
                or controller.frames is not snapshot["frames"]):
            feedback.setText("井段、模型或施工时刻已变化，请重新审核当前建议。")
            update(controller.current or {})
            return
        if timeline.is_playing():
            timeline.pause()
            feedback.setText("已暂停播放，请核对当前建议后再次记录。")
            return
        frame = snapshot["frame"]
        recommended_flow, recommended_sand, action = recommendation(frame)
        if recommended_flow is None and recommended_sand is None:
            feedback.setText("当前时刻没有可审核的排量或砂比建议。")
            return
        dataset_id, model_id = snapshot["context"]
        flow, sand = recommended_flow, recommended_sand
        if decision == "修改后采用":
            flow = flow_value.value() if recommended_flow is not None else None
            sand = sand_value.value() if recommended_sand is not None else None
        elif decision == "暂不采用":
            flow = sand = None
        try:
            saved = append_decision({
            "dataset_id": dataset_id, "model_id": model_id,
            "frame_id": frame.get("frame_id", frame.get("index")),
            "time_s": frame.get("time_s"), "decision": decision,
            "recommended_flow_m3_min": recommended_flow,
            "recommended_sand_ratio_pct": recommended_sand,
            "flow_m3_min": flow, "sand_ratio_pct": sand,
            "advisory_source": ((frame.get("hmi") or {}).get("quality") or {}).get("source")
                or (frame.get("alignment") or {}).get("hmi_source") or frame.get("hmi_model_source"),
            "advisory_mode": snapshot["mode"], "risk": frame.get("decision", {}),
            "advisory_action": action, "execution_state": "人工已记录，未下发现场",
            })
        except (OSError, ValueError, TypeError) as exc:
            feedback.setText(f"记录未保存：{exc}")
            return
        feedback.setText(f"已记录：{decision} · {saved['record_id'][:10]}（未下发现场）")
        refresh_history()

    def selected_history_record():
        indexes = history.selectionModel().selectedRows()
        index = indexes[0].row() if indexes else -1
        return history_rows[index] if 0 <= index < len(history_rows) else None

    def update_rollback_control():
        row = selected_history_record()
        rollback.setEnabled(bool(row and row.get("record_id") and not row.get("_reversed")
                                 and row.get("record_type", "review") == "review"
                                 and row.get("decision") in {"确认采用", "修改后采用"}))

    def append_rollback():
        row = selected_history_record()
        if not row:
            feedback.setText("请先选择一条确认记录。")
            return
        dataset_id, model_id = current_context()
        try:
            rollback_decision(row.get("record_id", ""), dataset_id=dataset_id, model_id=model_id)
        except (OSError, ValueError) as exc:
            feedback.setText(f"未撤销：{exc}")
            return
        feedback.setText(f"已撤销确认记录 {row['record_id'][:10]}；原记录保留，未执行现场回滚。")
        refresh_history()

    def export_log():
        destination, _ = QFileDialog.getSaveFileName(page, "导出审核记录", "人工审核记录.jsonl", "JSON Lines (*.jsonl)")
        if destination:
            try:
                if Path(destination).resolve() == DEFAULT_LOG.resolve():
                    raise ValueError("导出路径不能覆盖原始审核日志")
                source_text = DEFAULT_LOG.read_text(encoding="utf-8") if DEFAULT_LOG.exists() else ""
                Path(destination).write_text(source_text, encoding="utf-8")
            except (OSError, ValueError) as exc:
                feedback.setText(f"导出失败：{exc}")
                return
            feedback.setText(f"全部审核记录已导出：{destination}")

    def selected_dataset_entry():
        dataset_id = str(getattr(registry, "dataset_id", "") or "")
        dataset = registry.dataset(dataset_id) if dataset_id else {}
        source = registry.path(dataset.get("pressure_source"))
        cache = registry.path(dataset.get("cache_source"))
        ready = bool(source and source.exists() and cache and cache.exists())
        return (dataset_id, dataset, ready) if dataset_id else None

    def update_data_controls():
        entry = selected_dataset_entry()
        busy = prediction_runner.process.state() != QProcess.NotRunning
        model_id = str(model_selector.currentData() or "").lower()
        if not entry:
            data_status.setText("暂无可用井段")
            predict_data.setEnabled(False)
            return
        dataset_id, dataset, ready = entry
        cached = dataset_agent_evaluation_path(registry, dataset_id, model_id)
        data_label.setText(str(dataset.get("display_name") or dataset.get("stage_id") or dataset_id))
        if dataset.get("adapter") == "raw_frac_construction":
            if cached:
                data_status.setText(f"{model_id.upper()} 预测已缓存")
            else:
                data_status.setText(f"尚未生成 {model_id.upper()} 预测")
            predict_data.setEnabled(bool(ready and model_id and not busy))
            predict_data.setToolTip("使用当前模型对所选单井段运行本地推理，并将结果写入该井段缓存。")
        else:
            data_status.setText(f"{model_id.upper()} 回放已加载")
            predict_data.setEnabled(False)
            predict_data.setToolTip("当前实时预测脚本仅支持无 DAS 单井段；有 DAS 井段使用独立登记回放。")

    def start_realtime_prediction():
        nonlocal prediction_dataset_id, prediction_model_id, prediction_cancelled
        entry = selected_dataset_entry()
        model_id = str(model_selector.currentData() or "").lower()
        if not entry or not model_id:
            return
        dataset_id, dataset, ready = entry
        if dataset.get("adapter") != "raw_frac_construction" or not ready:
            data_status.setText("实时预测仅支持已准备好的无 DAS 单井段")
            return
        if prediction_runner.process.state() != QProcess.NotRunning:
            return
        prediction_dataset_id = dataset_id
        prediction_model_id = model_id
        prediction_cancelled = False
        ml_python, runtime_note = resolve_ml_python()
        if not ml_python:
            data_status.setText(f"实时预测无法启动：{runtime_note}")
            return
        model_selector.setEnabled(False)
        predict_data.setEnabled(False)
        data_status.setText(f"正在对 {dataset.get('display_name', dataset_id)} 运行 {model_id.upper()} 实时预测…")
        prediction_runner.start(
            ml_python,
            [
                str(PATHS.app / "build_no_das_agent_cache.py"),
                "--dataset-id",
                dataset_id,
                "--algorithm",
                model_id,
            ],
            PATHS.root,
            environment={
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "TF_CPP_MIN_LOG_LEVEL": "2",
            },
        )
        stop_prediction.setEnabled(True)

    def cancel_realtime_prediction():
        nonlocal prediction_cancelled
        if prediction_runner.process.state() == QProcess.NotRunning:
            return
        prediction_cancelled = True
        prediction_runner.cancel()
        data_status.setText("正在停止当前计算；已写入的独立缓存文件将保留。")

    def refresh_visuals():
        frames = list(controller.frames)
        pressure_chart.set_series(_pressure_series(frames))
        _set_action_series(flow_chart, frames, "current_flow", "action_flow", PALETTE["cyan"], PALETTE["blue"])
        _set_action_series(sand_chart, frames, "current_sand", "action_sand", PALETTE["orange"], PALETTE["red"])
        for chart in (pressure_chart, flow_chart, sand_chart):
            _set_time_range(chart, frames)
        risk_timeline.set_frames(frames)

    def switch_agent_model(index):
        model_id = model_selector.itemData(index)
        meta = next((item for item in agent_models if item.get("model_id") == model_id), None)
        if not meta or not meta.get("ready"):
            model_status.setText("该模型没有可用的真实回放数据")
            return
        if str(getattr(controller, "agent_model_id", "")).lower() == str(model_id).lower():
            model_status.setText(str(meta.get("status", "可用")))
            return
        if getattr(timeline, "pause", None):
            timeline.pause()
        if not hasattr(controller, "set_agent_model"):
            model_status.setText("当前控制器不支持切换智能体模型")
            return
        model_selector.setEnabled(False)
        model_status.setText(f"正在载入 {meta.get('display_name', model_id)} 的真实回放…")
        try:
            controller.set_agent_model(str(model_id))
            model_status.setText(f"{meta.get('status', '可用')} · {meta.get('display_name', model_id)}")
        except Exception as exc:
            model_status.setText(f"模型切换失败：{type(exc).__name__}: {exc}")
            current = model_selector.findData(getattr(controller, "agent_model_id", ""))
            if current >= 0:
                model_selector.blockSignals(True)
                model_selector.setCurrentIndex(current)
                model_selector.blockSignals(False)
        finally:
            model_selector.setEnabled(True)
            update_data_controls()

    def prediction_output():
        output = bytes(prediction_runner.process.readAllStandardOutput()).decode("utf-8", errors="replace").strip()
        if output:
            data_status.setText(output.splitlines()[-1][:500])

    def prediction_finished(exit_code, _exit_status):
        nonlocal prediction_dataset_id, prediction_model_id, prediction_cancelled
        dataset_id = prediction_dataset_id
        model_id = prediction_model_id
        prediction_dataset_id = None
        prediction_model_id = None
        stop_prediction.setEnabled(False)
        model_selector.setEnabled(True)
        if prediction_cancelled:
            prediction_cancelled = False
            data_status.setText("计算已停止；没有把未完成结果载入当前页面。")
            update_data_controls()
            return
        if (dataset_id, str(model_id or "").lower()) != current_context():
            data_status.setText("后台计算已完成，但当前井段或模型已切换；结果未自动载入。")
            update_data_controls()
            return
        if exit_code != 0:
            data_status.setText(f"{model_id.upper() if model_id else '当前模型'} 预测失败（返回码 {exit_code}）")
            update_data_controls()
            return
        if dataset_id and hasattr(registry, "refresh_catalog"):
            registry.refresh_catalog()
        try:
            if dataset_id and hasattr(controller, "set_dataset"):
                controller.set_dataset(dataset_id)
            data_status.setText(f"{model_id.upper() if model_id else '当前模型'} 预测完成，结果已写入井段缓存。")
            status.setText("预测缓存已加载，可继续播放、审核和导出。")
        except Exception as exc:
            data_status.setText(f"预测完成但回放加载失败：{type(exc).__name__}: {exc}")
        update_data_controls()

    last_frames_token = id(controller.frames)

    def update(frame):
        nonlocal last_frames_token, review_snapshot
        if id(controller.frames) != last_frames_token:
            refresh_visuals()
            last_frames_token = id(controller.frames)
        for chart in (pressure_chart, flow_chart, sand_chart):
            chart.set_index(controller.index)
        risk_timeline.set_index(controller.index)
        active_model = (frame or {}).get("hmi_model_id")
        if active_model:
            active_index = model_selector.findData(active_model)
            if active_index >= 0 and model_selector.currentIndex() != active_index:
                model_selector.blockSignals(True)
                model_selector.setCurrentIndex(active_index)
                model_selector.blockSignals(False)
            active_meta = next((item for item in agent_models if item.get("model_id") == active_model), None)
            if active_meta:
                model_status.setText(str(active_meta.get("status", "可用")))
        flow, sand, action = recommendation(frame)
        available = flow is not None or sand is not None
        for button in (accept, reject, modify):
            button.setEnabled(available)
        if flow is not None:
            flow_value.setValue(flow)
        if sand is not None:
            sand_value.setValue(sand)
        if available:
            reason.setText(str(action or "调整控制量"))
            status.setText(
                f"{_seconds((frame or {}).get('time_s'))} · 当前/建议排量 {_value((frame or {}).get('current_flow'))}/{_value(flow)} m³/min · "
                f"当前/建议砂比 {_value((frame or {}).get('current_sand'))}/{_value(sand)} %"
            )
        else:
            reason.setText("暂无建议")
            status.setText(f"{_seconds((frame or {}).get('time_s'))} · 暂无建议")
        review_snapshot = {
            "context": current_context(),
            "frame": frame or {},
            "original": controller.current,
            "frames": controller.frames,
            "mode": "历史/缓存回放",
        }

    accept.clicked.connect(lambda: record("确认采用"))
    reject.clicked.connect(lambda: record("暂不采用"))
    modify.clicked.connect(lambda: record("修改后采用"))
    rollback.clicked.connect(append_rollback)
    export.clicked.connect(export_log)
    model_selector.currentIndexChanged.connect(switch_agent_model)
    predict_data.clicked.connect(start_realtime_prediction)
    stop_prediction.clicked.connect(cancel_realtime_prediction)
    prediction_runner.process.readyReadStandardOutput.connect(prediction_output)
    prediction_runner.process.finished.connect(prediction_finished)
    risk_timeline.intervalSelected.connect(
        lambda interval: risk_detail.setText(
            "当前暂无可用风险判断"
            if not interval
            else (
                f"风险区间：{interval.get('start_s', 0):.0f}–{interval.get('end_s', 0):.0f} s　|　"
                f"状态：{_risk_name(interval.get('level'))}　|　工况：{interval.get('condition', '未标注')}　|　"
                f"触发原因：{interval.get('reason', '--')}　|　对建议的影响：{interval.get('recommendation', '--')}"
            )
        )
    )
    refresh_visuals()
    refresh_history()
    controller.frameChanged.connect(update)
    update(controller.current or {})
    update_data_controls()
    def set_global_dataset(_dataset_id):
        if prediction_runner.process.state() != QProcess.NotRunning:
            cancel_realtime_prediction()
        refresh_visuals()
        refresh_history()
        update_data_controls()
        update(controller.current or {})

    page.set_global_dataset = set_global_dataset
    page.pause_playback = timeline.pause
    page.prepare_dataset_change = cancel_realtime_prediction
    page._hmi_prediction_runner = prediction_runner
    return page


def _pressure_series(frames):
    candidates = [
        ("PKN先验", [_frame_value(f, "prior_bhp", "dt", "prior_bottomhole_pressure_mpa") for f in frames], PALETTE["orange"]),
        ("观测压力", [_frame_value(f, "observed_bhp", "dt", "observed_bottomhole_pressure_mpa") for f in frames], PALETTE["blue"]),
        ("EnKF后验", [_frame_value(f, "posterior_bhp", "dt", "bottomhole_pressure_mpa") for f in frames], PALETTE["cyan"]),
    ]
    return [item for item in candidates if any(_number(value) is not None for value in item[1])]


def _set_action_series(chart, frames, current_key, recommended_key, current_color, recommended_color):
    series = []
    current = [frame.get(current_key) for frame in frames]
    recommended = [frame.get(recommended_key) for frame in frames]
    if any(_number(value) is not None for value in current):
        series.append(("当前值", current, current_color))
    if any(_number(value) is not None for value in recommended):
        series.append(("建议值", recommended, recommended_color))
    chart.set_series(series)


def _set_time_range(chart, frames):
    if frames:
        start, end = _number(frames[0].get("time_s")), _number(frames[-1].get("time_s"))
        if start is not None and end is not None:
            chart.set_time_range(start, end)


def _frame_value(frame, flat_key, *nested_path):
    value = frame.get(flat_key)
    if _number(value) is not None:
        return value
    current = frame
    for key in nested_path:
        if not isinstance(current, dict):
            return float("nan")
        current = current.get(key)
    return current if _number(current) is not None else float("nan")


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _value(value):
    numeric = _number(value)
    return "--" if numeric is None else f"{numeric:.2f}"


def _seconds(value):
    numeric = _number(value)
    return "t=-- s" if numeric is None else f"t={numeric:.0f} s"


def _risk_name(value):
    return {"normal": "正常", "attention": "关注", "high": "高风险"}.get(str(value), "--")


def _label(text, name=None):
    from PySide6.QtWidgets import QLabel
    label = QLabel(text)
    if name:
        label.setObjectName(name)
    label.setWordWrap(True)
    return label


__all__ = ["build_hmi_page"]
