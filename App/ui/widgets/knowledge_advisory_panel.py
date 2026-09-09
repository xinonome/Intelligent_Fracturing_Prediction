"""API-gated knowledge-enhanced high-level advisory for the HMI page."""

from __future__ import annotations

import math

from ...services.knowledge_graph_service import (
    append_knowledge_advisory,
    load_api_config,
    load_saved_knowledge_sources,
    request_api_advisory,
)
from .status_card import Panel


def build_knowledge_advisory_panel(registry):
    from PySide6.QtCore import QThread, QTimer, Signal
    from PySide6.QtWidgets import (
        QCheckBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSpinBox,
        QTextBrowser,
    )

    panel, layout = Panel.create("知识增强决策")
    panel.setObjectName("knowledgeAdvisoryPanel")
    state_label = QLabel("等待当前施工帧")
    state_label.setObjectName("knowledgeAdvisoryState")
    state_label.setWordWrap(True)
    layout.addWidget(state_label)

    controls = QHBoxLayout()
    run_button = QPushButton("根据当前工况生成建议")
    run_button.setObjectName("knowledgeAdvisoryRun")
    auto_check = QCheckBox("周期研判")
    auto_check.setObjectName("knowledgeAdvisoryAuto")
    interval = QSpinBox()
    interval.setObjectName("knowledgeAdvisoryInterval")
    interval.setRange(1, 30)
    interval.setValue(5)
    interval.setSuffix(" min")
    status = QLabel("")
    status.setObjectName("muted")
    status.setWordWrap(True)
    controls.addWidget(run_button)
    controls.addWidget(auto_check)
    controls.addWidget(interval)
    controls.addWidget(status, 1)
    layout.addLayout(controls)

    result = QTextBrowser()
    result.setObjectName("knowledgeAdvisoryResult")
    result.setPlaceholderText("建议作为人工审核依据")
    result.setMinimumHeight(170)
    layout.addWidget(result)

    current_frame: dict = {}
    pending_state: dict | None = None
    worker = None
    last_bucket: tuple[str, int] | None = None

    class AdvisoryWorker(QThread):
        result_ready = Signal(str)
        error = Signal(str)

        def __init__(self, state: dict):
            super().__init__(panel)
            self.state = dict(state)
            self.config = load_api_config()

        def run(self):
            try:
                sources = load_saved_knowledge_sources()
                self.result_ready.emit(request_api_advisory(self.state, self.config, sources))
            except Exception as exc:
                self.error.emit(str(exc))

    def number(value):
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) else None
        except (TypeError, ValueError):
            return None

    def first_number(*values):
        for value in values:
            parsed = number(value)
            if parsed is not None:
                return parsed
        return None

    def frame_state(frame: dict) -> dict:
        fsl = frame.get("fsl") if isinstance(frame.get("fsl"), dict) else {}
        hmi = frame.get("hmi") if isinstance(frame.get("hmi"), dict) else {}
        dt = frame.get("dt") if isinstance(frame.get("dt"), dict) else {}
        decision = frame.get("decision") if isinstance(frame.get("decision"), dict) else {}
        condition = "未标注"
        for candidate in (fsl.get("working_type"), frame.get("working_type"), decision.get("main_risk")):
            text = str(candidate or "").strip()
            if text and text.lower() not in {"unknown", "none"} and text not in {"未标注", "\\"}:
                condition = text
                break
        return {
            "dataset_id": str(getattr(registry, "dataset_id", "") or ""),
            "time_s": number(frame.get("time_s")),
            "condition": condition,
            "risk_level": str(hmi.get("risk_level") or decision.get("risk_level") or ""),
            "risk_reason": str(decision.get("reason") or decision.get("main_risk") or ""),
            "surface_pressure_mpa": first_number(dt.get("surface_pressure_mpa"), frame.get("surface_pressure")),
            "bottomhole_pressure_mpa": first_number(dt.get("bottomhole_pressure_mpa"), frame.get("posterior_bhp")),
            "current_flow_m3_min": first_number(hmi.get("current_flow_m3_min"), frame.get("current_flow")),
            "current_sand_ratio_percent": first_number(
                hmi.get("current_sand_ratio_percent"), frame.get("current_sand")
            ),
            "existing_high_level_option": str(
                hmi.get("high_level_action") or frame.get("hmi_option") or ""
            ),
            "data_mode": "历史/缓存顺序回放",
        }

    def refresh_capability() -> bool:
        configured = bool(load_api_config().endpoint)
        panel.setVisible(configured)
        if not configured:
            auto_check.setChecked(False)
        return configured

    def finish_success(advisory: str) -> None:
        nonlocal pending_state
        result.setPlainText(advisory)
        config = load_api_config()
        try:
            saved = append_knowledge_advisory(
                pending_state or {}, advisory, model=config.model
            )
            status.setText(f"建议已记录 · {saved['record_id'][:8]} · 仅供人工审核")
        except OSError as exc:
            status.setText(f"建议已生成，但记录保存失败：{exc}")
        pending_state = None

    def finish_error(message: str) -> None:
        nonlocal pending_state
        result.setPlainText(f"知识增强建议生成失败：{message}")
        status.setText("当前未生成知识增强建议；智能体回放继续使用现有结果")
        pending_state = None

    def worker_finished() -> None:
        nonlocal worker
        run_button.setEnabled(True)
        worker = None

    def run_advisory() -> None:
        nonlocal worker, pending_state
        if worker is not None and worker.isRunning():
            return
        if not refresh_capability():
            return
        pending_state = frame_state(current_frame)
        run_button.setEnabled(False)
        status.setText("正在按当前工况检索知识图谱并调用 API…")
        worker = AdvisoryWorker(pending_state)
        worker.result_ready.connect(finish_success)
        worker.error.connect(finish_error)
        worker.finished.connect(worker_finished)
        panel._knowledge_advisory_worker = worker
        worker.start()

    def update_frame(frame: dict | None) -> None:
        nonlocal current_frame, last_bucket
        current_frame = dict(frame or {})
        state = frame_state(current_frame)
        pieces = [f"当前工况：{state['condition']}"]
        if state["risk_level"]:
            pieces.append(f"风险：{state['risk_level']}")
        if state["time_s"] is not None:
            pieces.append(f"t={state['time_s']:.0f} s")
        state_label.setText("　|　".join(pieces))
        if not (panel.isVisible() and auto_check.isChecked() and state["time_s"] is not None):
            return
        seconds = interval.value() * 60
        bucket = (state["dataset_id"], int(state["time_s"] // max(seconds, 1)))
        if bucket != last_bucket:
            last_bucket = bucket
            run_advisory()

    capability_timer = QTimer(panel)
    capability_timer.setInterval(2000)
    capability_timer.timeout.connect(refresh_capability)
    capability_timer.start()
    run_button.clicked.connect(run_advisory)
    auto_check.toggled.connect(lambda _checked: update_frame(current_frame))
    interval.valueChanged.connect(lambda _value: update_frame(current_frame))
    refresh_capability()

    panel.update_frame = update_frame
    panel.refresh_api_capability = refresh_capability
    panel._knowledge_capability_timer = capability_timer
    return panel


__all__ = ["build_knowledge_advisory_panel"]
