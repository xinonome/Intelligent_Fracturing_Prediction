"""Operational cross-well transfer workbench for the first APP section."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from ...core.paths import PATHS
from ...services.task_runner import TaskRunner
from ...services.ml_runtime import resolve_ml_python
from ..theme import PALETTE
from .chart_panel import build_chart
from .status_card import Panel


def _well_records(rows: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for row in rows:
        well = str(row.get("well_id") or Path(str(row.get("source_file") or "未知井")).stem).strip()
        if well:
            result.setdefault(well, []).append(dict(row))
    return result


def _compatible_models() -> list[tuple[str, Path, Path]]:
    root = PATHS.app_outputs / "transfer_runs"
    models = []
    if not root.exists():
        return models
    for package in sorted(root.rglob("transfer_package.joblib"), reverse=True):
        if package.parent.name.lower().startswith("smoke"):
            continue
        model = package.with_name("finetuned_model.pt")
        if model.exists():
            models.append((package.parent.name, model, package))
    return models


def create_fsl_transfer_workbench(registry, timeline_rows: list[dict]):
    from PySide6.QtCore import QProcess, Qt
    from PySide6.QtWidgets import (
        QComboBox,
        QFileDialog,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QPlainTextEdit,
        QPushButton,
        QSizePolicy,
        QSpinBox,
        QWidget,
    )

    panel, layout = Panel.create("跨井迁移工作台")
    layout.setContentsMargins(10, 8, 10, 10)
    layout.setSpacing(6)
    records = _well_records(timeline_rows)
    wells = sorted(records)

    form = QGridLayout()
    form.setContentsMargins(0, 0, 0, 0)
    form.setHorizontalSpacing(10)
    form.setVerticalSpacing(3)
    source_box = QComboBox()
    target_box = QComboBox()
    for well in wells:
        source_box.addItem(well, well)
        target_box.addItem(well, well)
    if target_box.count() > 1:
        target_box.setCurrentIndex(1)
    mode_box = QComboBox()
    mode_box.addItem("重新训练模型", "train")
    mode_box.addItem("使用已有模型", "apply")
    model_box = QComboBox()
    epoch_box = QSpinBox()
    epoch_box.setRange(1, 50)
    epoch_box.setValue(8)
    finetune_box = QSpinBox()
    finetune_box.setRange(1, 30)
    finetune_box.setValue(5)
    for column, (label, widget) in enumerate(
        (
            ("迁移井", source_box),
            ("目标井", target_box),
            ("模型方式", mode_box),
            ("已有模型", model_box),
            ("基础训练轮次", epoch_box),
            ("目标井微调轮次", finetune_box),
        )
    ):
        title = QLabel(label)
        title.setObjectName("key")
        form.addWidget(title, 0, column)
        form.addWidget(widget, 1, column)
    form.setColumnStretch(0, 2)
    form.setColumnStretch(1, 2)
    form.setColumnStretch(2, 1)
    form.setColumnStretch(3, 2)
    # Keep the two form rows at their size hint.  A bare grid layout can
    # absorb the tab's spare height and make the labels/combos look widely
    # separated on a large monitor.
    form_host = QWidget()
    form_host.setLayout(form)
    form_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    layout.addWidget(form_host)

    availability = QLabel()
    availability.setObjectName("muted")
    availability.setWordWrap(True)
    availability.setMaximumHeight(34)
    layout.addWidget(availability)

    actions = QHBoxLayout()
    actions.setContentsMargins(0, 0, 0, 0)
    actions.setSpacing(6)
    train_button = QPushButton("开始训练")
    apply_button = QPushButton("应用已有模型")
    locate_button = QPushButton("查看预测曲线")
    save_button = QPushButton("保存迁移模型")
    export_button = QPushButton("导出迁移结果")
    cancel_button = QPushButton("停止任务")
    for button in (train_button, apply_button, locate_button, save_button, export_button, cancel_button):
        actions.addWidget(button)
    actions.addStretch(1)
    layout.addLayout(actions)

    status = QLabel("待执行")
    status.setObjectName("notice")
    status.setWordWrap(True)
    status.setMaximumHeight(52)
    layout.addWidget(status)

    console = QPlainTextEdit()
    console.setObjectName("taskConsole")
    console.setReadOnly(True)
    console.setPlaceholderText("训练或应用模型后，子进程的实时输出会显示在这里。")
    console.setLineWrapMode(QPlainTextEdit.NoWrap)
    console.setMinimumHeight(108)
    console.setMaximumHeight(170)
    console.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    layout.addWidget(console)

    chart = build_chart("目标井工况轨迹 · 实际 / 迁移前 / 迁移后", 285, y_min=0.0)
    layout.addWidget(chart)

    runner = TaskRunner(panel)
    current_result: Path | None = None
    current_model: Path | None = None
    current_package: Path | None = None
    active_run_root: Path | None = None

    def refresh_models(select_dir: Path | None = None):
        model_box.blockSignals(True)
        model_box.clear()
        models = _compatible_models()
        if not models:
            model_box.addItem("暂无可直接应用的模型", None)
            model_box.model().item(0).setEnabled(False)
        for label, model, package in models:
            model_box.addItem(f"{label} · {model.name}", (str(model), str(package)))
            if select_dir and model.parent == select_dir:
                model_box.setCurrentIndex(model_box.count() - 1)
        model_box.blockSignals(False)

    def update_availability():
        source = str(source_box.currentData() or "")
        target = str(target_box.currentData() or "")
        source_rows = records.get(source, [])
        target_rows = records.get(target, [])
        source_points = sum(int(row.get("sample_count") or 0) for row in source_rows)
        target_points = sum(int(row.get("sample_count") or 0) for row in target_rows)
        source_ready = source_points >= 36
        target_ready = target_points >= 36
        same = bool(source and source == target)
        availability.setText(
            f"迁移井：{len(source_rows)} 个井段 / {source_points} 个有效点；"
            f"目标井：{len(target_rows)} 个井段 / {target_points} 个有效点。"
            + ("迁移井与目标井不能相同。" if same else "")
        )
        busy = runner.process.state() != QProcess.NotRunning
        train_button.setEnabled(bool(source_ready and target_ready and not same and not busy and mode_box.currentData() == "train"))
        model_data = model_box.currentData()
        apply_button.setEnabled(bool(model_data and target_ready and not same and not busy and mode_box.currentData() == "apply"))
        cancel_button.setEnabled(busy)
        epoch_box.setEnabled(mode_box.currentData() == "train" and not busy)
        finetune_box.setEnabled(mode_box.currentData() == "train" and not busy)
        model_box.setEnabled(mode_box.currentData() == "apply" and not busy)
        for selector in (source_box, target_box, mode_box):
            selector.setEnabled(not busy)
        locate_button.setEnabled(current_result is not None)
        save_button.setEnabled(current_model is not None and current_model.exists())
        export_button.setEnabled(current_result is not None and current_result.exists())

    def load_result(result_path: Path):
        nonlocal current_result, current_model, current_package
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise ValueError(payload.get("error") or "迁移任务失败")
        traces = payload.get("traces", {}) or {}
        series = []
        observed = traces.get("observed", [])
        before = traces.get("before", [])
        after = traces.get("after", [])
        if observed:
            series.append(("目标井实际工况", observed, PALETTE["blue"]))
        if before:
            series.append(("迁移前识别", before, PALETTE["orange"]))
        if after:
            series.append(("迁移后识别", after, PALETTE["cyan"]))
        chart.set_series(series)
        windows = traces.get("window_end", [])
        if windows:
            chart.set_time_range(float(windows[0]), float(windows[-1]))
        current_result = result_path
        model_value = payload.get("model_path")
        package_value = payload.get("package_path")
        current_model = Path(model_value) if model_value else None
        current_package = Path(package_value) if package_value else None
        mode = "训练完成" if payload.get("mode") == "train" else "模型应用完成"
        status.setText(
            f"{mode}：{payload.get('source_well', '--')} → {payload.get('target_well', '--')}；"
            f"结果已写入 {result_path.parent}。"
        )
        refresh_models(result_path.parent)
        update_availability()

    def start_job(mode: str):
        nonlocal current_result, current_model, current_package, active_run_root
        source = str(source_box.currentData() or "")
        target = str(target_box.currentData() or "")
        if not source or not target or source == target:
            status.setText("请选择两个不同且数据可用的井。")
            return
        ml_python, runtime_note = resolve_ml_python()
        if not ml_python:
            status.setText(f"训练环境不可用：{runtime_note}")
            return
        run_root = PATHS.app_outputs / "transfer_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root.mkdir(parents=True, exist_ok=True)
        active_run_root = run_root
        arguments = [
            "-m",
            "App.services.fsl_transfer_service",
            "--mode",
            mode,
            "--source-well",
            source,
            "--target-well",
            target,
            "--output-dir",
            str(run_root),
            "--pretrain-epochs",
            str(epoch_box.value()),
            "--finetune-epochs",
            str(finetune_box.value()),
        ]
        if mode == "apply":
            model_data = model_box.currentData()
            if not model_data:
                status.setText("没有可直接应用的模型。请先完成一次训练。")
                return
            model, package = model_data
            arguments.extend(["--model", model, "--package", package])
        current_result = None
        current_model = None
        current_package = None
        chart.set_series([])
        console.clear()
        console.appendPlainText(
            f"[启动] {'训练迁移模型' if mode == 'train' else '应用已有模型'}："
            f"{source} → {target}"
        )
        status.setText("任务运行中 · 正在等待子进程输出…")
        runner.start(ml_python, arguments, PATHS.root, environment={"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
        update_availability()

    def finish_job(exit_code, _exit_status):
        process_output()
        result_path = active_run_root / "transfer_result.json" if active_run_root else None
        if result_path is None or not result_path.exists():
            console.appendPlainText(f"[结束] 返回码 {exit_code}，未生成结果文件。")
            status.setText(f"任务结束（返回码 {exit_code}），但没有生成结果文件。")
            update_availability()
            return
        try:
            load_result(result_path)
            console.appendPlainText(f"[完成] 结果文件：{result_path}")
        except Exception as exc:
            console.appendPlainText(f"[失败] {type(exc).__name__}: {exc}")
            status.setText(f"任务失败：{exc}")
            update_availability()

    def process_output():
        text = bytes(runner.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not text.strip():
            return
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if not lines:
            return
        console.appendPlainText("\n".join(lines))
        console.verticalScrollBar().setValue(console.verticalScrollBar().maximum())
        status.setText(f"任务运行中 · {lines[-1][:180]}")

    def locate_chart():
        chart.setFocus(Qt.OtherFocusReason)
        chart.update()

    def save_model():
        if current_model is None or not current_model.exists():
            return
        destination, _ = QFileDialog.getSaveFileName(panel, "保存迁移模型", current_model.name, "PyTorch 模型 (*.pt)")
        if destination:
            shutil.copy2(current_model, destination)
            if current_package and current_package.exists():
                shutil.copy2(current_package, str(Path(destination).with_name("transfer_package.joblib")))
            status.setText(f"模型及配套预处理参数已保存到 {Path(destination).parent}。")

    def export_result():
        if current_result is None or not current_result.exists():
            return
        destination, _ = QFileDialog.getSaveFileName(panel, "导出迁移结果", "跨井迁移结果.json", "JSON (*.json)")
        if destination:
            shutil.copy2(current_result, destination)
            status.setText(f"迁移结果已导出：{destination}")

    refresh_models()
    for combo in (source_box, target_box, mode_box, model_box):
        combo.currentIndexChanged.connect(update_availability)
    train_button.clicked.connect(lambda: start_job("train"))
    apply_button.clicked.connect(lambda: start_job("apply"))
    cancel_button.clicked.connect(runner.cancel)
    locate_button.clicked.connect(locate_chart)
    save_button.clicked.connect(save_model)
    export_button.clicked.connect(export_result)
    runner.process.readyReadStandardOutput.connect(process_output)
    runner.process.finished.connect(finish_job)
    runner.process.errorOccurred.connect(lambda _error: status.setText(f"任务进程未能启动：{runner.process.errorString()}"))
    update_availability()

    panel._transfer_runner = runner
    panel._transfer_chart = chart
    panel._transfer_console = console
    return panel


__all__ = ["create_fsl_transfer_workbench"]
