"""Native PyFrac playback, parameter editing and isolated rerun controls."""

from __future__ import annotations

import json
import math
from pathlib import Path

from ...data.pyfrac_runtime_loader import PyFracRuntime, load_pyfrac_runtime
from ...services.pyfrac_run_service import (
    prepare_native_run, save_parameter_scheme, native_context_reason, native_run_reason,
    load_context_runtime, native_run_completed,
)
from ...services.task_runner import TaskRunner
from ..theme import PALETTE
from .native_time_chart import NativeTimeChart as build_chart
from .status_card import Panel


def create_pyfrac_workbench(parent=None, dataset=None):
    from PySide6.QtCore import QPointF, QTimer, Qt
    from PySide6.QtGui import QColor, QPainter, QPen
    from PySide6.QtWidgets import (
        QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
        QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget, QTabWidget,
        QSplitter, QPlainTextEdit,
    )

    class FieldView(QFrame):
        def __init__(self):
            super().__init__()
            self.setObjectName("chartPanel")
            self.setMinimumHeight(180)
            self.frame = None

        def set_frame(self, frame):
            self.frame = frame
            self.update()

        def paintEvent(self, _event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor(PALETTE["chart_bg"]))
            painter.setPen(QColor(PALETTE["text"]))
            painter.drawText(14, 24, "PyFrac内生压力场与裂缝前缘")
            if self.frame is None:
                painter.setPen(QColor(PALETTE["muted"]))
                painter.drawText(14, 52, "暂无原生内部状态")
                return
            field = self.frame.pressure_field
            front = self.frame.front_geometry
            points = [(item["x_m"], item["y_m"]) for item in field] + [tuple(item[:2]) for item in front]
            if not points:
                painter.setPen(QColor(PALETTE["muted"]))
                painter.drawText(14, 52, "当前内部点没有场与前缘数据")
                return
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            x_span = max(max(xs) - min(xs), 1.0)
            y_span = max(max(ys) - min(ys), 1.0)
            plot = self.rect().adjusted(42, 42, -18, -28)
            def xy(x, y):
                return QPointF(plot.left() + (x - min(xs)) / x_span * plot.width(), plot.bottom() - (y - min(ys)) / y_span * plot.height())
            pressures = [item["net_pressure_mpa"] for item in field]
            p_min, p_max = (min(pressures), max(pressures)) if pressures else (0.0, 1.0)
            for item in field:
                ratio = (item["net_pressure_mpa"] - p_min) / max(p_max - p_min, 1e-9)
                color = QColor.fromRgbF(0.15 + 0.75 * ratio, 0.55 - 0.30 * ratio, 0.95 - 0.70 * ratio)
                painter.setPen(Qt.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(xy(item["x_m"], item["y_m"]), 3.0, 3.0)
            if front:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(PALETTE["orange"]), 2))
                centre_x = sum(float(item[0]) for item in front) / len(front)
                centre_y = sum(float(item[1]) for item in front) / len(front)
                ordered_front = sorted(
                    front,
                    key=lambda item: math.atan2(float(item[1]) - centre_y, float(item[0]) - centre_x),
                )
                front_points = [xy(float(item[0]), float(item[1])) for item in ordered_front]
                for first, second in zip(front_points, front_points[1:] + front_points[:1]):
                    painter.drawLine(first, second)

    root = QWidget(parent)
    root_layout = QVBoxLayout(root)
    root_layout.setContentsMargins(0, 0, 0, 0)
    root_layout.setSpacing(10)

    context = dict(dataset or {})
    context_label = QLabel()
    context_label.setWordWrap(True)
    root_layout.addWidget(context_label)
    tabs = QTabWidget()
    tabs.setObjectName("pyfracTabs")
    root_layout.addWidget(tabs, 1)
    evolution_page = QWidget()
    evolution_layout = QVBoxLayout(evolution_page)
    parameter_page = QWidget()
    parameter_layout = QVBoxLayout(parameter_page)
    history_page = QWidget()
    history_layout = QVBoxLayout(history_page)
    tabs.addTab(evolution_page, "压力与演化")
    tabs.addTab(parameter_page, "参数全流程")
    tabs.addTab(history_page, "内部步与体积")
    controls, controls_layout = Panel.create("PyFrac 原生推演")
    first_row = QHBoxLayout()
    sigma = QDoubleSpinBox()
    sigma.setObjectName("pyfracSigmaMin")
    sigma.setRange(20.0, 120.0)
    sigma.setDecimals(2)
    sigma.setValue(112.5)
    sigma.setSuffix(" MPa")
    target = QDoubleSpinBox()
    target.setObjectName("pyfracTargetTime")
    target.setRange(2.0, 4435.0)
    target.setDecimals(0)
    target.setValue(4435.0)
    target.setSuffix(" s")
    start = QPushButton("开始重新推演")
    start.setObjectName("pyfracStart")
    stop = QPushButton("停止计算")
    stop.setObjectName("pyfracStop")
    stop.setEnabled(False)
    save = QPushButton("保存参数方案")
    rollback = QPushButton("回退参数方案")
    export_frame = QPushButton("导出当前帧")
    export_trace = QPushButton("导出完整轨迹")
    for widget in (QLabel("最小水平应力"), sigma, QLabel("目标模拟时间"), target):
        first_row.addWidget(widget)
    first_row.addStretch(1)
    controls_layout.addLayout(first_row)
    scheme_row = QHBoxLayout()
    for widget in (start, stop, save, rollback):
        scheme_row.addWidget(widget)
    scheme_row.addStretch(1)
    controls_layout.addLayout(scheme_row)
    start.setToolTip("从初始状态重新执行 PyFrac 推演")
    run_status = QLabel()
    run_status.setObjectName("notice")
    run_status.setWordWrap(True)
    controls_layout.addWidget(run_status)
    parameter_layout.addWidget(controls)
    parameter_note = QLabel("可调参数：最小水平应力、目标模拟时间")
    parameter_note.setWordWrap(True)
    parameter_layout.addWidget(parameter_note)
    parameter_details = QPlainTextEdit()
    parameter_details.setReadOnly(True)
    parameter_details.setObjectName("pyfracParameterDetails")
    parameter_layout.addWidget(parameter_details, 1)

    playback, playback_layout = Panel.create("内部计算点回放")
    row = QHBoxLayout()
    play = QPushButton("播放")
    reset = QPushButton("回到初始点")
    back = QPushButton("上一个内部点")
    forward = QPushButton("下一个内部点")
    slider = QSlider(Qt.Horizontal)
    jump = QDoubleSpinBox()
    jump.setRange(0.0, 4435.0)
    jump.setSuffix(" s")
    jump_button = QPushButton("跳转")
    frame_label = QLabel("")
    for widget in (play, reset, back, forward):
        row.addWidget(widget)
    row.addWidget(slider, 1)
    row.addWidget(frame_label)
    playback_layout.addLayout(row)
    jump_row = QHBoxLayout()
    jump_row.addWidget(QLabel("跳转到模拟时间"))
    jump_row.addWidget(jump)
    jump_row.addWidget(jump_button)
    jump_row.addStretch(1)
    playback_layout.addLayout(jump_row)
    root_layout.insertWidget(1, playback)
    jump_row.addWidget(export_frame)
    jump_row.addWidget(export_trace)

    time_step_chart = build_chart("PyFrac真实自适应时间步 / s", 230, y_min=0.0)
    volume_chart = build_chart("体积历史 / m³ · 注入 / 裂缝 / 滤失", 230, y_min=0.0)
    length_chart = build_chart("裂缝半缝长 / m", 230, y_min=0.0)
    width_chart = build_chart("裂缝最大缝宽 / mm", 230, y_min=0.0)
    pressure_chart = build_chart("净压力 / MPa · 内部场均值 / 最大值", 230)
    field_view = FieldView()
    evolution_split = QSplitter(Qt.Horizontal)
    curves = QWidget()
    curves_layout = QVBoxLayout(curves)
    curves_layout.setContentsMargins(0, 0, 0, 0)
    curves_layout.addWidget(pressure_chart)
    geometry_tabs = QTabWidget()
    geometry_tabs.addTab(length_chart, "半缝长")
    geometry_tabs.addTab(width_chart, "最大缝宽")
    curves_layout.addWidget(geometry_tabs)
    evolution_split.addWidget(curves)
    evolution_split.addWidget(field_view)
    evolution_split.setSizes([500, 500])
    evolution_layout.addWidget(evolution_split, 1)
    history_split = QSplitter(Qt.Vertical)
    history_split.addWidget(time_step_chart)
    history_split.addWidget(volume_chart)
    history_layout.addWidget(history_split, 1)
    # Minimums must fit a normal application viewport, not force a long page.
    for chart in (time_step_chart, volume_chart, length_chart, width_chart, pressure_chart):
        chart.setMinimumHeight(150)

    detail = QLabel()
    detail.setObjectName("notice")
    detail.setWordWrap(True)
    root_layout.addWidget(detail)

    runtime = load_context_runtime(context)
    sigma.setValue(float(runtime.parameters.get("sigma_min_mpa", 112.5)))
    previous_scheme = (sigma.value(), target.value())
    last_completed_scheme = previous_scheme
    previous_runtime: PyFracRuntime | None = None
    timer = QTimer(root)
    timer.setInterval(180)
    progress_timer = QTimer(root)
    progress_timer.setInterval(1000)
    runner = TaskRunner(root)
    active_root: Path | None = None
    active_context = None
    active_scheme = None
    running = False
    stopped = False

    def finite(value):
        return value is not None and math.isfinite(float(value))

    def values_for(value: PyFracRuntime, name):
        return [getattr(frame, name) if finite(getattr(frame, name)) else None for frame in value.frames]

    def compared_series(name, current_label, current_color):
        series = [(current_label, values_for(runtime, name), current_color)]
        if previous_runtime is not None and previous_runtime.frames:
            series.insert(0, (f"上次·{current_label}", values_for(previous_runtime, name), PALETTE["muted"], [frame.time_s for frame in previous_runtime.frames]))
        return series

    def refresh_runtime(value: PyFracRuntime):
        nonlocal runtime
        runtime = value
        root._pyfrac_runtime = value
        slider.setRange(0, max(runtime.point_count - 1, 0))
        jump.setMaximum(max(runtime.target_time_s or 4435.0, 1.0))
        times = [frame.time_s for frame in runtime.frames]
        time_step_chart.set_series(compared_series("time_step_s", "内部Δt", PALETTE["blue"]))
        volume_chart.set_series([
            ("累计注入", values_for(runtime, "injected_volume_m3"), PALETTE["blue"]),
            ("裂缝体积", values_for(runtime, "fracture_volume_m3"), PALETTE["cyan"]),
            ("累计滤失", values_for(runtime, "leakoff_volume_m3"), PALETTE["orange"]),
        ])
        length_chart.set_series(compared_series("half_length_m", "本次半缝长", PALETTE["cyan"]))
        width_chart.set_series(compared_series("max_aperture_mm", "本次最大缝宽", PALETTE["orange"]))
        pressure_series = [
            ("场均值", values_for(runtime, "mean_net_pressure_mpa"), PALETTE["blue"]),
            ("场最大值", values_for(runtime, "max_net_pressure_mpa"), PALETTE["red"]),
        ]
        if previous_runtime is not None and previous_runtime.frames:
            pressure_series.insert(0, ("上次场均值", values_for(previous_runtime, "mean_net_pressure_mpa"), PALETTE["muted"], [frame.time_s for frame in previous_runtime.frames]))
        pressure_chart.set_series(pressure_series)
        for chart in (time_step_chart, volume_chart, length_chart, width_chart, pressure_chart):
            chart.set_time_points(times)
        parameter_details.setPlainText(json.dumps({
            "dataset_id": context.get("dataset_id"), "run_root": str(value.run_root or ""),
            "actual_run_parameters": value.parameters,
            "result_status": {key: value.result.get(key) for key in ("success", "target_reached", "engine_mode", "final_time_s")},
        }, ensure_ascii=False, indent=2))
        run_status.setText(runtime.note or "暂无可回放的PyFrac原生运行")
        set_index(max(runtime.point_count - 1, 0))
        for button in (play, reset, back, forward, jump_button, export_frame, export_trace):
            button.setEnabled(bool(runtime.frames))

    def set_index(index):
        if not runtime.frames:
            frame_label.clear()
            detail.setText("当前没有真实原生内部计算点。")
            field_view.set_frame(None)
            return
        index = max(0, min(int(index), runtime.point_count - 1))
        slider.blockSignals(True)
        slider.setValue(index)
        slider.blockSignals(False)
        frame = runtime.frames[index]
        jump.setValue(frame.time_s)
        for chart in (time_step_chart, volume_chart, length_chart, width_chart, pressure_chart):
            chart.set_index(index)
        field_view.set_frame(frame)
        frame_label.setText(f"内部点 {index + 1}/{runtime.point_count}")
        detail.setText(
            f"内部点 {index + 1}/{runtime.point_count}　|　模拟时间 {frame.time_s:.3f} s　|　"
            f"Δt {_fmt(frame.time_step_s)} s　|　裂缝单元 {frame.crack_cells or ''}　|　"
            f"前缘单元 {frame.tip_cells or ''}"
        )

    def toggle_play():
        if timer.isActive():
            timer.stop()
            play.setText("播放")
        else:
            timer.start()
            play.setText("暂停")

    def advance():
        if slider.value() >= slider.maximum():
            timer.stop()
            play.setText("播放")
        else:
            set_index(slider.value() + 1)

    def jump_to_time():
        if runtime.frames:
            value = jump.value()
            index = min(range(runtime.point_count), key=lambda i: abs(runtime.frames[i].time_s - value))
            set_index(index)

    def start_run():
        nonlocal active_root, previous_scheme, previous_runtime, active_context, active_scheme, running, stopped
        if running:
            return
        try:
            active_root, command, environment = prepare_native_run(sigma.value(), target.value(), dataset=context)
        except Exception as exc:
            run_status.setText(f"无法启动PyFrac：{exc}")
            return
        pause()
        previous_scheme = last_completed_scheme
        previous_runtime = runtime
        active_context = dict(context)
        active_scheme = (sigma.value(), target.value())
        running, stopped = True, False
        refresh_runtime(PyFracRuntime(run_root=active_root, note="等待本次运行的真实内部点；旧轨迹仅作对照。"))
        start.setEnabled(False)
        stop.setEnabled(True)
        sigma.setEnabled(False)
        target.setEnabled(False)
        rollback.setEnabled(False)
        run_status.setText(f"后台从初始状态推演；输出目录：{active_root}")
        runner.start(command[0], command[1:], Path(__file__).resolve().parents[3], environment)
        progress_timer.start()

    def poll_progress():
        if active_root is None or active_context != context:
            return
        path = active_root / "progress.json"
        if not path.is_file():
            return
        try:
            progress = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        partial = load_pyfrac_runtime(active_root)
        if partial.frames:
            refresh_runtime(partial)
        run_status.setText(
            f"后台推演中：内部时间 {float(progress.get('time_s', 0.0)):.2f}/{active_scheme[1]:.0f} s · "
            f"成功内部步 {int(progress.get('successful_time_steps', 0))} · 输出 {active_root}"
        )

    def finished(exit_code, _status):
        nonlocal last_completed_scheme, running
        progress_timer.stop()
        running = False
        update_availability()
        stop.setEnabled(False)
        if active_root is not None and active_context == context:
            loaded = load_pyfrac_runtime(active_root)
            refresh_runtime(loaded)
            complete = exit_code == 0 and not stopped and native_run_completed(loaded, active_scheme[1])
            if complete:
                last_completed_scheme = active_scheme
            run_status.setText(("本井段推演达到目标时间。" if complete else f"本次未完成：失败、停止或未达到目标（返回码 {exit_code}）。") + f" {loaded.note}")

    def stop_run():
        nonlocal stopped
        stopped = True
        runner.cancel()
        run_status.setText("正在停止后台PyFrac进程；已完成的内部点保留在独立运行目录。")

    def save_scheme():
        destination, _ = QFileDialog.getSaveFileName(root, "保存PyFrac参数方案", "PyFrac参数方案.json", "JSON (*.json)")
        if destination:
            save_parameter_scheme(destination, {"dataset_id": context.get("dataset_id"), "well_id": context.get("well_id"), "stage_id": context.get("stage_id"), "pressure_source": context.get("pressure_source"), "sigma_min_mpa": sigma.value(), "target_time_s": target.value(), "template_parameters": runtime.parameters})
            run_status.setText(f"参数方案已保存：{destination}")

    def export_current():
        if not runtime.frames:
            return
        destination, _ = QFileDialog.getSaveFileName(root, "导出PyFrac当前帧", "PyFrac当前帧.json", "JSON (*.json)")
        if destination:
            Path(destination).write_text(json.dumps({"dataset": context, "run_root": str(runtime.run_root), "frame": runtime.frames[slider.value()].__dict__}, ensure_ascii=False, indent=2), encoding="utf-8")

    def export_all():
        if not runtime.frames:
            return
        destination, _ = QFileDialog.getSaveFileName(root, "导出PyFrac完整轨迹", "PyFrac完整轨迹.json", "JSON (*.json)")
        if destination:
            Path(destination).write_text(json.dumps({"dataset": context, "run_root": str(runtime.run_root), "parameters": runtime.parameters, "frames": [frame.__dict__ for frame in runtime.frames]}, ensure_ascii=False, indent=2), encoding="utf-8")

    def pause():
        timer.stop()
        play.setText("播放")

    def update_availability():
        reason = native_context_reason(context)
        run_reason = native_run_reason(context)
        context_label.setText(reason or "当前任务输入：JY84-Z1 · Stage 08")
        start.setEnabled(not run_reason and not running)
        start.setToolTip(run_reason or "使用本井段已验证注入历史，从初始状态重新推演。")
        for control in (sigma, target, save, rollback):
            control.setEnabled(not reason and not running)

    def set_dataset(value):
        nonlocal context, previous_runtime, previous_scheme, last_completed_scheme
        pause()
        new_context = dict(value or {})
        if new_context != context:
            if running:
                stop_run()
            context = new_context
            previous_runtime = None
            loaded = load_context_runtime(context)
            sigma.setValue(float(loaded.parameters.get("sigma_min_mpa", 112.5)))
            target.setValue(float(loaded.parameters.get("target_time_s", 4435)))
            previous_scheme = last_completed_scheme = (sigma.value(), target.value())
            refresh_runtime(loaded)
        update_availability()

    def restore_scheme():
        sigma.setValue(previous_scheme[0])
        target.setValue(previous_scheme[1])
        run_status.setText("回退方案仅作为参数草案")

    def process_error(error):
        nonlocal running
        from PySide6.QtCore import QProcess
        if error == QProcess.FailedToStart:
            progress_timer.stop()
            running = False
            stop.setEnabled(False)
            update_availability()
            run_status.setText(f"原生进程未启动：{runner.process.errorString()}")

    slider.valueChanged.connect(set_index)
    play.clicked.connect(toggle_play)
    reset.clicked.connect(lambda: set_index(0))
    back.clicked.connect(lambda: set_index(slider.value() - 1))
    forward.clicked.connect(lambda: set_index(slider.value() + 1))
    jump_button.clicked.connect(jump_to_time)
    timer.timeout.connect(advance)
    start.clicked.connect(start_run)
    stop.clicked.connect(stop_run)
    save.clicked.connect(save_scheme)
    rollback.clicked.connect(restore_scheme)
    export_frame.clicked.connect(export_current)
    export_trace.clicked.connect(export_all)
    runner.process.finished.connect(finished)
    runner.process.errorOccurred.connect(process_error)
    refresh_runtime(runtime)
    root._pyfrac_runtime = runtime
    root._refresh_pyfrac_runtime = refresh_runtime
    root.set_dataset = set_dataset
    root.pause = pause
    root.stop_active_task = lambda: stop_run() if running else None
    root._runner = runner
    root._timer = timer
    root._start_run = start_run
    root._finish_run = finished
    update_availability()
    return root


def _fmt(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:.3f}" if math.isfinite(number) else ""


__all__ = ["create_pyfrac_workbench"]
