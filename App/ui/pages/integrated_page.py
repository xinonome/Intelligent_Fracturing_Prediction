from __future__ import annotations

from bisect import bisect_left
from pathlib import Path

from ..theme import PALETTE
from ..layout_config import DEFAULT_LAYOUT_FILE, apply_layout_ratios, load_layout_ratios
from ...data.dt_loader import DTLoader, number
from ...data.hmi_loader import HMILoader
from ...data.no_das_animation import find_no_das_gif
from ..web_view import Embedded3DView
from ..widgets.chart_panel import build_chart
from ..widgets.cluster_view import create_cluster_share_chart
from ..widgets.decision_card import create_decision_card, update_decision_card
from ..widgets.gif_view import create_no_das_gif_view
from ..widgets.parameter_panel import create_parameter_panel, update_parameter_panel
from ..widgets.status_card import Panel
from ..widgets.timeline_control import create_timeline_control
from ..parameter_format import format_parameter_map


def build_integrated_page(controller, registry, html_path: Path | None, on_dataset_changed=None):
    from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
    from PySide6.QtWidgets import QComboBox, QLabel, QScrollArea, QSplitter, QStackedWidget, QVBoxLayout, QWidget

    class _ScenarioLoadWorker(QObject):
        loaded = Signal(int, str, str, object)
        failed = Signal(int, str, str)

        def __init__(self, request_id: int, scenario_id: str, dataset_id: str):
            super().__init__()
            self.request_id = request_id
            self.scenario_id = scenario_id
            self.dataset_id = dataset_id

        @Slot()
        def run(self):
            try:
                frames = controller.prepare_scenario_frames(
                    self.scenario_id,
                    dataset_id=self.dataset_id,
                )
                self.loaded.emit(self.request_id, self.scenario_id, self.dataset_id, frames)
            except Exception as exc:  # pragma: no cover - depends on local cache
                self.failed.emit(self.request_id, self.scenario_id, f"{type(exc).__name__}: {exc}")

    class _ScenarioLoadReceiver(QObject):
        @Slot(int, str, str, object)
        def loaded(self, request_id: int, scenario_id: str, dataset_id: str, frames: object):
            finish_scenario_switch(
                int(request_id),
                str(scenario_id),
                str(dataset_id),
                frames if isinstance(frames, list) else [],
            )

        @Slot(int, str, str)
        def failed(self, request_id: int, scenario_id: str, error: str):
            fail_scenario_switch(int(request_id), str(scenario_id), error)

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
    title = QLabel("全流程联动回放")
    title.setObjectName("pageTitle")
    root.addWidget(title)

    selected_scenario = getattr(registry, "scenario_id", "das_cluster_observation")
    dataset_row = QVBoxLayout()

    def _dataset_option_label(dataset: dict) -> str:
        """Keep the selector concise while retaining the physical identity."""

        if dataset.get("adapter") == "raw_frac_construction":
            return str(dataset.get("stage_id") or dataset.get("dataset_id") or "未知井段")
        well = str(dataset.get("well_id") or dataset.get("display_name") or dataset.get("dataset_id"))
        stage = str(dataset.get("stage_id") or "")
        return f"{well} · Stage {stage}" if stage else well

    dataset_title = QLabel()
    dataset_title.setVisible(False)

    scenario_row = QVBoxLayout()
    scenario_box = QComboBox()
    scenario_box.addItem("◌  无 DAS：压力在线校正", "no_das_pressure_only")
    scenario_box.addItem("◉  有 DAS：压力 + 分簇观测校验", "das_cluster_observation")
    scenario_box.setCurrentIndex(max(0, scenario_box.findData(selected_scenario)))
    source_label = QLabel()
    source_label.setObjectName("muted")
    source_label.setWordWrap(True)
    scenario_row.addWidget(scenario_box)
    scenario_row.addWidget(source_label)
    root.addLayout(scenario_row)

    def sync_scenario_options():
        """Keep scene options tied to the globally selected dataset."""

        dataset = registry.dataset()
        dataset_title.setText(_dataset_option_label(dataset))
        for index in range(scenario_box.count()):
            scenario_id = str(scenario_box.itemData(index) or "")
            item = scenario_box.model().item(index)
            if item is not None:
                item.setEnabled(bool(registry.dataset_ready_for_scenario(registry.dataset_id, scenario_id)))
        current_index = scenario_box.findData(str(getattr(registry, "scenario_id", "")))
        if current_index >= 0:
            scenario_box.blockSignals(True)
            scenario_box.setCurrentIndex(current_index)
            scenario_box.blockSignals(False)

    sync_scenario_options()
    scenario_thread = None
    scenario_worker = None
    scenario_pending = False
    scenario_request_id = 0
    scenario_was_playing = False
    scenario_receiver = _ScenarioLoadReceiver(page)

    # Keep the production hierarchy identical to the developer layout tester:
    # three top-level vertical regions, a chart/3D split above, and a
    # cluster/decision split below.  This makes a saved tester layout usable
    # by the formal APP instead of being a cosmetic preview only.
    main_splitter = QSplitter(Qt.Vertical)
    main_splitter.setObjectName("mainVerticalSplitter")
    main_splitter.setChildrenCollapsible(False)
    main_splitter.setHandleWidth(8)
    upper = QSplitter(Qt.Horizontal)
    upper.setObjectName("upperHorizontalSplitter")
    upper.setChildrenCollapsible(False)
    upper.setHandleWidth(8)
    charts = QSplitter(Qt.Vertical)
    charts.setObjectName("chartsVerticalSplitter")
    charts.setChildrenCollapsible(False)
    charts.setHandleWidth(8)

    chart_height = 270
    pressure_chart = build_chart("压力对比 · 观测 / PKN先验 / EnKF后验", chart_height, y_min=0.0)
    flow_chart = build_chart("排量对比 · 当前 / 推荐", chart_height, y_min=0.0)
    sand_chart = build_chart("砂比对比 · 当前 / 推荐", chart_height, y_min=0.0)
    for chart in (pressure_chart, flow_chart, sand_chart):
        charts.addWidget(chart)
    charts.setSizes([300, 300, 300])
    cluster_chart = create_cluster_share_chart()

    center, center_layout = Panel.create()
    center_title = QLabel("数字孪生三维状态")
    center_title.setObjectName("sectionTitle")
    center_layout.addWidget(center_title)
    initial_html = registry.html(getattr(registry, "scenario_id", None)) or html_path
    model = Embedded3DView.create(initial_html)
    gif_view = create_no_das_gif_view()
    visual_stack = QStackedWidget()
    visual_stack.setObjectName("dtVisualStack")
    visual_stack.addWidget(model)
    visual_stack.addWidget(gif_view)
    model.setMinimumHeight(690)
    gif_view.setMinimumHeight(690)
    center_layout.addWidget(visual_stack, 1)
    upper.addWidget(charts)
    upper.addWidget(center)
    upper.setSizes([1040, 520])
    upper.setStretchFactor(0, 2)
    upper.setStretchFactor(1, 1)

    decision = create_decision_card()
    middle = QSplitter(Qt.Horizontal)
    middle.setObjectName("middleHorizontalSplitter")
    middle.setChildrenCollapsible(False)
    middle.setHandleWidth(8)
    middle.addWidget(cluster_chart)
    middle.addWidget(decision)
    middle.setSizes([760, 800])
    middle.setStretchFactor(0, 1)
    middle.setStretchFactor(1, 1)

    params = create_parameter_panel(
        "EnKF 参数更新",
        columns=[
            ("先验参数", [("prior", "")]),
            ("后验参数", [("posterior", "")]),
            (
                "状态与簇级结果",
                [
                    ("cluster_balance", "分簇均衡指数"),
                    ("fracture_length", "裂缝长度(m)"),
                    ("fracture_width", "最大缝宽(mm)"),
                    ("prior_half_lengths", "先验半长(m)"),
                    ("posterior_half_lengths", "后验半长(m)"),
                    ("error", "后验误差"),
                    ("runtime", "计算时间"),
                ],
            ),
        ],
        column_stretches=[3, 3, 2],
    )
    main_splitter.addWidget(upper)
    main_splitter.addWidget(middle)
    main_splitter.addWidget(params)
    main_splitter.setSizes([720, 220, 260])
    main_splitter.setStretchFactor(0, 3)
    main_splitter.setStretchFactor(1, 1)
    main_splitter.setStretchFactor(2, 1)
    root.addWidget(main_splitter, 1)

    timeline = create_timeline_control(controller)
    # Camera gestures are deliberately disabled during playback.  A frame
    # update may still animate the model, but mouse rotation/zoom only becomes
    # available after the operator pauses the timeline.
    if getattr(timeline, "set_playback_callback", None):
        timeline.set_playback_callback(
            lambda playing: model.set_interaction_enabled(not playing)
            if getattr(model, "set_interaction_enabled", None)
            else None
        )
    root.addWidget(timeline)
    events = QLabel("关键事件：" + " · ".join(f"{name} 第{index + 1}帧" for name, index in _events(controller.frames).items()))
    events.setObjectName("muted")
    events.setWordWrap(True)
    root.addWidget(events)

    # The tester writes this file after a manual drag.  Loading is deferred
    # until the widget has a real size, otherwise QSplitter can discard the
    # ratios while it is still being laid out by the scroll area.
    production_splitters = {
        "main_vertical": main_splitter,
        "upper_horizontal": upper,
        "charts_vertical": charts,
        "middle_horizontal": middle,
    }

    def apply_saved_layout():
        apply_layout_ratios(production_splitters, load_layout_ratios(DEFAULT_LAYOUT_FILE))

    QTimer.singleShot(120, apply_saved_layout)

    def refresh_scenario_view():
        event_values = _events(controller.frames)
        events.setText(
            "关键事件："
            + (" · ".join(f"{name} 第{index + 1}帧" for name, index in event_values.items()) or "暂无标记")
        )
        chart_values, chart_end_s = _chart_series(registry, controller.frames)
        pressure_chart.set_series(chart_values)
        if chart_end_s is not None:
            pressure_chart.set_time_range(1.0, chart_end_s)
        sample_count = len(chart_values[0][1]) if chart_values else len(controller.frames)
        flow_values, sand_values = _control_series(
            registry,
            chart_end_s or float(max(len(controller.frames), 1)),
            sample_count,
            agent_model=getattr(controller, "agent_model_id", None),
        )
        flow_chart.set_series(flow_values)
        sand_chart.set_series(sand_values)
        cluster_chart.set_frame(controller.current or {})
        for control in (flow_chart, sand_chart):
            if chart_end_s is not None:
                control.set_time_range(1.0, chart_end_s)
        scenario = registry.scenario()
        scenario_cache = DTLoader(registry).cache
        scenario_meta = scenario_cache.get("meta", {}) if isinstance(scenario_cache, dict) else {}
        if scenario.get("observation_mode") == "pressure_only":
            cluster_note = f"无 DAS · 压力校正 · {int(scenario.get('assumed_cluster_count', 0))} 簇演变"
            observation_status = ""
            center_title.setText("无 DAS 压力校正 · 阶段级裂缝演变")
            dataset = registry.dataset()
            gif_path = find_no_das_gif(
                str(getattr(registry, "dataset_id", "")),
                stage_id=str(dataset.get("stage_id", "")),
            )
            gif_ready = gif_view.set_animation_path(gif_path, str(dataset.get("display_name", "")))
            visual_stack.setCurrentWidget(gif_view if gif_ready else model)
        else:
            cluster_note = "有 DAS · 压力 + 分簇观测"
            observation_status = ""
            geometry_status = str(scenario_meta.get("cluster_geometry", {}).get("status", ""))
            center_title.setText(
                "数字孪生三维状态 · 阶段级裂缝演化"
                if geometry_status in {"", "not_available"}
                else "数字孪生三维状态 · 六簇裂缝演化"
            )
            visual_stack.setCurrentWidget(model)
        coverage_end = scenario_meta.get("source_end_s") or scenario.get("source_end_s") or chart_end_s
        source_label.setText(
            f"{_dataset_option_label(registry.dataset())} · {cluster_note} · "
            f"1–{_format_seconds(coverage_end)} s"
        )

    refresh_scenario_view()

    def _end_scenario_switch(request_id: int, message: str | None = None):
        nonlocal scenario_pending
        if request_id != scenario_request_id:
            return
        scenario_pending = False
        scenario_box.setEnabled(True)
        if scenario_was_playing and getattr(timeline, "resume", None):
            timeline.resume()
        if message:
            source_label.setText(message)

    def _scenario_visual_timeout(request_id: int):
        if request_id != scenario_request_id or not scenario_pending:
            return
        _end_scenario_switch(
            request_id,
            "场景已切换 · 三维视图加载中",
        )

    def finish_scenario_switch(
        request_id: int,
        scenario_id: str,
        dataset_id: str,
        frames: list[dict],
    ):
        nonlocal scenario_pending
        if request_id != scenario_request_id:
            return
        if not frames:
            fail_scenario_switch(request_id, scenario_id, "场景回放数据为空")
            return
        controller.apply_scenario_frames(scenario_id, frames, dataset_id=dataset_id)
        refresh_scenario_view()
        # The no-DAS GIF is a local, non-WebGL view.  Do not navigate Chromium
        # when it is selected; this keeps the scenario switch responsive on
        # machines affected by WebEngine/GPU driver resets.
        if (
            registry.scenario().get("observation_mode") == "pressure_only"
            and visual_stack.currentWidget() is gif_view
        ):
            _end_scenario_switch(request_id)
            return
        scenario_html = registry.html(str(scenario_id))
        if scenario_html and getattr(model, "set_html_path", None):
            # Let the committed frame and clock repaint before Chromium
            # starts loading the next multi-megabyte Plotly document.
            QTimer.singleShot(
                250,
                lambda: model.set_html_path(
                    scenario_html,
                    lambda ok: _end_scenario_switch(
                        request_id,
                        None if ok else "场景数据已切换，但 3D 视图加载失败；其余结果仍可用。",
                    ),
                ),
            )
            # A broken WebEngine/GPU subprocess must not leave the controls
            # disabled forever.  The data switch is already committed.
            QTimer.singleShot(16000, lambda: _scenario_visual_timeout(request_id))
        else:
            _end_scenario_switch(request_id)

    def fail_scenario_switch(request_id: int, scenario_id: str, error: str):
        nonlocal scenario_pending
        if request_id != scenario_request_id:
            return
        scenario_pending = False
        scenario_box.blockSignals(True)
        scenario_box.setCurrentIndex(max(0, scenario_box.findData(getattr(registry, "scenario_id", ""))))
        scenario_box.blockSignals(False)
        sync_scenario_options()
        scenario_box.setEnabled(True)
        refresh_scenario_view()
        source_label.setText(f"场景切换失败：{error}")
        if scenario_was_playing and getattr(timeline, "resume", None):
            timeline.resume()

    def _scenario_thread_finished():
        nonlocal scenario_thread, scenario_worker
        scenario_worker = None
        scenario_thread = None

    def change_scenario(index):
        nonlocal scenario_thread, scenario_worker, scenario_pending, scenario_request_id, scenario_was_playing
        scenario_id = scenario_box.itemData(index)
        if not scenario_id or scenario_id == getattr(registry, "scenario_id", None):
            return
        if scenario_pending:
            return
        target_dataset_id = str(getattr(registry, "dataset_id", ""))
        if not target_dataset_id or not registry.dataset_ready_for_scenario(target_dataset_id, str(scenario_id)):
            scenario_box.blockSignals(True)
            scenario_box.setCurrentIndex(max(0, scenario_box.findData(getattr(registry, "scenario_id", ""))))
            scenario_box.blockSignals(False)
            sync_scenario_options()
            source_label.setText("当前井段不支持该场景")
            return
        if not hasattr(controller, "prepare_scenario_frames"):
            # Compatibility path for a non-Qt/headless controller.
            if hasattr(controller, "set_dataset") and target_dataset_id != getattr(registry, "dataset_id", ""):
                controller.set_dataset(str(target_dataset_id))
            controller.set_scenario(str(scenario_id))
            refresh_scenario_view()
            scenario_html = registry.html(str(scenario_id))
            if (
                visual_stack.currentWidget() is not gif_view
                and scenario_html
                and getattr(model, "set_html_path", None)
            ):
                model.set_html_path(scenario_html)
            return

        scenario_pending = True
        scenario_request_id += 1
        request_id = scenario_request_id
        scenario_was_playing = bool(getattr(timeline, "is_playing", lambda: False)())
        if getattr(timeline, "pause", None):
            timeline.pause()
        scenario_box.setEnabled(False)
        source_label.setText("正在切换场景，请稍候…")
        scenario_thread = QThread(page)
        scenario_worker = _ScenarioLoadWorker(
            request_id,
            str(scenario_id),
            str(target_dataset_id),
        )
        scenario_worker.moveToThread(scenario_thread)
        scenario_thread.started.connect(scenario_worker.run)
        scenario_worker.loaded.connect(scenario_receiver.loaded)
        scenario_worker.failed.connect(scenario_receiver.failed)
        scenario_worker.loaded.connect(scenario_thread.quit)
        scenario_worker.failed.connect(scenario_thread.quit)
        scenario_thread.finished.connect(scenario_worker.deleteLater)
        scenario_thread.finished.connect(scenario_thread.deleteLater)
        scenario_thread.finished.connect(_scenario_thread_finished)
        scenario_thread.start()

    scenario_box.currentIndexChanged.connect(change_scenario)

    def update(frame):
        frame = frame or {}
        progress = controller.index / max(len(controller.frames) - 1, 1)
        pressure_chart.set_progress(progress)
        flow_chart.set_progress(progress)
        sand_chart.set_progress(progress)
        cluster_chart.set_frame(frame)
        dt = frame.get("dt", {}) or {}
        update_parameter_panel(params, {
            "prior": _parameter_text(dt.get("prior_parameters")),
            "posterior": _parameter_text(dt.get("posterior_parameters")),
            "cluster_balance": dt.get("cluster_balance_degree"),
            "fracture_length": dt.get("fracture_length_m") or _total_half_length(dt.get("posterior_half_lengths_m")),
            "fracture_width": (
                float(dt.get("fracture_width_m")) * 1000.0
                if dt.get("fracture_width_m") is not None else None
            ),
            "prior_half_lengths": _half_length_text(dt.get("prior_half_lengths_m")),
            "posterior_half_lengths": _half_length_text(dt.get("posterior_half_lengths_m")),
            "error": dt.get("posterior_error"),
            "runtime": dt.get("runtime_ms"),
        }, {
            "cluster_balance": "{:.3f}",
            "fracture_length": "{:.2f} m",
            "fracture_width": "{:.3f} mm",
            "error": "{:.3f}",
            "runtime": "{:.1f} ms",
        })
        update_decision_card(decision, frame)
        if getattr(model, "set_time_index", None):
            progress = controller.index / max(len(controller.frames) - 1, 1)
            if visual_stack.currentWidget() is gif_view:
                gif_view.set_progress(progress)
            else:
                model.set_time_index(frame.get("time_s", 0))

    controller.frameChanged.connect(update)
    update(controller.current or {})
    pending_dataset_visual = {"value": False}

    def refresh_dataset_visual():
        """Avoid reloading hidden WebEngine documents during a global switch."""

        if not page.isVisible():
            pending_dataset_visual["value"] = True
            return
        # A global stage switch can arrive while this page is not the active
        # page.  Do not synchronously replace a QWebEngineView with the GIF
        # view from inside the dataset worker completion callback: on Windows
        # that can race Chromium's native repaint and terminate the process.
        # Once the page is visible, refresh the scenario on the next event
        # turn, after all pages have received the new frame list.
        if pending_dataset_visual["value"]:
            pending_dataset_visual["value"] = False
            QTimer.singleShot(150, _refresh_after_global_dataset)
            return
        pending_dataset_visual["value"] = False
        scenario_html = registry.html(getattr(registry, "scenario_id", None))
        if visual_stack.currentWidget() is not gif_view and scenario_html and getattr(model, "set_html_path", None):
            model.set_html_path(scenario_html)

    def _refresh_after_global_dataset():
        """Commit the integrated visual change outside the dataset callback."""

        if not page.isVisible():
            pending_dataset_visual["value"] = True
            return
        pending_dataset_visual["value"] = False
        refresh_scenario_view()
        refresh_dataset_visual()

    def set_global_dataset(dataset_id):
        """Refresh the joint presentation after the global stage changes."""

        nonlocal scenario_pending, scenario_request_id
        if scenario_pending:
            # An in-flight scene worker must not be allowed to commit the old
            # dataset after the global selector has already moved on.
            scenario_request_id += 1
            scenario_pending = False
            scenario_box.setEnabled(True)
        if str(dataset_id or "") != str(getattr(registry, "dataset_id", "")):
            registry.set_dataset(str(dataset_id))
        sync_scenario_options()
        # The frame controller will emit the first frame on the next event
        # turn.  Defer the heavy visual switch until the integrated page is
        # actually visible; this prevents a hidden QWebEngineView/GIF
        # QStackedWidget transition from causing a native access violation.
        pending_dataset_visual["value"] = True
        if page.isVisible():
            QTimer.singleShot(150, _refresh_after_global_dataset)
        if on_dataset_changed:
            on_dataset_changed(registry.dataset())

    page.set_global_dataset = set_global_dataset
    page.refresh_dataset_visual = refresh_dataset_visual
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


def _control_series(registry, end_s, sample_count, agent_model=None):
    loader = HMILoader(registry, model_id=agent_model)
    dt_loader = DTLoader(registry)
    rows = loader.rows if dt_loader.hmi_available() else []
    if not rows or sample_count <= 0:
        # Other registered well sections may only have construction curves.
        # Show measured flow/sand and leave recommendation curves explicitly
        # unavailable instead of reusing another well's HMI actions.
        cache = dt_loader.cache
        arrays = cache.get("arrays", {}) if isinstance(cache, dict) else {}
        flow = arrays.get("flow_rate_m3_min", [])
        sand = arrays.get("sand_ratio_percent", [])
        if not flow and not sand:
            return [], []
        flow_series = []
        sand_series = []
        if flow:
            flow_series.append(("当前排量", _resample_sequence([number(value, float("nan")) for value in flow], sample_count), PALETTE["cyan"]))
        if sand:
            sand_series.append(("当前砂比", _resample_sequence([number(value, float("nan")) for value in sand], sample_count), PALETTE["orange"]))
        return flow_series, sand_series
    fields = [
        ("当前排量", "current_flow_m3_min", PALETTE["cyan"]),
        ("推荐排量", "flow_m3_min", PALETTE["blue"]),
        ("当前砂比", "current_sand_ratio_percent", PALETTE["orange"]),
        ("推荐砂比", "recommended_sand_ratio_percent", PALETTE["red"]),
    ]
    result = []
    for name, key, color in fields:
        source = [number(row.get(key), float("nan")) for row in rows]
        result.append((name, _resample_sequence(source, sample_count), color))
    return result[:2], result[2:]


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
    return format_parameter_map(value)


def _half_length_text(values):
    if not values:
        return "当前井段无该项数据"
    return "；".join(f"簇{index + 1}={_fmt(value)} m" for index, value in enumerate(values))


def _total_half_length(values):
    if not values:
        return None
    try:
        return float(sum(float(value) for value in values))
    except (TypeError, ValueError):
        return None


def _fmt(value):
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "--"


def _format_seconds(value):
    try:
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return "--"


def _events(frames):
    result = {}
    for index, frame in enumerate(frames):
        option = frame.get("hmi_option")
        if option and option not in result:
            result[option] = index
    return result
