from __future__ import annotations


def create_timeline_control(controller):
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox, QWidget

    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    timer = QTimer(widget)
    playback_callbacks = []
    play = QPushButton("▶  播放")
    reset = QPushButton("↺  重置")
    back = QPushButton("‹  上一步")
    forward = QPushButton("下一步  ›")
    speed = QComboBox()
    for value in (0.25, 0.5, 1.0, 2.0, 4.0):
        speed.addItem(f"{value:g}×", value)
    speed.setCurrentIndex(2)
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, max(len(controller.frames) - 1, 0))
    jump = QSpinBox()
    jump.setRange(0, max(len(controller.frames) - 1, 0))
    time_label = QLabel("")
    layout.addWidget(play)
    layout.addWidget(reset)
    layout.addWidget(back)
    layout.addWidget(forward)
    layout.addWidget(QLabel("速度"))
    layout.addWidget(speed)
    layout.addWidget(slider, 1)
    layout.addWidget(QLabel("跳转"))
    layout.addWidget(jump)
    layout.addWidget(time_label)

    def interval():
        return int(1000 / max(float(speed.currentData()), 0.25))

    def show(frame):
        if not frame:
            time_label.clear()
            return
        index = controller.index
        slider.blockSignals(True)
        jump.blockSignals(True)
        slider.setValue(index)
        jump.setValue(index)
        slider.blockSignals(False)
        jump.blockSignals(False)
        time_label.setText(f"t={float(frame.get('time_s', 0)):.0f}s / {len(controller.frames)}帧")

    def toggle():
        if timer.isActive():
            timer.stop()
            play.setText("▶  播放")
            notify_playback_changed(False)
        else:
            timer.start(interval())
            play.setText("⏸  暂停")
            notify_playback_changed(True)

    def notify_playback_changed(playing):
        for callback in list(playback_callbacks):
            try:
                callback(bool(playing))
            except Exception:
                continue

    timer.timeout.connect(lambda: controller.set_index((controller.index + 1) % max(len(controller.frames), 1)))
    play.clicked.connect(toggle)
    reset.clicked.connect(lambda: controller.set_index(0))
    back.clicked.connect(lambda: controller.step(-1))
    forward.clicked.connect(lambda: controller.step(1))
    speed.currentIndexChanged.connect(lambda: timer.start(interval()) if timer.isActive() else None)
    slider.valueChanged.connect(controller.set_index)
    jump.valueChanged.connect(controller.set_index)
    controller.frameChanged.connect(show)

    def pause():
        """Pause playback without changing the current frame."""

        if timer.isActive():
            timer.stop()
            play.setText("▶  播放")
            notify_playback_changed(False)

    def resume():
        """Resume playback after a temporary UI/data transition."""

        if not timer.isActive():
            timer.start(interval())
            play.setText("⏸  暂停")
            notify_playback_changed(True)

    def set_playback_callback(callback):
        if callable(callback):
            playback_callbacks.append(callback)
            callback(bool(timer.isActive()))

    widget.pause = pause
    widget.resume = resume
    widget.is_playing = timer.isActive
    widget.set_playback_callback = set_playback_callback
    widget._timer = timer
    widget._show_frame = show
    show(controller.current)
    return widget
