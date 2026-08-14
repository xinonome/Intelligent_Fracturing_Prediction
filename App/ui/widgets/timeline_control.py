from __future__ import annotations


def create_timeline_control(controller):
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox, QWidget

    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    timer = QTimer(widget)
    play = QPushButton("▶ 播放")
    reset = QPushButton("重置")
    back = QPushButton("−1步")
    forward = QPushButton("+1步")
    speed = QComboBox()
    for value in (0.25, 0.5, 1.0, 2.0, 4.0):
        speed.addItem(f"{value:g}×", value)
    speed.setCurrentIndex(2)
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, max(len(controller.frames) - 1, 0))
    jump = QSpinBox()
    jump.setRange(0, max(len(controller.frames) - 1, 0))
    time_label = QLabel("t=-- / --")
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
            time_label.setText("t=-- / --")
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
            play.setText("▶ 播放")
        else:
            timer.start(interval())
            play.setText("⏸ 暂停")

    timer.timeout.connect(lambda: controller.set_index((controller.index + 1) % max(len(controller.frames), 1)))
    play.clicked.connect(toggle)
    reset.clicked.connect(lambda: controller.set_index(0))
    back.clicked.connect(lambda: controller.step(-1))
    forward.clicked.connect(lambda: controller.step(1))
    speed.currentIndexChanged.connect(lambda: timer.start(interval()) if timer.isActive() else None)
    slider.valueChanged.connect(controller.set_index)
    jump.valueChanged.connect(controller.set_index)
    controller.frameChanged.connect(show)
    widget._timer = timer
    widget._show_frame = show
    show(controller.current)
    return widget
