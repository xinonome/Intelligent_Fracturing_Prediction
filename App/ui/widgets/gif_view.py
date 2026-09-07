"""Interactive timeline-controlled GIF view used by the no-DAS path."""

from __future__ import annotations

from pathlib import Path


def create_no_das_gif_view():
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtGui import QMovie, QPainter
    from PySide6.QtWidgets import (
        QFrame,
        QGraphicsPixmapItem,
        QGraphicsScene,
        QGraphicsView,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
    )

    class _ZoomView(QGraphicsView):
        zoomed = Signal(int)

        def wheelEvent(self, event):
            delta = event.angleDelta().y()
            if delta:
                self.zoomed.emit(1 if delta > 0 else -1)
                event.accept()
                return
            super().wheelEvent(event)

    class NoDasGifView(QFrame):
        def __init__(self):
            super().__init__()
            self.setObjectName("noDasGifPanel")
            self._movie = None
            self._path = None
            self._progress = 0.0
            self._frame_index = -1
            self._fit_mode = True
            layout = QVBoxLayout(self)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(4)

            controls = QHBoxLayout()
            controls.setContentsMargins(0, 0, 0, 0)
            controls.setSpacing(4)
            self.zoom_out = QPushButton("−")
            self.zoom_out.setToolTip("缩小")
            self.zoom_in = QPushButton("+")
            self.zoom_in.setToolTip("放大")
            self.fit_button = QPushButton("适应窗口")
            self.fit_button.setToolTip("恢复到适应窗口")
            self.zoom_label = QLabel("适应窗口")
            self.zoom_label.setObjectName("muted")
            controls.addWidget(self.zoom_out)
            controls.addWidget(self.zoom_in)
            controls.addWidget(self.fit_button)
            controls.addWidget(self.zoom_label)
            controls.addStretch(1)
            hint = QLabel("滚轮缩放 · 按住左键拖动平移")
            hint.setObjectName("muted")
            controls.addWidget(hint)
            layout.addLayout(controls)

            self.view = _ZoomView()
            self.view.setObjectName("noDasGifImage")
            self.view.setMinimumSize(320, 240)
            self.view.setAlignment(Qt.AlignCenter)
            self.view.setDragMode(QGraphicsView.ScrollHandDrag)
            self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.view.setResizeAnchor(QGraphicsView.AnchorViewCenter)
            self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
            self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.scene = QGraphicsScene(self.view)
            self.image_item = QGraphicsPixmapItem()
            self.scene.addItem(self.image_item)
            self.view.setScene(self.scene)
            layout.addWidget(self.view, 1)

            self.zoom_out.clicked.connect(lambda: self._zoom_by(1 / 1.2))
            self.zoom_in.clicked.connect(lambda: self._zoom_by(1.2))
            self.fit_button.clicked.connect(self._fit_to_view)
            self.view.zoomed.connect(lambda direction: self._zoom_by(1.2 if direction > 0 else 1 / 1.2))

        def set_animation_path(self, path: Path | None, dataset_label: str = ""):
            resolved = path.resolve() if path else None
            if resolved == self._path and self._movie is not None:
                return bool(self._movie)
            if self._movie is not None:
                self._movie.stop()
                self._movie.deleteLater()
                self._movie = None
            self._path = resolved
            self._frame_index = -1
            if not resolved or not resolved.exists():
                self.image_item.setPixmap(self._placeholder_pixmap("当前井段暂无无 DAS 动画"))
                self._fit_to_view()
                return False
            movie = QMovie(str(resolved))
            if not movie.isValid():
                self.image_item.setPixmap(self._placeholder_pixmap("无 DAS 动画无法读取"))
                self._fit_to_view()
                return False
            # Cache only the active dataset's frames.  QMovie cannot reliably
            # seek to an arbitrary frame with CacheNone; the main APP timeline
            # needs random access for dragging and single-step playback.  The
            # active movie is still rendered in system memory and does not
            # allocate a WebGL/GPU document.
            movie.setCacheMode(QMovie.CacheAll)
            self._movie = movie
            movie.frameChanged.connect(self._render_frame)
            movie.start()
            movie.setPaused(True)
            self.set_progress(self._progress)
            return True

        def _placeholder_pixmap(self, text: str):
            from PySide6.QtGui import QPixmap

            pixmap = QPixmap(720, 420)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setPen(Qt.white)
            painter.drawText(pixmap.rect(), Qt.AlignCenter, text)
            painter.end()
            return pixmap

        def _render_frame(self, _frame_number: int = 0):
            movie = self._movie
            if movie is None:
                return
            pixmap = movie.currentPixmap()
            if pixmap.isNull():
                return
            self.image_item.setPixmap(pixmap)
            self.scene.setSceneRect(self.image_item.boundingRect())
            if self._fit_mode:
                self._fit_to_view()

        def _zoom_by(self, factor: float):
            if self._movie is None or self.image_item.pixmap().isNull():
                return
            self._fit_mode = False
            self.view.scale(float(factor), float(factor))
            self.zoom_label.setText(f"{self.view.transform().m11() * 100:.0f}%")

        def _fit_to_view(self):
            if self.image_item.pixmap().isNull():
                return
            self._fit_mode = True
            self.view.fitInView(self.image_item, Qt.KeepAspectRatio)
            self.zoom_label.setText("适应窗口")

        def set_progress(self, progress: float):
            self._progress = min(max(float(progress), 0.0), 1.0)
            movie = self._movie
            if movie is None or movie.frameCount() <= 0:
                return
            target = round(self._progress * max(movie.frameCount() - 1, 0))
            if target != self._frame_index:
                self._frame_index = target
                movie.jumpToFrame(target)
                self._render_frame(target)

        def resizeEvent(self, event):
            super().resizeEvent(event)
            if self._fit_mode:
                self._fit_to_view()

    return NoDasGifView()
