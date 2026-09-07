from __future__ import annotations

import math

from ..theme import PALETTE


def create_cluster_view():
    from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem

    table = QTableWidget(0, 5)
    table.setObjectName("panel")
    table.setSelectionMode(QAbstractItemView.ExtendedSelection)
    table.setSelectionBehavior(QAbstractItemView.SelectItems)
    table.setHorizontalHeaderLabels(["簇", "PKN先验半长(m)", "EnKF后验半长(m)", "液量份额", "砂量份额"])
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    table.verticalHeader().setVisible(False)
    table.setAlternatingRowColors(True)
    table.setWordWrap(False)
    table.setMinimumHeight(168)
    table.setMaximumHeight(230)
    table.keyPressEvent = lambda event: _copy_on_shortcut(table, event)
    table._set_frame = lambda frame: update_cluster_view(table, frame)
    return table


def create_cluster_share_chart():
    """Create a compact current-frame visual for six-cluster allocation.

    The table remains available for exact values.  This chart is deliberately
    small and state-focused: it makes the observed/model/equal-share relation
    visible without introducing another heavy plotting dependency.
    """

    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QColor, QPainter, QPen
    from PySide6.QtWidgets import QFrame, QMenu, QApplication

    class ClusterShareChart(QFrame):
        def __init__(self):
            super().__init__()
            self.setObjectName("chartPanel")
            self.setMinimumHeight(250)
            self._frame = {}
            self.setContextMenuPolicy(Qt.CustomContextMenu)
            self.customContextMenuRequested.connect(self._show_copy_menu)

        def set_frame(self, frame):
            self._frame = dict(frame or {})
            self.update()

        def _clusters(self):
            values = self._frame.get("clusters", [])
            if values:
                return values
            dt = self._frame.get("dt", {}) or {}
            return dt.get("clusters", []) or []

        def _show_copy_menu(self, position):
            menu = QMenu(self)
            action = menu.addAction("复制当前六簇分配")
            action.triggered.connect(self._copy_data)
            menu.exec(self.mapToGlobal(position))

        def _copy_data(self):
            rows = [["簇", "模型液量份额", "观测液量份额", "模型砂量份额", "观测砂量份额"]]
            for index, item in enumerate(self._clusters(), start=1):
                rows.append([
                    f"簇{item.get('id', index)}",
                    _copy_number(item.get("liquid")),
                    _copy_number(item.get("observed_liquid")),
                    _copy_number(item.get("sand")),
                    _copy_number(item.get("observed_sand")),
                ])
            QApplication.clipboard().setText("\n".join("\t".join(row) for row in rows))

        def paintEvent(self, _event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor(PALETTE["panel"]))
            painter.setPen(QColor(PALETTE["text"]))
            painter.drawText(14, 23, "六簇液量分配 · 当前帧")
            clusters = self._clusters()
            if not clusters:
                painter.setPen(QColor(PALETTE["muted"]))
                painter.drawText(14, 52, "暂无分簇数据")
                return

            has_observed = any(_finite(item.get("observed_liquid")) for item in clusters)
            legend_x = 14
            legend_y = 45
            legend = [("模型", PALETTE["cyan"])]
            if has_observed:
                legend.append(("观测", PALETTE["blue"]))
            legend.append(("均衡基准", PALETTE["orange"]))
            for name, color in legend:
                painter.setPen(QPen(QColor(color), 3))
                if name == "均衡基准":
                    painter.setPen(QPen(QColor(color), 2, Qt.DashLine))
                painter.drawLine(QPointF(legend_x, legend_y - 4), QPointF(legend_x + 15, legend_y - 4))
                painter.setPen(QColor(color))
                painter.drawText(legend_x + 21, legend_y, name)
                legend_x += 72 if name != "均衡基准" else 92

            model_values = [_number(item.get("liquid")) for item in clusters]
            observed_values = [_number(item.get("observed_liquid")) for item in clusters]
            valid_values = [value for value in model_values + observed_values if value is not None]
            equal_share = 1.0 / max(len(clusters), 1)
            maximum = max([equal_share * 1.6, *valid_values, 0.01])
            maximum = _nice_upper(maximum)
            plot = self.rect().adjusted(48, 60, -16, -34)
            painter.setPen(QPen(QColor(PALETTE["border"]), 1))
            for tick in range(0, 4):
                value = maximum * tick / 3.0
                y = plot.bottom() - plot.height() * value / max(maximum, 1e-9)
                painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
                painter.setPen(QColor(PALETTE["muted"]))
                painter.drawText(8, int(y + 4), f"{value:.2f}")
                painter.setPen(QPen(QColor(PALETTE["border"]), 1))

            equal_y = plot.bottom() - plot.height() * equal_share / max(maximum, 1e-9)
            painter.setPen(QPen(QColor(PALETTE["orange"]), 2, Qt.DashLine))
            painter.drawLine(QPointF(plot.left(), equal_y), QPointF(plot.right(), equal_y))
            group_width = plot.width() / max(len(clusters), 1)
            bar_width = min(18.0, group_width / (3.0 if has_observed else 2.0))
            for index, item in enumerate(clusters):
                center = plot.left() + group_width * (index + 0.5)
                series = [(model_values[index], PALETTE["cyan"], -bar_width / 2)]
                if has_observed:
                    series.append((observed_values[index], PALETTE["blue"], bar_width / 2))
                for value, color, offset in series:
                    if value is None:
                        continue
                    bar_height = plot.height() * value / max(maximum, 1e-9)
                    painter.fillRect(
                        int(center + offset),
                        int(plot.bottom() - bar_height),
                        max(4, int(bar_width)),
                        max(1, int(bar_height)),
                        QColor(color),
                    )
                painter.setPen(QColor(PALETTE["text"]))
                painter.drawText(int(center - 12), self.height() - 12, f"簇{item.get('id', index + 1)}")

    return ClusterShareChart()


def _copy_on_shortcut(table, event):
    from PySide6.QtGui import QKeySequence
    from PySide6.QtWidgets import QApplication, QTableWidget

    if event.matches(QKeySequence.Copy):
        ranges = table.selectedRanges()
        if not ranges:
            return
        selected = {(item.row(), item.column()) for item in table.selectedItems()}
        rows = []
        for row in range(ranges[0].topRow(), ranges[0].bottomRow() + 1):
            cells = []
            for column in range(ranges[0].leftColumn(), ranges[0].rightColumn() + 1):
                if (row, column) not in selected:
                    cells.append("")
                else:
                    item = table.item(row, column)
                    cells.append(item.text() if item else "")
            rows.append("\t".join(cells))
        QApplication.clipboard().setText("\n".join(rows))
        return
    QTableWidget.keyPressEvent(table, event)


def update_cluster_view(table, frame: dict):
    values = frame.get("clusters", [])
    if not values:
        dt = frame.get("dt", {}) or {}
        posterior = dt.get("posterior_half_lengths_m") or []
        prior = dt.get("prior_half_lengths_m") or []
        values = [
            {
                "id": i + 1,
                "prior_length": prior[i] if i < len(prior) else 0.0,
                "length": value,
                "liquid": "--",
                "sand": "--",
            }
            for i, value in enumerate(posterior)
        ]
    table.setRowCount(len(values))
    for row, item in enumerate(values):
        cells = [row + 1, item.get("prior_length", 0.0), item.get("length", 0.0), item.get("liquid", "--"), item.get("sand", "--")]
        for column, value in enumerate(cells):
            table.setItem(row, column, __import__("PySide6.QtWidgets", fromlist=["QTableWidgetItem"]).QTableWidgetItem(_format(value)))


def _format(value):
    try:
        number = float(value)
        if number == 0:
            return "0"
        if abs(number) < 0.01:
            return f"{number:.2e}"
        return f"{number:.2f}"
    except (TypeError, ValueError):
        return str(value)


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _finite(value):
    return _number(value) is not None


def _nice_upper(value):
    if value <= 0.1:
        return 0.1
    if value <= 0.2:
        return 0.2
    if value <= 0.5:
        return 0.5
    return math.ceil(value * 10.0) / 10.0


def _copy_number(value):
    number = _number(value)
    return "" if number is None else f"{number:.6g}"
