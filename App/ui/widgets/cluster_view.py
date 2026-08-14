from __future__ import annotations


def create_cluster_view():
    from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

    table = QTableWidget(0, 5)
    table.setObjectName("panel")
    table.setHorizontalHeaderLabels(["簇", "PKN先验半长(m)", "EnKF后验半长(m)", "液量份额", "砂量份额"])
    table.horizontalHeader().setStretchLastSection(True)
    table._set_frame = lambda frame: update_cluster_view(table, frame)
    return table


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
