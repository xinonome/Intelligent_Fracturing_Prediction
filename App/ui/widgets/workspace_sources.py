"""Operator-visible inventory of registered data, models and run outputs."""

from __future__ import annotations

import json
from pathlib import Path

from ...core.paths import PATHS
from ...data.resource_catalog import collect_resource_inventory


def show_workspace_sources(parent, registry, controller, *, edition: str = "integrated") -> None:
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import (
        QApplication, QDialog, QFileDialog, QHBoxLayout, QLabel, QPushButton,
        QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    )

    dialog = QDialog(parent)
    dialog.setWindowTitle("数据、模型与运行成果")
    dialog.resize(980, 620)
    root = QVBoxLayout(dialog)
    root.addWidget(QLabel("所列内容均来自当前项目注册表、模型目录或输出目录。"))
    tabs = QTabWidget()
    root.addWidget(tabs, 1)

    datasets = []
    for dataset_id, value in sorted((registry.dataset_catalog().get("datasets", {}) or {}).items()):
        if value.get("data_scope") == "composite":
            continue
        source = registry.path(value.get("pressure_source"))
        cache = registry.path(value.get("cache_source"))
        datasets.append([
            dataset_id,
            str(value.get("well_id") or value.get("stage_id") or value.get("display_name") or ""),
            "有 DAS" if value.get("fiber_source") else "无 DAS",
            "原始数据可用" if source and source.exists() else "缺少原始数据",
            "已有孪生缓存" if cache and cache.exists() else "等待计算",
            str(source or ""),
        ])
    dataset_table = _table(["数据集", "井/段", "场景", "数据状态", "结果状态", "来源路径"], datasets)
    tabs.addTab(dataset_table, "数据")

    inventory = collect_resource_inventory(registry)
    models = [
        [
            str(item.get("kind") or ""),
            str(item.get("name") or ""),
            str(item.get("status") or ""),
            str(item.get("purpose") or ""),
            str(item.get("context") or ""),
            str(item.get("size") or ""),
            str(item.get("path") or ""),
        ]
        for item in inventory
    ]
    model_table = _table(["类型", "模型 / 运行", "状态", "用途", "算法 / 场景", "大小", "文件 / 运行来源"], models)
    tabs.addTab(model_table, "模型")

    run_rows = []
    roots = [PATHS.app_runs, PATHS.app_outputs / "pyfrac_runs", PATHS.app_outputs / "transfer_runs"]
    for run_root in roots:
        if not run_root.exists():
            continue
        for path in sorted(run_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
            if path.is_dir():
                run_rows.append([path.name, run_root.name, _run_state(path), str(path)])
    run_table = _table(["运行编号", "类型", "状态", "输出目录"], run_rows)
    tabs.addTab(run_table, "运行与成果")

    status = QLabel("")
    status.setObjectName("muted")
    actions = QHBoxLayout()
    open_button = QPushButton("打开所选路径")
    copy_button = QPushButton("复制所选路径")
    export_button = QPushButton("导出来源清单")
    close_button = QPushButton("关闭")
    for button in (open_button, copy_button, export_button):
        actions.addWidget(button)
    actions.addStretch(1)
    actions.addWidget(close_button)
    root.addLayout(actions)
    root.addWidget(status)

    def selected_path() -> Path | None:
        table = tabs.currentWidget()
        if not isinstance(table, QTableWidget) or table.currentRow() < 0:
            return None
        for column in reversed(range(table.columnCount())):
            item = table.item(table.currentRow(), column)
            if item and item.text():
                candidate = Path(item.text())
                if candidate.exists():
                    return candidate
        return None

    def open_selected() -> None:
        path = selected_path()
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path if path.is_dir() else path.parent)))
        else:
            status.setText("请先选择包含有效本地路径的记录。")

    def copy_selected() -> None:
        path = selected_path()
        if path:
            QApplication.clipboard().setText(str(path))
            status.setText("路径已复制。")
        else:
            status.setText("当前记录没有可用路径。")

    def export_inventory() -> None:
        destination, _ = QFileDialog.getSaveFileName(dialog, "导出来源清单", "数据模型与运行来源.json", "JSON (*.json)")
        if not destination:
            return
        payload = {
            "active_dataset": str(getattr(registry, "dataset_id", "")),
            "active_scenario": str(getattr(registry, "scenario_id", "")),
            "active_agent_model": str(getattr(controller, "agent_model_id", "")),
            "datasets": datasets,
            "models": models,
            "runs": run_rows,
        }
        Path(destination).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        status.setText(f"来源清单已导出：{destination}")

    open_button.clicked.connect(open_selected)
    copy_button.clicked.connect(copy_selected)
    export_button.clicked.connect(export_inventory)
    close_button.clicked.connect(dialog.accept)
    dialog.exec()


def _table(headers: list[str], rows: list[list[str]]):
    from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem

    table = QTableWidget(len(rows), len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setAlternatingRowColors(True)
    for row_index, row in enumerate(rows):
        for column, value in enumerate(row):
            item = QTableWidgetItem(str(value))
            item.setToolTip(str(value))
            table.setItem(row_index, column, item)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    return table


def _run_state(path: Path) -> str:
    for filename in ("validation_manifest.json", "run_manifest.json", "summary.json"):
        candidate = path / filename
        if candidate.exists():
            try:
                value = json.loads(candidate.read_text(encoding="utf-8-sig"))
                return str(value.get("status") or value.get("run_status") or "已生成")
            except (OSError, json.JSONDecodeError):
                return "文件可用"
    return "输出目录"


__all__ = ["show_workspace_sources"]
