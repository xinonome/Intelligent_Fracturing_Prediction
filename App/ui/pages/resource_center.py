"""Shared data, knowledge, model and run resources for the integrated APP."""

from __future__ import annotations

from pathlib import Path

from ...core.paths import PATHS
from ...data.hmi_loader import discover_agent_models
from ..widgets.knowledge_graph_panel import build_knowledge_graph_panel
from ..widgets.status_card import Panel
from ..widgets.workspace_sources import show_workspace_sources
from .data_import_page import build_data_import_page


def build_resource_center(
    registry,
    controller,
    *,
    on_imported=None,
    on_dataset_requested=None,
    research_mode: bool = False,
):
    from PySide6.QtCore import Qt, QUrl
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QPushButton,
        QScrollArea,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(20, 16, 20, 18)
    layout.setSpacing(10)
    title = QLabel("资源中心" if not research_mode else "资源中心 · 模型研发")
    title.setObjectName("pageTitle")
    layout.addWidget(title)
    tabs = QTabWidget()
    tabs.setObjectName("resourceCenterTabs")
    layout.addWidget(tabs, 1)

    dataset_page = QWidget()
    dataset_layout = QVBoxLayout(dataset_page)
    dataset_layout.setContentsMargins(8, 8, 8, 8)
    dataset_table = QTableWidget(0, 6)
    dataset_table.setObjectName("resourceDatasetTable")
    dataset_table.setHorizontalHeaderLabels(["数据集", "井 / 段", "场景", "原始数据", "孪生缓存", "来源"])
    dataset_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    dataset_table.setSelectionBehavior(QAbstractItemView.SelectRows)
    dataset_table.setSelectionMode(QAbstractItemView.SingleSelection)
    dataset_table.setAlternatingRowColors(True)
    dataset_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    dataset_table.horizontalHeader().setStretchLastSection(True)
    dataset_layout.addWidget(dataset_table, 1)
    dataset_actions = QHBoxLayout()
    apply_dataset = QPushButton("应用所选井段")
    refresh_dataset = QPushButton("刷新数据目录")
    dataset_status = QLabel("")
    dataset_status.setObjectName("muted")
    dataset_actions.addWidget(apply_dataset)
    dataset_actions.addWidget(refresh_dataset)
    dataset_actions.addWidget(dataset_status, 1)
    dataset_layout.addLayout(dataset_actions)
    tabs.addTab(dataset_page, "井段管理")

    def imported():
        if hasattr(registry, "refresh_catalog"):
            registry.refresh_catalog()
        refresh_catalog()
        if on_imported:
            on_imported()

    tabs.addTab(build_data_import_page(registry, on_imported=imported), "数据导入")
    tabs.addTab(build_knowledge_graph_panel(registry), "知识查询")

    resources = QScrollArea()
    resources.setWidgetResizable(True)
    resource_content = QWidget()
    resources.setWidget(resource_content)
    resource_layout = QVBoxLayout(resource_content)
    resource_layout.setContentsMargins(8, 8, 8, 8)
    model_panel, model_layout = Panel.create("可用模型")
    model_table = QTableWidget(0, 4)
    model_table.setHorizontalHeaderLabels(["模型", "状态", "用途", "文件 / 运行来源"])
    model_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    model_table.setSelectionBehavior(QAbstractItemView.SelectRows)
    model_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    model_table.horizontalHeader().setStretchLastSection(True)
    model_layout.addWidget(model_table)
    resource_layout.addWidget(model_panel)
    resource_actions = QHBoxLayout()
    open_inventory = QPushButton("查看完整数据、模型与运行清单")
    open_outputs = QPushButton("打开项目输出目录")
    resource_actions.addWidget(open_inventory)
    resource_actions.addWidget(open_outputs)
    resource_actions.addStretch(1)
    resource_layout.addLayout(resource_actions)
    if research_mode:
        audit_panel, audit_layout = Panel.create("运行与审计")
        audit_actions = QHBoxLayout()
        open_runs = QPushButton("打开任务运行目录")
        open_audit = QPushButton("打开人工审核记录")
        audit_actions.addWidget(open_runs)
        audit_actions.addWidget(open_audit)
        audit_actions.addStretch(1)
        audit_layout.addLayout(audit_actions)
        resource_layout.addWidget(audit_panel)
    resource_layout.addStretch(1)
    tabs.addTab(resources, "模型与结果")

    dataset_ids = []

    def refresh_catalog():
        nonlocal dataset_ids
        if hasattr(registry, "refresh_catalog"):
            registry.refresh_catalog()
        dataset_ids = []
        rows = []
        for dataset_id, value in sorted((registry.dataset_catalog().get("datasets", {}) or {}).items()):
            if value.get("data_scope") == "composite":
                continue
            source = registry.path(value.get("pressure_source"))
            cache = registry.path(value.get("cache_source"))
            dataset_ids.append(str(dataset_id))
            rows.append([
                str(dataset_id),
                str(value.get("display_name") or value.get("stage_id") or value.get("well_id") or "--"),
                "有 DAS" if value.get("fiber_source") else "无 DAS",
                "可用" if source and source.exists() else "缺失",
                "可用" if cache and cache.exists() else "等待计算",
                str(source or ""),
            ])
        dataset_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column, value in enumerate(row):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                dataset_table.setItem(row_index, column, item)
        active = str(getattr(registry, "dataset_id", ""))
        if active in dataset_ids:
            dataset_table.selectRow(dataset_ids.index(active))
        dataset_status.setText(f"{len(rows)} 个独立井段")

        model_rows = []
        runtime = registry.runtime_selection() if hasattr(registry, "runtime_selection") else {}
        model_rows.append(["PKN / KG-EnKF", "已登记", "裂缝数字孪生", str(runtime.get("enkf_run_dir") or "项目注册表")])
        for item in discover_agent_models(registry):
            model_rows.append([
                str(item.get("display_name") or item.get("model_id") or "--"),
                "可用" if item.get("ready") else "不可用",
                "智能调控离线回放",
                str(item.get("policy_path") or item.get("evaluation_path") or ""),
            ])
        model_table.setRowCount(len(model_rows))
        for row_index, row in enumerate(model_rows):
            for column, value in enumerate(row):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                model_table.setItem(row_index, column, item)

    def select_dataset():
        row = dataset_table.currentRow()
        if row < 0 or row >= len(dataset_ids):
            dataset_status.setText("请先选择井段")
            return
        dataset_id = dataset_ids[row]
        if on_dataset_requested:
            on_dataset_requested(dataset_id)
            dataset_status.setText("井段切换请求已提交")

    def open_path(path: Path):
        target = path if path.is_dir() else path.parent
        target.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    apply_dataset.clicked.connect(select_dataset)
    dataset_table.doubleClicked.connect(lambda _index: select_dataset())
    refresh_dataset.clicked.connect(refresh_catalog)
    open_inventory.clicked.connect(lambda: show_workspace_sources(page, registry, controller, edition="integrated"))
    open_outputs.clicked.connect(lambda: open_path(PATHS.outputs))
    if research_mode:
        open_runs.clicked.connect(lambda: open_path(PATHS.app_runs))
        open_audit.clicked.connect(lambda: open_path(PATHS.app_outputs / "operator_decisions.jsonl"))

    def set_global_dataset(dataset_id):
        dataset_id = str(dataset_id or "")
        if dataset_id in dataset_ids:
            dataset_table.selectRow(dataset_ids.index(dataset_id))

    refresh_catalog()
    page.set_global_dataset = set_global_dataset
    page.refresh_catalog = refresh_catalog
    page.pause_playback = lambda: None
    page.prepare_dataset_change = lambda: None
    page._resource_tabs = tabs
    return page


__all__ = ["build_resource_center"]
