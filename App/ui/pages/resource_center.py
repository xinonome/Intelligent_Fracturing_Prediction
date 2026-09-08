"""Shared data, knowledge, model and run resources for the integrated APP."""

from __future__ import annotations

from pathlib import Path

from ...core.paths import PATHS
from ...data.resource_catalog import collect_resource_inventory
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
        QApplication,
        QComboBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
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
    model_panel, model_layout = Panel.create("完整模型与结果清单")
    model_toolbar = QHBoxLayout()
    model_filter = QComboBox()
    model_filter.addItems(["全部资源", "模型", "运行结果", "缓存"])
    model_filter.setToolTip("按资源类型筛选，清单内容仍来自项目实际文件")
    model_search = QLineEdit()
    model_search.setPlaceholderText("搜索模型名、算法、井段或路径")
    model_search.setClearButtonEnabled(True)
    model_refresh = QPushButton("刷新模型清单")
    model_summary = QLabel("")
    model_summary.setObjectName("muted")
    model_toolbar.addWidget(model_filter)
    model_toolbar.addWidget(model_search, 1)
    model_toolbar.addWidget(model_refresh)
    model_toolbar.addWidget(model_summary)
    model_layout.addLayout(model_toolbar)
    model_table = QTableWidget(0, 7)
    model_table.setObjectName("resourceModelInventory")
    model_table.setHorizontalHeaderLabels(["类型", "模型 / 运行", "状态", "用途", "算法 / 场景", "大小", "文件 / 运行来源"])
    model_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    model_table.setSelectionBehavior(QAbstractItemView.SelectRows)
    model_table.setSelectionMode(QAbstractItemView.SingleSelection)
    model_table.setAlternatingRowColors(True)
    model_table.setMinimumHeight(480)
    header = model_table.horizontalHeader()
    header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
    header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
    header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
    header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
    header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
    header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
    model_table.horizontalHeader().setStretchLastSection(True)
    model_layout.addWidget(model_table)
    resource_layout.addWidget(model_panel)
    resource_actions = QHBoxLayout()
    open_selected = QPushButton("打开所选资源")
    copy_selected = QPushButton("复制所选路径")
    open_inventory = QPushButton("打开来源总览")
    open_outputs = QPushButton("打开项目输出目录")
    resource_actions.addWidget(open_selected)
    resource_actions.addWidget(copy_selected)
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
                str(value.get("display_name") or value.get("stage_id") or value.get("well_id") or ""),
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

        refresh_model_inventory()

    inventory_rows = []
    visible_inventory_rows = []

    def refresh_model_inventory():
        nonlocal inventory_rows
        inventory_rows = collect_resource_inventory(registry)
        apply_model_filter()

    def apply_model_filter():
        nonlocal visible_inventory_rows
        selected_kind = model_filter.currentText()
        query = model_search.text().strip().lower()
        visible = []
        for row in inventory_rows:
            if selected_kind != "全部资源" and row.get("kind") != selected_kind:
                continue
            haystack = " ".join(str(row.get(key, "")) for key in ("kind", "name", "status", "purpose", "context", "display_path")).lower()
            if query and query not in haystack:
                continue
            visible.append(row)
        visible_inventory_rows = visible
        model_table.setRowCount(len(visible))
        for row_index, row in enumerate(visible):
            values = [
                str(row.get("kind") or ""),
                str(row.get("name") or ""),
                str(row.get("status") or ""),
                str(row.get("purpose") or ""),
                str(row.get("context") or ""),
                str(row.get("size") or ""),
                str(row.get("display_path") or row.get("path") or ""),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(str(row.get("path") or value))
                if column == 6:
                    item.setData(Qt.UserRole, str(row.get("path") or ""))
                model_table.setItem(row_index, column, item)
        if not query and selected_kind == "全部资源":
            counts = {kind: sum(1 for item in inventory_rows if item.get("kind") == kind) for kind in ("模型", "运行结果", "缓存")}
            model_summary.setText(
                f"共 {len(visible)} 项 · 模型 {counts['模型']} · 结果 {counts['运行结果']} · 缓存 {counts['缓存']}"
            )
        else:
            model_summary.setText(f"显示 {len(visible)} / 共 {len(inventory_rows)} 项")
        update_resource_actions()

    def update_resource_actions():
        path = selected_inventory_path()
        available = bool(path and path.exists())
        open_selected.setEnabled(available)
        copy_selected.setEnabled(available)

    def selected_inventory_path() -> Path | None:
        row = model_table.currentRow()
        if row < 0:
            return None
        item = model_table.item(row, 6)
        if item is None:
            return None
        value = str(item.data(Qt.UserRole) or item.text()).strip()
        path = Path(value) if value else None
        return path if path and path.exists() else None

    def open_selected_resource():
        path = selected_inventory_path()
        if path:
            open_path(path)
        else:
            model_summary.setText("请选择包含有效路径的资源")

    def copy_selected_resource():
        path = selected_inventory_path()
        if path:
            QApplication.clipboard().setText(str(path))
            model_summary.setText("已复制所选资源路径")
        else:
            model_summary.setText("当前记录没有可用路径")

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
        if not path.exists():
            return
        target = path if path.is_dir() else path.parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    apply_dataset.clicked.connect(select_dataset)
    dataset_table.doubleClicked.connect(lambda _index: select_dataset())
    refresh_dataset.clicked.connect(refresh_catalog)
    model_filter.currentTextChanged.connect(lambda _value: apply_model_filter())
    model_search.textChanged.connect(lambda _value: apply_model_filter())
    model_refresh.clicked.connect(refresh_model_inventory)
    open_selected.clicked.connect(open_selected_resource)
    copy_selected.clicked.connect(copy_selected_resource)
    model_table.cellDoubleClicked.connect(lambda _row, _column: open_selected_resource())
    model_table.itemSelectionChanged.connect(update_resource_actions)
    open_inventory.clicked.connect(lambda: show_workspace_sources(page, registry, controller, edition="integrated"))
    open_outputs.clicked.connect(lambda: open_path(PATHS.outputs))
    if research_mode:
        open_runs.setEnabled(PATHS.app_runs.exists())
        open_audit.setEnabled((PATHS.app_outputs / "operator_decisions.jsonl").exists())
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
