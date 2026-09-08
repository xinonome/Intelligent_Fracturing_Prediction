"""Application-facing table import and source preflight page."""

from __future__ import annotations

from pathlib import Path

from ...services.data_import_service import ImportInspection, import_tables, inspect_table
from ..widgets.status_card import Panel


def build_data_import_page(registry, on_imported=None):
    from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
    from PySide6.QtWidgets import (
        QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QTableWidget,
        QTableWidgetItem, QVBoxLayout, QWidget,
    )

    page = QScrollArea()
    page.setObjectName("dataImportPageScroll")
    page.setWidgetResizable(True)
    content = QWidget()
    page.setWidget(content)
    layout = QVBoxLayout(content)
    layout.setContentsMargins(20, 16, 20, 18)
    layout.setSpacing(10)

    title = QLabel("数据导入")
    title.setObjectName("pageTitle")
    layout.addWidget(title)
    operations, operations_layout = Panel.create("新增施工表")
    operation_row = QHBoxLayout()
    choose = QPushButton("选择表格")
    remove = QPushButton("移除选中")
    clear = QPushButton("清空列表")
    import_button = QPushButton("导入可识别文件")
    import_button.setObjectName("importTablesButton")
    import_button.setEnabled(False)
    for button in (choose, remove, clear, import_button):
        operation_row.addWidget(button)
    operation_row.addStretch(1)
    operations_layout.addLayout(operation_row)
    txt_hint = QLabel(
        "TXT：支持原有九列、逗号或制表符分隔，可带中文表头。类型列填写数字编码，时间须为完整日期和时分秒且严格递增。\n"
        "请逐文件填写真实井名/井段（TXT 必填，不从文件名推断）；Excel/CSV 沿用原字段，不被这些输入覆盖。"
    )
    txt_hint.setWordWrap(True)
    operations_layout.addWidget(txt_hint)
    layout.addWidget(operations)

    files_panel, files_layout = Panel.create("待导入文件与字段检查")
    files_table = QTableWidget(0, 7)
    files_table.setObjectName("importFilesTable")
    files_table.setHorizontalHeaderLabels(["文件", "格式", "识别结果", "可用场景", "真实井名（TXT）", "井段（TXT）", "检查说明"])
    files_table.setSelectionBehavior(QTableWidget.SelectRows)
    files_table.setSelectionMode(QTableWidget.ExtendedSelection)
    files_table.setEditTriggers(QTableWidget.NoEditTriggers)
    files_table.setAlternatingRowColors(True)
    files_table.horizontalHeader().setStretchLastSection(True)
    files_table.setColumnWidth(0, 290)
    files_table.setColumnWidth(1, 80)
    files_table.setColumnWidth(2, 150)
    files_table.setColumnWidth(3, 220)
    files_table.setColumnWidth(4, 170)
    files_table.setColumnWidth(5, 130)
    files_table.setMinimumHeight(260)
    files_layout.addWidget(files_table)
    layout.addWidget(files_panel)

    status = QLabel("尚未选择表格。")
    status.setObjectName("notice")
    status.setWordWrap(True)
    layout.addWidget(status)

    layout.addStretch(1)

    selected: list[Path] = []
    inspections: dict[str, ImportInspection] = {}
    metadata: dict[str, dict[str, str]] = {}
    operation_thread = None
    operation_worker = None
    operation_sources: list[Path] = []
    operation_poll_timer = QTimer(page)
    operation_poll_timer.setInterval(50)

    class ImportWorker(QObject):
        done = Signal()

        def __init__(self, mode: str, sources: list[Path], metadata_snapshot=None):
            super().__init__()
            self.mode = mode
            self.sources = list(sources)
            self.metadata_snapshot = dict(metadata_snapshot or {})
            self.result = None
            self.error = ""

        @Slot()
        def run(self):
            try:
                if self.mode == "inspect":
                    self.result = [(source, inspect_table(source)) for source in self.sources]
                else:
                    self.result = import_tables(self.sources, metadata_by_path=self.metadata_snapshot)
            except Exception as exc:
                self.error = str(exc)
            finally:
                self.done.emit()

    def key(path: Path) -> str:
        return str(path.resolve()).lower()

    def update_import_button() -> None:
        import_button.setEnabled(any(
            inspections[key(source)].can_import and (
                source.suffix.lower() != ".txt" or all(
                    metadata.get(key(source), {}).get(field, "").strip() for field in ("well_name", "stage_id")
                )
            ) for source in selected if key(source) in inspections
        ))

    def save_identity(source_key: str, field: str, value: str) -> None:
        metadata.setdefault(source_key, {})[field] = value
        update_import_button()

    def refresh_table() -> None:
        files_table.setRowCount(0)
        valid_count = 0
        for source in selected:
            inspection = inspections.get(key(source))
            row = files_table.rowCount()
            files_table.insertRow(row)
            if inspection is None:
                values = (source.name, source.suffix.lower(), "检查中…", "", "", "")
                for column, value in enumerate(values):
                    files_table.setItem(row, column, QTableWidgetItem(str(value)))
                files_table.setItem(row, 6, QTableWidgetItem("正在读取字段"))
                if source.suffix.lower() == ".txt":
                    source_key = key(source)
                    for column, field, placeholder in ((4, "well_name", "必填：真实井名"), (5, "stage_id", "必填：井段")):
                        edit = QLineEdit()
                        edit.setObjectName(f"txt_{field}_{row}")
                        edit.setPlaceholderText(placeholder)
                        edit.setText(metadata.get(source_key, {}).get(field, ""))
                        edit.textChanged.connect(
                            lambda value, source_key=source_key, field=field: save_identity(source_key, field, value)
                        )
                        files_table.setCellWidget(row, column, edit)
                continue
            values = (
                source.name,
                inspection.extension or "",
                inspection.field_profile,
                "、".join(inspection.suggested_scenarios) if inspection.suggested_scenarios else "",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                files_table.setItem(row, column, item)
            reason_item = QTableWidgetItem(inspection.reason)
            reason_item.setToolTip(inspection.reason)
            files_table.setItem(row, 6, reason_item)
            if inspection.extension == ".txt":
                source_key = key(source)
                for column, field, placeholder in ((4, "well_name", "必填：真实井名"), (5, "stage_id", "必填：井段")):
                    edit = QLineEdit()
                    edit.setObjectName(f"txt_{field}_{row}")
                    edit.setPlaceholderText(placeholder)
                    edit.setText(metadata.get(source_key, {}).get(field, ""))
                    edit.textChanged.connect(lambda value, source_key=source_key, field=field: save_identity(source_key, field, value))
                    files_table.setCellWidget(row, column, edit)
            else:
                for column in (4, 5):
                    files_table.setItem(row, column, QTableWidgetItem("沿用原表字段"))
            if inspection.can_import:
                valid_count += 1
        update_import_button()
        status.setText(f"已选择 {len(selected)} 个文件，其中 {valid_count} 个通过检查。TXT 需填写真实井名和井段后导入。")

    def choose_files() -> None:
        import_demo_dir = registry.path("Data/import_demo")
        initial_dir = import_demo_dir if import_demo_dir and import_demo_dir.exists() else registry.path("Data/raw_frac")
        paths, _ = QFileDialog.getOpenFileNames(
            page,
            "选择施工表格",
            str(initial_dir or Path.cwd()),
            "施工表格 (*.xlsx *.xls *.csv *.txt)",
        )
        for raw in paths:
            source = Path(raw).resolve()
            if key(source) not in {key(item) for item in selected}:
                selected.append(source)
        refresh_table()
        start_inspection([source for source in selected if key(source) not in inspections])

    def remove_selected() -> None:
        rows = sorted({item.row() for item in files_table.selectedItems()}, reverse=True)
        for row in rows:
            if 0 <= row < len(selected):
                inspections.pop(key(selected[row]), None)
                metadata.pop(key(selected[row]), None)
                selected.pop(row)
        refresh_table()

    def clear_selected() -> None:
        selected.clear()
        inspections.clear()
        metadata.clear()
        refresh_table()

    def do_import() -> None:
        valid = []
        for source in selected:
            inspection = inspections.get(key(source))
            if inspection is not None and inspection.can_import:
                valid.append(source)
        if not valid:
            status.setText("没有通过基础检查的文件可导入。")
            return
        set_busy(True, "正在导入并建立数据索引…")
        start_worker("import", valid, dict(metadata))

    def apply_import_result(result) -> None:
        imported_names = "、".join(path.name for path in result.imported) or "无"
        rejected_names = "；".join(f"{path.name}：{reason}" for path, reason in result.rejected)
        message = f"已导入 {len(result.imported)} 个文件：{imported_names}。"
        if rejected_names:
            message += f" 未导入：{rejected_names}。"
        rejected_keys = {key(path) for path, _ in result.rejected}
        for source in list(operation_sources):
            if key(source) not in rejected_keys:
                selected.remove(source)
                inspections.pop(key(source), None)
                metadata.pop(key(source), None)
        for source, _reason in result.rejected:
            inspections.pop(key(source), None)
        refresh_table()
        status.setText(message)
        if result.imported and on_imported:
            on_imported()

    def set_busy(busy: bool, message: str = "") -> None:
        choose.setEnabled(not busy)
        remove.setEnabled(not busy)
        clear.setEnabled(not busy)
        files_table.setEnabled(not busy)
        if busy:
            import_button.setEnabled(False)
            status.setText(message)
        else:
            update_import_button()

    def start_inspection(sources: list[Path]) -> None:
        if not sources:
            return
        set_busy(True, f"正在检查 {len(sources)} 个文件的字段与数据格式…")
        start_worker("inspect", sources)

    def apply_inspections(results) -> None:
        for source, inspection in results:
            inspections[key(source)] = inspection
        refresh_table()

    def worker_failed(message: str) -> None:
        status.setText(f"操作失败：{message}")

    def worker_finished() -> None:
        nonlocal operation_thread, operation_worker, operation_sources
        set_busy(False)
        if operation_thread is not None:
            operation_thread.deleteLater()
        operation_thread = None
        operation_worker = None
        operation_sources = []

    def poll_worker() -> None:
        if operation_thread is None or operation_worker is None or operation_thread.isRunning():
            return
        operation_poll_timer.stop()
        if operation_worker.error:
            worker_failed(operation_worker.error)
        elif operation_worker.mode == "inspect":
            apply_inspections(operation_worker.result or [])
        else:
            apply_import_result(operation_worker.result)
        worker_finished()

    def start_worker(mode: str, sources: list[Path], metadata_snapshot=None) -> None:
        nonlocal operation_thread, operation_worker, operation_sources
        if operation_thread is not None:
            return
        operation_sources = list(sources)
        # Tiny files complete faster than thread startup and keeping this path
        # synchronous preserves immediate field feedback.  Large workbooks,
        # which caused the reported UI freeze, always use the worker thread.
        total_size = sum(source.stat().st_size for source in sources if source.exists())
        if total_size <= 64 * 1024:
            try:
                if mode == "inspect":
                    apply_inspections([(source, inspect_table(source)) for source in sources])
                else:
                    apply_import_result(import_tables(sources, metadata_by_path=dict(metadata_snapshot or {})))
            except Exception as exc:
                worker_failed(str(exc))
            finally:
                operation_sources = []
                set_busy(False)
            return
        operation_thread = QThread(page)
        operation_worker = ImportWorker(mode, sources, metadata_snapshot)
        operation_worker.moveToThread(operation_thread)
        operation_thread.started.connect(operation_worker.run)
        operation_worker.done.connect(operation_thread.quit)
        operation_thread.start()
        operation_poll_timer.start()

    def refresh_catalog() -> None:
        """Refresh pending inspections without losing user-entered identities."""
        inspections.clear()
        refresh_table()
        start_inspection(list(selected))

    def set_global_dataset(dataset_id: str) -> None:
        # This page owns no timeline/player. Switching never discards pending
        # imports or infers well/stage metadata from the selected dataset ID.
        page.setProperty("globalDatasetId", dataset_id)

    page.refresh_catalog = refresh_catalog
    page.set_global_dataset = set_global_dataset
    page.active_import_thread = lambda: operation_thread
    operation_poll_timer.timeout.connect(poll_worker)

    choose.clicked.connect(choose_files)
    remove.clicked.connect(remove_selected)
    clear.clicked.connect(clear_selected)
    import_button.clicked.connect(do_import)
    refresh_table()
    return page


__all__ = ["build_data_import_page"]
