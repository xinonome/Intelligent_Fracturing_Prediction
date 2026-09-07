"""Knowledge-graph workbench embedded in the first application section."""

from __future__ import annotations

from pathlib import Path
import shutil
from datetime import datetime

from ...core.paths import PATHS
from ...services.knowledge_graph_service import (
    KnowledgeGraphAPIConfig,
    ParsedKnowledgeSource,
    answer_question,
    load_api_config,
    parse_knowledge_source,
    request_api_answer,
    save_api_config,
)
from ..web_view import create_local_html_view
from .status_card import Panel


def build_knowledge_graph_panel(registry):
    from PySide6.QtCore import QThread, Qt, Signal
    from PySide6.QtWidgets import (
        QFileDialog,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QPushButton,
        QSplitter,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )

    root = QWidget()
    root_layout = QVBoxLayout(root)
    root_layout.setContentsMargins(8, 8, 8, 8)
    root_layout.setSpacing(8)

    api_config = load_api_config()
    api_panel, api_layout = Panel.create("问答 API 配置")
    api_form = QGridLayout()
    endpoint_box = QLineEdit(api_config.endpoint)
    endpoint_box.setPlaceholderText("OpenAI 兼容接口地址，例如 https://.../v1")
    model_box = QLineEdit(api_config.model)
    model_box.setPlaceholderText("模型名称（可选）")
    key_box = QLineEdit(api_config.api_key)
    key_box.setEchoMode(QLineEdit.Password)
    key_box.setPlaceholderText("API Key（可选，不在界面明文显示）")
    save_api_button = QPushButton("保存 API 配置")
    api_status = QLabel("API 未配置" if not api_config.endpoint else "API 已配置")
    api_status.setObjectName("muted")
    api_status.setWordWrap(True)
    for column, (label, widget) in enumerate(
        (("接口地址", endpoint_box), ("模型", model_box), ("API Key", key_box))
    ):
        title = QLabel(label)
        title.setObjectName("key")
        api_form.addWidget(title, 0, column)
        api_form.addWidget(widget, 1, column)
    api_form.addWidget(save_api_button, 1, 3)
    api_form.setColumnStretch(0, 3)
    api_form.setColumnStretch(1, 1)
    api_form.setColumnStretch(2, 2)
    api_form.setColumnStretch(3, 1)
    api_layout.addLayout(api_form)
    api_layout.addWidget(api_status)
    root_layout.addWidget(api_panel)

    graph_path = registry.path(registry.module("fsl").get("html"))
    graph_view = create_local_html_view(graph_path, title="知识图谱")
    graph_view.setMinimumHeight(440)
    graph_view.setToolTip("图谱支持点击节点、逐步展开、搜索和拖动节点位置")
    graph_panel, graph_layout = Panel.create("知识图谱浏览")
    graph_layout.addWidget(graph_view)
    root_layout.addWidget(graph_panel, 1)

    lower_splitter = QSplitter(Qt.Horizontal)
    source_panel, source_layout = Panel.create("导入数据并解析")
    source_actions = QHBoxLayout()
    choose_button = QPushButton("选择资料")
    import_button = QPushButton("导入并解析")
    clear_button = QPushButton("清空")
    source_actions.addWidget(choose_button)
    source_actions.addWidget(import_button)
    source_actions.addWidget(clear_button)
    source_actions.addStretch(1)
    source_layout.addLayout(source_actions)
    source_list = QListWidget()
    source_list.setMinimumHeight(120)
    source_layout.addWidget(source_list)
    source_status = QLabel("未选择资料")
    source_status.setObjectName("muted")
    source_status.setWordWrap(True)
    source_layout.addWidget(source_status)
    lower_splitter.addWidget(source_panel)

    qa_panel, qa_layout = Panel.create("询问并回答")
    question_box = QLineEdit()
    question_box.setPlaceholderText("例如：每口井出现了哪些工况？")
    question_actions = QHBoxLayout()
    ask_local_button = QPushButton("查询本地知识")
    ask_api_button = QPushButton("调用 API 回答")
    question_actions.addWidget(question_box, 1)
    question_actions.addWidget(ask_local_button)
    question_actions.addWidget(ask_api_button)
    qa_layout.addLayout(question_actions)
    answer_box = QTextBrowser()
    answer_box.setOpenExternalLinks(False)
    answer_box.setPlaceholderText("回答会显示在这里。")
    answer_box.setMinimumHeight(120)
    qa_layout.addWidget(answer_box)
    lower_splitter.addWidget(qa_panel)
    lower_splitter.setStretchFactor(0, 1)
    lower_splitter.setStretchFactor(1, 2)
    root_layout.addWidget(lower_splitter)

    selected_paths: list[Path] = []
    parsed_sources: list[ParsedKnowledgeSource] = []
    api_worker = None

    def refresh_source_list() -> None:
        source_list.clear()
        for source in parsed_sources:
            source_list.addItem(f"{source.path.name} · {source.summary()}")
        import_button.setEnabled(bool(selected_paths))
        ask_local_button.setEnabled(bool(parsed_sources) or True)

    def choose_sources() -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            root,
            "选择知识资料",
            str(PATHS.root),
            "知识资料 (*.json *.csv *.txt *.md *.docx)",
        )
        known = {str(item).lower() for item in selected_paths}
        for raw in paths:
            source = Path(raw).resolve()
            if str(source).lower() not in known:
                selected_paths.append(source)
        source_status.setText(f"已选择 {len(selected_paths)} 个资料文件")
        refresh_source_list()

    def import_and_parse() -> None:
        nonlocal parsed_sources
        destination_dir = PATHS.app_outputs / "knowledge_sources"
        destination_dir.mkdir(parents=True, exist_ok=True)
        parsed: list[ParsedKnowledgeSource] = []
        failures: list[str] = []
        for source in selected_paths:
            try:
                parsed_source = parse_knowledge_source(source)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                destination = destination_dir / f"{stamp}_{source.name}"
                shutil.copy2(source, destination)
                parsed_source.path = destination
                parsed.append(parsed_source)
            except (OSError, ValueError) as exc:
                failures.append(f"{source.name}：{exc}")
        parsed_sources = parsed
        refresh_source_list()
        if failures:
            source_status.setText("已解析部分资料；未处理：" + "；".join(failures))
        else:
            source_status.setText(f"已解析 {len(parsed_sources)} 个资料文件，可用于本地问答。")

    def clear_sources() -> None:
        selected_paths.clear()
        parsed_sources.clear()
        refresh_source_list()
        source_status.setText("已清空")

    def save_api() -> None:
        config = KnowledgeGraphAPIConfig(
            endpoint=endpoint_box.text().strip(),
            model=model_box.text().strip(),
            api_key=key_box.text(),
        )
        try:
            path = save_api_config(config)
        except OSError as exc:
            api_status.setText(f"API 配置保存失败：{exc}")
            return
        api_status.setText(f"API 配置已保存到 {path}；API Key 仅用于实际调用。")

    def ask_local() -> None:
        answer_box.setPlainText(answer_question(question_box.text(), parsed_sources))

    class APIWorker(QThread):
        result_ready = Signal(str)
        error = Signal(str)

        def __init__(self, question: str, config: KnowledgeGraphAPIConfig, sources: list[ParsedKnowledgeSource]):
            super().__init__(root)
            self.question = question
            self.config = config
            self.sources = sources

        def run(self):
            try:
                self.result_ready.emit(request_api_answer(self.question, self.config, self.sources))
            except Exception as exc:  # network and endpoint-specific failures
                self.error.emit(str(exc))

    def ask_api() -> None:
        nonlocal api_worker
        config = KnowledgeGraphAPIConfig(
            endpoint=endpoint_box.text().strip(),
            model=model_box.text().strip(),
            api_key=key_box.text(),
        )
        if not config.endpoint:
            api_status.setText("API 未配置")
            return
        ask_api_button.setEnabled(False)
        answer_box.setPlainText("正在调用 API…")
        api_worker = APIWorker(question_box.text(), config, list(parsed_sources))
        api_worker.result_ready.connect(answer_box.setPlainText)
        api_worker.error.connect(lambda message: answer_box.setPlainText(f"API 调用失败：{message}"))
        api_worker.finished.connect(lambda: ask_api_button.setEnabled(True))
        api_worker.finished.connect(lambda: setattr(root, "_kg_api_worker", None))
        root._kg_api_worker = api_worker
        api_worker.start()

    choose_button.clicked.connect(choose_sources)
    import_button.clicked.connect(import_and_parse)
    clear_button.clicked.connect(clear_sources)
    save_api_button.clicked.connect(save_api)
    ask_local_button.clicked.connect(ask_local)
    ask_api_button.clicked.connect(ask_api)
    question_box.returnPressed.connect(ask_local)
    refresh_source_list()
    root._kg_graph_view = graph_view
    root._kg_parsed_sources = parsed_sources
    return root


__all__ = ["build_knowledge_graph_panel"]
