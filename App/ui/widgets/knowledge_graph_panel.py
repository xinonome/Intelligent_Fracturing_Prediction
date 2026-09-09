"""Knowledge-graph workbench embedded in the first application section."""

from __future__ import annotations

from pathlib import Path
import shutil
from datetime import datetime

from ...core.paths import PATHS
from ...services.knowledge_graph_service import (
    KnowledgeGraphAPIConfig,
    ParsedKnowledgeSource,
    append_expert_correction,
    answer_question,
    load_expert_corrections,
    load_api_config,
    load_saved_knowledge_sources,
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
        QSizePolicy,
        QTabWidget,
        QTextBrowser,
        QTextEdit,
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
    key_box.setPlaceholderText("API Key（可选，界面加密显示）")
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
    api_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    api_panel.setMinimumHeight(142)
    root_layout.addWidget(api_panel)

    graph_path = registry.path(registry.module("fsl").get("html"))
    graph_view = create_local_html_view(graph_path, title="知识图谱")
    graph_view.setMinimumSize(0, 560)
    graph_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    graph_view.setToolTip("图谱支持点击节点、逐步展开、搜索和拖动节点位置")
    graph_panel, graph_layout = Panel.create("知识图谱浏览")
    graph_layout.addWidget(graph_view)
    graph_panel.setMinimumSize(560, 560)
    graph_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

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
    source_status = QLabel("内置知识图谱已加载；可继续导入补充资料")
    source_status.setObjectName("muted")
    source_status.setWordWrap(True)
    source_layout.addWidget(source_status)
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

    correction_panel, correction_layout = Panel.create("专家修正")
    correction_topic = QLineEdit()
    correction_topic.setPlaceholderText("工况或知识主题，例如：砂堵")
    correction_original = QTextEdit()
    correction_original.setPlaceholderText("原知识内容或当前回答（可选）")
    correction_original.setMinimumHeight(90)
    correction_revised = QTextEdit()
    correction_revised.setPlaceholderText("填写经专家确认的修正内容")
    correction_revised.setMinimumHeight(110)
    correction_reason = QLineEdit()
    correction_reason.setPlaceholderText("修正原因或现场依据（可选）")
    correction_expert = QLineEdit()
    correction_expert.setPlaceholderText("专家/审核人（可选）")
    correction_actions = QHBoxLayout()
    load_answer_button = QPushButton("载入当前回答")
    save_correction_button = QPushButton("保存专家修正")
    correction_actions.addWidget(load_answer_button)
    correction_actions.addWidget(save_correction_button)
    correction_actions.addStretch(1)
    correction_history = QListWidget()
    correction_history.setMinimumHeight(120)
    correction_status = QLabel("专家修正保存后将优先参与本地查询和 API 检索。")
    correction_status.setObjectName("muted")
    correction_status.setWordWrap(True)
    correction_layout.addWidget(QLabel("主题"))
    correction_layout.addWidget(correction_topic)
    correction_layout.addWidget(QLabel("原内容"))
    correction_layout.addWidget(correction_original)
    correction_layout.addWidget(QLabel("修正内容"))
    correction_layout.addWidget(correction_revised)
    correction_layout.addWidget(correction_reason)
    correction_layout.addWidget(correction_expert)
    correction_layout.addLayout(correction_actions)
    correction_layout.addWidget(correction_history)
    correction_layout.addWidget(correction_status)
    # Keep the graph tall and nearly square on the left. The import controls
    # stay at the upper-right, with the question panel directly underneath.
    workspace_splitter = QSplitter(Qt.Horizontal)
    workspace_splitter.setChildrenCollapsible(False)
    workspace_splitter.addWidget(graph_panel)
    right_column = QWidget()
    right_layout = QVBoxLayout(right_column)
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.setSpacing(8)
    right_tabs = QTabWidget()
    right_tabs.addTab(qa_panel, "知识问答")
    right_tabs.addTab(source_panel, "资料导入")
    right_tabs.addTab(correction_panel, "专家修正")
    right_layout.addWidget(right_tabs, 1)
    workspace_splitter.addWidget(right_column)
    workspace_splitter.setStretchFactor(0, 0)
    workspace_splitter.setStretchFactor(1, 1)
    workspace_splitter.setSizes([680, 1180])
    root_layout.addWidget(workspace_splitter, 1)

    selected_paths: list[Path] = []
    parsed_sources: list[ParsedKnowledgeSource] = load_saved_knowledge_sources()
    api_worker = None
    parse_worker = None

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
        source_status.setText(f"已选择 {len(selected_paths)} 个补充资料文件；内置图谱始终参与问答")
        refresh_source_list()

    class ParseWorker(QThread):
        result_ready = Signal(object, object)

        def __init__(self, paths: list[Path]):
            super().__init__(root)
            self.paths = list(paths)

        def run(self):
            destination_dir = PATHS.app_outputs / "knowledge_sources"
            destination_dir.mkdir(parents=True, exist_ok=True)
            parsed: list[ParsedKnowledgeSource] = []
            failures: list[str] = []
            for source in self.paths:
                try:
                    parsed_source = parse_knowledge_source(source)
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    destination = destination_dir / f"{stamp}_{source.name}"
                    shutil.copy2(source, destination)
                    parsed_source.path = destination
                    parsed.append(parsed_source)
                except (OSError, ValueError) as exc:
                    failures.append(f"{source.name}：{exc}")
            self.result_ready.emit(parsed, failures)

    def finish_parse(parsed, failures) -> None:
        known = {str(item.path).lower() for item in parsed_sources}
        parsed_sources.extend(item for item in parsed if str(item.path).lower() not in known)
        refresh_source_list()
        if failures:
            source_status.setText("已解析部分资料；未处理：" + "；".join(failures))
        else:
            source_status.setText(f"已解析 {len(parsed_sources)} 个补充资料文件，将与内置图谱共同用于本地和 API 问答。")

    def import_and_parse() -> None:
        nonlocal parse_worker
        if parse_worker is not None and parse_worker.isRunning():
            return
        import_button.setEnabled(False)
        source_status.setText(f"正在解析 {len(selected_paths)} 个资料文件…")
        parse_worker = ParseWorker(list(selected_paths))
        parse_worker.result_ready.connect(finish_parse)
        parse_worker.finished.connect(lambda: import_button.setEnabled(bool(selected_paths)))
        parse_worker.finished.connect(lambda: setattr(root, "_kg_parse_worker", None))
        root._kg_parse_worker = parse_worker
        parse_worker.start()

    def clear_sources() -> None:
        selected_paths.clear()
        parsed_sources.clear()
        refresh_source_list()
        source_status.setText("已清空补充资料；内置知识图谱仍可用于问答")

    def save_api() -> None:
        config = KnowledgeGraphAPIConfig(
            endpoint=endpoint_box.text().strip(),
            model=("deepseek-chat" if "deepseek" in endpoint_box.text().lower()
                   and model_box.text().strip().lower() in {"", "deepseek", "default"}
                   else model_box.text().strip()),
            api_key=key_box.text(),
        )
        try:
            path = save_api_config(config)
        except OSError as exc:
            api_status.setText(f"API 配置保存失败：{exc}")
            return
        api_status.setText("API 配置已保存；密钥已使用当前 Windows 账户加密。")
        model_box.setText(config.model)
        setattr(registry, "_knowledge_api_configured", bool(config.endpoint))

    def ask_local() -> None:
        answer_box.setPlainText(answer_question(question_box.text(), parsed_sources))

    def refresh_correction_history() -> None:
        correction_history.clear()
        for item in reversed(load_expert_corrections()[-50:]):
            correction_history.addItem(
                f"{item.recorded_at} · {item.topic} · {item.corrected_content[:80]}"
            )

    def load_current_answer() -> None:
        if not correction_topic.text().strip():
            correction_topic.setText(question_box.text().strip())
        correction_original.setPlainText(answer_box.toPlainText())
        right_tabs.setCurrentWidget(correction_panel)

    def save_correction() -> None:
        try:
            saved = append_expert_correction(
                correction_topic.text(),
                correction_original.toPlainText(),
                correction_revised.toPlainText(),
                reason=correction_reason.text(),
                expert=correction_expert.text(),
            )
        except (OSError, ValueError) as exc:
            correction_status.setText(f"修正未保存：{exc}")
            return
        correction_status.setText(f"已保存专家修正：{saved.topic}；后续检索将优先采用该内容。")
        correction_original.clear()
        correction_revised.clear()
        correction_reason.clear()
        refresh_correction_history()

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
        answer_box.setPlainText("正在检索内置图谱、规则和已导入资料，并调用 API…")
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
    load_answer_button.clicked.connect(load_current_answer)
    save_correction_button.clicked.connect(save_correction)
    question_box.returnPressed.connect(ask_local)
    refresh_source_list()
    refresh_correction_history()
    root._kg_graph_view = graph_view
    root._kg_parsed_sources = parsed_sources
    return root


__all__ = ["build_knowledge_graph_panel"]
