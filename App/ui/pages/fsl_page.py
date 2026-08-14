from __future__ import annotations

from ..theme import PALETTE
from ..web_view import create_local_html_view
from ..widgets.status_card import MetricCardMixin, Panel


def build_fsl_page(registry):
    from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(20, 16, 20, 16)
    title = QLabel("第一部分 · 专家知识与小样本学习")
    title.setObjectName("pageTitle")
    layout.addWidget(title)
    layout.addWidget(_label("知识图谱、两阶段工况识别与迁移学习。当前页读取冻结产物，不在现场训练。", "subtitle"))
    module = registry.module("fsl")
    summary = registry.summary("fsl")
    metrics = summary.get("window_results", {}).get("4", {})
    cards = QGridLayout()
    cards.addWidget(MetricCardMixin.create("第一层 Macro-F1", _fmt(metrics.get("test_binary_macro_f1")), "正常/异常二分类", PALETTE["blue"]), 0, 0)
    cards.addWidget(MetricCardMixin.create("第二层 Macro-F1", _fmt(metrics.get("test_two_stage_grouped_macro_f1")), "五类段级输出", PALETTE["cyan"]), 0, 1)
    cards.addWidget(MetricCardMixin.create("测试 Accuracy", _pct(metrics.get("test_two_stage_grouped_accuracy")), "段级划分", PALETTE["orange"]), 0, 2)
    cards.addWidget(MetricCardMixin.create("知识规则", "可追溯", "砂堵相关规则与关键词", PALETTE["yellow"]), 0, 3)
    layout.addLayout(cards)
    graph_path = registry.path(module.get("html"))
    graph = create_local_html_view(graph_path, title="知识图谱不可用")
    graph.setMinimumHeight(360)
    layout.addWidget(graph, 1)
    panel, panel_layout = Panel.create("模型口径")
    panel_layout.addWidget(_label("正常/正常工况/主缝延伸不计入异常正类；空白或 ?? 不作为异常正类。标签、规则命中和测试样本量均以注册产物为准。", "notice"))
    layout.addWidget(panel)
    return page


def _label(text, name=None):
    from PySide6.QtWidgets import QLabel

    label = QLabel(text)
    if name:
        label.setObjectName(name)
    label.setWordWrap(True)
    return label


def _fmt(value):
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "--"


def _pct(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "--"
