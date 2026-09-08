from pathlib import Path
from tempfile import TemporaryDirectory

import json

from App.services.knowledge_graph_service import (
    KnowledgeGraphAPIConfig,
    append_expert_correction,
    answer_question,
    build_knowledge_context,
    load_expert_corrections,
    parse_knowledge_source,
    request_api_advisory,
    request_api_answer,
)


ROOT = Path(__file__).resolve().parents[2]


def test_shipped_knowledge_graph_json_is_parsed():
    source = ROOT / "FSL-Expert" / "knowledge_graph" / "full_book_qwen_output" / "kg_full_book_qwen.json"

    parsed = parse_knowledge_source(source)

    assert parsed.entity_count > 0
    assert parsed.relation_count > 0
    assert parsed.file_type == "JSON 知识数据"


def test_imported_text_is_available_to_local_question_answering():
    with TemporaryDirectory() as directory:
        source = Path(directory) / "notes.txt"
        source.write_text("压裂施工过程中应关注施工压力、排量和砂比的连续变化。", encoding="utf-8")
        parsed = parse_knowledge_source(source)

    answer = answer_question("施工压力和排量需要关注什么？", [parsed])

    assert "施工压力" in answer


def test_shipped_qa_preset_is_available_without_api():
    answer = answer_question("每口井出现了哪些工况？")

    assert "主缝延伸" in answer


def test_graph_node_question_returns_definition_relations_and_rules_without_import():
    answer = answer_question("砂堵")

    assert "砂子堵塞井筒或裂缝" in answer
    assert "图谱关联关系" in answer
    assert "内置规则与处置" in answer


def test_api_context_contains_builtin_graph_when_no_supplementary_file(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "已根据图谱回答"}}]}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    answer = request_api_answer(
        "砂堵是什么？",
        KnowledgeGraphAPIConfig(endpoint="https://api.deepseek.com", model="deepseek-chat", api_key="test"),
    )

    sent = captured["body"]["messages"][1]["content"]
    assert answer == "已根据图谱回答"
    assert "内置知识图谱节点" in sent
    assert "砂子堵塞井筒或裂缝" in sent
    assert "内置规则与处置" in sent


def test_imported_evidence_is_merged_with_builtin_graph(tmp_path):
    source = tmp_path / "supplement.txt"
    source.write_text("现场砂堵复核时应同步核对压力突升和排量变化。", encoding="utf-8")
    parsed = parse_knowledge_source(source)

    context = build_knowledge_context("砂堵如何复核？", [parsed])

    assert "内置知识图谱节点" in context
    assert "已导入资料" in context
    assert "压力突升" in context


def test_expert_correction_is_persisted_and_precedes_builtin_knowledge(tmp_path, monkeypatch):
    correction_path = tmp_path / "corrections.jsonl"
    monkeypatch.setattr(
        "App.services.knowledge_graph_service.EXPERT_CORRECTIONS_PATH", correction_path
    )
    append_expert_correction(
        "砂堵",
        "原处置内容",
        "现场确认后应先稳排量并核对压力响应，再决定是否降砂。",
        reason="适用于本区块设备响应",
    )

    rows = load_expert_corrections()
    answer = answer_question("砂堵怎么处理？")

    assert len(rows) == 1
    assert "专家修正（优先采用）" in answer
    assert "先稳排量" in answer
    assert answer.index("专家修正") < answer.index("内置知识图谱")


def test_api_advisory_receives_state_and_graph_evidence(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "建议先人工核对压力趋势"}}]}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    answer = request_api_advisory(
        {"condition": "砂堵", "risk_level": "high", "current_flow_m3_min": 12.0},
        KnowledgeGraphAPIConfig(endpoint="https://api.deepseek.com", model="deepseek-chat", api_key="test"),
    )

    sent = captured["body"]["messages"][1]["content"]
    assert "建议先人工核对压力趋势" in answer
    assert '"condition": "砂堵"' in sent
    assert "内置知识图谱" in sent
