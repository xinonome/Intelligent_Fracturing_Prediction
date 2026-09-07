from pathlib import Path
from tempfile import TemporaryDirectory

from App.services.knowledge_graph_service import answer_question, parse_knowledge_source


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
