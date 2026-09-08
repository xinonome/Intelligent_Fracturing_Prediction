"""Local knowledge-graph parsing, API settings and question answering."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import base64
import ctypes
from datetime import datetime
import json
from functools import lru_cache
import os
from pathlib import Path
import re
import urllib.error
import urllib.request
import uuid
import zipfile
import xml.etree.ElementTree as ET

from ..core.paths import PATHS


API_CONFIG_PATH = PATHS.app_outputs / "knowledge_graph_api.json"
EXPERT_CORRECTIONS_PATH = PATHS.app_outputs / "knowledge_graph_expert_corrections.jsonl"
KNOWLEDGE_ADVISORY_LOG_PATH = PATHS.app_outputs / "knowledge_agent_advisories.jsonl"
QA_PATH = PATHS.root / "FSL-Expert" / "knowledge_graph" / "qa.json"
KNOWLEDGE_GRAPH_PATH = (
    PATHS.root
    / "FSL-Expert"
    / "knowledge_graph"
    / "full_book_qwen_output"
    / "kg_full_book_qwen.json"
)
SUPPORTED_KNOWLEDGE_EXTENSIONS = {".json", ".csv", ".txt", ".md", ".docx"}


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_ulong), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _protect_secret(value: str) -> str:
    """Protect a local API key with the current Windows user credentials."""

    if not value or os.name != "nt":
        return ""
    raw = value.encode("utf-8")
    buffer = ctypes.create_string_buffer(raw)
    source = _DataBlob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    target = _DataBlob()
    if not ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(source), None, None, None, None, 0, ctypes.byref(target)
    ):
        raise OSError("Windows 无法加密 API Key")
    try:
        encrypted = ctypes.string_at(target.pbData, target.cbData)
        return base64.b64encode(encrypted).decode("ascii")
    finally:
        ctypes.windll.kernel32.LocalFree(target.pbData)


def _unprotect_secret(value: str) -> str:
    if not value or os.name != "nt":
        return ""
    try:
        raw = base64.b64decode(value)
        buffer = ctypes.create_string_buffer(raw)
        source = _DataBlob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
        target = _DataBlob()
        if not ctypes.windll.crypt32.CryptUnprotectData(
            ctypes.byref(source), None, None, None, None, 0, ctypes.byref(target)
        ):
            return ""
        try:
            return ctypes.string_at(target.pbData, target.cbData).decode("utf-8")
        finally:
            ctypes.windll.kernel32.LocalFree(target.pbData)
    except (ValueError, OSError, UnicodeDecodeError):
        return ""


@dataclass
class KnowledgeGraphAPIConfig:
    endpoint: str = ""
    model: str = ""
    api_key: str = ""
    timeout_s: int = 20


@dataclass
class ParsedKnowledgeSource:
    path: Path
    file_type: str
    record_count: int = 0
    entity_count: int = 0
    relation_count: int = 0
    fields: list[str] | None = None
    preview: str = ""
    text: str = ""

    def summary(self) -> str:
        details = [self.file_type, f"记录 {self.record_count} 条"]
        if self.entity_count or self.relation_count:
            details.append(f"实体 {self.entity_count} 个 / 关系 {self.relation_count} 条")
        if self.fields:
            details.append("字段：" + "、".join(self.fields[:8]))
        return " · ".join(details)

    def as_dict(self) -> dict:
        value = asdict(self)
        value["path"] = str(self.path)
        return value


@dataclass
class ExpertCorrection:
    record_id: str
    recorded_at: str
    topic: str
    original_content: str
    corrected_content: str
    reason: str = ""
    expert: str = ""
    status: str = "active"

    def as_dict(self) -> dict:
        return asdict(self)


def load_api_config(path: Path | None = None) -> KnowledgeGraphAPIConfig:
    selected = path or API_CONFIG_PATH
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return KnowledgeGraphAPIConfig()
    if not isinstance(payload, dict):
        return KnowledgeGraphAPIConfig()
    try:
        timeout_s = int(payload.get("timeout_s") or 20)
    except (TypeError, ValueError):
        timeout_s = 20
    api_key = os.environ.get("IFP_KG_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
    if not api_key:
        api_key = _unprotect_secret(str(payload.get("api_key_protected") or ""))
    if not api_key:
        # Backward compatibility: a legacy plaintext value is read once and
        # will be replaced by DPAPI ciphertext on the next save.
        api_key = str(payload.get("api_key") or "")
    return KnowledgeGraphAPIConfig(
        endpoint=str(payload.get("endpoint") or "").strip(),
        model=str(payload.get("model") or "").strip(),
        api_key=api_key,
        timeout_s=max(3, min(timeout_s, 120)),
    )


def save_api_config(config: KnowledgeGraphAPIConfig, path: Path | None = None) -> Path:
    selected = path or API_CONFIG_PATH
    selected.parent.mkdir(parents=True, exist_ok=True)
    selected.write_text(
        json.dumps(
            {
                "endpoint": config.endpoint.strip(),
                "model": config.model.strip(),
                "api_key_protected": _protect_secret(config.api_key),
                "timeout_s": int(config.timeout_s),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return selected


def parse_knowledge_source(path: str | Path) -> ParsedKnowledgeSource:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("文件不存在")
    if source.suffix.lower() not in SUPPORTED_KNOWLEDGE_EXTENSIONS:
        raise ValueError("支持 JSON、CSV、TXT、MD 和 DOCX 文件")
    if source.suffix.lower() == ".json":
        return _parse_json(source)
    if source.suffix.lower() == ".csv":
        return _parse_csv(source)
    if source.suffix.lower() == ".docx":
        text = _read_docx_text(source)
        return _parse_text(source, text, "DOCX 文本")
    text = _read_text(source)
    return _parse_text(source, text, "文本资料")


def load_saved_knowledge_sources(directory: Path | None = None) -> list[ParsedKnowledgeSource]:
    """Reload supplementary sources copied into the application output area."""

    selected = directory or (PATHS.app_outputs / "knowledge_sources")
    if not selected.is_dir():
        return []
    parsed: list[ParsedKnowledgeSource] = []
    for source in sorted(selected.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if not source.is_file() or source.suffix.lower() not in SUPPORTED_KNOWLEDGE_EXTENSIONS:
            continue
        try:
            parsed.append(parse_knowledge_source(source))
        except (OSError, ValueError):
            continue
    return parsed


def load_expert_corrections(path: Path | None = None) -> list[ExpertCorrection]:
    selected = path or EXPERT_CORRECTIONS_PATH
    if not selected.is_file():
        return []
    rows: list[ExpertCorrection] = []
    try:
        lines = selected.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            correction = ExpertCorrection(
                record_id=str(payload.get("record_id") or ""),
                recorded_at=str(payload.get("recorded_at") or ""),
                topic=str(payload.get("topic") or "").strip(),
                original_content=str(payload.get("original_content") or "").strip(),
                corrected_content=str(payload.get("corrected_content") or "").strip(),
                reason=str(payload.get("reason") or "").strip(),
                expert=str(payload.get("expert") or "").strip(),
                status=str(payload.get("status") or "active"),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if correction.record_id and correction.topic and correction.corrected_content:
            rows.append(correction)
    return rows


def append_expert_correction(
    topic: str,
    original_content: str,
    corrected_content: str,
    *,
    reason: str = "",
    expert: str = "",
    path: Path | None = None,
) -> ExpertCorrection:
    topic = str(topic or "").strip()
    corrected_content = str(corrected_content or "").strip()
    if not topic:
        raise ValueError("请填写需要修正的工况或知识主题")
    if not corrected_content:
        raise ValueError("请填写修正后的知识内容")
    correction = ExpertCorrection(
        record_id=str(uuid.uuid4()),
        recorded_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        topic=topic,
        original_content=str(original_content or "").strip(),
        corrected_content=corrected_content,
        reason=str(reason or "").strip(),
        expert=str(expert or "").strip(),
    )
    selected = path or EXPERT_CORRECTIONS_PATH
    selected.parent.mkdir(parents=True, exist_ok=True)
    with selected.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(correction.as_dict(), ensure_ascii=False) + "\n")
    return correction


def answer_question(question: str, sources: list[ParsedKnowledgeSource] | None = None) -> str:
    """Answer from built-in QA, graph nodes/relations/rules and imported text."""

    prompt = str(question or "").strip()
    if not prompt:
        return "请输入问题。"
    correction_context = _expert_correction_context(prompt)
    if correction_context:
        supporting = build_knowledge_context(prompt, sources, include_corrections=False)
        return correction_context + ("\n\n" + supporting if supporting else "")
    qa = _load_qa()
    normalized = _normalize(prompt)
    for item in qa.get("presets", []) or []:
        if not isinstance(item, dict):
            continue
        known = str(item.get("question") or "")
        if normalized == _normalize(known) or normalized in _normalize(known) or _normalize(known) in normalized:
            return str(item.get("answer") or "未找到答案。")

    segment_answer = _answer_segment_statistic(prompt, qa)
    if segment_answer:
        return segment_answer

    context = build_knowledge_context(prompt, sources)
    if context:
        return context
    return "本地知识库中未找到与该问题直接匹配的内容。可导入相关资料，或在 API 配置中填写可用的问答接口后重试。"


def build_knowledge_context(
    question: str,
    sources: list[ParsedKnowledgeSource] | None = None,
    *,
    max_chars: int = 12000,
    include_corrections: bool = True,
) -> str:
    """Retrieve compact evidence from every local knowledge source.

    The returned text is shared by local answering and external API calls, so
    both buttons operate on the same evidence. Only matching excerpts are
    returned; the complete graph or complete imported documents are never
    uploaded wholesale.
    """

    prompt = str(question or "").strip()
    if not prompt:
        return ""
    terms = _question_terms(prompt)
    normalized = _normalize(prompt)
    sections: list[str] = []

    if include_corrections:
        correction_context = _expert_correction_context(prompt)
        if correction_context:
            sections.append(correction_context)

    qa_matches = []
    for item in _load_qa().get("presets", []) or []:
        if not isinstance(item, dict):
            continue
        known = str(item.get("question") or "")
        answer = str(item.get("answer") or "")
        searchable = _normalize(known + answer)
        if searchable and (any(term in searchable for term in terms) or normalized in searchable):
            qa_matches.append(f"问：{known}\n答：{answer}")
    if qa_matches:
        sections.append("【内置问答库】\n" + "\n\n".join(qa_matches[:3]))

    graph = _load_graph()
    nodes = graph.get("nodes", []) if isinstance(graph.get("nodes"), list) else []
    edges = graph.get("edges", []) if isinstance(graph.get("edges"), list) else []
    rules = graph.get("rules", []) if isinstance(graph.get("rules"), list) else []
    by_id = {str(node.get("id")): node for node in nodes if isinstance(node, dict) and node.get("id") is not None}

    matched_nodes = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        label = str(node.get("label") or "")
        title = _plain_graph_text(node.get("title"))
        searchable = (label + " " + title + " " + str(node.get("type") or "")).lower()
        hits = sum(term in searchable for term in terms)
        normalized_label = _normalize(label)
        if hits or (normalized_label and normalized_label in normalized) or (normalized and normalized in _normalize(searchable)):
            exact = int(_normalize(label) == normalized or label.lower() in terms)
            matched_nodes.append((exact, hits, len(label), node))
    matched_nodes.sort(key=lambda item: (-item[0], -item[1], item[2]))
    selected_nodes = [item[3] for item in matched_nodes[:6]]
    graph_terms = list(terms)
    for node in selected_nodes:
        label_term = str(node.get("label") or "").lower()
        if label_term and label_term not in graph_terms:
            graph_terms.append(label_term)
    if selected_nodes:
        node_lines = []
        for node in selected_nodes:
            description = _plain_graph_text(node.get("title"))
            node_lines.append(
                f"- {node.get('label')}（{node.get('type') or '知识节点'}）"
                + (f"：{description}" if description else "")
            )
        sections.append("【内置知识图谱节点】\n" + "\n".join(node_lines))

        selected_ids = {str(node.get("id")) for node in selected_nodes}
        relation_rows = []
        priority = {"表现为": 0, "原因是": 1, "导致": 2, "监测指标": 3, "处置措施": 4}
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source_id, target_id = str(edge.get("from")), str(edge.get("to"))
            if source_id not in selected_ids and target_id not in selected_ids:
                continue
            source_node, target_node = by_id.get(source_id), by_id.get(target_id)
            if not source_node or not target_node:
                continue
            relation = str(edge.get("label") or "相关")
            row = f"- {source_node.get('label')} —{relation}→ {target_node.get('label')}"
            # Prefer operational/symptom relations over a long list of wells.
            other = target_node if source_id in selected_ids else source_node
            rank = priority.get(relation, 8) + (6 if other.get("type") == "井" else 0)
            relation_rows.append((rank, row))
        relation_lines = []
        for _rank, row in sorted(relation_rows, key=lambda item: item[0]):
            if row not in relation_lines:
                relation_lines.append(row)
            if len(relation_lines) >= 12:
                break
        if relation_lines:
            sections.append("【图谱关联关系】\n" + "\n".join(relation_lines))

    rule_matches = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        searchable = " ".join(str(rule.get(key) or "") for key in ("condition", "conclusion", "action", "evidence")).lower()
        hits = sum(term in searchable for term in graph_terms)
        if not hits:
            continue
        line = (
            f"- 条件：{rule.get('condition') or ''}；结论：{rule.get('conclusion') or ''}；"
            f"处置：{rule.get('action') or ''}"
        )
        if rule.get("page") is not None:
            line += f"；来源页：{rule.get('page')}"
        rule_matches.append((hits, line))
    if rule_matches:
        sections.append(
            "【内置规则与处置】\n"
            + "\n".join(line for _score, line in sorted(rule_matches, key=lambda item: -item[0])[:8])
        )

    imported = []
    for source in sources or []:
        text = source.text or source.preview or ""
        if text and (not terms or any(term in text.lower() for term in terms)):
            imported.append(f"【{source.path.name}】\n{_first_matching_snippet(text, terms)}")
    if imported:
        sections.append("【已导入资料】\n" + "\n\n".join(imported[:5]))

    return "\n\n".join(sections)[:max(1000, int(max_chars))]


def request_api_answer(
    question: str,
    config: KnowledgeGraphAPIConfig,
    sources: list[ParsedKnowledgeSource] | None = None,
) -> str:
    """Call an OpenAI-compatible chat endpoint only when the user configures it."""

    endpoint = config.endpoint.strip()
    if not endpoint:
        raise ValueError("尚未配置 API 地址")
    url = endpoint if endpoint.endswith("/chat/completions") else endpoint.rstrip("/") + "/chat/completions"
    context = build_knowledge_context(question, sources, max_chars=12000)
    configured_model = config.model.strip()
    if "deepseek" in endpoint.lower() and configured_model.lower() in {"", "default", "deepseek"}:
        configured_model = "deepseek-chat"
    body = {
        "model": configured_model or "default",
        "messages": [
            {
                "role": "system",
                "content": "你是压裂施工知识图谱问答助手。请根据检索到的内置知识图谱、规则、问答库和用户导入资料回答，并说明主要依据；资料不足时明确说明。",
            },
            {"role": "user", "content": f"问题：{question.strip()}\n\n检索到的本地知识：\n{context or '没有检索到相关本地知识。'}"},
        ],
        "temperature": 0.2,
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=max(3, int(config.timeout_s))) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"API 返回 HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"API 连接失败：{exc.reason if hasattr(exc, 'reason') else exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("API 返回内容不是有效 JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("API 返回结构无法识别")
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict) and message.get("content"):
            return str(message["content"]).strip()
    for key in ("answer", "output", "text"):
        if payload.get(key):
            return str(payload[key]).strip()
    raise RuntimeError("API 返回中没有可显示的回答")


def request_api_advisory(
    state: dict,
    config: KnowledgeGraphAPIConfig,
    sources: list[ParsedKnowledgeSource] | None = None,
) -> str:
    """Generate a knowledge-grounded advisory without claiming field execution."""

    condition = str(state.get("condition") or "").strip()
    risk = str(state.get("risk_level") or "").strip()
    if not condition or condition in {"unknown", "未标注", "\\"}:
        condition = "当前工况尚未标注"
    retrieval_query = " ".join(item for item in (condition, risk, "处置 调控建议") if item)
    evidence = build_knowledge_context(retrieval_query, sources, max_chars=12000)
    endpoint = config.endpoint.strip()
    if not endpoint:
        raise ValueError("尚未配置 API 地址")
    url = endpoint if endpoint.endswith("/chat/completions") else endpoint.rstrip("/") + "/chat/completions"
    configured_model = config.model.strip()
    if "deepseek" in endpoint.lower() and configured_model.lower() in {"", "default", "deepseek"}:
        configured_model = "deepseek-chat"
    body = {
        "model": configured_model or "default",
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是压裂施工知识增强高层决策助手。只能依据给定状态和本地知识提出人工审核建议，"
                    "不得声称已经控制设备。专家修正的优先级高于内置图谱。请依次输出：当前判断、"
                    "调控建议、知识依据、需要人工核对的事项。资料不足时明确说明，不编造数值。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "当前结构化状态：\n"
                    + json.dumps(state, ensure_ascii=False, indent=2)
                    + "\n\n检索到的本地知识：\n"
                    + (evidence or "没有检索到与当前工况直接相关的知识。")
                ),
            },
        ],
        "temperature": 0.1,
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=max(3, int(config.timeout_s))) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"API 返回 HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"API 连接失败：{exc.reason if hasattr(exc, 'reason') else exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("API 返回内容不是有效 JSON") from exc
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict) and message.get("content"):
                return str(message["content"]).strip()
        for key in ("answer", "output", "text"):
            if payload.get(key):
                return str(payload[key]).strip()
    raise RuntimeError("API 返回中没有可显示的建议")


def append_knowledge_advisory(
    state: dict,
    advisory: str,
    *,
    model: str,
    path: Path | None = None,
) -> dict:
    row = {
        "record_id": str(uuid.uuid4()),
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": str(model or ""),
        "state": dict(state),
        "advisory": str(advisory or "").strip(),
        "execution_state": "仅供人工审核",
    }
    selected = path or KNOWLEDGE_ADVISORY_LOG_PATH
    selected.parent.mkdir(parents=True, exist_ok=True)
    with selected.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def _parse_json(source: Path) -> ParsedKnowledgeSource:
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON 解析失败：{exc}") from exc
    if not isinstance(payload, dict):
        return _parse_text(source, json.dumps(payload, ensure_ascii=False), "JSON 数据")
    nodes = payload.get("nodes") or payload.get("entities") or []
    edges = payload.get("edges") or payload.get("triples") or payload.get("relations") or []
    text = json.dumps(payload, ensure_ascii=False)
    return ParsedKnowledgeSource(
        path=source,
        file_type="JSON 知识数据",
        record_count=len(payload),
        entity_count=len(nodes) if isinstance(nodes, list) else 0,
        relation_count=len(edges) if isinstance(edges, list) else 0,
        fields=list(payload.keys()),
        preview=text[:900],
        text=text[:200000],
    )


def _parse_csv(source: Path) -> ParsedKnowledgeSource:
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = [str(item) for item in (reader.fieldnames or [])]
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"CSV 解析失败：{exc}") from exc
    text = "\n".join("；".join(f"{key}={value}" for key, value in row.items()) for row in rows)
    return ParsedKnowledgeSource(
        path=source,
        file_type="CSV 数据",
        record_count=len(rows),
        fields=fields,
        preview=(text or "；".join(fields))[:900],
        text=("；".join(fields) + "\n" + text)[:200000],
    )


def _parse_text(source: Path, text: str, file_type: str) -> ParsedKnowledgeSource:
    clean = re.sub(r"\s+", " ", text or "").strip()
    entities = set(re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_-]{2,}", clean))
    return ParsedKnowledgeSource(
        path=source,
        file_type=file_type,
        record_count=len(clean.splitlines()) if clean else 0,
        entity_count=len(entities),
        preview=clean[:900],
        text=clean[:200000],
    )


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _read_docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
        root = ET.fromstring(xml)
        return " ".join(item.text or "" for item in root.iter() if item.tag.endswith("}t"))
    except (OSError, KeyError, ET.ParseError, zipfile.BadZipFile) as exc:
        raise ValueError(f"DOCX 解析失败：{exc}") from exc


@lru_cache(maxsize=1)
def _load_qa() -> dict:
    try:
        payload = json.loads(QA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=1)
def _load_graph() -> dict:
    try:
        payload = json.loads(KNOWLEDGE_GRAPH_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _plain_graph_text(value: object) -> str:
    text = re.sub(r"<br\s*/?>", "；", str(value or ""), flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip(" ；")


def _expert_correction_context(question: str) -> str:
    normalized = _normalize(question)
    terms = _question_terms(question)
    matches: list[ExpertCorrection] = []
    for correction in reversed(load_expert_corrections()):
        if correction.status != "active":
            continue
        searchable = _normalize(
            " ".join((correction.topic, correction.original_content, correction.corrected_content, correction.reason))
        )
        topic = _normalize(correction.topic)
        if (
            (topic and (topic in normalized or normalized in topic))
            or any(term in searchable for term in terms)
        ):
            matches.append(correction)
        if len(matches) >= 5:
            break
    if not matches:
        return ""
    lines = []
    for item in matches:
        line = f"- {item.topic}：{item.corrected_content}"
        if item.reason:
            line += f"（修正原因：{item.reason}）"
        line += f"；修订时间：{item.recorded_at}"
        lines.append(line)
    return "【专家修正（优先采用）】\n" + "\n".join(lines)


def _answer_segment_statistic(question: str, qa: dict) -> str | None:
    match = re.search(r"(?:段号|段)\s*(\d+).*?(砂比|排量|施工泵压).*?(平均值|均值|方差)", question)
    if not match:
        return None
    segment, metric, statistic = match.groups()
    stats = qa.get("segmentStats", {})
    if not isinstance(stats, dict):
        return None
    for well, well_stats in stats.items():
        item = well_stats.get(segment, {}) if isinstance(well_stats, dict) else {}
        values = item.get(metric, {}) if isinstance(item, dict) else {}
        key = "mean" if statistic in {"平均值", "均值"} else "var"
        if isinstance(values, dict) and values.get(key) is not None:
            return f"{well} 段 {segment} 的{metric}{statistic}为 {float(values[key]):.4f}。"
    return "知识库中没有找到该井段和指标的已保存统计结果。"


def _first_matching_snippet(text: str, terms: list[str]) -> str:
    sentences = re.split(r"(?<=[。！？.!?])\s*", text)
    for sentence in sentences:
        if not terms or any(term in sentence.lower() for term in terms):
            return sentence[:600]
    return text[:600]


def _question_terms(question: str) -> list[str]:
    """Split common Chinese question glue words into searchable terms."""

    raw = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", str(question))
    terms: list[str] = []
    for value in raw:
        parts = re.split(r"和|与|及|以及|关于|什么|如何|是否|需要|关注|哪些|哪个|怎么", value)
        for part in parts:
            if len(part) >= 2 and part not in terms:
                terms.append(part.lower())
    return terms


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", str(value).lower())


__all__ = [
    "API_CONFIG_PATH",
    "EXPERT_CORRECTIONS_PATH",
    "KNOWLEDGE_GRAPH_PATH",
    "KNOWLEDGE_ADVISORY_LOG_PATH",
    "ExpertCorrection",
    "KnowledgeGraphAPIConfig",
    "ParsedKnowledgeSource",
    "SUPPORTED_KNOWLEDGE_EXTENSIONS",
    "append_expert_correction",
    "append_knowledge_advisory",
    "answer_question",
    "build_knowledge_context",
    "load_expert_corrections",
    "load_api_config",
    "load_saved_knowledge_sources",
    "parse_knowledge_source",
    "request_api_advisory",
    "request_api_answer",
    "save_api_config",
]
