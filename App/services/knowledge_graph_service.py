"""Local knowledge-graph parsing, API settings and question answering."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from functools import lru_cache
from pathlib import Path
import re
import urllib.error
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

from ..core.paths import PATHS


API_CONFIG_PATH = PATHS.app_outputs / "knowledge_graph_api.json"
QA_PATH = PATHS.root / "FSL-Expert" / "knowledge_graph" / "qa.json"
SUPPORTED_KNOWLEDGE_EXTENSIONS = {".json", ".csv", ".txt", ".md", ".docx"}


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
    return KnowledgeGraphAPIConfig(
        endpoint=str(payload.get("endpoint") or "").strip(),
        model=str(payload.get("model") or "").strip(),
        api_key=str(payload.get("api_key") or ""),
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
                "api_key": config.api_key,
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


def answer_question(question: str, sources: list[ParsedKnowledgeSource] | None = None) -> str:
    """Answer from the shipped QA knowledge and explicitly imported text."""

    prompt = str(question or "").strip()
    if not prompt:
        return "请输入问题。"
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

    snippets = []
    terms = _question_terms(prompt)
    for source in sources or []:
        text = source.text or ""
        if not text:
            continue
        if not terms or any(term in text.lower() for term in terms):
            snippets.append(f"【{source.path.name}】\n{_first_matching_snippet(text, terms)}")
    if snippets:
        return "本地导入资料中找到以下相关内容：\n\n" + "\n\n".join(snippets[:3])
    return "本地知识库中未找到与该问题直接匹配的内容。可导入相关资料，或在 API 配置中填写可用的问答接口后重试。"


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
    context = "\n\n".join(
        f"来源：{source.path.name}\n{source.preview}" for source in (sources or []) if source.preview
    )
    body = {
        "model": config.model.strip() or "default",
        "messages": [
            {"role": "system", "content": "你是压裂施工知识图谱问答助手。只根据给定资料回答，不确定时明确说明。"},
            {"role": "user", "content": f"问题：{question.strip()}\n\n资料：{context[:12000]}"},
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
    "KnowledgeGraphAPIConfig",
    "ParsedKnowledgeSource",
    "SUPPORTED_KNOWLEDGE_EXTENSIONS",
    "answer_question",
    "load_api_config",
    "parse_knowledge_source",
    "request_api_answer",
    "save_api_config",
]
