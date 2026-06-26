#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Build a full-book knowledge graph from the parsed fracturing book with Qwen API.

This script reads the existing MinerU/hybrid_auto parsed output and calls the
DashScope OpenAI-compatible chat API to extract entities, triples, and rules.
It is designed to be restartable: each page result is cached before final graph
fusion.

Environment:
    DASHSCOPE_API_KEY=your_key

Typical usage:
    python DT-Crack\Knowledge_Graph\build_full_book_kg_with_qwen.py --max-pages 10
    python DT-Crack\Knowledge_Graph\build_full_book_kg_with_qwen.py --pages-before 63
    python DT-Crack\Knowledge_Graph\build_full_book_kg_with_qwen.py --include-images --pages-before 63
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOOK_DIR = (
    PROJECT_ROOT
    / "Data"
    / "multimodal"
    / "陪陵页岩气田试气压裂作业井复杂情况与故障案例分析271"
    / "hybrid_auto"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "full_book_qwen_output"

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_TEXT_MODEL = os.environ.get("QWEN_TEXT_MODEL", "qwen-plus")
DEFAULT_VISION_MODEL = os.environ.get("QWEN_VISION_MODEL", "qwen-vl-plus")


ENTITY_TYPES = {
    "故障类型",
    "工况",
    "现象",
    "原因",
    "处置措施",
    "施工参数",
    "设备",
    "材料",
    "井",
    "地层",
    "指标",
    "风险",
    "其他",
}


RELATION_TYPES = {
    "表现为",
    "导致",
    "原因是",
    "处置措施",
    "影响",
    "监测指标",
    "发生于",
    "包含",
    "同义",
    "相关",
}


@dataclass
class PageInput:
    page_no: int
    text: str
    image_paths: list[Path]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(read_text(path))


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_name(value: Any) -> str:
    text = normalize_space(value)
    text = text.strip("：:;；,.，。[]【】()（）<>《》\"'")
    return text[:120]


def find_book_files(book_dir: Path) -> dict[str, Path]:
    if not book_dir.exists():
        raise FileNotFoundError(f"Book directory not found: {book_dir}")

    files = {}
    for path in book_dir.iterdir():
        name = path.name
        if name.endswith(".md"):
            files["md"] = path
        elif name.endswith("_content_list_v2.json"):
            files["content_v2"] = path
        elif name.endswith("_content_list.json"):
            files.setdefault("content", path)
        elif name.endswith("_model.json"):
            files["model"] = path
    return files


def text_from_content_item(item: Any) -> str:
    """Best-effort extraction from MinerU content JSON variants."""
    parts: list[str] = []
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        item_type = item.get("type")
        if item_type in {"text", "equation_inline", "equation_block"} and isinstance(item.get("content"), str):
            parts.append(item["content"])
        for key in (
            "text",
            "caption",
            "html",
            "latex",
            "title",
            "title_content",
            "paragraph_content",
            "page_header_content",
            "page_footer_content",
            "table_content",
            "image_caption",
            "list_items",
            "item_content",
            "content",
        ):
            value = item.get(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, (dict, list)):
                parts.append(text_from_content_item(value))
        for key in ("children", "items", "blocks", "spans"):
            value = item.get(key)
            if isinstance(value, list):
                parts.extend(text_from_content_item(child) for child in value)
    elif isinstance(item, list):
        parts.extend(text_from_content_item(child) for child in item)
    return "\n".join(part for part in parts if part)


def image_paths_from_content_item(item: Any, image_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if isinstance(item, dict):
        for key in ("img_path", "image_path", "path", "src"):
            value = item.get(key)
            if value and isinstance(value, str):
                candidate = image_dir / Path(value).name
                if candidate.exists():
                    paths.append(candidate)
        for key in ("children", "items", "blocks", "spans"):
            value = item.get(key)
            if isinstance(value, list):
                for child in value:
                    paths.extend(image_paths_from_content_item(child, image_dir))
    elif isinstance(item, list):
        for child in item:
            paths.extend(image_paths_from_content_item(child, image_dir))
    return list(dict.fromkeys(paths))


def load_pages(book_dir: Path, pages_before: int | None = None) -> list[PageInput]:
    files = find_book_files(book_dir)
    image_dir = book_dir / "images"

    if "content_v2" in files:
        data = load_json(files["content_v2"])
        pages: list[PageInput] = []
        if isinstance(data, list):
            for idx, item in enumerate(data, start=1):
                if pages_before is not None and idx > pages_before:
                    break
                text = text_from_content_item(item)
                images = image_paths_from_content_item(item, image_dir)
                pages.append(PageInput(idx, normalize_space(text), images))
        if pages and any(page.text for page in pages):
            return pages

    if "model" in files:
        data = load_json(files["model"])
        pages = []
        if isinstance(data, list):
            for idx, item in enumerate(data, start=1):
                if pages_before is not None and idx > pages_before:
                    break
                pages.append(PageInput(idx, normalize_space(text_from_content_item(item)), []))
        if pages and any(page.text for page in pages):
            return pages

    if "md" not in files:
        raise FileNotFoundError(f"No usable content JSON or markdown found in {book_dir}")

    md_text = read_text(files["md"])
    return split_markdown_into_pages(md_text, pages_before)


def split_markdown_into_pages(md_text: str, pages_before: int | None = None) -> list[PageInput]:
    # Fallback: split by major headings or fixed character blocks when page marks are absent.
    page_matches = list(re.finditer(r"(?:^|\n)#{1,3}\s*(?:第\s*)?(\d+)\s*(?:页|PAGE|Page)?", md_text))
    pages: list[PageInput] = []
    if len(page_matches) >= 10:
        for idx, match in enumerate(page_matches):
            page_no = int(match.group(1))
            if pages_before is not None and page_no > pages_before:
                break
            start = match.start()
            end = page_matches[idx + 1].start() if idx + 1 < len(page_matches) else len(md_text)
            pages.append(PageInput(page_no, normalize_space(md_text[start:end]), []))
        return pages

    chunk_size = 3500
    chunks = [md_text[i : i + chunk_size] for i in range(0, len(md_text), chunk_size)]
    for idx, chunk in enumerate(chunks, start=1):
        if pages_before is not None and idx > pages_before:
            break
        pages.append(PageInput(idx, normalize_space(chunk), []))
    return pages


def truncate_text(text: str, max_chars: int) -> str:
    text = normalize_space(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "……"


def make_extraction_prompt(page: PageInput, max_chars: int) -> str:
    page_text = truncate_text(page.text, max_chars)
    return f"""
你是页岩气压裂工程知识图谱抽取助手。请从下面书籍第 {page.page_no} 页内容中抽取结构化知识。

抽取范围优先关注：砂堵、缝口暂堵、缝内暂堵、主缝延伸、滤失、泵压异常、排量、砂比、暂堵剂、施工参数、故障原因、处置措施。

请严格输出 JSON，不要输出 Markdown。JSON 格式如下：
{{
  "page": {page.page_no},
  "entities": [
    {{"name": "实体名称", "type": "故障类型/工况/现象/原因/处置措施/施工参数/设备/材料/井/地层/指标/风险/其他", "description": "一句话解释", "evidence": "原文依据短句"}}
  ],
  "triples": [
    {{"head": "头实体", "relation": "表现为/导致/原因是/处置措施/影响/监测指标/发生于/包含/同义/相关", "tail": "尾实体", "evidence": "原文依据短句"}}
  ],
  "rules": [
    {{"condition": "触发条件", "conclusion": "判断或风险", "action": "建议措施", "evidence": "原文依据短句"}}
  ]
}}

要求：
1. 只抽取原文能够支持的内容，不要编造。
2. 实体名称要短，避免整句作为实体。
3. 没有内容时返回空数组。
4. evidence 控制在 60 字以内。

页面内容：
{page_text}
""".strip()


def encode_image(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "jpeg"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


def build_messages(prompt: str, images: list[Path], include_images: bool) -> list[dict[str, Any]]:
    if not include_images or not images:
        return [{"role": "user", "content": prompt}]

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_path in images[:3]:
        content.append({"type": "image_url", "image_url": {"url": encode_image(image_path)}})
    return [{"role": "user", "content": content}]


def call_qwen_api(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    timeout: int,
    retries: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(base_url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            return data["choices"][0]["message"]["content"]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    detail = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    detail = str(exc)
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Qwen API call failed after retries: {last_error}")


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"No JSON object found: {text[:200]}")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Extracted JSON is not an object")
    return data


def fallback_rule_extract(page: PageInput) -> dict[str, Any]:
    """Offline fallback for dry-run or API unavailable demos."""
    keywords = ["砂堵", "暂堵", "缝口暂堵", "缝内暂堵", "主缝延伸", "泵压", "排量", "砂比", "滤失"]
    entities = []
    triples = []
    found = [kw for kw in keywords if kw in page.text]
    for kw in found:
        ent_type = "故障类型" if "堵" in kw or kw == "滤失" else "施工参数"
        entities.append({"name": kw, "type": ent_type, "description": f"第{page.page_no}页出现的{kw}相关概念", "evidence": kw})
    if "砂堵" in found and "泵压" in found:
        triples.append({"head": "砂堵", "relation": "表现为", "tail": "泵压异常", "evidence": "关键词共现"})
    if "暂堵" in found and "泵压" in found:
        triples.append({"head": "暂堵", "relation": "影响", "tail": "泵压", "evidence": "关键词共现"})
    return {"page": page.page_no, "entities": entities, "triples": triples, "rules": []}


def validate_result(data: dict[str, Any], page_no: int) -> dict[str, Any]:
    entities = []
    for item in data.get("entities", []) or []:
        if not isinstance(item, dict):
            continue
        name = normalize_name(item.get("name"))
        if not name:
            continue
        ent_type = normalize_name(item.get("type")) or "其他"
        if ent_type not in ENTITY_TYPES:
            ent_type = "其他"
        entities.append(
            {
                "name": name,
                "type": ent_type,
                "description": normalize_space(item.get("description"))[:240],
                "evidence": normalize_space(item.get("evidence"))[:120],
                "page": page_no,
            }
        )

    triples = []
    for item in data.get("triples", []) or []:
        if not isinstance(item, dict):
            continue
        head = normalize_name(item.get("head"))
        tail = normalize_name(item.get("tail"))
        if not head or not tail or head == tail:
            continue
        relation = normalize_name(item.get("relation")) or "相关"
        if relation not in RELATION_TYPES:
            relation = "相关"
        triples.append(
            {
                "head": head,
                "relation": relation,
                "tail": tail,
                "evidence": normalize_space(item.get("evidence"))[:120],
                "page": page_no,
            }
        )

    rules = []
    for item in data.get("rules", []) or []:
        if not isinstance(item, dict):
            continue
        condition = normalize_space(item.get("condition"))
        conclusion = normalize_space(item.get("conclusion"))
        action = normalize_space(item.get("action"))
        if not condition and not conclusion and not action:
            continue
        rules.append(
            {
                "condition": condition[:240],
                "conclusion": conclusion[:240],
                "action": action[:240],
                "evidence": normalize_space(item.get("evidence"))[:120],
                "page": page_no,
            }
        )

    return {"page": page_no, "entities": entities, "triples": triples, "rules": rules}


def dedupe_entities(page_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    entities: dict[str, dict[str, Any]] = {}
    for result in page_results:
        for item in result.get("entities", []):
            name = normalize_name(item.get("name"))
            if not name:
                continue
            current = entities.setdefault(
                name,
                {
                    "id": f"E{len(entities) + 1:05d}",
                    "name": name,
                    "type": item.get("type") or "其他",
                    "description": item.get("description") or "",
                    "pages": [],
                    "evidence": [],
                },
            )
            page = item.get("page")
            if page not in current["pages"]:
                current["pages"].append(page)
            evidence = item.get("evidence")
            if evidence and evidence not in current["evidence"]:
                current["evidence"].append(evidence)
            if not current.get("description") and item.get("description"):
                current["description"] = item["description"]
            if current.get("type") == "其他" and item.get("type") != "其他":
                current["type"] = item["type"]
    return entities


def dedupe_triples(page_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    triples: dict[tuple[str, str, str], dict[str, Any]] = {}
    for result in page_results:
        for item in result.get("triples", []):
            head = normalize_name(item.get("head"))
            relation = normalize_name(item.get("relation")) or "相关"
            tail = normalize_name(item.get("tail"))
            if not head or not tail or head == tail:
                continue
            key = (head, relation, tail)
            current = triples.setdefault(
                key,
                {
                    "id": f"T{len(triples) + 1:05d}",
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                    "pages": [],
                    "evidence": [],
                    "weight": 0,
                },
            )
            current["weight"] += 1
            page = item.get("page")
            if page not in current["pages"]:
                current["pages"].append(page)
            evidence = item.get("evidence")
            if evidence and evidence not in current["evidence"]:
                current["evidence"].append(evidence)
    return list(triples.values())


def dedupe_rules(page_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    rules: list[dict[str, Any]] = []
    for result in page_results:
        for item in result.get("rules", []):
            key = (
                normalize_space(item.get("condition")),
                normalize_space(item.get("conclusion")),
                normalize_space(item.get("action")),
            )
            if key in seen:
                continue
            seen.add(key)
            rules.append(
                {
                    "id": f"R{len(rules) + 1:05d}",
                    "condition": key[0],
                    "conclusion": key[1],
                    "action": key[2],
                    "page": item.get("page"),
                    "evidence": item.get("evidence") or "",
                }
            )
    return rules


def build_graph_payload(
    entities: dict[str, dict[str, Any]], triples: list[dict[str, Any]], rules: list[dict[str, Any]]
) -> dict[str, Any]:
    nodes = []
    entity_lookup = {name: entity["id"] for name, entity in entities.items()}
    for entity in entities.values():
        nodes.append(
            {
                "id": entity["id"],
                "label": entity["name"],
                "type": entity["type"],
                "title": f"{entity['type']}<br>页码: {', '.join(map(str, entity['pages'][:12]))}<br>{html.escape(entity.get('description') or '')}",
            }
        )

    edges = []
    for triple in triples:
        head_id = entity_lookup.get(triple["head"])
        tail_id = entity_lookup.get(triple["tail"])
        if not head_id:
            head_id = f"AUTO_{len(nodes) + 1:05d}"
            entity_lookup[triple["head"]] = head_id
            nodes.append({"id": head_id, "label": triple["head"], "type": "其他"})
        if not tail_id:
            tail_id = f"AUTO_{len(nodes) + 1:05d}"
            entity_lookup[triple["tail"]] = tail_id
            nodes.append({"id": tail_id, "label": triple["tail"], "type": "其他"})
        edges.append(
            {
                "from": head_id,
                "to": tail_id,
                "label": triple["relation"],
                "weight": triple.get("weight", 1),
                "title": f"页码: {', '.join(map(str, triple.get('pages', [])[:12]))}<br>{html.escape('; '.join(triple.get('evidence', [])[:3]))}",
            }
        )

    return {"nodes": nodes, "edges": edges, "rules": rules}


def write_csvs(output_dir: Path, entities: dict[str, dict[str, Any]], triples: list[dict[str, Any]], rules: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "entities_full_book_qwen.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "type", "description", "pages", "evidence"])
        writer.writeheader()
        for item in entities.values():
            row = dict(item)
            row["pages"] = ",".join(map(str, row.get("pages", [])))
            row["evidence"] = " | ".join(row.get("evidence", [])[:5])
            writer.writerow(row)

    with (output_dir / "triples_full_book_qwen.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "head", "relation", "tail", "pages", "evidence", "weight"])
        writer.writeheader()
        for item in triples:
            row = dict(item)
            row["pages"] = ",".join(map(str, row.get("pages", [])))
            row["evidence"] = " | ".join(row.get("evidence", [])[:5])
            writer.writerow(row)

    with (output_dir / "rules_full_book_qwen.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "condition", "conclusion", "action", "page", "evidence"])
        writer.writeheader()
        writer.writerows(rules)


def write_html(output_dir: Path, graph: dict[str, Any]) -> None:
    data_json = json.dumps({"nodes": graph["nodes"], "edges": graph["edges"]}, ensure_ascii=False)
    type_counts = Counter(node.get("type", "其他") for node in graph["nodes"])
    relation_counts = Counter(edge.get("label", "相关") for edge in graph["edges"])
    type_html = "".join(f"<li>{html.escape(k)}：{v}</li>" for k, v in type_counts.most_common())
    relation_html = "".join(f"<li>{html.escape(k)}：{v}</li>" for k, v in relation_counts.most_common(12))
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>全书知识图谱 - Qwen 抽取</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ margin: 0; font-family: "Microsoft YaHei", sans-serif; background: #f5f1e8; color: #1e1b16; }}
    header {{ padding: 22px 28px; background: linear-gradient(120deg, #254f5f, #b96536); color: white; }}
    header h1 {{ margin: 0 0 8px; font-size: 26px; }}
    main {{ display: grid; grid-template-columns: 1fr 320px; gap: 18px; padding: 18px; }}
    #graph {{ height: calc(100vh - 125px); min-height: 620px; border-radius: 16px; background: white; border: 1px solid #ddd; }}
    aside {{ background: white; border-radius: 16px; padding: 18px; border: 1px solid #ddd; overflow: auto; max-height: calc(100vh - 125px); }}
    .metric {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 16px; }}
    .card {{ padding: 12px; background: #f8f5ef; border-radius: 12px; }}
    .card b {{ display: block; font-size: 24px; color: #b96536; }}
    li {{ margin: 6px 0; }}
  </style>
</head>
<body>
  <header>
    <h1>全书知识图谱 - Qwen 抽取版</h1>
    <div>数据来源：陪陵页岩气田试气压裂作业井复杂情况与故障案例分析271；生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
  </header>
  <main>
    <div id="graph"></div>
    <aside>
      <div class="metric">
        <div class="card">节点数<b>{len(graph['nodes'])}</b></div>
        <div class="card">关系数<b>{len(graph['edges'])}</b></div>
      </div>
      <h3>实体类型分布</h3>
      <ul>{type_html}</ul>
      <h3>关系类型分布</h3>
      <ul>{relation_html}</ul>
      <p>提示：拖拽节点可查看结构，悬停节点/边可查看页码和证据。</p>
    </aside>
  </main>
  <script>
    const data = {data_json};
    const colors = {{
      "故障类型": "#d94f45", "工况": "#d98c32", "现象": "#557eb1", "原因": "#8064a2",
      "处置措施": "#54a24b", "施工参数": "#2f9aa0", "设备": "#8c6d31", "材料": "#b279a2",
      "井": "#4c78a8", "地层": "#72b7b2", "指标": "#f58518", "风险": "#e45756", "其他": "#999999"
    }};
    const nodes = new vis.DataSet(data.nodes.map(n => ({{
      ...n,
      color: colors[n.type] || colors["其他"],
      shape: n.type === "故障类型" ? "diamond" : "dot",
      size: n.type === "故障类型" ? 24 : 14
    }})));
    const edges = new vis.DataSet(data.edges.map(e => ({{
      ...e,
      arrows: "to",
      font: {{ align: "middle", size: 11 }},
      width: Math.min(5, 1 + Math.log((e.weight || 1) + 1))
    }})));
    new vis.Network(document.getElementById("graph"), {{ nodes, edges }}, {{
      physics: {{ stabilization: true, barnesHut: {{ gravitationalConstant: -4200, springLength: 130 }} }},
      interaction: {{ hover: true, tooltipDelay: 120 }},
      nodes: {{ font: {{ size: 14, face: "Microsoft YaHei" }} }},
      edges: {{ smooth: {{ type: "dynamic" }} }}
    }});
  </script>
</body>
</html>"""
    (output_dir / "index_full_book_qwen.html").write_text(page, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    book_dir = Path(args.book_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    cache_dir = output_dir / "page_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pages = load_pages(book_dir, args.pages_before)
    if args.page_start:
        pages = [page for page in pages if page.page_no >= args.page_start]
    if args.max_pages:
        pages = pages[: args.max_pages]
    pages = [page for page in pages if page.text or page.image_paths]

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key and not args.dry_run:
        raise RuntimeError("DASHSCOPE_API_KEY is not set. Use --dry-run for offline rule extraction.")

    page_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, page in enumerate(pages, start=1):
        cache_path = cache_dir / f"page_{page.page_no:04d}.json"
        if cache_path.exists() and not args.force:
            result = load_json(cache_path)
            page_results.append(result)
            print(f"[cache] page={page.page_no} ({index}/{len(pages)})")
            continue

        try:
            if args.dry_run:
                raw_data = fallback_rule_extract(page)
            else:
                model = args.vision_model if args.include_images and page.image_paths else args.model
                prompt = make_extraction_prompt(page, args.max_chars)
                messages = build_messages(prompt, page.image_paths, args.include_images)
                content = call_qwen_api(
                    api_key=api_key,
                    base_url=args.base_url,
                    model=model,
                    messages=messages,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    retries=args.retries,
                )
                raw_data = extract_json_object(content)
            result = validate_result(raw_data, page.page_no)
            write_json(cache_path, result)
            page_results.append(result)
            print(
                f"[ok] page={page.page_no} ({index}/{len(pages)}) "
                f"entities={len(result['entities'])} triples={len(result['triples'])} rules={len(result['rules'])}"
            )
            if args.sleep > 0:
                time.sleep(args.sleep)
        except Exception as exc:
            failure = {"page": page.page_no, "error": str(exc)}
            failures.append(failure)
            write_json(cache_dir / f"page_{page.page_no:04d}.error.json", failure)
            print(f"[fail] page={page.page_no}: {exc}")
            if not args.continue_on_error:
                raise

    entities = dedupe_entities(page_results)
    triples = dedupe_triples(page_results)
    rules = dedupe_rules(page_results)
    graph = build_graph_payload(entities, triples, rules)

    write_json(output_dir / "page_results_full_book_qwen.json", page_results)
    write_json(output_dir / "entities_full_book_qwen.json", list(entities.values()))
    write_json(output_dir / "triples_full_book_qwen.json", triples)
    write_json(output_dir / "rules_full_book_qwen.json", rules)
    write_json(output_dir / "kg_full_book_qwen.json", graph)
    write_csvs(output_dir, entities, triples, rules)
    write_html(output_dir, graph)

    summary = {
        "book_dir": str(book_dir),
        "output_dir": str(output_dir),
        "processed_pages": len(page_results),
        "failed_pages": len(failures),
        "entity_count": len(entities),
        "triple_count": len(triples),
        "rule_count": len(rules),
        "include_images": args.include_images,
        "model": args.model,
        "vision_model": args.vision_model,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files": {
            "kg": str(output_dir / "kg_full_book_qwen.json"),
            "html": str(output_dir / "index_full_book_qwen.html"),
            "entities_csv": str(output_dir / "entities_full_book_qwen.csv"),
            "triples_csv": str(output_dir / "triples_full_book_qwen.csv"),
            "rules_csv": str(output_dir / "rules_full_book_qwen.csv"),
        },
        "failures": failures,
    }
    write_json(output_dir / "summary_full_book_qwen.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full-book KG with Qwen/DashScope API.")
    parser.add_argument("--book-dir", default=str(DEFAULT_BOOK_DIR), help="hybrid_auto directory of parsed book")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="output directory")
    parser.add_argument("--model", default=DEFAULT_TEXT_MODEL, help="text model name, e.g. qwen-plus or qwen3.5-plus")
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL, help="vision model name, e.g. qwen-vl-plus")
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--include-images", action="store_true", help="send page images to vision model when available")
    parser.add_argument("--pages-before", type=int, default=None, help="only process pages <= N")
    parser.add_argument("--page-start", type=int, default=None, help="only process pages >= N")
    parser.add_argument("--max-pages", type=int, default=None, help="process first N loaded pages")
    parser.add_argument("--max-chars", type=int, default=4500, help="max page text chars sent to model")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.2, help="sleep seconds between API calls")
    parser.add_argument("--force", action="store_true", help="ignore page cache and call API again")
    parser.add_argument("--continue-on-error", action="store_true", help="continue when one page fails")
    parser.add_argument("--dry-run", action="store_true", help="do offline keyword extraction without API")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
