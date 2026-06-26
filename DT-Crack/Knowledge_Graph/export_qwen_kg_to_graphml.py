#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export the Qwen full-book KG JSON to GraphML for the existing demo server.

The original visualization server is optimized around GraphML.  This converter
keeps the Qwen entity-relation graph in the same format so it can be loaded by
`main.py -s --graph ...` without changing the front-end rendering path.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import networkx as nx


DEFAULT_INPUT = Path(__file__).resolve().parent / "full_book_qwen_output" / "kg_full_book_qwen.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "full_book_qwen_output" / "kg_full_book_qwen.graphml"
XML_ILLEGAL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(value: object, max_len: int | None = None) -> str:
    text = XML_ILLEGAL_RE.sub("", str(value or "")).strip()
    if max_len and len(text) > max_len:
        return text[:max_len] + "..."
    return text


def export_qwen_graphml(input_path: Path, output_path: Path) -> dict:
    data = json.loads(input_path.read_text(encoding="utf-8-sig"))
    graph = nx.MultiDiGraph()

    for node in data.get("nodes", []):
        node_id = clean_text(node.get("id"))
        if not node_id:
            continue
        label = clean_text(node.get("label") or node_id)
        entity_type = clean_text(node.get("type") or "其他")
        title = clean_text(node.get("title"))
        graph.add_node(
            node_id,
            label=label,
            entity_type=entity_type,
            description=title,
            source_id="qwen_full_book",
        )

    skipped_edges = 0
    for idx, edge in enumerate(data.get("edges", []), start=1):
        source = clean_text(edge.get("from") or edge.get("source"))
        target = clean_text(edge.get("to") or edge.get("target"))
        if not source or not target or source not in graph or target not in graph:
            skipped_edges += 1
            continue
        relation = clean_text(edge.get("label") or edge.get("relation") or "相关")
        description = clean_text(edge.get("title") or relation)
        try:
            weight = float(edge.get("weight", 1.0) or 1.0)
        except (TypeError, ValueError):
            weight = 1.0
        graph.add_edge(
            source,
            target,
            key=f"qwen_edge_{idx}",
            relation=relation,
            label=relation,
            description=description,
            weight=weight,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    nx.write_graphml(graph, tmp_path)
    tmp_path.replace(output_path)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "rules_not_embedded": len(data.get("rules", [])),
        "skipped_edges": skipped_edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Qwen full-book KG JSON to GraphML.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = export_qwen_graphml(args.input, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
