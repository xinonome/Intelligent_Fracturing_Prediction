#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare the legacy GraphML demo KG with the Qwen full-book KG."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KG_DIR = Path(__file__).resolve().parent

# This is the GraphML file loaded by DT-Crack/run_demo.ps1 in the old demo page.
GRAPHML_PATH = PROJECT_ROOT / "active_run_data" / "output" / "example_mmkg.graphml"
QWEN_DIR = KG_DIR / "full_book_qwen_output"
QWEN_KG = QWEN_DIR / "kg_full_book_qwen.json"
QWEN_HTML = QWEN_DIR / "index_full_book_qwen.html"
QWEN_SUMMARY = QWEN_DIR / "summary_full_book_qwen.json"

OUT_GRAPHML_HTML = KG_DIR / "graphml_book_sample_view.html"
OUT_COMPARE_HTML = KG_DIR / "graphml_vs_qwen_full_book_comparison.html"
OUT_MD = KG_DIR / "graphml_vs_qwen_full_book_comparison.md"
OUT_SPEECH = KG_DIR / "graphml_vs_qwen_full_book_speech.md"


def clean(value: object) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1]
    return text


def graphml_stats(path: Path) -> tuple[nx.Graph, dict]:
    graph = nx.read_graphml(path)
    node_types = Counter(clean(data.get("entity_type", "UNKNOWN")) for _, data in graph.nodes(data=True))
    edge_labels = Counter(clean(data.get("description", ""))[:40] or "关系" for _, _, data in graph.edges(data=True))
    return graph, {
        "path": path,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": nx.number_connected_components(graph.to_undirected()),
        "node_types": node_types.most_common(14),
        "edge_labels": edge_labels.most_common(8),
        "size": path.stat().st_size,
    }


def qwen_stats(path: Path) -> tuple[dict, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    return data, {
        "path": path,
        "nodes": len(nodes),
        "edges": len(edges),
        "rules": len(data.get("rules", [])),
        "node_types": Counter(clean(node.get("type", "其他")) for node in nodes).most_common(14),
        "edge_labels": Counter(clean(edge.get("label", "相关")) for edge in edges).most_common(8),
        "size": path.stat().st_size,
    }


def rel(path: Path) -> str:
    try:
        return path.relative_to(KG_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def html_rows(items: list[tuple[str, int]]) -> str:
    return "\n".join(f"<li><span>{name}</span><b>{count}</b></li>" for name, count in items)


def markdown_rows(items: list[tuple[str, int]]) -> str:
    return "\n".join(f"- {name}: {count}" for name, count in items)


def write_graphml_view(graph: nx.Graph, stats: dict) -> None:
    nodes = []
    for node, data in graph.nodes(data=True):
        label = clean(node)
        node_type = clean(data.get("entity_type", "UNKNOWN"))
        desc = clean(data.get("description", ""))
        nodes.append({
            "id": label,
            "label": label[:36],
            "type": node_type,
            "title": f"{node_type}<br>{desc[:240]}",
            "value": max(1, int(graph.degree[node])),
        })

    edges = []
    for source, target, data in graph.edges(data=True):
        edges.append({
            "from": clean(source),
            "to": clean(target),
            "title": clean(data.get("description", ""))[:240],
            "value": float(data.get("weight", 1.0) or 1.0),
        })

    payload = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>旧版 GraphML 局部知识图谱</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ margin:0; font-family:"Microsoft YaHei", sans-serif; background:#0f172a; color:#e5e7eb; }}
    header {{ padding:18px 24px; border-bottom:1px solid rgba(255,255,255,.14); }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    .wrap {{ display:grid; grid-template-columns:1fr 320px; gap:16px; padding:16px; }}
    #graph {{ height:720px; background:#111827; border:1px solid rgba(255,255,255,.16); border-radius:16px; }}
    aside {{ background:#111827; border:1px solid rgba(255,255,255,.16); border-radius:16px; padding:16px; }}
    .metrics {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
    .card {{ background:#1f2937; padding:12px; border-radius:12px; }}
    .card b {{ display:block; font-size:24px; color:#60a5fa; }}
    li {{ display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px dashed rgba(255,255,255,.15); }}
    ul {{ list-style:none; padding:0; }}
  </style>
</head>
<body>
  <header>
    <h1>旧版 GraphML 局部知识图谱</h1>
    <div>这是原展示页默认加载的 GraphML：{stats["nodes"]} 个节点，{stats["edges"]} 条边。</div>
  </header>
  <div class="wrap">
    <div id="graph"></div>
    <aside>
      <div class="metrics">
        <div class="card">节点数<b>{stats["nodes"]}</b></div>
        <div class="card">边数<b>{stats["edges"]}</b></div>
        <div class="card">连通分量<b>{stats["components"]}</b></div>
        <div class="card">文件大小<b>{stats["size"] // 1024}KB</b></div>
      </div>
      <h3>主要实体类型</h3>
      <ul>{html_rows(stats["node_types"])}</ul>
    </aside>
  </div>
  <script>
    const data = {payload};
    const palette = ["#60a5fa", "#34d399", "#fbbf24", "#f87171", "#a78bfa", "#22d3ee", "#fb7185"];
    const typeColor = new Map();
    function colorOf(type) {{
      if (!typeColor.has(type)) typeColor.set(type, palette[typeColor.size % palette.length]);
      return typeColor.get(type);
    }}
    const nodes = new vis.DataSet(data.nodes.map(n => ({{
      ...n, color: colorOf(n.type), size: Math.min(34, 8 + Math.sqrt(n.value) * 2)
    }})));
    const edges = new vis.DataSet(data.edges.map(e => ({{
      ...e, arrows: "to", color: "rgba(180,190,210,.45)"
    }})));
    new vis.Network(document.getElementById("graph"), {{nodes, edges}}, {{
      physics: {{ stabilization: true, barnesHut: {{ gravitationalConstant: -4500, springLength: 130 }} }},
      interaction: {{ hover: true, tooltipDelay: 120 }},
      nodes: {{ font: {{ color: "#e5e7eb", size: 13 }} }}
    }});
  </script>
</body>
</html>"""
    OUT_GRAPHML_HTML.write_text(html, encoding="utf-8-sig")


def write_comparison(old: dict, new: dict, summary: dict) -> None:
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>GraphML 与 Qwen 全书知识图谱对比</title>
  <style>
    body {{ margin:0; font-family:"Microsoft YaHei", sans-serif; background:#f5efe5; color:#1f2937; }}
    header {{ padding:26px 34px; background:linear-gradient(120deg,#1f4e5f,#b96536); color:white; }}
    h1 {{ margin:0 0 8px; font-size:30px; }}
    main {{ padding:22px 34px 34px; display:grid; gap:22px; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
    .card {{ background:#fffaf2; border:1px solid rgba(0,0,0,.12); border-radius:18px; padding:18px; box-shadow:0 12px 28px rgba(0,0,0,.08); }}
    .old {{ border-top:6px solid #1f4e5f; }}
    .new {{ border-top:6px solid #b96536; }}
    .metrics {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:12px 0 16px; }}
    .metric {{ background:white; border:1px solid rgba(0,0,0,.10); border-radius:12px; padding:10px; }}
    .metric span {{ display:block; color:#667085; font-size:13px; }}
    .metric b {{ display:block; font-size:25px; margin-top:4px; }}
    .old .metric b {{ color:#1f4e5f; }} .new .metric b {{ color:#b96536; }}
    li {{ display:flex; justify-content:space-between; gap:10px; padding:6px 0; border-bottom:1px dashed rgba(0,0,0,.16); }}
    ul {{ list-style:none; padding:0; margin:8px 0 0; }}
    iframe {{ width:100%; height:640px; border:1px solid rgba(0,0,0,.14); border-radius:18px; background:white; }}
    .note {{ background:white; border-left:6px solid #2f9aa0; border-radius:14px; padding:16px 18px; line-height:1.8; }}
    a.btn {{ display:inline-block; margin-right:10px; padding:8px 12px; border-radius:999px; background:#1f4e5f; color:white; text-decoration:none; }}
  </style>
</head>
<body>
  <header>
    <h1>书籍知识图谱版本对比：GraphML 局部图谱 → Qwen 全书图谱</h1>
    <div>旧版来自原展示页默认 GraphML；新版来自 Qwen3.5-Plus 对全书文本的批量抽取。</div>
  </header>
  <main>
    <section class="grid">
      <div class="card old">
        <h2>旧版：GraphML 局部书籍图谱</h2>
        <div class="metrics">
          <div class="metric"><span>节点数</span><b>{old["nodes"]}</b></div>
          <div class="metric"><span>边数</span><b>{old["edges"]}</b></div>
          <div class="metric"><span>规则数</span><b>0</b></div>
        </div>
        <p>这是之前展示页默认加载的局部 GraphML 图谱，规模约一百多个节点，主要用于证明书籍图谱构建流程已经跑通。</p>
        <h3>主要实体类型</h3><ul>{html_rows(old["node_types"])}</ul>
      </div>
      <div class="card new">
        <h2>新版：Qwen 全书文本知识图谱</h2>
        <div class="metrics">
          <div class="metric"><span>节点数</span><b>{new["nodes"]}</b></div>
          <div class="metric"><span>边数</span><b>{new["edges"]}</b></div>
          <div class="metric"><span>规则数</span><b>{new["rules"]}</b></div>
        </div>
        <p>新版对全书 {summary.get("processed_pages", "-")} 页文本进行批量抽取，重点沉淀故障类型、现象、原因、处置措施、风险和专家规则。</p>
        <h3>主要实体类型</h3><ul>{html_rows(new["node_types"])}</ul>
      </div>
    </section>
    <section class="note">
      <b>汇报口径：</b>旧版 GraphML 证明我们已跑通书籍图谱构建和展示流程，但覆盖范围较小；新版 Qwen 全书图谱把覆盖范围扩展到 268 页文本，并新增 621 条专家规则，更适合支撑后续工况解释、风险判断和处置建议。
    </section>
    <section>
      <a class="btn" href="{OUT_GRAPHML_HTML.name}" target="_blank">打开旧版 GraphML 图谱</a>
      <a class="btn" href="{rel(QWEN_HTML)}" target="_blank">打开新版 Qwen 全书图谱</a>
      <a class="btn" href="{rel(QWEN_DIR / 'rules_full_book_qwen.csv')}" target="_blank">查看新版规则 CSV</a>
    </section>
    <section class="grid">
      <div><h2>旧版 GraphML 局部图谱</h2><iframe src="{OUT_GRAPHML_HTML.name}"></iframe></div>
      <div><h2>新版 Qwen 全书图谱</h2><iframe src="{rel(QWEN_HTML)}"></iframe></div>
    </section>
  </main>
</body>
</html>"""
    OUT_COMPARE_HTML.write_text(html, encoding="utf-8-sig")

    md = f"""# GraphML 与 Qwen 全书知识图谱对比

## 旧版 GraphML

- 文件：`{old["path"]}`
- 节点数：{old["nodes"]}
- 边数：{old["edges"]}
- 连通分量：{old["components"]}
- 说明：这是之前展示页默认加载的局部书籍 GraphML 图谱，规模约一百多个节点。

主要实体类型：

{markdown_rows(old["node_types"])}

## 新版 Qwen 全书图谱

- 文件：`{new["path"]}`
- 处理页数：{summary.get("processed_pages", "-")}
- 节点数：{new["nodes"]}
- 边数：{new["edges"]}
- 规则数：{new["rules"]}
- 说明：新版对全书文本进行批量抽取，覆盖故障、现象、原因、处置措施和专家规则。

主要实体类型：

{markdown_rows(new["node_types"])}

## 展示文件

- 对比页面：`{OUT_COMPARE_HTML}`
- 旧版 GraphML 图谱页面：`{OUT_GRAPHML_HTML}`
- 新版 Qwen 图谱页面：`{QWEN_HTML}`

## 结论

之前展示的 GraphML 不是数千节点版本，而是 `active_run_data/output/example_mmkg.graphml`，实际为 {old["nodes"]} 个节点、{old["edges"]} 条边。新版 Qwen 全书图谱为 {new["nodes"]} 个节点、{new["edges"]} 条边、{new["rules"]} 条规则，能体现本周“从局部展示图谱扩展到全书知识抽取”的工作增量。
"""
    OUT_MD.write_text(md, encoding="utf-8-sig")

    speech = f"""# 汇报讲稿：GraphML 局部图谱与 Qwen 全书图谱对比

这一页主要说明我们本周在知识图谱上的迭代。

之前展示页里加载的是一个 GraphML 局部图谱，文件是 `active_run_data/output/example_mmkg.graphml`。这个图谱一共有 {old["nodes"]} 个节点、{old["edges"]} 条边，主要覆盖少量案例页中的事件、参数、井、阶段、设备和图表信息。它的价值是验证了书籍图谱抽取、存储和可视化流程可以跑通，但覆盖范围还比较有限。

本周我们用 Qwen3.5-Plus 对书籍文本做了全书级抽取，当前完成 {summary.get("processed_pages", "-")} 页，形成 {new["nodes"]} 个节点、{new["edges"]} 条边，并额外沉淀了 {new["rules"]} 条专家规则。新版图谱更偏向工程知识表达，包含故障类型、现象、原因、处置措施、风险、施工参数和材料等内容。

所以这里的核心进展不是单纯节点数增加，而是知识覆盖从局部案例展示扩展到了全书文本，并且从普通实体关系进一步扩展到可用于工况解释和智能决策的专家规则。
"""
    OUT_SPEECH.write_text(speech, encoding="utf-8-sig")


def main() -> None:
    graphml, old = graphml_stats(GRAPHML_PATH)
    _, new = qwen_stats(QWEN_KG)
    summary = json.loads(QWEN_SUMMARY.read_text(encoding="utf-8")) if QWEN_SUMMARY.exists() else {}
    write_graphml_view(graphml, old)
    write_comparison(old, new, summary)
    print(json.dumps({
        "old_graphml": {"nodes": old["nodes"], "edges": old["edges"], "path": str(old["path"])},
        "new_qwen": {"nodes": new["nodes"], "edges": new["edges"], "rules": new["rules"], "path": str(new["path"])},
        "comparison_html": str(OUT_COMPARE_HTML),
        "graphml_html": str(OUT_GRAPHML_HTML),
        "comparison_md": str(OUT_MD),
        "speech": str(OUT_SPEECH),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
