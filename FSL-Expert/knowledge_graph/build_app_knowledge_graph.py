"""Build the offline knowledge-graph view used by the acceptance APP.

The full-book Qwen graph is useful as a data artifact, but its old browser
page depends on a remote vis-network script and tries to force-layout too many
nodes for an embedded Qt panel.  This builder keeps the complete local data
available while generating a compact canvas view that reveals a small core and
expands by connected neighborhoods.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "full_book_qwen_output" / "kg_full_book_qwen.json"
DEFAULT_OUTPUT = ROOT / "full_book_qwen_output" / "index_app_knowledge_graph.html"


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>知识图谱</title>
  <style>
    :root { color-scheme: dark; --bg:#0d1821; --panel:#172936; --panel2:#1d3341; --line:#34505f; --text:#e8f0f5; --muted:#9eb2c1; --cyan:#20c7c2; --blue:#4d9de0; --orange:#f2a93b; }
    * { box-sizing: border-box; }
    html, body { width:100%; height:100%; margin:0; }
    body { overflow:hidden; background:var(--bg); color:var(--text); font-family:"Microsoft YaHei","Noto Sans SC",sans-serif; font-size:13px; }
    .kg-app { width:100%; height:100%; min-height:280px; display:flex; flex-direction:column; padding:10px; gap:8px; }
    .toolbar { display:flex; flex-wrap:wrap; align-items:center; gap:6px; min-height:30px; }
    button, select, input { border:1px solid var(--line); border-radius:4px; background:var(--panel2); color:var(--text); padding:5px 8px; font:inherit; }
    button { cursor:pointer; }
    button:hover, button:focus-visible, select:focus-visible, input:focus-visible { border-color:var(--cyan); outline:1px solid var(--cyan); }
    button.primary { background:#175d68; border-color:#20aeb0; }
    input { min-width:130px; flex:1 1 150px; }
    .graph-layout { min-height:0; flex:1; display:grid; grid-template-columns:minmax(0, 1fr) 190px; gap:8px; }
    .graph-wrap { min-width:0; min-height:0; background:#0a1117; border:1px solid var(--line); border-radius:4px; position:relative; }
    canvas { display:block; width:100%; height:100%; cursor:grab; touch-action:none; }
    canvas.dragging { cursor:grabbing; }
    aside { min-width:0; overflow:auto; background:var(--panel); border:1px solid var(--line); border-radius:4px; padding:9px; }
    .label { color:var(--cyan); font-weight:600; margin-bottom:5px; }
    .muted { color:var(--muted); line-height:1.45; }
    .selected { color:var(--text); line-height:1.5; word-break:break-word; }
    .selected strong { color:var(--orange); }
    .relation { border-top:1px solid rgba(52,80,95,.55); padding:5px 0; line-height:1.35; }
    .relation span { color:var(--cyan); }
    .footer { color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    @media (max-width:650px) { .graph-layout { grid-template-columns:minmax(0, 1fr); grid-template-rows:minmax(230px, 1fr) auto; } aside { max-height:130px; } }
  </style>
</head>
<body>
  <div class="kg-app" id="kgApp">
    <div class="toolbar" aria-label="知识图谱筛选和展开">
      <select id="viewSelect" aria-label="图谱视图">
        <option value="core">核心风险链</option>
      </select>
      <button id="expandButton" class="primary" type="button">逐步展开</button>
      <button id="resetButton" type="button">重置</button>
      <input id="searchInput" type="search" placeholder="搜索节点" aria-label="搜索知识节点">
      <button id="searchButton" type="button">定位</button>
    </div>
    <div class="graph-layout">
      <div class="graph-wrap"><canvas id="graphCanvas" role="img" aria-label="可交互知识图谱"></canvas></div>
      <aside>
        <div class="label">当前节点</div>
        <div id="nodeDetail" class="muted">点击节点查看关联关系</div>
        <div class="label" style="margin-top:12px">当前视图</div>
        <div id="viewDetail" class="muted"></div>
      </aside>
    </div>
    <div id="footer" class="footer"></div>
  </div>
  <script>
    const DATA = __DATA__;
    const canvas = document.getElementById("graphCanvas");
    const ctx = canvas.getContext("2d");
    const viewSelect = document.getElementById("viewSelect");
    const expandButton = document.getElementById("expandButton");
    const resetButton = document.getElementById("resetButton");
    const searchInput = document.getElementById("searchInput");
    const searchButton = document.getElementById("searchButton");
    const nodeDetail = document.getElementById("nodeDetail");
    const viewDetail = document.getElementById("viewDetail");
    const footer = document.getElementById("footer");
    const colors = {
      "故障类型":"#d95d55", "工况":"#d98c32", "现象":"#6a98cb", "原因":"#a180c4",
      "处置措施":"#65b96b", "施工参数":"#36afb0", "设备":"#b18c4d", "材料":"#be8eae",
      "井":"#4d9de0", "地层":"#65bdb7", "指标":"#f2a93b", "风险":"#e45756", "其他":"#91a2ab"
    };
    const nodes = Array.isArray(DATA.nodes) ? DATA.nodes : [];
    const edges = Array.isArray(DATA.edges) ? DATA.edges : [];
    const byId = new Map(nodes.map(node => [String(node.id), node]));
    const degree = new Map(nodes.map(node => [String(node.id), 0]));
    const adjacency = new Map(nodes.map(node => [String(node.id), []]));
    edges.forEach(edge => {
      const from = String(edge.from), to = String(edge.to);
      if (!byId.has(from) || !byId.has(to)) return;
      degree.set(from, (degree.get(from) || 0) + 1);
      degree.set(to, (degree.get(to) || 0) + 1);
      adjacency.get(from).push({ other:to, edge, direction:"→" });
      adjacency.get(to).push({ other:from, edge, direction:"←" });
    });
    const ranked = [...nodes].sort((a,b) => (degree.get(String(b.id)) || 0) - (degree.get(String(a.id)) || 0));
    const typeCounts = new Map();
    nodes.forEach(node => typeCounts.set(node.type || "其他", (typeCounts.get(node.type || "其他") || 0) + 1));
    [...typeCounts.keys()].sort((a,b) => (typeCounts.get(b) || 0) - (typeCounts.get(a) || 0)).forEach(type => {
      const option = document.createElement("option");
      option.value = `type:${type}`;
      option.textContent = `${type}（${typeCounts.get(type)}）`;
      viewSelect.appendChild(option);
    });
    let expansion = 0;
    let selectedId = null;
    let positions = new Map();
    const manualPositions = new Map();
    let dragState = null;
    let panOffset = { x:0, y:0 };
    let zoomScale = 1;
    let visibleIds = [];
    const focusPattern = /砂堵|压力|排量|砂比|滤失|裂缝|应力|风险|异常|施工|处置/;

    function coreIds() {
      const focused = nodes.filter(node => focusPattern.test(String(node.label || "")))
        .sort((a,b) => (degree.get(String(b.id)) || 0) - (degree.get(String(a.id)) || 0));
      const result = [];
      [...focused.slice(0, 30), ...ranked.slice(0, 30)].forEach(node => {
        const id = String(node.id);
        if (!result.includes(id)) result.push(id);
      });
      return result;
    }

    function neighborhood(seedIds, limit) {
      const result = [...seedIds];
      const resultSet = new Set(result);
      for (let step = 0; step < 4 && result.length < limit; step += 1) {
        const candidates = [];
        result.forEach(id => (adjacency.get(id) || []).forEach(item => {
          if (!resultSet.has(item.other)) candidates.push(item.other);
        }));
        candidates.sort((a,b) => (degree.get(b) || 0) - (degree.get(a) || 0));
        for (const id of candidates) {
          if (result.length >= limit) break;
          if (!resultSet.has(id)) { resultSet.add(id); result.push(id); }
        }
      }
      return result;
    }

    function selectedIds() {
      const mode = viewSelect.value;
      const limit = Math.min(48 + expansion * 26, 150);
      if (mode === "core") return neighborhood(coreIds().slice(0, 48), limit);
      const type = mode.slice(5);
      const seeds = nodes.filter(node => (node.type || "其他") === type)
        .sort((a,b) => (degree.get(String(b.id)) || 0) - (degree.get(String(a.id)) || 0))
        .slice(0, Math.min(48, limit)).map(node => String(node.id));
      return neighborhood(seeds, limit);
    }

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }

    function layoutNodes(ids) {
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(rect.width, 100), height = Math.max(rect.height, 180);
      const centerX = width / 2, centerY = height / 2;
      const sorted = [...ids].sort((a,b) => (degree.get(b) || 0) - (degree.get(a) || 0));
      const result = new Map();
      if (sorted.length) result.set(sorted[0], { x:centerX, y:centerY, r:22 });
      sorted.slice(1).forEach((id, index) => {
        const ring = Math.floor(index / 16) + 1;
        const ringCount = Math.min(16, sorted.length - 1 - (ring - 1) * 16);
        const slot = index % 16;
        const angle = (Math.PI * 2 * slot / Math.max(ringCount, 1)) - Math.PI / 2;
        const radius = Math.min(width, height) * (0.18 + ring * 0.14) * zoomScale;
        const saved = manualPositions.get(id);
        result.set(id, saved ? {
          x: saved.x + panOffset.x,
          y: saved.y + panOffset.y,
          r: id === selectedId ? 18 : 14,
        } : { x:centerX + Math.cos(angle) * radius + panOffset.x, y:centerY + Math.sin(angle) * radius + panOffset.y, r: id === selectedId ? 18 : 14 });
      });
      if (sorted.length && manualPositions.has(sorted[0])) {
        const saved = manualPositions.get(sorted[0]);
        result.set(sorted[0], {
          x: saved.x + panOffset.x,
          y: saved.y + panOffset.y,
          r: 22,
        });
      }
      return result;
    }

    function draw() {
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(rect.width, 100), height = Math.max(rect.height, 180);
      ctx.clearRect(0, 0, width, height);
      visibleIds = selectedIds();
      const visibleSet = new Set(visibleIds);
      positions = layoutNodes(visibleIds);
      ctx.lineWidth = 1;
      edges.forEach(edge => {
        const from = positions.get(String(edge.from)), to = positions.get(String(edge.to));
        if (!from || !to) return;
        ctx.strokeStyle = "rgba(130,160,175,.32)";
        ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y); ctx.stroke();
      });
      visibleIds.forEach(id => {
        const node = byId.get(id), point = positions.get(id);
        if (!node || !point) return;
        const active = id === selectedId;
        point.r = active ? 19 : ((degree.get(id) || 0) > 8 ? 16 : 12);
        ctx.fillStyle = colors[node.type] || colors["其他"];
        ctx.globalAlpha = active ? 1 : .86;
        ctx.beginPath(); ctx.arc(point.x, point.y, point.r, 0, Math.PI * 2); ctx.fill();
        if (active) { ctx.strokeStyle = "#f2a93b"; ctx.lineWidth = 3; ctx.stroke(); }
        ctx.globalAlpha = 1;
        const text = String(node.label || "");
        const label = text.length > 10 ? `${text.slice(0,10)}…` : text;
        ctx.fillStyle = "#e8f0f5";
        ctx.font = "12px Microsoft YaHei, sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "top";
        ctx.fillText(label, point.x, point.y + point.r + 3);
      });
      const mode = viewSelect.value === "core" ? "核心风险链" : viewSelect.value.slice(5);
      viewDetail.textContent = `${mode} · ${visibleIds.length} 个节点 · ${edges.filter(edge => visibleSet.has(String(edge.from)) && visibleSet.has(String(edge.to))).length} 条关系`;
      footer.textContent = `全书数据 ${nodes.length} 个节点 / ${edges.length} 条关系 · 当前仅绘制可读子图`;
    }

    function showNode(id) {
      selectedId = id;
      const node = byId.get(id);
      if (!node) { nodeDetail.textContent = "点击节点查看关联关系"; draw(); return; }
      const links = (adjacency.get(id) || []).slice().sort((a,b) => (degree.get(b.other) || 0) - (degree.get(a.other) || 0)).slice(0, 8);
      nodeDetail.innerHTML = `<div class="selected"><strong>${node.label || "未命名"}</strong><br>${node.type || "其他"} · ${degree.get(id) || 0} 条关系</div>` +
        (links.length ? links.map(item => `<div class="relation"><span>${item.direction} ${item.edge.label || "相关"}</span><br>${byId.get(item.other)?.label || item.other}</div>`).join("") : `<div class="muted">暂无已抽取关系</div>`);
      draw();
    }

    function locate() {
      const query = searchInput.value.trim().toLowerCase();
      if (!query) return;
      const match = nodes.find(node => String(node.label || "").toLowerCase().includes(query));
      if (!match) { nodeDetail.textContent = "未找到匹配节点"; return; }
      const id = String(match.id);
      if (!visibleIds.includes(id)) {
        expansion = Math.min(expansion + 1, 4);
        draw();
      }
      showNode(id);
    }

    function canvasPoint(event) {
      const rect = canvas.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    }

    function nearestNode(x, y) {
      let nearest = null, distance = Infinity;
      positions.forEach((point, id) => {
        const current = Math.hypot(point.x - x, point.y - y);
        if (current <= point.r + 8 && current < distance) { nearest = id; distance = current; }
      });
      return nearest;
    }

    canvas.addEventListener("pointerdown", event => {
      const point = canvasPoint(event);
      const nodeId = nearestNode(point.x, point.y);
      if (!nodeId) {
        dragState = { mode:"pan", start:point, origin:{...panOffset}, moved:false };
        canvas.classList.add("dragging");
        canvas.setPointerCapture(event.pointerId);
        event.preventDefault();
        return;
      }
      selectedId = nodeId;
      dragState = { mode:"node", id: nodeId, moved: false };
      canvas.classList.add("dragging");
      canvas.setPointerCapture(event.pointerId);
      showNode(nodeId);
      event.preventDefault();
    });
    canvas.addEventListener("pointermove", event => {
      if (!dragState) return;
      const point = canvasPoint(event);
      if (dragState.mode === "pan") {
        panOffset = {
          x: dragState.origin.x + point.x - dragState.start.x,
          y: dragState.origin.y + point.y - dragState.start.y,
        };
      } else {
        manualPositions.set(dragState.id, { x: point.x - panOffset.x, y: point.y - panOffset.y });
      }
      dragState.moved = true;
      draw();
      event.preventDefault();
    });
    canvas.addEventListener("pointerup", event => {
      if (!dragState) return;
      dragState = null;
      canvas.classList.remove("dragging");
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      event.preventDefault();
    });
    canvas.addEventListener("pointercancel", () => {
      dragState = null;
      canvas.classList.remove("dragging");
    });
    canvas.addEventListener("wheel", event => {
      const previous = zoomScale;
      zoomScale = Math.max(0.55, Math.min(2.2, zoomScale * (event.deltaY < 0 ? 1.12 : 0.89)));
      if (zoomScale !== previous) draw();
      event.preventDefault();
    }, { passive:false });
    canvas.addEventListener("click", event => {
      const point = canvasPoint(event);
      const nearest = nearestNode(point.x, point.y);
      if (nearest) showNode(nearest);
    });
    expandButton.addEventListener("click", () => { expansion = Math.min(expansion + 1, 4); draw(); });
    resetButton.addEventListener("click", () => { expansion = 0; selectedId = null; manualPositions.clear(); panOffset = {x:0,y:0}; zoomScale = 1; searchInput.value = ""; viewSelect.value = "core"; nodeDetail.textContent = "点击节点查看关联关系"; draw(); });
    searchButton.addEventListener("click", locate);
    searchInput.addEventListener("keydown", event => { if (event.key === "Enter") locate(); });
    viewSelect.addEventListener("change", () => { expansion = 0; selectedId = null; nodeDetail.textContent = "点击节点查看关联关系"; draw(); });
    new ResizeObserver(resizeCanvas).observe(canvas);
    resizeCanvas();
  </script>
</body>
</html>
'''


def build(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> Path:
    graph = json.loads(input_path.read_text(encoding="utf-8"))
    payload = {
        "nodes": graph.get("nodes", []),
        "edges": graph.get("edges", []),
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(HTML_TEMPLATE.replace("__DATA__", serialized), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the offline APP knowledge-graph view")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build(args.input, args.output))


if __name__ == "__main__":
    main()
