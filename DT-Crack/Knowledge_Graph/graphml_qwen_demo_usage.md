# 新旧知识图谱展示说明

## 目的

原知识图谱展示页基于 GraphML 和 Sigma.js，加载大图比单独生成的 HTML 页面更快。因此新版 Qwen 全书知识图谱已导出为 GraphML，并接入原来的展示入口。

## 文件

- 旧版局部 GraphML：`active_run_data/output/example_mmkg.graphml`
- 新版 Qwen JSON：`Knowledge_Graph/full_book_qwen_output/kg_full_book_qwen.json`
- 新版 Qwen GraphML：`Knowledge_Graph/full_book_qwen_output/kg_full_book_qwen.graphml`
- JSON 转 GraphML 脚本：`Knowledge_Graph/export_qwen_kg_to_graphml.py`

## 启动命令

在 `DT-Crack` 目录下运行：

```powershell
.\run_demo.ps1 -Version qwen -Port 8080
```

打开浏览器：

```text
http://127.0.0.1:8080/
```

旧版局部图谱：

```powershell
.\run_demo.ps1 -Version legacy -Port 8080
```

旧的 parsed271 大图谱：

```powershell
.\run_demo.ps1 -Version parsed -Port 8080
```

## 说明

新版默认展示的是 Qwen 全书实体关系图，包含 2487 个节点和 2057 条边。621 条专家规则没有直接拆成图节点，否则会额外增加大量规则文本节点，影响前端加载性能；规则仍保留在 `rules_full_book_qwen.csv` 和原始 JSON 中，用于汇报、检索和后续智能决策。
