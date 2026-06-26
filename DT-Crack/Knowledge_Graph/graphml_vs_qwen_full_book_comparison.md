# GraphML 与 Qwen 全书知识图谱对比

## 旧版 GraphML

- 文件：`C:\Workspace\Intelligent_Fracturing_Prediction\DT-Crack\active_run_data\output\example_mmkg.graphml`
- 节点数：161
- 边数：158
- 连通分量：32
- 说明：这是之前展示页默认加载的局部书籍 GraphML 图谱，规模约一百多个节点。

主要实体类型：

- EVENT: 27
- PARAMETER: 21
- OPERATION: 17
- TIME: 15
- FIGURE: 13
- EQUIPMENT: 12
- WELL: 10
- FLUID_SYSTEM: 10
- STAGE: 9
- FORMATION_LAYER: 9
- PROPPANT: 9
- CHEMICAL: 3
- DOCUMENT: 3
- ROCK_PROPERTY: 2

## 新版 Qwen 全书图谱

- 文件：`C:\Workspace\Intelligent_Fracturing_Prediction\DT-Crack\Knowledge_Graph\full_book_qwen_output\kg_full_book_qwen.json`
- 处理页数：268
- 节点数：2487
- 边数：2057
- 规则数：621
- 说明：新版对全书文本进行批量抽取，覆盖故障、现象、原因、处置措施和专家规则。

主要实体类型：

- 处置措施: 475
- 原因: 345
- 其他: 321
- 现象: 273
- 故障类型: 220
- 施工参数: 212
- 设备: 191
- 井: 120
- 指标: 107
- 材料: 88
- 工况: 81
- 风险: 28
- 地层: 26

## 展示文件

- 对比页面：`C:\Workspace\Intelligent_Fracturing_Prediction\DT-Crack\Knowledge_Graph\graphml_vs_qwen_full_book_comparison.html`
- 旧版 GraphML 图谱页面：`C:\Workspace\Intelligent_Fracturing_Prediction\DT-Crack\Knowledge_Graph\graphml_book_sample_view.html`
- 新版 Qwen 图谱页面：`C:\Workspace\Intelligent_Fracturing_Prediction\DT-Crack\Knowledge_Graph\full_book_qwen_output\index_full_book_qwen.html`

## 结论

之前展示的 GraphML 不是数千节点版本，而是 `active_run_data/output/example_mmkg.graphml`，实际为 161 个节点、158 条边。新版 Qwen 全书图谱为 2487 个节点、2057 条边、621 条规则，能体现本周“从局部展示图谱扩展到全书知识抽取”的工作增量。
