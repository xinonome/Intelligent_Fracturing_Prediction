# 全书知识图谱 Qwen 抽取脚本使用说明

## 目标

`build_full_book_kg_with_qwen.py` 用于把已经扫描解析完成的 271 页书籍内容，批量送入阿里云 Qwen/DashScope API，抽取实体、关系和专家规则，并生成可展示的知识图谱。

默认输入目录：

```text
Data/multimodal/陪陵页岩气田试气压裂作业井复杂情况与故障案例分析271/hybrid_auto
```

默认输出目录：

```text
DT-Crack/Knowledge_Graph/full_book_qwen_output
```

## API 配置

不要把 API Key 写进代码，建议在 PowerShell 中设置环境变量：

```powershell
$env:DASHSCOPE_API_KEY="你的阿里云API_KEY"
```

默认使用 DashScope OpenAI 兼容接口：

```text
https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
```

## 推荐运行方式

先小范围测试 10 页：

```powershell
python DT-Crack\Knowledge_Graph\build_full_book_kg_with_qwen.py `
  --page-start 14 `
  --max-pages 10 `
  --model qwen-plus `
  --continue-on-error
```

只抽取与砂堵关系更密切的前 63 页：

```powershell
python DT-Crack\Knowledge_Graph\build_full_book_kg_with_qwen.py `
  --pages-before 63 `
  --model qwen-plus `
  --continue-on-error
```

如果确认你的模型支持视觉输入，再启用图片理解：

```powershell
python DT-Crack\Knowledge_Graph\build_full_book_kg_with_qwen.py `
  --pages-before 63 `
  --model qwen-plus `
  --vision-model qwen-vl-plus `
  --include-images `
  --continue-on-error
```

如果你控制台中模型名确实是 `qwen3.5-plus`，可以替换：

```powershell
python DT-Crack\Knowledge_Graph\build_full_book_kg_with_qwen.py `
  --pages-before 63 `
  --model qwen3.5-plus `
  --continue-on-error
```

## 输出文件

脚本会生成：

```text
summary_full_book_qwen.json
kg_full_book_qwen.json
entities_full_book_qwen.json
triples_full_book_qwen.json
rules_full_book_qwen.json
entities_full_book_qwen.csv
triples_full_book_qwen.csv
rules_full_book_qwen.csv
index_full_book_qwen.html
page_cache/
```

其中：

- `kg_full_book_qwen.json`：前端展示用图谱数据。
- `index_full_book_qwen.html`：可直接双击打开的图谱可视化页面。
- `entities_full_book_qwen.csv`：实体表。
- `triples_full_book_qwen.csv`：三元组关系表。
- `rules_full_book_qwen.csv`：规则表，包括触发条件、判断结论和建议措施。
- `page_cache/`：每页抽取缓存，API 中断后可继续运行，不会重复消耗已经完成的页面。

## 注意事项

- 当前脚本是“知识抽取”，不是重新训练 Qwen 大模型。
- 文本抽取用 `qwen-plus` 或 `qwen3.5-plus` 即可。
- 图像、流程图、表格截图理解需要使用支持视觉输入的 VL 模型，例如 `qwen-vl-plus`。
- 第一次建议先跑 `--max-pages 10`，确认输出质量后再跑 `--pages-before 63` 或全书。
