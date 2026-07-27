# FSL-Expert：专家知识与小样本学习

保留五条主实验：GNN工况识别、迁移学习、两阶段分类、同口径直接分类基线和下一工况转移预测。`knowledge_graph` 保存全书Qwen抽取、GraphML和快速展示页面。

```powershell
python run_project.py fsl knowledge-graph
python run_project.py fsl gnn
python run_project.py fsl transfer
python run_project.py fsl train
python run_project.py fsl direct
python run_project.py fsl transition
```

数据按段划分，标签为 `WORKING_TYPE`，空白标签按“正常”处理。同一段不得跨训练、验证、测试或迁移集合。
