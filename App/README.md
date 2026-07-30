# 联合验收APP

## 定位

该APP面向最终验收展示，不承担生产级实时控制。界面统一展示合同三部分的代表性模型、指标、图表、运行入口和技术边界。

## 页面

1. 项目总览：三部分核心指标和联合数据流。
2. 第一部分：知识图谱、两阶段工况识别、迁移学习。
3. 第二部分：真实数据链路、PKN-EnKF观测验证、3D数字孪生。
4. 第三部分：强化学习动作、奖励、风险和180秒验证。
5. 联合演示：按验收顺序准备知识图谱、数字孪生和决策结果。
6. 运行日志：显示算法子进程输出，支持停止任务。
7. 验收与边界：区分已验证能力、阶段性结果和待补现场数据。

## 启动

推荐使用：

```powershell
powershell -ExecutionPolicy Bypass -File App\launch_app.ps1
```

也可以从base环境启动，程序会自动检测并切换到`frac_app`：

```powershell
python App\run_app.py
```

只做环境和产物检查：

```powershell
python App\run_app.py --preflight
```

只生成联合摘要：

```powershell
python App\run_app.py --no-gui
```

## 环境隔离

- PySide界面：`C:\Users\xinonome\anaconda3\envs\frac_app\python.exe`
- 算法子进程：默认使用base Python，可通过`FRACTURING_ALGORITHM_PYTHON`覆盖。
- 不再静默切换Tkinter备用界面。Qt失败时会明确报错。

## 验收现场原则

- 现场只运行知识图谱打开、数字孪生轻量验证和3D页面生成。
- GNN、迁移学习、PPO/SAC等长时间训练使用冻结产物展示，不现场重新训练。
- 所有结果统一写入`outputs`，真实数据只从`Data`读取。
