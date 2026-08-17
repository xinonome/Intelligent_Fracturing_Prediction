# 联合验收APP

## 定位

该APP面向最终验收展示，不承担生产级实时控制。界面统一展示合同三部分的代表性模型、指标、图表、运行入口和技术边界。

## 页面

1. 联合动态演示：默认页，单一时间轴联动曲线、参数、决策和内嵌3D。
2. 第一部分：知识图谱、两阶段工况识别、迁移学习。
3. 第二部分：真实数据链路、PKN-EnKF观测验证、3D数字孪生。
4. 第三部分：强化学习动作、奖励、风险和180秒验证。

第三部分页面的风险卡片采用“异常风险、压力安全、分簇不均衡、模型不确定性”分栏口径。
正常/正常工况/主缝延伸只作为正常基线，不进入异常风险正类；空白或`??`不作为异常正类，
但应在后续数据治理中补充标注。详细阈值见`App/HMI风险判定说明.md`。

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

## 产物注册与联调口径

APP 读取 `App/config/demo_registry.json`，不再根据历史目录猜测模型结果。每次启动会在
`outputs/app/runs/<timestamp>/` 保存 `app_run.json`、`preflight.json`、三部分快照、
`dt_to_hmi.json`、`hmi_decision.json` 和任务元数据。

产物状态只有四种：`validated`（注册文件完整且通过当前验证）、`development_only`
（可运行但质量门禁或现场条件未满足）、`not_available`（缺少文件）和 `invalid`（文件损坏）。
当前 HMI 结果即使训练步数已经达到 100000，只要 `quality_gate.passed=false` 或
`scientific_status=demo_only`，页面仍显示为 `development_only`。

DT 到 HMI 的桥接只传递可追溯的后验状态和来源；当前 DT 没有异常概率时，字段明确记为
`null` 并列入 `unavailable_fields`，不会用数字填充制造“已接入”假象。

## 验收现场原则

- 现场只运行知识图谱内嵌展示、数字孪生轻量验证和3D页面生成。
- GNN、迁移学习、PPO/SAC等长时间训练使用冻结产物展示，不现场重新训练。
- 所有结果统一写入`outputs`，真实数据只从`Data`读取。
- 3D 固定使用 `outputs/app/dt_realtime_3d.html`，禁止外部浏览器路径；QtWebEngine 缺失时显示环境错误。
