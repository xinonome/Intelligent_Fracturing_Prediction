# 联合动态演示使用说明

## 启动

```powershell
powershell -ExecutionPolicy Bypass -File App\launch_app.ps1
```

进入“联合动态演示”页面后，动态回放默认读取注册表中的冻结结果，不修改 `Data` 和
`artifacts`，也不会现场训练 GNN、EnKF 或 PPO。APP 使用单一主时间轴，所有二维曲线、
决策卡片、参数面板和 Plotly 3D 视图共用同一帧。

## 交互方式

- `播放/暂停`：按时间顺序回放统一帧流。
- `倍速`：支持 0.25x、0.5x、1x、2x、4x。
- `时间轴`：拖动到任意时间步，所有图表、决策卡片和内嵌 3D 同步刷新。
- `跳转`：输入帧号，或使用前进/后退一个时间步。

## 每帧展示

- 左侧上图：PKN 先验井底压力、现场观测压力、EnKF 参数更新后重新正演的后验压力。
- 左侧下图：液量和砂量观测空间 TVD 误差。
- 右侧柱图：6 个分簇的后验裂缝半长。
- 下方决策卡片：未来 60 秒排量/砂比动作、异常/砂堵概率、风险等级和人工确认。

## 数据口径

DT 回放来自 `artifacts/dt/direct_observation_enkf` 的历史 CSV；HMI 动作和风险来自
`outputs/hmi/contract3_acceptance/ppo_20260729_222332`。二者按回放进度对齐，用于展示
跨模块联调过程，不代表 HMI 已经在现场实时调用 DT 服务。

3D 裂缝演化固定使用 `outputs/app/dt_realtime_3d.html`，由 QtWebEngine 在 APP 内嵌加载，
不打开外部浏览器。HTML 自包含 Plotly JS、真实井轨迹、六簇位置、橙色 PKN 先验面和青色
EnKF 后验面，并通过 `window.setTimeIndex(time)` 接收 APP 主时间轴的时间同步。

QtWebEngine 不可用时，3D 区域显示 Python、Qt、PySide6、WebEngine 状态和建议命令；
其它二维页面以及 `--no-gui` 摘要仍可运行。
