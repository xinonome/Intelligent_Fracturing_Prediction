# HMI-KE：知识嵌入智能体

当前实现包含泵序约束、综合奖励、Gymnasium环境、数字孪生动作响应、PPO/SAC、轻量分层策略、六类工况和动作后180秒验证。数字孪生响应当前用于平层PPO/SAC；`--hierarchical` 是独立的轻量分层原型，尚未与数字孪生响应联合训练。

```powershell
python run_project.py hmi train --total-timesteps 5000
python run_project.py hmi scenarios --total-timesteps 1000
```

状态窗口为当前300秒压力、排量和砂比；动作是未来60秒排量和砂比。数字孪生响应调用 `DT-Crack/inversion` 的PKN-EnKF公开接口。当前为离线仿真验证，不能解释为现场自动控制。
