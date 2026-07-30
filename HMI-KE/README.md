# HMI-KE：知识嵌入智能体

当前实现包含泵序约束、综合奖励、Gymnasium环境、PKN-EnKF数字孪生动作响应、PPO/SAC、轻量分层策略、六类工况和动作后180秒严格验证。分层策略已可与数字孪生响应联合训练，高风险会抢占进入安全选项，并允许应急降低排量和砂比。

合同第三部分新增统一验收评估：离线数字孪生滚动5分钟预警、本机“策略推理+PKN-EnKF效果计算”15秒门禁、预防型与既有异常恢复型180秒验证，以及需要人工确认的结构化决策卡片。

```powershell
python run_project.py hmi train --total-timesteps 5000
python run_project.py hmi scenarios --total-timesteps 1000
python run_project.py hmi validate-env --strict
python run_project.py hmi curriculum
python run_project.py hmi full-train --total-timesteps 100000 --seeds 2026 2027 2028

python run_project.py hmi acceptance
```

状态窗口为当前300秒压力、排量和砂比；动作是未来60秒排量和砂比。数字孪生响应调用 `DT-Crack/inversion` 的PKN-EnKF公开接口。当前为离线仿真验证，不能解释为现场自动控制。

180秒验证同时报告两个口径：动作前正常窗口用于检验“调整后不产生异常”，动作前已经异常的窗口单独检验180秒恢复率；原始总体安全率继续保留。2026-07-29现有100000步策略复评结果为：总体安全率99.43%，680个已知动作前正常窗口的预防型安全率100%，4个既有异常恢复窗口的180秒恢复率75%。因此只能表述为离线验收候选，不能替代现场验收。

5分钟预警当前是基于数字孪生策略滚动的离线验证，不是现场实测提前量；15秒指标不包含一次性Excel扫描和报告渲染，需在甲方现场硬件复测端到端时延。
## 真实数据动作响应仿真

主训练环境不再依赖六类固定扰动复制样本。当前推荐流程是先从真实分段秒点学习
“过去300秒状态 + 候选排量/砂比 -> 未来60秒施工响应”，再与 PKN-EnKF 组成混合数字孪生：

```powershell
python run_project.py hmi train-surrogate --max-samples 50000

python HMI-KE\train_rl_control_agent.py `
  --data-path Data\raw_frac `
  --response-model learned_hybrid `
  --response-surrogate-path outputs\hmi\response_surrogate\<run>\response_surrogate.joblib `
  --scenario-source historical `
  --hierarchical `
  --total-timesteps 100000
```

代理模型只预测未来施工压力、异常概率和砂堵概率；裂缝长度、缝宽、净压力基线及
EnKF物理参数仍由 PKN-EnKF 计算。六类场景保留用于训练后压力测试和180秒验收，
不再作为主训练数据。该代理属于观察数据上的反事实近似，超出历史排量/砂比范围时
会输出 `surrogate_ood_score`，现场使用前必须补充受控试验或高保真正演样本校准。
