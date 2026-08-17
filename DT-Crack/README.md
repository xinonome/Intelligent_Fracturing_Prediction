# DT-Crack：裂缝数字孪生

## 主流程

```text
DAS分簇液砂数据 + 施工压力 + 井轨迹
  -> 时空同步与井底/净压力换算
  -> 耦合增强PKN、面板离散BEM或物理引导数据代理正演
  -> EnKF更新 E'、C_L、黏度、最小水平应力和分簇参数
  -> 正演模型重新计算裂缝状态
  -> 直接观测空间验证与3D展示
```

`inversion.physics` 是稳定公开接口。EnKF 更新的是物理参数，不直接覆盖裂缝长度。

## 当前默认正演

默认 PKN 使用累计注入量计算裂缝历史生长，并以当前总排量计算瞬时缝宽和净压力；簇间排量由可学习导流能力、缝宽反馈与邻簇应力阴影迭代分配。该模型不把观测到的分簇流量直接作为正演输入，分簇液砂数据只进入 EnKF 残差。

在第 8 段、60 步在线验证中，`coupled` 相比 `legacy` 的后验液量 TVD 由 3.30% 降至 2.22%，砂量 TVD 由 2.98% 降至 2.42%，井底压力误差由 7.73% 降至 6.91%；单步 P95 约为 70--90 ms。该指标仍属于观测空间验证，不代表独立真实缝长精度。

## 运行

```powershell
python run_project.py dt validate
python run_project.py dt benchmark
python run_project.py dt visualize --open
```

如需复现实验对照：

```powershell
$env:PYTHONPATH = (Resolve-Path DT-Crack).Path
python DT-Crack\inversion\validate_direct_observations.py `
  --frac-monitor-text Data\3Dfrac\光纤本井监测08.txt `
  --construction-pressure-xls Data\3Dfrac\JY84-Z1-stage08-f1.xls `
  --hydraulic-coupling-mode legacy

python DT-Crack\inversion\validate_direct_observations.py `
  --frac-monitor-text Data\3Dfrac\光纤本井监测08.txt `
  --construction-pressure-xls Data\3Dfrac\JY84-Z1-stage08-f1.xls `
  --hydraulic-coupling-mode coupled
```

正式保存结果位于 `artifacts/dt/direct_observation_enkf`。验证集平均误差为：液量分布6.19%、砂量分布7.10%、井底压力4.04%；单步计算P95约57.3 ms。该结论只针对观测空间，不代表真实缝长误差。

当前增强在线配置增加动态近井筒压力修正、分通道稳健同化和50%累计滤失质量守恒先验。在第8段42步校准、18步在线留出、多随机种子复核中，后验液量TVD约1.7%--1.8%、砂量TVD约1.9%--2.0%、井底压力误差约1.6%--1.7%，每步15%达标率为100%，单步P95约227--238 ms。详见 `docs/正反演模型缺点与精度优化.md`。滤失上限仍需甲方地层参数校准。

## 正演模型对比

`bem_reduced` 现采用每条裂缝36个常开度边界面板，构造平面应变弹性影响矩阵，叠加簇间应力遮挡，并迭代求解润滑压力与缝宽。它属于固定缝高、对称双翼的降维数值BEM，不是完整三维生产级边界元求解器。

`data_surrogate` 以面板BEM为教师模型，在排量、黏度、弹性模量、缝高、时间、簇间距和分簇入口系数联合变化的1100组工况上训练三隐层神经网络代理。代理学习相对PKN基线的对数修正量，并按完整工况分组留出验证，避免同一工况的不同簇泄漏到训练集和验证集。

```powershell
python DT-Crack\inversion\benchmark_forward_models.py `
  --models pkn4 bem_reduced data_surrogate `
  --max-playback-steps 60 `
  --repeats 50
```

输出包括单次正演P50/P95、EnKF单步P95、观测空间后验误差、15秒达标状态，以及代理模型对面板BEM的独立留出误差。

### PyFrac 离线教师与残差代理

PyFrac 扩展链路位于 `forward_models/pyfrac_surrogate.py` 和
`inversion/benchmark_pyfrac_surrogate.py`。当前教师样本覆盖 96 个场景组、15 类排量/滤失/应力/黏度和六簇液量分配组合，采用严格场景组 70%/15%/15% 留出。残差代理使用独立长度、缝宽、压力输出头，并在长度和压力测试 R2 均不低于 0.80 且 P95 小于 15 秒时才允许作为候选在线修正层。

```powershell
python DT-Crack\forward_models\pyfrac_surrogate.py `
  --mode generate `
  --teacher-csv outputs\dt\pyfrac_teacher_iteration_v3\teacher_samples.csv `
  --samples 96 --steps-per-scenario 4 --n-clusters 6

python DT-Crack\forward_models\pyfrac_surrogate.py `
  --mode train `
  --teacher-csv outputs\dt\pyfrac_teacher_iteration_v3\teacher_samples.csv `
  --model-out outputs\dt\pyfrac_teacher_iteration_v3\pyfrac_residual_surrogate_independent.joblib
```

本轮代理留出 R2 为：长度 `0.9985`、压力 `0.8295`，候选门禁通过；但默认在线主链仍是 PKN + EnKF。真实第8段主链结果位于 `outputs/dt/pyfrac_enkf_validation_pkn_main_v3`，代理只有在显式传入 `--surrogate-path` 时才参与验证。

### 知识图谱增强 EnKF

KG-EnKF 已从“只放大不确定性”的旧实验升级为四种可审计模式：`off` 基线、
`uncertainty_only` 不确定性扩展、`soft_prior` 软先验均值偏移、
`soft_correlated` 先验均值 + 参数协方差 + 压力观测置信度 + 单步更新限幅。

知识图谱读取 `FSL-Expert/rule_fusion/rule_fusion/fused_sand_plug_rules.json`，
只使用校准区间提取压力上升、压力斜率、高砂比和排量下降信号。它影响
`[log E', log C_L, log mu, sigma_min, log K_IC]` 的先验分布，不直接写入缝长，
EnKF 仍使用观测残差更新，更新后重新运行 PKN。详细公式、字段和限制见
`DT-Crack/docs/知识图谱增强EnKF实现说明.md`。

单次完整模式运行：

```powershell
python DT-Crack\inversion\validate_direct_observations.py `
  --frac-monitor-text Data\3Dfrac\光纤本井监测08.txt `
  --construction-pressure-xls Data\3Dfrac\JY84-Z1-stage08-f1.xls `
  --physics-profile enhanced `
  --validation-mode frozen `
  --knowledge-guided-mode soft_correlated `
  --knowledge-guided-strength 0.35 `
  --max-steps 60 --ensemble-size 80 --seed 2026
```

同一数据划分和随机种子对比全部模式：

```powershell
python DT-Crack\inversion\compare_knowledge_guided_modes.py `
  --frac-monitor-text Data\3Dfrac\光纤本井监测08.txt `
  --construction-pressure-xls Data\3Dfrac\JY84-Z1-stage08-f1.xls `
  --max-steps 60 --ensemble-size 80
```

2026-08-13 的单段正式对照中，`soft_correlated` 将井底压力留出误差由约
`16.69%` 降至 `13.87%`，三项观测同时不超过 15% 的留出比例由 `38.89%`
升至 `50.00%`，P95 约 `149.6 ms`。这是单段、单种子证据，不能外推为跨井普遍提升。
