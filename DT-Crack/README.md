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
