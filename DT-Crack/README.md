# DT-Crack：裂缝数字孪生

## 主流程

```text
DAS分簇液砂数据 + 施工压力 + 井轨迹
  -> 时空同步与井底/净压力换算
  -> 耦合增强PKN、reduced BEM或数据代理正演
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
