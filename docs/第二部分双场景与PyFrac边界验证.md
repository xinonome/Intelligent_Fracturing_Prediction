# 第二部分双场景与 PyFrac 边界验证

## 场景边界

`no_das_pressure_only` 只使用施工曲线和井口—井底压力换算。它的观测向量只有井底压力，六簇液量、砂量和簇级几何显示为“未接入”，不会用 0 或等距簇位代替缺失数据。

`das_cluster_observation` 使用施工曲线、压力换算和 FracMonitor 解释后的六簇观测。FracMonitor 是分簇解释结果，不等同于原始 DAS 振幅。时间间隙、重复簇行、簇不完整、累计量回退和负值会被标记并排除出 EnKF 同化。

六簇三维位置通过 `App/config/cluster_geometry.csv` 接入，最低字段为 `stage_id,cluster_id,md_m`。未提供实际位置时只画井轨迹和阶段级模型结果，不再按井段等距生成伪造簇位。

## 压力换算与在线校正

```text
P_bhp = P_wellhead + ΔP_hydrostatic - ΔP_pipe - ΔP_perforation + pressure_bias
P_net_raw = P_bhp - σ_hmin
P_net_display = max(P_net_raw, 0)
```

`pressure_bias` 是有界随机游走状态，只有在无 DAS 场景中作为压力转换不确定性进行校正。深度、密度、摩阻和最小水平应力仍为工程默认参数时，系统状态为“待校准”，不表示现场验收通过。

## PyFrac 验收边界

原生动态 PyFrac 输出注入体积、裂缝体积和滤失体积，并计算：

```text
R = V_injected - V_fracture - V_leakoff
|R| / V_injected <= 10%
```

网格和时间步的中—细方案主要结果相对差异均需不超过 5%。`snapshot` 只能做快速状态比较，不能作为动态时间推进或守恒证明。

残差代理必须同时满足留出集 R²、P95 推理时间、PyFrac 收敛/守恒和 PKN 非劣性门槛。未通过时默认链路仍为 `PKN + EnKF`，代理只能显式用于实验。

## 主要命令

```powershell
python App/build_dt_realtime_cache.py
python App/build_dt_3d_realtime_html.py
python DT-Crack/inversion/validate_direct_observations.py --observation-mode pressure_only --construction-pressure-xls Data/3Dfrac/JY84-Z1-stage08-f1.xls
python DT-Crack/inversion/validate_direct_observations.py --observation-mode pressure_plus_cluster --frac-monitor-text Data/3Dfrac/光纤本井监测08.txt --construction-pressure-xls Data/3Dfrac/JY84-Z1-stage08-f1.xls
python -m multistage.cli convergence --config <config.yaml>
```
