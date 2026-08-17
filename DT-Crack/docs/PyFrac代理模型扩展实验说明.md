# PyFrac 教师样本与残差代理模型扩展实验说明

## 1. 本轮目标

本轮不是把 PyFrac 直接替换为在线正演，而是扩大离线教师模型覆盖，并验证“PKN 基线 + PyFrac 残差代理 + EnKF 参数更新”是否具备进入在线链路的条件。

在线主链仍然是：

```text
真实数据 -> PKN 正演 -> EnKF 更新 E'、C_L、mu、sigma_min、K_IC
        -> 用更新后的参数重新运行 PKN
```

PyFrac 只在代理模型通过质量门禁后作为可选的残差修正层：

```text
PyFrac(theta, Q, t) - PKN(theta, Q, t)
        -> 离线教师残差
        -> 场景组留出训练
        -> 在线代理预测残差
        -> PKN 输出 + 代理残差
```

没有独立的真实裂缝长度标注时，代理模型的 R2 是对 PyFrac 教师残差的留出拟合能力，不是现场裂缝长度精度。

## 2. 教师样本扩展

旧版本只有少量参数组合，容易让相邻时间点或相似工况同时出现在训练和测试中。本轮将 `samples` 定义为“场景组数量”，每个场景组包含 4 个时间状态，每个时间状态包含 6 个簇，因此每组最多产生 24 条簇级样本。

正式增量命令：

```powershell
python DT-Crack\forward_models\pyfrac_surrogate.py `
  --mode generate `
  --teacher-csv outputs\dt\pyfrac_teacher_iteration_v3\teacher_samples.csv `
  --samples 96 `
  --steps-per-scenario 4 `
  --n-clusters 6 `
  --pyfrac-mode snapshot
```

实际结果：

| 项目 | 结果 |
|---|---:|
| 请求场景组 | 96 |
| 有效教师行 | 2001 |
| 时间状态/场景 | 4 |
| 簇数/状态 | 6 |
| 场景类型 | 15 类 |

有效行少于 `96 x 4 x 6 = 2304` 的原因是 PyFrac 个别快照在前缘重构时可能失败；失败样本没有被伪造为成功结果，保留成功样本进入教师表。

场景类型包括：

- 排量突变：`rate_step_up`、`rate_step_down`、`rate_pulse`；
- 滤失变化：`enhanced_leakoff`、`combined_leakoff_viscosity`；
- 应力变化：`stress_shift`、`combined_stress_rate`；
- 黏度变化：`viscosity_shift`、`combined_leakoff_viscosity`；
- 六簇分配：`uniform_allocation`、`heel_dominant`、`toe_dominant`、`middle_dominant`、`alternating_allocation`、`edge_dominant`。

六簇分配先生成非负权重，再归一化：

\[
 a_i\ge 0,\qquad \sum_{i=1}^{6}a_i=1,\qquad Q_i(t)=Q_{total}(t)a_i.
\]

因此教师样本不会因为分簇权重变化而改变总排量边界。在线真实数据链路中，光纤累计液量增量优先于这些合成分配组合：

\[
 a_i(t)=\frac{\Delta V_i(t)}{\sum_{j=1}^{6}\Delta V_j(t)}.
\]

## 3. 代理模型输入和目标

代理不直接从零预测裂缝状态，而是学习 PyFrac 相对 PKN 的残差：

\[
 \Delta z=z_{PyFrac}-z_{PKN},
 \qquad
 \hat z_{online}=z_{PKN}+\widehat{\Delta z}.
\]

输入特征分为五组：

1. 物性：`E'`、滤失系数、黏度、最小水平主应力、断裂韧度、缝高；
2. 工况：簇排量、总排量、排量变化率、时间、累计注入量；
3. 几何与分簇：簇号、簇间距、6 个液量分配权重、分配熵、最大/最小分配和不均衡度；
4. 物理派生量：滤失体积分数、应力变化、黏度比、应力阴影、排量/黏度比、PKN压力/应力比、时间尺度；
5. PKN先验：PKN半长、最大缝宽、压力。

输出三个残差：

\[
 [\Delta L,\Delta w,\Delta p]
 =
 [L_{PyFrac}-L_{PKN},w_{PyFrac}-w_{PKN},p_{PyFrac}-p_{PKN}].
\]

为避免长度残差的数量级压制压力残差，训练前按 PKN 结果归一化：

\[
 \tilde{\Delta L}=\frac{\Delta L}{\max(|L_{PKN}|,25)},\quad
 \tilde{\Delta w}=\frac{\Delta w}{\max(|w_{PKN}|,0.25)},\quad
 \tilde{\Delta p}=\frac{\Delta p}{\max(|p_{PKN}|,5)}.
\]

网络采用三个独立的 MLP 残差头，而不是一个共享输出头：

```text
每个输出头：StandardScaler -> MLP(256, 128, 64, 32) -> residual
```

这样长度、缝宽、压力分别拟合，避免长度残差的大数值和更强梯度影响压力头。

## 4. 严格场景组留出

数据按 `scenario_group` 划分，而不是按单条时间点随机划分：

```text
训练组       70%：1358 行、67 组
验证组       15%：309 行、14 组
测试组       15%：334 行、15 组
```

同一个场景的 4 个时间状态和 6 个簇始终属于同一个集合。这样测试集代表未参与训练的参数组合，避免“记住同一场景邻近时间点”造成虚高指标。

## 5. 正式增量结果

结果文件：

```text
outputs/dt/pyfrac_teacher_iteration_v3/teacher_samples.csv
outputs/dt/pyfrac_teacher_iteration_v3/pyfrac_residual_surrogate_independent.joblib
outputs/dt/pyfrac_benchmark_iteration_v3/benchmark_summary.json
```

严格场景组测试结果：

| 留出目标 | 测试 R2 | 测试 MAE | 门禁要求 |
|---|---:|---:|---:|
| 长度残差 `delta_length_m` | 0.9985 | 3.06 m | >= 0.80，达到 |
| 缝宽残差 `delta_aperture_mm` | 0.9458 | 0.181 mm | 展示，不作为在线必要门禁 |
| 压力残差 `delta_pressure_mpa` | 0.8295 | 0.138 MPa | >= 0.80，达到 |

门禁逻辑为：

\[
 approved
 = (R^2_{L,test}\ge 0.80)
 \land (R^2_{p,test}\ge 0.80)
 \land (T_{surrogate,P95}<15s).
\]

本轮代理模型满足该门禁。benchmark 实测：

| 模型 | P95 单次耗时 |
|---|---:|
| PKN | 约 0.076 ms |
| PyFrac snapshot | 约 192 ms |
| 独立头残差代理 | 约 37.6 ms |

因此代理推理有足够的 15 秒预算，但“通过速度门禁”不等于已经完成现场真实裂缝几何验收。

## 6. 真实第8段兼容性验证

使用真实输入运行：

```powershell
python DT-Crack\inversion\validate_direct_observations.py `
  --frac-monitor-text Data\3Dfrac\光纤本井监测08.txt `
  --construction-pressure-xls Data\3Dfrac\JY84-Z1-stage08-f1.xls `
  --max-steps 60 `
  --calibration-ratio 0.70 `
  --ensemble-size 40 `
  --physics-profile enhanced `
  --validation-mode frozen `
  --surrogate-path outputs\dt\pyfrac_benchmark_iteration_v3\pyfrac_residual_surrogate.joblib `
  --run-dir outputs\dt\pyfrac_enkf_validation_iteration_v3
```

本次运行确认：

- EnKF 状态维度为 5：`E'`、`C_L`、`mu`、`sigma_min`、`K_IC`；
- 光纤液量分配仍是观测边界，不被代理模型改写；
- 代理模型先通过自身严格留出门禁，再作为可选残差层；
- EnKF更新后仍需重新正演，不能直接把后验缝长写回；
- 单步 P95 约 4.80 s，低于 15 s，但本次真实留出链路整体 `validation_pass=false`，原因是砂量约 15.28%、井底压力约 13.76%且所有观测同时满足 15% 的比例约 50%。

因此当前工程口径分成两层：

1. **稳定在线主链**：默认仍为 PKN + EnKF；
2. **候选高保真修正层**：本轮代理已通过教师残差 R2 和速度门禁，可在开发联调中显式启用，但还不能把真实第8段整体验证 `validation_pass=false` 说成合同验收完成。

作为对照，不传入 `--surrogate-path` 的稳定主链结果位于：

```text
outputs/dt/pyfrac_enkf_validation_pkn_main_v3/20260810_114053/summary.json
```

该结果明确使用 `PKN + EnKF`，代理未请求、未启用：

- 60 步计算 P95 约 `53.6 ms`；
- 液量 TVD `0%`；
- 砂量 TVD `13.31%`；
- 井底压力相对误差 `14.62%`；
- 超过 5% 的缝长回缩次数为 `0`；
- 当前脚本按平均观测误差口径给出 `validation_pass=true`；若采用“每个观测点全部同时低于15%”的更严格口径，达标率为 `44.44%`，两者不能混用。

显式启用代理的结果是候选链路，不是默认链路。该次运行的代理自身 R2 门禁通过，但真实综合验证 `validation_pass=false`，说明“代理对 PyFrac 残差拟合良好”和“代理改善真实第8段观测验证”是两个不同条件。只有后者也稳定通过，才应考虑将代理设为默认前向修正层。

## 7. 当前短板和下一步

压力残差总体 R2 达标，但按场景类型拆分后，`toe_dominant`、`combined_leakoff_viscosity` 等场景仍不稳定。这说明总体 R2 不能代替场景级可靠性评估。后续应：

- 增加端部偏置和联合场景教师样本；
- 对 PyFrac 前缘重构失败样本做数值稳定性处理，但不把失败样本静默删除；
- 增加独立多井或真实裂缝解释标签；
- 将场景级门禁作为更严格的上线条件；
- 在代理层稳定后，再比较在线 PKN+EnKF 与 PKN+代理+EnKF 的真实观测改善。

当前不建议关闭 PKN 主链，也不建议把 PyFrac 离线教师结果称作现场真值。
