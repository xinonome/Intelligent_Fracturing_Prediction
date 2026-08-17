# PKN 与 PyFrac 对照实验说明

## 1. 对照目标

对照实验回答的是“复杂高保真参考模型相对快速 PKN 的差异在哪里”，不是在没有裂缝几何真值时直接宣布某个模型更准确。每个时间点使用相同的排量、时间、流体黏度、平面应变模量、滤失系数、最小水平应力、断裂韧度和裂缝高度。

## 2. 输出量

两类模型统一整理为：

\[
L_{1/2},\quad w_{max},\quad A,\quad V,\quad p_{net},\quad p_{bh}.
\]

其中 PyFrac 原生 footprint/开度/压力由适配器提取；若字段需要计算，结果中标记为 `derived`。当前光纤没有独立的裂缝半长标签，因此下列误差只表示模型间差异：

\[
e_L=\frac{|L_{PyFrac}-L_{PKN}|}{\max(|L_{PyFrac}|,\epsilon)},
\quad
e_w=\frac{|w_{PyFrac}-w_{PKN}|}{\max(|w_{PyFrac}|,\epsilon)},
\quad
e_p=\frac{|p_{PyFrac}-p_{PKN}|}{\max(|p_{PyFrac}|,\epsilon)}.
\]

## 3. 教师数据与代理模型

代理模型不是从零生成裂缝，而是学习：

\[
\Delta z=z_{PyFrac}-z_{PKN},
\qquad
\hat z_{online}=z_{PKN}+\widehat{\Delta z}.
\]

输入包含 (E')、(C_L)、\(\mu\)、\(\sigma_{min}\)、\(K_{IC}\)、高度、簇排量、时间、簇距和真实液量分配权重。场景按参数组合分组后做 70%/15%/15% 划分，避免相邻时间点泄漏到测试集。

## 4. EnKF 位置

新 EnKF 状态为：

\[
\theta=[\log E',\log C_L,\log\mu,\sigma_{min},\log K_{IC}].
\]

观测更新：

\[
K=P_{\theta y}(P_{yy}+R)^{-1},
\qquad
\theta^{post}=\theta^{prior}+K(y-H(\theta^{prior})).
\]

随后必须重新调用 PKN、代理模型或离线 PyFrac：

\[
L^{post}=F(\theta^{post},Q,t).
\]

EnKF不能把后验缝长直接写回结果表。分簇液量比例已由光纤给出，不再作为自由生长因子参与二次更新。

## 5. 解释口径

- PKN：在线速度优先，适合作为主正演和 EnKF 快速前向算子。
- PyFrac：离线高保真参考/教师模型；其耗时不纳入在线 15 秒指标。
- 残差代理：在线近似 PyFrac 与 PKN 的差异，只有测试集误差和 P95 耗时达标后才进入在线 EnKF。
- 砂量：当前为光纤外部观测、分簇约束和风险校验，不是 PyFrac 原生砂输运结果。

## 6. 当前正式基准结论

正式教师样本位于 `outputs/dt/pyfrac_teacher_formal/teacher_samples.csv`，有效样本为
31 条、12 个参数场景组，按场景组划分为 70%/15%/15%。当前结果位于
`outputs/dt/pyfrac_benchmark_formal_final/benchmark_summary.json`：

- PKN 单次正演 P95 约 `0.085 ms`；
- PyFrac snapshot 参考计算 P95 约 `245.7 ms`；
- 残差代理单次推理 P95 约 `1.015 ms`；
- 代理测试集长度残差 `R²=-0.603`，压力残差 `R²=-0.569`；
- 因此 `online_enkf_approval=false`，当前代理模型不能作为正式在线 EnKF 前向模型。

这不是代码链路失败，而是质量门禁按预期阻止了一个留出泛化能力不足的代理模型进入在线闭环。后续应扩大 PyFrac 教师场景，增加排量突变、滤失、应力和黏度组合，并保持场景组留出测试。

> 注：上述“31 条、12 组”是早期正式基准，不代表当前最新结果。当前增量实验已升级到 `outputs/dt/pyfrac_teacher_iteration_v3`：96 个场景组、2001 条有效样本，采用独立残差头后，严格场景组测试长度 R2=`0.9985`、压力 R2=`0.8295`，候选代理门禁通过。默认在线主链仍为 PKN + EnKF，代理只在显式指定路径且通过门禁时作为候选修正层。完整扩展过程见 `docs/PyFrac代理模型扩展实验说明.md`。

## 7. 原生 PyFrac 对照

为区分“snapshot 快照初始化”和“原始 PyFrac 动态求解”，另行使用 `--pyfrac-mode native` 调用 PyFrac 的 `Controller.run()`。在真实第 8 段簇 1、161×101 网格和 360 s 暖启动条件下：

- 887 s 目标点原生推进到目标，耗时约 70.7 s，18 个成功时间步；
- 4435 s 目标点在 30 步默认上限下只能推进到 2445.99 s；提高到 70 步后推进到目标，耗时约 290.8 s，45 个成功时间步；
- 因此 `242.5 ms` 只能解释为 snapshot/网格化初始化耗时，不能作为原生 PyFrac 完整计算耗时；
- 原生 PyFrac 适合离线高保真参考和教师数据生成，不能直接作为在线 EnKF 前向算子满足 15 s 指标。

原生输出位于：

```text
outputs/dt/pyfrac_native_quality_checked_two_points/
outputs/dt/pyfrac_native_late_70steps/
```

当前 PKN 与 PyFrac 的差异还不能作为现场精度结论。比较前必须统一 PyFrac 的平行板等效黏度 `muPrime=12*mu`、完整排量历史、累计滤失定义和压力边界。适配器现在会检查 native 最终时刻是否达到目标；未达到时输出 `pyfrac_success=false`，避免把未完成的数值推进当成有效结果。

验证脚本支持以下可选接入方式：

```powershell
python DT-Crack\inversion\validate_direct_observations.py `
  --frac-monitor-text Data\3Dfrac\光纤本井监测08.txt `
  --construction-pressure-xls Data\3Dfrac\JY84-Z1-stage08-f1.xls `
  --surrogate-path outputs\dt\pyfrac_benchmark_formal_final\pyfrac_residual_surrogate.joblib `
  --run-dir outputs\dt\pyfrac_enkf_with_surrogate
```

若代理没有通过 `R²>=0.80` 门禁，脚本会记录质量原因并自动回退到 PKN。只有开发调试时明确增加
`--allow-unapproved-surrogate` 才会强制使用未达标代理；这种结果必须标记为 development-only，不能用于验收结论。
