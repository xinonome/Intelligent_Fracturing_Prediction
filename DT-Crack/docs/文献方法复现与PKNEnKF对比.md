# 领域文献方法复现与当前 PKN/EnKF 对比

## 1. 对比目的

本次工作不是简单罗列论文指标，而是把与合同第二部分最接近的研究路线拆成可验证的算法模块，在同一份真实第 8 段数据上进行对比：

```text
真实光纤分簇液量/砂量 + 施工压力
  -> 统一观测空间
  -> 同一增强 PKN 正演
  -> 文献型全时间步反演 / 当前参数空间 EnKF
  -> 后30%留出验证
```

这样可以区分三个问题：

1. 正演模型本身能否计算；
2. 采用一次性历史拟合还是在线数据同化更适合当前数据；
3. 论文中依赖的原始 DAS/DTS 数据，在当前项目中哪些已经具备，哪些还没有。

## 2. 参考文献与方法拆解

### 2.1 DAS + PKN + 迭代反演

参考：Hu 等，*A Hydraulic Fracture Geometry Inversion Model Based on Distributed-Acoustic-Sensing Data*，SPE Journal，DOI: [10.2118/214306-PA](https://doi.org/10.2118/214306-PA)。论文使用 LF-DAS、HF-DAS 和注入排量等多源数据，在 PKN 或类似裂缝模型基础上，对裂缝长度、宽度随时间进行迭代反演。

论文方法的核心结构是：

```text
DAS 信号特征 + 注入排量
  -> 裂缝几何正演
  -> 计算信号/观测残差
  -> 迭代调整几何或物理参数
  -> 重新正演
```

当前项目可以真实复现的部分：

- PKN 几何正演；
- 多时间步拟合；
- 分簇液量、砂量和井底压力残差；
- 参数更新后重新运行 PKN。

当前不能声称完全复现的部分：

- 当前 `光纤本井监测08.txt` 没有原始 LF-DAS/HF-DAS 应变率矩阵；
- 没有原始 DAS 的空间采样井深坐标和信号预处理结果；
- 因此本次使用“分簇液量/砂量份额 + 压力”的工程观测算子，而不是论文的原始 DAS 信号观测算子。

### 2.2 DTS + EnKF 参数同化

参考：Tarrahi 等，*Dynamic Integration of DTS Data for Hydraulically Fractured Reservoir Characterization with the Ensemble Kalman Filter*，SPE 169990-MS，DOI: [10.2118/169990-MS](https://doi.org/10.2118/169990-MS)。该方法建立非等温正演模型，将裂缝几何、导流能力等参数映射为 DTS 温度响应，再用 EnKF 按时间顺序同化温度数据。

其关键思想不是直接改裂缝长度，而是：

\[
\mathbf{x}_{t}^{a}
=\mathbf{x}_{t}^{f}
+K_t\left(\mathbf{y}_t-H(F(\mathbf{x}_{t}^{f},\mathbf{u}_t))\right)
\]

其中：

- \(\mathbf{x}\)：裂缝长度、导流能力、渗透率或其他物理参数；
- \(F\)：物理正演模型；
- \(H\)：温度观测算子；
- \(\mathbf{y}_t\)：当前 DTS 观测；
- \(K_t\)：由集合协方差计算的 Kalman Gain。

当前项目与该路线的对应关系：

```text
PKN 参数集合
  -> PKN 裂缝几何和压力正演
  -> 分簇液量/砂量/压力观测算子
  -> EnKF 更新 E'、CL、mu、sigma_min、分簇参数
  -> 更新参数重新运行 PKN
```

区别在于当前没有 DTS 温度，因此不能宣称已完成 DTS 温度反演；当前实现是“DTS-EnKF 的参数同化结构迁移到现有液量/砂量/压力观测”的工程版本。

### 2.3 VAE/KAN + EnKF 快速代理

参考：Zhou 等，*Data-Driven Real-Time Inversion of Hydraulic Fracture Geometry and Stress Fields Using a Deep-Kolmogorov-Arnold Network Assisted Data Assimilation Approach*，SPE 228112-MS，以及其期刊扩展路线 *Characterizing the Hydrodynamic and Mechanical Properties of Hydraulic Fractured Shale Plays Using a Kolmogorov-Arnold-Network-Assisted Data Assimilation Approach*，Engineering Applications of Artificial Intelligence，DOI: [10.1016/j.engappai.2025.110380](https://doi.org/10.1016/j.engappai.2025.110380)。

该路线大致为：

```text
大量裂缝网络图像
  -> VAE 压缩为低维裂缝潜变量
  -> KAN 代理模型预测施工压力
  -> EnKF 同化真实压力
  -> VAE 解码得到裂缝网络和应力场
```

论文摘要给出的路线包括约 80,000 个合成裂缝网络图像和应力场 realizations，再训练 KAN 代理。当前项目缺少：

- 80,000 级裂缝网络训练样本；
- 原始裂缝网络图像或网格；
- 现场应力场真值；
- 足够的压力-几何配对样本。

因此本次没有伪称“实现 KAN/VAE 方法”。项目已有的 `data_surrogate` 是基于 PKN/BEM 生成样本的轻量代理模型，只能作为接口和速度基线，不能等同于论文中的 VAE/KAN。

### 2.4 DAS 全时间步反演的补充参考

另一项可参考的工作是 Liu 等，*Enhancing Understanding of Hydraulic Fracture Tip Advancement through Inversion of Low-Frequency Distributed Acoustic Sensing Data*。该研究使用三维位移不连续/边界元风格正演和迭代反演从 LF-DAS 估计裂缝尖端推进，并强调全时间步裂缝演化的一致性。当前项目保留 reduced-order BEM 接口，但本次正式对比采用 PKN，以保证和实际第 8 段液砂/压力观测算子一致。

## 3. 本次实现的文献型基线

新增脚本：

`DT-Crack/inversion/compare_literature_inversion_methods.py`

### 3.1 全时间步 PKN 最小二乘反演

该方法对应 DAS-PKN 文献中的“多时间步迭代拟合”思想，但根据当前可用数据采用最小二乘/MAP形式：

\[
\hat{\mathbf{x}}
=\arg\min_{\mathbf{x}}
\sum_{t\in\mathcal{C}}
\left\|R_t^{-1/2}
\left[\mathbf{y}_t-H(F_{PKN}(\mathbf{x},\mathbf{u}_t))\right]\right\|_2^2
+\lambda\left\|S_x^{-1}(\mathbf{x}-\mathbf{x}_0)\right\|_2^2
\]

其中：

- \(\mathcal{C}\) 是前 70% 校准时间步；
- \(\mathbf{x}\) 包含 PKN 全局参数和分簇参数；
- \(\mathbf{u}_t\) 包含累计注入量、当前排量和时间；
- \(\mathbf{y}_t\) 包含分簇液量份额、分簇砂量份额和井底压力；
- \(R_t\) 是观测噪声尺度；
- 正则项用于避免优化器选择不合理的等效参数组合。

拟合完成后固定一组参数，在后 30%时间步运行 PKN，不吸收新的验证观测。这是一个严格的历史匹配/留出外推基线。

### 3.2 当前参数空间 EnKF

当前方法使用相同增强 PKN 和相同观测算子，但在每个新观测到达时更新参数：

\[
P_{xy}=\frac{1}{N-1}X'Y'^T
\]

\[
P_{yy}=\frac{1}{N-1}Y'Y'^T+R
\]

\[
K=P_{xy}P_{yy}^{-1}
\]

\[
\mathbf{x}_t^a=\mathbf{x}_t^f
+K\left(\mathbf{y}_t-H(F_{PKN}(\mathbf{x}_t^f,\mathbf{u}_t))\right)
\]

随后使用 \(\mathbf{x}_t^a\) 重新运行 PKN。裂缝长度始终是正演输出，不是 EnKF 直接修改的状态变量。

## 4. 真实第 8 段对比结果

运行数据：

- `Data/3Dfrac/光纤本井监测08.txt`
- `Data/3Dfrac/JY84-Z1-stage08-f1.xls`
- 60 个时间步；42 步校准，18 步验证；增强耦合 PKN；400 个 EnKF 集合成员。

| 方法 | 液量 TVD | 砂量 TVD | 压力相对误差 | 验证步≤15%比例 | 计算说明 |
|---|---:|---:|---:|---:|---|
| 当前参数空间 EnKF | 1.563% | 1.890% | 1.643% | 100% | 每个新观测到达后在线更新参数；P95约1.03 s |
| 全时间步 PKN 最小二乘 | 43.995% | 53.947% | 6.088% | 0% | 前70%拟合静态参数；全拟合耗时约17.48 s |

结果说明：

1. 全时间步拟合在压力上仍有一定拟合能力，但对液量和砂量分配明显失配，说明单组静态分簇参数无法解释后续施工阶段的分簇动态变化。
2. 参数空间 EnKF 通过逐步吸收新观测，能够跟踪动态入口能力、砂量输运和压力偏置，因此液砂分配误差明显更低。
3. 全时间步拟合的单次 PKN 正演很快，但整体拟合耗时约 17.48 秒；如果把整段历史重新拟合放进在线循环，不能直接满足 15 秒更新口径。
4. 当前 EnKF 单步 P95 约 1.03 秒，保留了进一步增加观测算子、集合规模或 BEM 代理复杂度的计算空间。

## 5. 结果文件

完整对比运行目录：

`outputs/dt/literature_method_comparison/20260804_114004/`

主要产物：

- `summary.json`：两类方法、数据口径和限制说明；
- `method_comparison.csv`：数值对比表；
- `method_accuracy_comparison.png`：液量、砂量、压力误差对比；
- `method_runtime_comparison.png`：耗时对比；
- `full_time_step_lsq_history.csv`：全时间步拟合逐步结果；
- `online_enkf_history.csv`：当前在线 EnKF 逐步结果。

复现命令：

```powershell
python DT-Crack\inversion\compare_literature_inversion_methods.py `
  --frac-monitor-text Data\3Dfrac\光纤本井监测08.txt `
  --construction-pressure-xls Data\3Dfrac\JY84-Z1-stage08-f1.xls `
  --max-steps 60 `
  --lsq-max-nfev 80 `
  --run-dir outputs\dt\literature_method_comparison
```

## 6. 当前结论与下一步

当前最适合作为主线的是：

```text
增强 PKN 正演
  + 参数空间 EnKF 在线更新
  + 真实 DAS/光纤观测算子
  + 井底压力同步校正
```

下一步可以按数据到位情况推进：

1. 甲方提供原始 DAS 应变率矩阵后，增加 LF-DAS/HF-DAS 信号特征观测算子，替换当前分簇等效观测；
2. 甲方提供 DTS 后，增加温度正演子模型，将温度作为额外观测通道；
3. 获得裂缝长度/宽度独立解释结果后，增加真实几何空间验证，区分观测空间误差与几何误差；
4. 增加更高保真 BEM 训练样本后，再评估 VAE/KAN 或其他代理模型；
5. 在现有 `LengthForwardModel` 接口上比较 PKN、BEM 和代理模型，而不是改变 EnKF 的参数更新原则。

当前结论不能表述为“完整复现论文模型”，应表述为：

> 参考 DAS-PKN 的全时间步反演思想和 DTS-EnKF 的参数同化思想，结合当前可获得的分簇液量、砂量、施工压力和井轨迹数据，构建了一个可复现的 PKN 参数空间 EnKF 在线反演流程，并与全时间步静态 PKN 反演基线完成了真实第 8 段对比。
