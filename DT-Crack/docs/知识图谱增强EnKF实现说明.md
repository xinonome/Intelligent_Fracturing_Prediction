# 知识图谱增强 EnKF 实现说明

## 1. 设计边界

本版本实现的是 **KG-EnKF prior bridge**：知识图谱影响 EnKF 的先验分布、参数协方差、压力观测置信度和单步参数更新幅度；实时观测仍由 EnKF 完成同化，PKN 仍是物理前向算子。

知识图谱不直接指定最终缝长、排量或砂比，也不把等效观测缝长直接写入模型状态。每一次分析更新后，都必须重新调用 PKN，得到新的裂缝长度、开度、压力和体积。

```text
规则/工况信号
    -> 参数先验均值、方差和相关性
    -> 生成带物理边界的 ensemble
    -> PKN 正演得到观测空间预测
    -> EnKF 依据实时残差更新参数
    -> 更新后的参数重新运行 PKN
    -> 输出后验裂缝状态
```

## 2. EnKF 状态、观测和更新公式

当前全局物理状态为五维向量：

\[
\theta = [\log E',\;\log C_L,\;\log\mu,\;\sigma_{min},\;\log K_{IC}]^T
\]

其中 (E') 是平面应变模量，(C_L) 是滤失系数，\(\mu\) 是流体黏度，\(\sigma_{min}) 是最小水平主应力，\(K_{IC}) 是断裂韧度。簇间液量分配不再作为自由增长因子，而是来自光纤累计液量及其轻量平滑后的分配比例。

第 (m) 个 ensemble 成员由先验分布生成：

\[
\theta_0^{(m)} \sim \mathcal N(\mu_{KG}, P_{KG})
\]

知识图谱只改变 (\mu_{KG}) 和 (P_{KG})，然后通过 PKN 产生预测观测：

\[
\hat y^{(m)} = H(F_{PKN}(\theta_0^{(m)}, Q_t, t))
\]

标准 EnKF 更新为：

\[
K_t=P_{\theta y}(P_{yy}+R_t)^{-1}
\]

\[
\theta_t^{post,(m)}=\theta_t^{prior,(m)}+K_t(y_t+\epsilon_t^{(m)}-\hat y_t^{(m)})
\]

其中 (y_t) 包含分簇液量份额、分簇砂量份额和井底压力。更新后重新正演：

\[
S_t^{post}=F_{PKN}(\bar\theta_t^{post}, Q_t, t)
\]

这里 (S_t^{post}) 才是后验裂缝状态。代码中没有“EnKF 后验缝长 = 观测缝长”的赋值路径。

## 3. 知识图谱如何进入先验

输入规则来自：

```text
FSL-Expert/rule_fusion/rule_fusion/fused_sand_plug_rules.json
```

当前从校准区间提取四类可审计信号：

| 信号 | 计算方式 | 影响 |
|---|---|---|
| 压力上升 | `max(BHP)-BHP_start` 归一化 | 增大滤失/应力解释的不确定性 |
| 压力斜率 | 相邻压力差除以时间间隔 | 提高压力异常风险 |
| 高砂比 | 校准区间最大砂比 | 增大黏度与滤失耦合的不确定性 |
| 排量下降 | 排量下降的时间比例 | 增大堵塞/近井筒损失解释的不确定性 |

综合风险分数为：

\[
r=clip(0.40r_p+0.30r_{dp/dt}+0.20r_s+0.10r_q,0,1)
\]

该分数不是异常工况概率，而是“知识规则对当前先验应保持多大不确定性”的控制量。

### 3.1 先验均值

在 `soft_prior` 和 `soft_correlated` 下，使用小幅、可追溯的软偏移：

\[
\mu_{KG}=\mu_0+r\cdot\alpha\cdot
[0,\;0.10r_p,\;0.04\max(r_s,r_q),\;0.80r_p,\;0]^T
\]

这只是对“压力上升可能由滤失或最小主应力解释”的先验假设，不是最终参数答案。当前规则没有提供足够证据识别 (K_{IC})，因此该参数均值保持不偏移。

### 3.2 协方差和过程噪声

用标准差向量 (d) 和规则相关矩阵 (C_{KG}) 构造：

\[
P_{KG}=diag(d)C_{KG}diag(d)
\]

当前只加入小幅、可解释的相关关系：

- 压力异常使 (C_L) 与 \(\sigma_{min}\) 正相关，避免单个压力通道强行决定唯一原因；
- 高砂比/排量下降使 (C_L) 与 \(\mu\) 弱相关；
- 风险升高时 (E') 与 \(\sigma_{min}\) 保持弱相关。

协方差先投影到半正定矩阵，确保多元正态采样稳定。所有参数仍经过物理边界裁剪。

### 3.3 观测误差和更新幅度

在 `soft_correlated` 下，只放宽井底压力观测标准差：

\[
R_{p,t}^{1/2}=\gamma_pR_{p,0}^{1/2},\quad
\gamma_p=1+\alpha_R\cdot strength\cdot r
\]

因为当前规则主要描述压力上升、摩阻和堵塞解释不确定性；光纤液量分配是已知边界输入，不应被图谱再次“降权”。

同时限制一次分析更新：

\[
|\theta^{post}-\theta^{prior}|\leq
\gamma_\theta d_{KG}
\]

这是防止单次异常观测造成物理参数跳变，不是替代 Kalman 增益。

## 4. 四种模式

| 模式 | 知识图谱作用 | 用途 |
|---|---|---|
| `off` | 不接入 | PKN-EnKF 基线 |
| `uncertainty_only` | 只放大先验/过程不确定性 | 与旧实验兼容 |
| `soft_prior` | 增加小幅先验均值偏移，并调整方差 | 检查规则方向是否合理 |
| `soft_correlated` | 先验均值 + 协方差 + 压力观测置信度 + 更新限幅 | 当前最完整的 KG-EnKF 原型 |

所有模式都走同一个 EnKF 观测更新和 PKN 重算流程。

## 5. 复现实验

基线：

```powershell
python DT-Crack\inversion\validate_direct_observations.py `
  --frac-monitor-text Data\3Dfrac\光纤本井监测08.txt `
  --construction-pressure-xls Data\3Dfrac\JY84-Z1-stage08-f1.xls `
  --physics-profile enhanced `
  --validation-mode frozen `
  --knowledge-guided-mode off `
  --max-steps 60 --ensemble-size 80 --seed 2026
```

完整 KG-EnKF：

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

四种模式同口径对比：

```powershell
python DT-Crack\inversion\compare_knowledge_guided_modes.py `
  --frac-monitor-text Data\3Dfrac\光纤本井监测08.txt `
  --construction-pressure-xls Data\3Dfrac\JY84-Z1-stage08-f1.xls `
  --max-steps 60 --ensemble-size 80 `
  --output-root outputs\dt\kg_enkf_mode_comparison
```

输出包括每种模式的 `summary.json`、历史 CSV，以及总表：

```text
outputs/dt/kg_enkf_mode_comparison/kg_enkf_mode_comparison.csv
outputs/dt/kg_enkf_mode_comparison/kg_enkf_mode_comparison.json
```

## 6. 当前验证结果与边界

在 2026-08-13、同一随机种子和同一留出划分的正式对照中：

| 模式 | 砂量 TVD | 井底压力误差 | 三项同时≤15%比例 | P95 |
|---|---:|---:|---:|---:|
| `off` | 13.285% | 16.687% | 38.89% | 143.3 ms |
| `soft_correlated` | 13.340% | 13.869% | 50.00% | 149.6 ms |

液量 TVD 在该口径下为 0%，长度回缩超过 5% 的次数为 0。这个结果说明 KG 先验在该数据切分上改善了压力留出稳定性，但砂量误差略有上升，不能宣称知识图谱在所有井段都提升精度。

当前限制：

1. 规则到参数方向的映射是人工审计的工程桥接，需要跨井留出验证；
2. 当前实验主要是第 8 段，不能代表跨井泛化；
3. 规则不是异常概率模型，也不替代专业人员确认；
4. 只有观测空间误差可直接计算，不能把它等同于独立真实裂缝几何真值误差；
5. 如果知识规则与实时观测冲突，仍应由观测更新和物理边界决定后验，不能让规则覆盖观测。
