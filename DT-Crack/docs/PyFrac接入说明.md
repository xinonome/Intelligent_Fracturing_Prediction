# PyFrac 接入说明

## 1. 定位

本项目不把 PyFrac 直接替换在线 PKN。当前采用分层路线：

```text
PyFrac 离线高保真参考/教师模型
            -> 生成 PKN 残差样本
            -> 训练残差代理模型
            -> 代理模型进入在线 EnKF
            -> EnKF 更新物理参数后重新正演
```

PKN 仍是在线快速基线，PyFrac 用于检查复杂裂缝物理对 PKN 的偏差。没有独立裂缝几何真值时，PKN 与 PyFrac 的差异只能作为模型差异，不能直接称为现场误差。

## 2. 第三方源码与许可证

源码位于 `DT-Crack/third_party/PyFrac`，版本和 commit 记录在 `PROJECT_METADATA.json`。PyFrac 源码没有被本项目修改，适配代码位于 `DT-Crack/forward_models/pyfrac_adapter.py`。交付包必须同时保留 `GPL.txt`、`LICENSE.TXT`、版权信息和本说明。

## 3. 两种运行模式

### Snapshot

`snapshot` 为当前默认模式。适配器根据给定的排量、时间、黏度、弹性模量、滤失、应力、断裂韧度和裂缝高度建立 PyFrac 网格，并以 PKN 初始化状态提取裂缝 footprint、开度、面积、体积和压力。它适合批量教师数据和模型空间对照，不是完整时间推进。

### Native

`native` 显式调用 PyFrac `Controller.run()` 做时间推进。老版本 PyFrac 对 NumPy 版本、网格分辨率和前缘推进条件较敏感，失败时返回 `success=false` 和错误原因，不能自动改名为成功结果。

## 4. 真实数据映射

当前 Stage 08 输入为：

- `Data/3Dfrac/光纤本井监测08.txt`：六簇累计液量、累计砂量、均衡程度和时间。
- `Data/3Dfrac/JY84-Z1-stage08-f1.xls`：施工压力、排量等压力换算输入。
- `Data/3Dfrac/JY84-Z1HF-1011.csv`：测深、垂深、北向和东向井轨迹。

第 (i) 簇的液量边界按真实累计液量增量分配：

\[
Q_i(t)=Q_{total}(t)\frac{\Delta V_i(t)}{\sum_j\Delta V_j(t)}.
\]

当本步所有增量为零时沿用上一个有效比例，绝不默认六簇均分。砂量不被伪装成 PyFrac 内部支撑剂输运，而作为外部校验和风险约束。

## 5. 运行示例

```powershell
$env:PYTHONPATH = "$PWD\DT-Crack"
python DT-Crack\forward_models\run_pyfrac_single_cluster.py `
  --fiber-monitor Data\3Dfrac\光纤本井监测08.txt `
  --pressure-xls Data\3Dfrac\JY84-Z1-stage08-f1.xls `
  --trajectory-csv Data\3Dfrac\JY84-Z1HF-1011.csv `
  --stage 8 --cluster 1 --model pkn --max-points 12 `
  --output-dir outputs\dt\pyfrac_single_cluster\pkn

python DT-Crack\forward_models\run_pyfrac_single_cluster.py `
  --fiber-monitor Data\3Dfrac\光纤本井监测08.txt `
  --pressure-xls Data\3Dfrac\JY84-Z1-stage08-f1.xls `
  --trajectory-csv Data\3Dfrac\JY84-Z1HF-1011.csv `
  --stage 8 --cluster 1 --model pyfrac --pyfrac-mode snapshot --max-points 12 `
  --output-dir outputs\dt\pyfrac_single_cluster\pyfrac
```

六簇外部耦合 smoke：

```powershell
python DT-Crack\forward_models\run_pyfrac_multicluster.py `
  --fiber-monitor Data\3Dfrac\光纤本井监测08.txt `
  --pressure-xls Data\3Dfrac\JY84-Z1-stage08-f1.xls `
  --trajectory-csv Data\3Dfrac\JY84-Z1HF-1011.csv `
  --stage 8 --max-points 6 `
  --output-dir outputs\dt\pyfrac_multicluster
```

该命令把真实光纤累计液量增量转换为六簇排量边界，然后对每簇执行一个 PyFrac snapshot，再用外部应力阴影矩阵得到 `coupled_half_length_m`。`pyfrac_half_length_m` 是单簇参考，`coupled_half_length_m` 是耦合展示量，两者都不是现场裂缝几何真值。

第二条命令输出 PKN、PyFrac 和对照曲线；`single_cluster_metrics.json` 中的 `length_mean_relative_error` 是模型间相对差异，不是现场缝长精度。

## 6. 当前边界

- 当前优先验证单簇，六簇由多个单簇 PyFrac 实例和外部应力阴影矩阵耦合，不声称是 PyFrac 原生六簇水平井求解。
- PyFrac 离线计算可能较慢或因网格分辨率失败；在线 15 秒指标只评价 PKN 或代理模型加 EnKF。
- 若需要把 PyFrac直接用于验收交付，应由法务确认 GPLv3 对源码、衍生适配和分发方式的要求。

## 7. 本轮扩展状态

本轮教师数据扩展结果位于 `outputs/dt/pyfrac_teacher_iteration_v3`，共生成 96 个严格隔离的场景组、2001 条有效簇级样本，覆盖排量突变、增强滤失、应力变化、黏度变化和六簇液量分配组合。残差代理改为长度、缝宽、压力三个独立 MLP 输出头，并增加排量/黏度比、压力/应力比、时间尺度、滤失累计尺度和排量变化率等派生特征。

严格场景组留出结果为：长度残差测试 R2=`0.9985`，压力残差测试 R2=`0.8295`，代理推理 P95=`47.7 ms`（120点基准），因此代理通过了“长度、压力 R2 均不低于 0.80 且在线推理小于 15 秒”的候选门禁。该门禁只说明代理对 PyFrac 教师残差具备留出拟合能力，不能代替真实裂缝几何真值验证。

默认在线主链仍保持 PKN + EnKF。只有在显式提供 `--surrogate-path` 且门禁通过时，验证脚本才允许尝试叠加代理残差；真实第8段的整体验证仍需单独查看液量、砂量、井底压力和同时达标率，不能仅凭代理 R2 宣称合同指标完成。详细口径见 `docs/PyFrac代理模型扩展实验说明.md`。

## 8. 原生 PyFrac 直接对照结果

本轮新增了原生 `Controller.run()` 的直接对照，不再把 snapshot 初始化耗时当作原生 PyFrac 耗时。原生运行使用真实第 8 段、簇 1、161×101 网格，并从 360 s 的已分辨 PKN 状态暖启动；暖启动是为避免 1 s 裂缝小于网格单元而设置的数值条件，不代表现场工艺阶段被删除。

复现实验命令：

```powershell
python DT-Crack\forward_models\run_pyfrac_single_cluster.py `
  --fiber-monitor Data\3Dfrac\光纤本井监测08.txt `
  --pressure-xls Data\3Dfrac\JY84-Z1-stage08-f1.xls `
  --trajectory-csv Data\3Dfrac\JY84-Z1HF-1011.csv `
  --stage 8 --cluster 1 --model pyfrac --pyfrac-mode native `
  --min-time-s 887 --max-points 2 `
  --mesh-nx 161 --mesh-ny 101 `
  --native-start-time-s 360 --max-time-steps 70 `
  --output-dir outputs\dt\pyfrac_native_quality_checked_two_points
```

结果位于 `outputs/dt/pyfrac_native_quality_checked_two_points`：

| 目标时刻 | 原生推进结果 | 原生耗时 | 原生最终时刻 | 成功步数 |
|---:|---|---:|---:|---:|
| 887 s | 达到目标 | 约 70.7 s | 887 s | 18 |
| 4435 s | 在 30 步上限下未完成；提高到 70 步后达到目标 | 约 290.8 s | 4435 s | 45 |

这说明之前 benchmark 中的 `242.5 ms` 是 PyFrac snapshot 参考计算的耗时，不是原始 PyFrac 的完整时间推进耗时。原生模型可以作为离线高保真教师，但不能直接放入在线 EnKF 的每个时间步，更不能在 15 s 指标中把它当作在线模型。

在当前未完全标定的参数映射下，簇 1 的模型空间输出示例为：

| 时刻 | 项目 PKN 半缝长 | 原生 PyFrac 半缝长 | 项目 PKN 最大开度 | 原生 PyFrac 最大开度 |
|---:|---:|---:|---:|---:|
| 887 s | 54.62 m | 27.00 m | 0.329 mm | 1.635 mm |
| 4435 s | 102.73 m | 36.00 m | 0.236 mm | 1.323 mm |

这些差异首先用于发现模型和参数映射差异，不能直接解释为真实裂缝误差。当前对照还存在两个必须校准的问题：

1. PyFrac 使用平行板等效黏度 `muPrime=12*mu`，项目快速 PKN 的 `calc_pkn` 使用项目侧黏度约定；两者必须统一黏度口径后才能比较长度和开度。
2. 当前单簇脚本用该时刻簇排量做同条件模型对照，而现场施工是变化排量历史；严格对比应让 PKN 和 PyFrac都读取同一段完整排量历史，并使用同一累计滤失/注入量定义。

因此，本轮原生结果的正确结论是：PyFrac 已经真实接入并可在选定时刻完成原生推进，但其计算量远超 15 s，且尚未完成物理参数和历史排量口径标定。下一步应先统一黏度、压力、历史排量和滤失定义，再用少量原生 PyFrac 检查点训练残差代理，最后由质量门禁决定是否进入在线 EnKF。
