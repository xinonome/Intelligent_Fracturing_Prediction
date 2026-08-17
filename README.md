# 智能压裂预测项目

本目录是合同三部分主线的精简交付版：

1. `FSL-Expert`：知识图谱、小样本工况识别、迁移学习和状态转移预测。
2. `DT-Crack`：真实DAS/压力/井轨迹接入，PKN/BEM正演，EnKF物理参数反演和3D展示。
3. `HMI-KE`：知识约束决策、Gymnasium环境、PPO/SAC、多工况训练和180秒验证。

真实数据只存放在 `Data`，代表性结果存放在 `artifacts`，新运行结果统一写入 `outputs`。

## 安装

```powershell
pip install -r requirements-core.txt
pip install -r requirements-kg.txt
pip install -r requirements-rl.txt
pip install -r requirements-ui.txt
```

## 统一入口

```powershell
python run_project.py fsl knowledge-graph
python run_project.py fsl train
python run_project.py dt validate
python run_project.py dt benchmark
python run_project.py dt visualize --open
python run_project.py hmi train --total-timesteps 5000
python run_project.py hmi scenarios --total-timesteps 1000
python run_project.py app
python run_project.py test
```

无图形界面检查：

```powershell
python App\run_app.py --no-gui
```

## 科学口径

当前数字孪生正式验证使用分簇液量、砂量分布和井底压力直接观测空间。现有数据没有独立真实缝长标签，因此不得把观测空间误差表述为真实裂缝几何精度。

详细交接见 `docs/项目交接说明.md`。

## 授权发布包

甲方可运行的发布副本由 `tools/prepare_github_release.ps1` 生成。该脚本按白名单复制当前源码、`Data/raw_frac`、`Data/3Dfrac`、`artifacts` 以及 APP 注册表引用的冻结结果；不复制 `Data/multimodal` 原始书籍 PDF、历史实验目录和本地缓存。

在 Windows PowerShell 中从本项目目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File tools\prepare_github_release.ps1 `
  -SourceRoot "C:\Workspace\Intelligent_Fracturing_Prediction" `
  -ReleaseRoot "C:\Workspace\FSL_Expert_GitHub_Release" `
  -IncludeAuthorizedData `
  -PruneLegacyGenerated
```

发布目录生成后，甲方可先执行：

```powershell
python App\run_app.py --no-gui
python run_project.py hmi validate-env --strict
```
