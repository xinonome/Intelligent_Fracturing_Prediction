[CmdletBinding()]
param([switch]$SkipZip)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$patentRoot = Join-Path $projectRoot '专利与软著\deliverables\专利'
$packageRoot = Join-Path $patentRoot '测试软件\多簇压裂阶段液量均衡调控方法测试软件V1.0'
$stagingRoot = Join-Path $patentRoot '测试软件\.staging'

if (Test-Path -LiteralPath $stagingRoot) { Remove-Item -LiteralPath $stagingRoot -Recurse -Force }
New-Item -ItemType Directory -Path $stagingRoot -Force | Out-Null

Copy-Item -LiteralPath (Join-Path $patentRoot '02_Balancing_Control.py') -Destination (Join-Path $stagingRoot '02_Balancing_Control.py') -Force
Copy-Item -LiteralPath (Join-Path $patentRoot '03_Patent_Test_App.py') -Destination (Join-Path $stagingRoot '03_Patent_Test_App.py') -Force

$launcher = @'
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$conda = 'C:\Users\xinonome\anaconda3\Scripts\conda.exe'
if ($env:FRACTURING_QT_PYTHON) {
    & $env:FRACTURING_QT_PYTHON (Join-Path $PSScriptRoot '03_Patent_Test_App.py')
} elseif (Test-Path -LiteralPath $conda) {
    & $conda run -n base --no-capture-output python (Join-Path $PSScriptRoot '03_Patent_Test_App.py')
} else { throw '未找到运行环境，请设置 FRACTURING_QT_PYTHON。' }
'@
$selfTest = @'
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$conda = 'C:\Users\xinonome\anaconda3\Scripts\conda.exe'
if ($env:FRACTURING_QT_PYTHON) {
    & $env:FRACTURING_QT_PYTHON (Join-Path $PSScriptRoot '03_Patent_Test_App.py') --self-test
} elseif (Test-Path -LiteralPath $conda) {
    & $conda run -n base --no-capture-output python (Join-Path $PSScriptRoot '03_Patent_Test_App.py') --self-test
} else { throw '未找到运行环境，请设置 FRACTURING_QT_PYTHON。' }
if ($LASTEXITCODE -ne 0) { throw "自检失败，退出码：$LASTEXITCODE" }
'@
$readme = @'
# 多簇压裂阶段液量均衡调控方法测试软件 V1.0

运行 `START_SOFTWARE.ps1` 打开测试界面，运行 `SELF_TEST.ps1` 执行完整计算自检。

软件可选择有 DAS、无 DAS 或自动识别场景；支持导入施工压力表和分簇观测表。留空时使用随程序生成的可复现实验数据。计算结果写入 `runs`，包括压力与状态历史、三类参数轨迹、六簇结果、Piggy-Bank 审计记录和总览图。

当前版本用于专利方法的离线功能验证，不连接现场泵注控制系统；阶段液量输出为建议和审计记录，不代表设备已经执行。
'@
$requirements = @'
PySide6>=6.5
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
openpyxl>=3.1
'@
Set-Content -LiteralPath (Join-Path $stagingRoot 'START_SOFTWARE.ps1') -Value $launcher -Encoding UTF8
Set-Content -LiteralPath (Join-Path $stagingRoot 'SELF_TEST.ps1') -Value $selfTest -Encoding UTF8
Set-Content -LiteralPath (Join-Path $stagingRoot 'README.md') -Value $readme -Encoding UTF8
Set-Content -LiteralPath (Join-Path $stagingRoot 'requirements.txt') -Value $requirements -Encoding UTF8

$manifest = [ordered]@{
    name = '多簇压裂阶段液量均衡调控方法测试软件'
    version = 'V1.0'
    patent = '一种基于知识引导协同同化的多簇压裂阶段液量均衡调控方法'
    entrypoint = 'START_SOFTWARE.ps1'
    self_test = 'SELF_TEST.ps1'
    scenarios = @('with_das', 'no_das', 'auto')
    outputs = @('history.csv', 'parameter_trajectory.csv', 'cluster_results.csv', 'piggy_bank_audit.csv', 'summary.json', 'patent_demo_overview.png')
    field_control_connected = $false
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $stagingRoot 'release_manifest.json') -Encoding UTF8

if (Test-Path -LiteralPath $packageRoot) { Remove-Item -LiteralPath $packageRoot -Recurse -Force }
New-Item -ItemType Directory -Path (Split-Path -Parent $packageRoot) -Force | Out-Null
Move-Item -LiteralPath $stagingRoot -Destination $packageRoot

if (-not $SkipZip) {
    $zip = "$packageRoot.zip"
    if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
    Compress-Archive -Path (Join-Path $packageRoot '*') -DestinationPath $zip -CompressionLevel Optimal
}

Get-ChildItem -LiteralPath $packageRoot -Recurse -File | Select-Object FullName, Length
