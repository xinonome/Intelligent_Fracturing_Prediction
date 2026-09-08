[CmdletBinding()]
param(
    [string]$SourceRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$ReleaseBase = "",
    [switch]$SkipZip
)

$ErrorActionPreference = 'Stop'

function Assert-SafeReleasePath {
    param([string]$Source, [string]$Target)
    $sourcePath = [IO.Path]::GetFullPath($Source).TrimEnd('\', '/')
    $targetPath = [IO.Path]::GetFullPath($Target).TrimEnd('\', '/')
    $allowedRoot = [IO.Path]::GetFullPath((Join-Path $sourcePath 'deliverables')).TrimEnd('\', '/')
    if (-not $targetPath.StartsWith($allowedRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "发行目录必须位于项目 deliverables 下：$targetPath"
    }
    if ($targetPath.Equals($sourcePath, [StringComparison]::OrdinalIgnoreCase)) {
        throw '发行目录不能是项目根目录。'
    }
}

function Copy-Tree {
    param([string]$Source, [string]$Destination)
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        throw "缺少目录：$Source"
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    & robocopy $Source $Destination /E /R:2 /W:1 /COPY:DAT /DCOPY:DAT /XJ `
        /XD '__pycache__' '.pytest_cache' '.mypy_cache' '.ruff_cache' 'runs' `
        /XF '*.pyc' '*.log' '*.tmp' | Out-Null
    if ($LASTEXITCODE -gt 7) {
        throw "目录复制失败（$LASTEXITCODE）：$Source"
    }
}

function Copy-FileSafe {
    param([string]$Source, [string]$Destination)
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "缺少文件：$Source"
    }
    $destinationDirectory = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

function Copy-OptionalTree {
    param([string]$Source, [string]$Destination)
    if (Test-Path -LiteralPath $Source -PathType Container) {
        Copy-Tree $Source $Destination
    }
}

function Write-Launcher {
    param([string]$Directory, [string]$Edition)
    $launcher = @"
`$ErrorActionPreference = 'Stop'
`$env:PYTHONUTF8 = '1'
`$env:PYTHONIOENCODING = 'utf-8'
`$packageRoot = `$PSScriptRoot
`$qtPython = if (`$env:FRACTURING_QT_PYTHON) { `$env:FRACTURING_QT_PYTHON } else { 'C:\Users\xinonome\anaconda3\envs\frac_app\python.exe' }
if (-not (Test-Path -LiteralPath `$qtPython -PathType Leaf)) {
    throw '未找到 PySide6 Python。请安装 requirements-ui.txt 中的依赖，并设置 FRACTURING_QT_PYTHON。'
}
Set-Location -LiteralPath `$packageRoot
& `$qtPython (Join-Path `$packageRoot 'App\run_app.py') '--no-auto-env' '--edition' '$Edition'
"@
    $selfTest = @"
`$ErrorActionPreference = 'Stop'
`$env:PYTHONUTF8 = '1'
`$env:PYTHONIOENCODING = 'utf-8'
`$packageRoot = `$PSScriptRoot
`$qtPython = if (`$env:FRACTURING_QT_PYTHON) { `$env:FRACTURING_QT_PYTHON } else { 'C:\Users\xinonome\anaconda3\envs\frac_app\python.exe' }
if (-not (Test-Path -LiteralPath `$qtPython -PathType Leaf)) {
    throw '未找到 PySide6 Python。'
}
Set-Location -LiteralPath `$packageRoot
& `$qtPython (Join-Path `$packageRoot 'App\run_app.py') '--no-auto-env' '--edition' '$Edition' '--smoke-gui'
if (`$LASTEXITCODE -ne 0) { throw "启动检查失败，退出码：`$LASTEXITCODE" }
Write-Host '启动检查通过。' -ForegroundColor Green
"@
    Set-Content -LiteralPath (Join-Path $Directory 'START_SOFTWARE.ps1') -Value $launcher -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $Directory 'SELF_TEST.ps1') -Value $selfTest -Encoding UTF8
}

function Write-Manifest {
    param(
        [string]$Directory,
        [string]$SoftwareName,
        [string]$Edition,
        [string[]]$VisibleModules,
        [string[]]$TestData
    )
    $files = @(Get-ChildItem -LiteralPath $Directory -Recurse -File)
    $manifest = [ordered]@{
        software_name = $SoftwareName
        version = 'V1.0'
        edition = $Edition
        release_type = '软件著作权申请测试版'
        generated_at = (Get-Date).ToUniversalTime().ToString('o')
        entrypoint = 'START_SOFTWARE.ps1'
        self_test = 'SELF_TEST.ps1'
        visible_modules = $VisibleModules
        included_test_data = $TestData
        execution_mode = 'frozen_replay_with_test_data_import'
        field_control_connected = $false
        file_count = $files.Count
        total_bytes = [int64](($files | Measure-Object Length -Sum).Sum)
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Directory 'release_manifest.json') -Encoding UTF8
}

$SourceRoot = (Resolve-Path -LiteralPath $SourceRoot).Path
if ([string]::IsNullOrWhiteSpace($ReleaseBase)) {
    $ReleaseBase = Join-Path $SourceRoot 'deliverables\software_copyright_test_editions'
}
$ReleaseBase = [IO.Path]::GetFullPath($ReleaseBase)
Assert-SafeReleasePath $SourceRoot $ReleaseBase

$fslName = '压裂施工工况识别与风险预测软件V1.0_申请测试版'
$dtName = '智能压裂双场景数字孪生与安全建议软件V1.0_申请测试版'
$fslRelease = Join-Path $ReleaseBase $fslName
$dtRelease = Join-Path $ReleaseBase $dtName

foreach ($target in @($fslRelease, $dtRelease)) {
    Assert-SafeReleasePath $SourceRoot $target
    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $target | Out-Null
}

foreach ($release in @($fslRelease, $dtRelease)) {
    Copy-Tree (Join-Path $SourceRoot 'App') (Join-Path $release 'App')
    foreach ($file in @('requirements-core.txt', 'requirements-ui.txt', 'project_config.yaml')) {
        Copy-FileSafe (Join-Path $SourceRoot $file) (Join-Path $release $file)
    }
}

# 第一套软件：工况识别、逐点预测、风险评估和知识规则。
Copy-Tree (Join-Path $SourceRoot 'FSL-Expert') (Join-Path $fslRelease 'FSL-Expert')
Copy-Tree (Join-Path $SourceRoot 'artifacts\fsl') (Join-Path $fslRelease 'artifacts\fsl')
Copy-FileSafe (Join-Path $SourceRoot 'outputs\app\fsl_timeline_cache.json') (Join-Path $fslRelease 'outputs\app\fsl_timeline_cache.json')
New-Item -ItemType Directory -Force -Path (Join-Path $fslRelease 'Data\raw_frac') | Out-Null
Copy-FileSafe (Join-Path $SourceRoot 'Data\raw_frac\FDBH14.xlsx') (Join-Path $fslRelease 'SampleData\FSL\FDBH14.xlsx')
Copy-FileSafe (Join-Path $SourceRoot 'Data\raw_frac\FDBH26.xlsx') (Join-Path $fslRelease 'SampleData\FSL\FDBH26.xlsx')
Copy-FileSafe (Join-Path $SourceRoot 'docs\software_copyright_test_editions\01_FSL_测试说明.md') (Join-Path $fslRelease '测试说明.md')
Copy-FileSafe (Join-Path $SourceRoot 'docs\software_copyright_test_editions\01_FSL_测试记录表.md') (Join-Path $fslRelease '测试记录表.md')
Write-Launcher $fslRelease 'fsl'
Write-Manifest $fslRelease '压裂施工工况识别与风险预测软件' 'fsl' @('预测功能', '知识图谱功能', '数据导入', '跨井迁移工作台') @('冻结井段缓存', 'SampleData/FSL/FDBH14.xlsx', 'SampleData/FSL/FDBH26.xlsx')

# 第二套软件：双场景数字孪生、智能建议与联合演示。
Copy-Tree (Join-Path $SourceRoot 'DT-Crack') (Join-Path $dtRelease 'DT-Crack')
Copy-Tree (Join-Path $SourceRoot 'HMI-KE') (Join-Path $dtRelease 'HMI-KE')
Copy-OptionalTree (Join-Path $SourceRoot 'artifacts\dt') (Join-Path $dtRelease 'artifacts\dt')
Copy-OptionalTree (Join-Path $SourceRoot 'artifacts\hmi') (Join-Path $dtRelease 'artifacts\hmi')
Copy-Tree (Join-Path $SourceRoot 'outputs\dt\second_part_kg_enkf_20260824\20260824_004357') (Join-Path $dtRelease 'outputs\dt\second_part_kg_enkf_20260824\20260824_004357')
Copy-Tree (Join-Path $SourceRoot 'outputs\hmi\model_comparison_formal_v3_20260818\20260818_192543') (Join-Path $dtRelease 'outputs\hmi\model_comparison_formal_v3_20260818\20260818_192543')
Copy-Tree (Join-Path $SourceRoot 'outputs\dt\pyfrac_sigma_4435_volume_projection_refine1125_20260830_r1') (Join-Path $dtRelease 'outputs\dt\pyfrac_sigma_4435_volume_projection_refine1125_20260830_r1')
foreach ($file in @('dt_realtime_cache.json', 'dt_realtime_3d.html', 'dt_realtime_3d_no_das.html')) {
    Copy-FileSafe (Join-Path $SourceRoot "outputs\app\$file") (Join-Path $dtRelease "outputs\app\$file")
}
Copy-Tree (Join-Path $SourceRoot 'outputs\app\fiber_six_cluster') (Join-Path $dtRelease 'outputs\app\fiber_six_cluster')
Copy-Tree (Join-Path $SourceRoot 'outputs\app\datasets\raw_fdbh14') (Join-Path $dtRelease 'outputs\app\datasets\raw_fdbh14')
Copy-Tree (Join-Path $SourceRoot 'outputs\app\no_das_ppt_delivery_all_single_stages') (Join-Path $dtRelease 'outputs\app\no_das_ppt_delivery_all_single_stages')
Copy-Tree (Join-Path $SourceRoot 'Data\3Dfrac') (Join-Path $dtRelease 'Data\3Dfrac')
New-Item -ItemType Directory -Force -Path (Join-Path $dtRelease 'Data\raw_frac') | Out-Null
Copy-FileSafe (Join-Path $SourceRoot 'Data\raw_frac\FDBH14.xlsx') (Join-Path $dtRelease 'Data\raw_frac\FDBH14.xlsx')
Copy-FileSafe (Join-Path $SourceRoot 'Data\raw_frac\FDBH26.xlsx') (Join-Path $dtRelease 'Data\raw_frac\FDBH26.xlsx')
Copy-FileSafe (Join-Path $SourceRoot 'App\services\ml_runtime.py') (Join-Path $dtRelease 'App\services\ml_runtime.py')
Copy-FileSafe (Join-Path $SourceRoot 'requirements-kg.txt') (Join-Path $dtRelease 'requirements-kg.txt')
Copy-FileSafe (Join-Path $SourceRoot 'requirements-rl.txt') (Join-Path $dtRelease 'requirements-rl.txt')
Copy-FileSafe (Join-Path $SourceRoot 'docs\software_copyright_test_editions\02_DT_HMI_测试说明.md') (Join-Path $dtRelease '测试说明.md')
Copy-FileSafe (Join-Path $SourceRoot 'docs\software_copyright_test_editions\02_DT_HMI_测试记录表.md') (Join-Path $dtRelease '测试记录表.md')
Write-Launcher $dtRelease 'dt_hmi'
Write-Manifest $dtRelease '智能压裂双场景数字孪生与安全建议软件' 'dt_hmi' @('裂缝数字孪生', '智能决策与安全建议', '联合动态演示') @('JY84-Z1 Stage 08 DAS 测试数据', 'FDBH14 无 DAS 测试数据', '冻结 KG-EnKF/智能体回放结果', 'PyFrac 4435 s 原生推演轨迹')

$results = @()
foreach ($release in @($fslRelease, $dtRelease)) {
    $zipPath = "$release.zip"
    if (-not $SkipZip) {
        if (Test-Path -LiteralPath $zipPath) {
            Remove-Item -LiteralPath $zipPath -Force
        }
        Compress-Archive -Path (Join-Path $release '*') -DestinationPath $zipPath -CompressionLevel Optimal -Force
    }
    $files = @(Get-ChildItem -LiteralPath $release -Recurse -File)
    $results += [ordered]@{
        release_root = $release
        zip_path = if ($SkipZip) { $null } else { $zipPath }
        file_count = $files.Count
        size_mb = [math]::Round((($files | Measure-Object Length -Sum).Sum) / 1MB, 2)
    }
}
$results | ConvertTo-Json -Depth 6
