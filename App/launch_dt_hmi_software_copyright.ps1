$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$projectRoot = Split-Path -Parent $PSScriptRoot
$qtPython = if ($env:FRACTURING_QT_PYTHON) {
    $env:FRACTURING_QT_PYTHON
} else {
    'C:\Users\xinonome\anaconda3\envs\frac_app\python.exe'
}
$appScript = Join-Path -Path $projectRoot -ChildPath 'App\run_app.py'

if (-not (Test-Path -LiteralPath $qtPython -PathType Leaf)) {
    throw '未找到 PySide6 运行环境。请设置 FRACTURING_QT_PYTHON 后重试。'
}

Set-Location -LiteralPath $projectRoot
& $qtPython $appScript '--no-auto-env' '--edition' 'dt_hmi'
