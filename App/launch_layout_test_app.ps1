$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$root = Split-Path -Parent $PSScriptRoot
$python = if ($env:FRACTURING_QT_PYTHON) { $env:FRACTURING_QT_PYTHON } else { 'C:\Users\xinonome\anaconda3\envs\frac_app\python.exe' }
$appScript = Join-Path -Path $root -ChildPath 'App\layout_test_app.py'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Qt Python environment not found: $python"
}

Set-Location -LiteralPath $root
& $python $appScript
