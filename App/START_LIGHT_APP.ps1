$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$pythonPath = if ($env:FRACTURING_QT_PYTHON) { $env:FRACTURING_QT_PYTHON } else { "C:\Users\xinonome\anaconda3\envs\frac_app\python.exe" }
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
& $pythonPath (Join-Path $projectRoot "App\run_app.py") --theme light @args
