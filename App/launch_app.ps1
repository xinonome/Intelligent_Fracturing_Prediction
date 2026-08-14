$ErrorActionPreference = 'Stop'
$env:HOME = $env:USERPROFILE
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$root = Split-Path -Parent $PSScriptRoot
$python = if ($env:FRACTURING_QT_PYTHON) {
    $env:FRACTURING_QT_PYTHON
} else {
    (Get-Command python -ErrorAction Stop).Source
}
$appScript = Join-Path -Path $root -ChildPath 'App\run_app.py'

if (-not (Test-Path -LiteralPath $python)) {
    throw 'PySide environment not found.'
}

& $python -c "from PySide6.QtCore import qVersion; print('Qt preflight OK:', qVersion())"
if ($LASTEXITCODE -ne 0) {
    throw 'Qt preflight failed in frac_app.'
}

Set-Location -LiteralPath $root
& $python $appScript '--no-auto-env'
