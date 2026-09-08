[CmdletBinding()]
param(
    [switch]$RebuildNoDasViews
)

$ErrorActionPreference = "Stop"
$ScriptRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
if ((Split-Path -Leaf $ScriptRoot) -ieq "tools") {
    $ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptRoot "..")).Path
} else {
    $ProjectRoot = $ScriptRoot
}
Set-Location -LiteralPath $ProjectRoot

function Find-Python {
    foreach ($name in @("python", "py")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) {
            if ($name -eq "py") {
                return @($command.Source, "-3.11")
            }
            return @($command.Source)
        }
    }
    throw "Python was not found. Install Python 3.10 or newer and add it to PATH."
}

$PythonParts = Find-Python
$Python = $PythonParts[0]
$PythonPrefix = @()
if ($PythonParts.Count -gt 1) {
    $PythonPrefix = $PythonParts[1..($PythonParts.Count - 1)]
}

function Invoke-ProjectPython {
    param([string[]]$Arguments)
    & $Python @PythonPrefix @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python task failed with exit code $LASTEXITCODE"
    }
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
# Use the interpreter selected on the recipient machine, not the developer's
# original Conda path.
$env:FRACTURING_QT_PYTHON = $Python
$env:FRACTURING_ALGORITHM_PYTHON = $Python

$rawDirectory = Join-Path $ProjectRoot "Data\raw_frac"
$dasDirectory = Join-Path $ProjectRoot "Data\3Dfrac"
$hasRaw = @(Get-ChildItem -LiteralPath $rawDirectory -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension.ToLowerInvariant() -in @(".xlsx", ".xls", ".csv") }).Count -gt 0
$hasDas = @(Get-ChildItem -LiteralPath $dasDirectory -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension.ToLowerInvariant() -in @(".xls", ".xlsx", ".txt", ".csv") }).Count -gt 0

if (-not $hasRaw -or -not $hasDas) {
    Write-Warning "Input files are incomplete. See README_RECORDING.md for the required folders and names."
}

$datasetBuildReport = Join-Path $ProjectRoot "outputs\app\datasets\raw_dataset_view_build.json"
if ($hasRaw -and ($RebuildNoDasViews -or -not (Test-Path -LiteralPath $datasetBuildReport))) {
    Write-Host "Building no-DAS playback views for independent stages..." -ForegroundColor Cyan
    Invoke-ProjectPython @("App\build_all_dt_dataset_views.py", "--frame-count", "120")
}

Write-Host "Running environment checks..." -ForegroundColor Cyan
Invoke-ProjectPython @("App\run_app.py", "--preflight", "--no-auto-env")
Write-Host "Starting the recording APP." -ForegroundColor Green
Invoke-ProjectPython @("App\run_app.py", "--no-auto-env")
