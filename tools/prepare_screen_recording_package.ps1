[CmdletBinding()]
param(
    [string]$SourceRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$ReleaseRoot = "",
    [string]$ZipPath = ""
)

$ErrorActionPreference = "Stop"

function Copy-Tree {
    param([string]$Source, [string]$Destination)
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        throw "Source directory does not exist: $Source"
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    & robocopy $Source $Destination /E /R:2 /W:1 /COPY:DAT /DCOPY:DAT /XJ `
        /XD "__pycache__" ".pytest_cache" ".mypy_cache" ".ruff_cache" "runs" "outputs" `
        /XF "*.pyc" "*.log" "*.tmp" | Out-Null
    if ($LASTEXITCODE -gt 7) {
        throw "Directory copy failed ($LASTEXITCODE): $Source -> $Destination"
    }
}

function Copy-FileSafe {
    param([string]$Source, [string]$Destination)
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Source file does not exist: $Source"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

function Copy-Optional {
    param([string]$Source, [string]$Destination)
    if (Test-Path -LiteralPath $Source -PathType Leaf) {
        Copy-FileSafe $Source $Destination
    } elseif (Test-Path -LiteralPath $Source -PathType Container) {
        Copy-Tree $Source $Destination
    } else {
        Write-Warning "Optional artifact not found; skipped: $Source"
    }
}

function Assert-SafeTarget {
    param([string]$Source, [string]$Target)
    $sourceFull = [IO.Path]::GetFullPath($Source).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $targetFull = [IO.Path]::GetFullPath($Target).TrimEnd([IO.Path]::DirectorySeparatorChar)
    if ($sourceFull.Equals($targetFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Release directory cannot be the source directory."
    }
    if ($targetFull.Length -lt 8) {
        throw "Release directory is too broad; refusing to continue: $targetFull"
    }
}

function Normalize-CopiedJsonPaths {
    param([string]$Directory, [string]$OriginalRoot)
    $sourceSlash = $OriginalRoot.TrimEnd('\', '/')
    $sourceForward = $sourceSlash.Replace('\', '/')
    foreach ($file in Get-ChildItem -LiteralPath $Directory -Recurse -File -Filter "*.json") {
        $text = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8
        $normalized = $text.Replace($sourceSlash + '\', '').Replace($sourceForward + '/', '')
        if ($normalized -ne $text) {
            Set-Content -LiteralPath $file.FullName -Value $normalized -Encoding UTF8
        }
    }
}

$SourceRoot = (Resolve-Path -LiteralPath $SourceRoot).Path
if ([string]::IsNullOrWhiteSpace($ReleaseRoot)) {
    $ReleaseRoot = Join-Path $SourceRoot "deliverables\IFP_screen_recording_package_no_raw_data"
}
if ([string]::IsNullOrWhiteSpace($ZipPath)) {
    $ZipPath = $ReleaseRoot.TrimEnd('\', '/') + ".zip"
}
Assert-SafeTarget $SourceRoot $ReleaseRoot

if (Test-Path -LiteralPath $ReleaseRoot) {
    $existing = (Resolve-Path -LiteralPath $ReleaseRoot).Path
    Assert-SafeTarget $SourceRoot $existing
    Remove-Item -LiteralPath $existing -Recurse -Force
}
if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath (Resolve-Path -LiteralPath $ZipPath).Path -Force
}
New-Item -ItemType Directory -Force -Path $ReleaseRoot | Out-Null

foreach ($name in @("App", "DT-Crack", "FSL-Expert", "HMI-KE", "tools", "docs")) {
    Copy-Tree (Join-Path $SourceRoot $name) (Join-Path $ReleaseRoot $name)
}
foreach ($name in @(
    "README.md", "project_config.yaml", "requirements-core.txt",
    "requirements-kg.txt", "requirements-rl.txt", "requirements-ui.txt",
    "run_project.py", "build_data_manifest.py"
)) {
    Copy-FileSafe (Join-Path $SourceRoot $name) (Join-Path $ReleaseRoot $name)
}

# Frozen model and APP artifacts are needed for immediate recording.  No
# source input data is copied by this script.
Copy-Tree (Join-Path $SourceRoot "artifacts") (Join-Path $ReleaseRoot "artifacts")
Copy-Tree (Join-Path $SourceRoot "outputs\dt\second_part_kg_enkf_20260824\20260824_004357") `
    (Join-Path $ReleaseRoot "outputs\dt\second_part_kg_enkf_20260824\20260824_004357")
Copy-Tree (Join-Path $SourceRoot "outputs\hmi\model_comparison_formal_v3_20260818\20260818_192543") `
    (Join-Path $ReleaseRoot "outputs\hmi\model_comparison_formal_v3_20260818\20260818_192543")
foreach ($file in @(
    "outputs\app\fsl_timeline_cache.json",
    "outputs\app\dt_realtime_cache.json",
    "outputs\app\dt_realtime_3d.html",
    "outputs\app\dt_realtime_3d_no_das.html",
    "outputs\app\fiber_six_cluster\fiber_six_cluster_timeseries.csv",
    "outputs\app\fiber_six_cluster\fiber_six_cluster_timeseries.png",
    "outputs\app\fiber_six_cluster\summary.json"
)) {
    Copy-Optional (Join-Path $SourceRoot $file) (Join-Path $ReleaseRoot $file)
}
Copy-Tree (Join-Path $SourceRoot "outputs\app\no_das_ppt_delivery_all_single_stages") `
    (Join-Path $ReleaseRoot "outputs\app\no_das_ppt_delivery_all_single_stages")

New-Item -ItemType Directory -Force -Path (Join-Path $ReleaseRoot "Data\raw_frac") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ReleaseRoot "Data\3Dfrac") | Out-Null
@"
Put the matching independent stage construction tables in this directory.
Supported extensions: .xlsx, .xls, .csv. Aggregate exports are not treated as one independent stage.
After copying data, run START_RECORDING.ps1 -RebuildNoDasViews from the package root.
"@ | Set-Content -LiteralPath (Join-Path $ReleaseRoot "Data\raw_frac\README_DATA_INPUT.txt") -Encoding UTF8
@"
Put the matching DAS/FracMonitor and well trajectory inputs in this directory.
The default registered names are JY84-Z1-stage08-f1.xls, fiber_monitor_stage08.txt, and JY84-Z1HF-1011.csv.
If names differ, update App\config\dt_dataset_registry.json.
"@ | Set-Content -LiteralPath (Join-Path $ReleaseRoot "Data\3Dfrac\README_DATA_INPUT.txt") -Encoding UTF8
Copy-FileSafe (Join-Path $SourceRoot "docs\recording_package_readme.md") (Join-Path $ReleaseRoot "README_RECORDING.md")
Copy-FileSafe (Join-Path $SourceRoot "tools\start_screen_recording.ps1") (Join-Path $ReleaseRoot "START_RECORDING.ps1")

# Summary files from local experiments can contain absolute developer paths.
# Convert those paths to package-relative paths before the archive is created.
Normalize-CopiedJsonPaths $ReleaseRoot $SourceRoot

$forbiddenExtensions = @(".xlsx", ".xls", ".csv", ".txt", ".pkl", ".pt", ".pth")
$sensitiveFiles = @(
    Get-ChildItem -LiteralPath (Join-Path $ReleaseRoot "Data\raw_frac") -Recurse -File -ErrorAction SilentlyContinue
    Get-ChildItem -LiteralPath (Join-Path $ReleaseRoot "Data\3Dfrac") -Recurse -File -ErrorAction SilentlyContinue
) | Where-Object { $_.Extension.ToLowerInvariant() -in $forbiddenExtensions -and $_.Name -notlike "README_*" }
if ($sensitiveFiles.Count -gt 0) {
    throw "Possible source data detected; packaging stopped: $($sensitiveFiles.FullName -join '; ')"
}

$files = @(Get-ChildItem -LiteralPath $ReleaseRoot -Recurse -File)
$manifest = [ordered]@{
    release_version = "recording-1.0.0"
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    purpose = "PPT screen-recording handoff"
    included_raw_data = $false
    excluded_roots = @("Data/raw_frac", "Data/3Dfrac")
    included_frozen_outputs = $true
    required_user_inputs = @(
        "Data/raw_frac/independent stage construction tables",
        "Data/3Dfrac/DAS, pressure, FracMonitor and trajectory inputs"
    )
    file_count = $files.Count
    total_bytes = [int64](($files | Measure-Object -Property Length -Sum).Sum)
    entrypoint = "START_RECORDING.ps1"
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $ReleaseRoot "release_manifest.json") -Encoding UTF8

Compress-Archive -Path (Join-Path $ReleaseRoot "*") -DestinationPath $ZipPath -CompressionLevel Optimal -Force
$zipInfo = Get-Item -LiteralPath $ZipPath
$result = [ordered]@{
    release_root = $ReleaseRoot
    zip_path = $zipInfo.FullName
    zip_size_mb = [math]::Round($zipInfo.Length / 1MB, 2)
    file_count = $files.Count
    included_raw_data = $false
}
$result | ConvertTo-Json -Depth 8
