[CmdletBinding()]
param(
    [string]$SourceRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$ReleaseRoot = "C:\Workspace\FSL_Expert_GitHub_Release",
    [switch]$IncludeAuthorizedData,
    [switch]$PruneLegacyGenerated
)

$ErrorActionPreference = "Stop"

function Copy-Tree {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        throw "Source directory does not exist: $Source"
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    & robocopy $Source $Destination /E /R:2 /W:1 /COPY:DAT /DCOPY:DAT /XJ `
        /XD "__pycache__" ".pytest_cache" ".mypy_cache" ".ruff_cache" "runs" "outputs" `
        /XF "*.pyc" "*.log" | Out-Null
    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed ($LASTEXITCODE): $Source -> $Destination"
    }
}

function Copy-FileSafe {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Source file does not exist: $Source"
    }
    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

$SourceRoot = (Resolve-Path -LiteralPath $SourceRoot).Path
New-Item -ItemType Directory -Force -Path $ReleaseRoot | Out-Null
$ReleaseRoot = (Resolve-Path -LiteralPath $ReleaseRoot).Path

if ($PruneLegacyGenerated) {
    # These directories belong to the old public-snapshot layout.  The
    # current runnable handoff uses root-level outputs/artifacts instead.
    # Keep the target list explicit and verify every target stays below the
    # requested release root before deleting it.
    $legacyTargets = @(
        (Join-Path $ReleaseRoot ".tmp"),
        (Join-Path $ReleaseRoot ".pytest_cache"),
        (Join-Path $ReleaseRoot "HMI-KE\outputs"),
        (Join-Path $ReleaseRoot "HMI-KE\runs")
    )
    foreach ($target in $legacyTargets) {
        if (Test-Path -LiteralPath $target) {
            $resolvedTarget = (Resolve-Path -LiteralPath $target).Path
            if (-not $resolvedTarget.StartsWith($ReleaseRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Refusing to prune path outside release root: $resolvedTarget"
            }
            Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
        }
    }
}

# Rebuild the code surface from the current source, while excluding local
# caches and historical generated runs. Existing Git history in ReleaseRoot
# is preserved; only managed paths are overwritten.
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

# The frozen registry is part of the runnable handoff. It points only to
# paths copied below and keeps the scientific status/limitations visible.
Copy-FileSafe (Join-Path $SourceRoot "App\config\demo_registry.json") `
    (Join-Path $ReleaseRoot "App\config\demo_registry.json")

# Representative, reproducible artifacts. These are small enough to version
# and are the files used by the APP for frozen-result playback.
Copy-Tree (Join-Path $SourceRoot "artifacts") (Join-Path $ReleaseRoot "artifacts")

# Runtime outputs referenced by App/config/demo_registry.json.
Copy-Tree (Join-Path $SourceRoot "outputs\dt\fiber_allocation_parameter_enkf\20260806_082250") `
    (Join-Path $ReleaseRoot "outputs\dt\fiber_allocation_parameter_enkf\20260806_082250")
Copy-Tree (Join-Path $SourceRoot "outputs\hmi\bugfix_validation\ppo_20260812_184048") `
    (Join-Path $ReleaseRoot "outputs\hmi\bugfix_validation\ppo_20260812_184048")
Copy-Tree (Join-Path $SourceRoot "outputs\hmi\response_surrogate_abnormal_only\20260805_195713") `
    (Join-Path $ReleaseRoot "outputs\hmi\response_surrogate_abnormal_only\20260805_195713")
Copy-Tree (Join-Path $SourceRoot "outputs\hmi\response_surrogate\20260728_205957") `
    (Join-Path $ReleaseRoot "outputs\hmi\response_surrogate\20260728_205957")
Copy-FileSafe (Join-Path $SourceRoot "outputs\app\dt_realtime_cache.json") `
    (Join-Path $ReleaseRoot "outputs\app\dt_realtime_cache.json")
Copy-FileSafe (Join-Path $SourceRoot "outputs\app\dt_realtime_3d.html") `
    (Join-Path $ReleaseRoot "outputs\app\dt_realtime_3d.html")

if ($IncludeAuthorizedData) {
    # These are the actual input roots required by the three workstreams.
    # The large multimodal source-book PDF is intentionally not copied: the
    # generated knowledge-graph page is already included in FSL-Expert.
    Copy-Tree (Join-Path $SourceRoot "Data\raw_frac") (Join-Path $ReleaseRoot "Data\raw_frac")
    Copy-Tree (Join-Path $SourceRoot "Data\3Dfrac") (Join-Path $ReleaseRoot "Data\3Dfrac")
}

$files = @(Get-ChildItem -LiteralPath $ReleaseRoot -Recurse -File -ErrorAction SilentlyContinue)
$manifest = [ordered]@{
    release_version = "2.0.0"
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    source_root = "external-source"
    release_root = "."
    runtime_mode = "authorized_data_and_frozen_artifacts"
    included_authorized_data = [bool]$IncludeAuthorizedData
    data_roots = @("Data/raw_frac", "Data/3Dfrac")
    excluded_data = @("Data/multimodal")
    required_artifacts = @(
        "artifacts/fsl",
        "artifacts/dt",
        "artifacts/hmi/ppo_policy",
        "outputs/dt/fiber_allocation_parameter_enkf/20260806_082250",
        "outputs/hmi/bugfix_validation/ppo_20260812_184048",
        "outputs/hmi/response_surrogate/20260728_205957",
        "outputs/app/dt_realtime_cache.json",
        "outputs/app/dt_realtime_3d.html"
    )
    file_count = $files.Count
    total_bytes = [int64](($files | Measure-Object -Property Length -Sum).Sum)
    entrypoint = "python run_project.py"
    smoke_commands = @(
        "python App/run_app.py --no-gui",
        "python run_project.py dt validate",
        "python run_project.py hmi validate-env --strict"
    )
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $ReleaseRoot "release_manifest.json") -Encoding UTF8

Write-Output ($manifest | ConvertTo-Json -Depth 8)
