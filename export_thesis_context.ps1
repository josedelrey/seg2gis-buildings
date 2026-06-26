param(
    [string]$OutputPath = "",
    [int]$MaxFileMB = 5,
    [switch]$IncludePresentationDecks
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $RepoRoot "thesis_context_$Timestamp.zip"
}

$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
$MaxBytes = [int64]$MaxFileMB * 1024 * 1024
$StageRoot = Join-Path $env:TEMP "seg2gis_thesis_context_$Timestamp"

function Get-RelativePath {
    param([string]$FullPath)

    $absolute = [System.IO.Path]::GetFullPath($FullPath)
    return $absolute.Substring($RepoRoot.Length).TrimStart("\", "/").Replace("\", "/")
}

function Test-IsUnderRepo {
    param([string]$Path)

    $absolute = [System.IO.Path]::GetFullPath($Path)
    return $absolute.StartsWith($RepoRoot, [System.StringComparison]::OrdinalIgnoreCase)
}

function Test-ShouldSkipRelativePath {
    param([string]$RelativePath)

    $path = $RelativePath.Replace("\", "/").TrimStart("/")

    if ($path -match "(^|/)__pycache__/" -or $path -match "\.pyc$") {
        return $true
    }

    $skippedPrefixes = @(
        "data/",
        "models/",
        "outputs/",
        "results/cache/",
        "results/full_predictions/",
        "results/qualitative/",
        "docs/presentation/builds/",
        ".git/"
    )

    foreach ($prefix in $skippedPrefixes) {
        if ($path.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }

    if ($path -match "(^|/)node_modules/") {
        return $true
    }

    return $false
}

function Add-IfExists {
    param(
        [string]$RelativePath,
        [System.Collections.Generic.List[object]]$Included,
        [System.Collections.Generic.List[object]]$Skipped
    )

    $RelativePath = $RelativePath.Replace("\", "/").TrimStart("/")
    if ($Included | Where-Object { $_.Path -eq $RelativePath }) {
        return
    }
    if (Test-ShouldSkipRelativePath -RelativePath $RelativePath) {
        return
    }

    $source = Join-Path $RepoRoot $RelativePath
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        $Skipped.Add([pscustomobject]@{
            Path = $RelativePath
            Reason = "missing"
            SizeBytes = 0
        })
        return
    }

    $item = Get-Item -LiteralPath $source
    if ($item.Length -gt $MaxBytes) {
        $Skipped.Add([pscustomobject]@{
            Path = $RelativePath
            Reason = "larger than $MaxFileMB MB"
            SizeBytes = $item.Length
        })
        return
    }

    $destination = Join-Path $StageRoot $RelativePath
    $destinationDir = Split-Path -Parent $destination
    if (-not (Test-Path -LiteralPath $destinationDir)) {
        New-Item -ItemType Directory -Path $destinationDir | Out-Null
    }

    Copy-Item -LiteralPath $source -Destination $destination
    $Included.Add([pscustomobject]@{
        Path = $RelativePath
        SizeBytes = $item.Length
    })
}

function Add-FilesFromDirectory {
    param(
        [string]$RelativeDirectory,
        [string[]]$Include = @("*"),
        [System.Collections.Generic.List[object]]$Included,
        [System.Collections.Generic.List[object]]$Skipped
    )

    $directory = Join-Path $RepoRoot $RelativeDirectory
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
        $Skipped.Add([pscustomobject]@{
            Path = $RelativeDirectory
            Reason = "missing directory"
            SizeBytes = 0
        })
        return
    }

    Get-ChildItem -LiteralPath $directory -Recurse -File -Include $Include | ForEach-Object {
        $relative = Get-RelativePath $_.FullName
        Add-IfExists -RelativePath $relative -Included $Included -Skipped $Skipped
    }
}

function Write-TextFile {
    param(
        [string]$RelativePath,
        [string]$Content
    )

    $destination = Join-Path $StageRoot $RelativePath
    $destinationDir = Split-Path -Parent $destination
    if (-not (Test-Path -LiteralPath $destinationDir)) {
        New-Item -ItemType Directory -Path $destinationDir | Out-Null
    }
    Set-Content -LiteralPath $destination -Value $Content -Encoding UTF8
}

if (-not (Test-IsUnderRepo $RepoRoot)) {
    throw "Could not resolve repository root."
}

if (Test-Path -LiteralPath $StageRoot) {
    Remove-Item -LiteralPath $StageRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $StageRoot | Out-Null

$included = [System.Collections.Generic.List[object]]::new()
$skipped = [System.Collections.Generic.List[object]]::new()

$importantFiles = @(
    "README.md",
    "requirements.txt",
    "LICENSE",
    ".gitignore",
    "configs/default.json",
    "configs/experiments_phase1_noaug_baseline.yaml",
    "configs/experiments_phase2_augmentation_boundary_loss.yaml",
    "scripts/postprocess_ablation_table.py",
    "results/tables/postprocess_ablation_validation.csv",
    "results/tables/postprocess_ablation_validation_summary.csv",
    "docs/presentation/thesis_presentation_source.md",
    "docs/presentation/assets/README.md"
)

foreach ($file in $importantFiles) {
    Add-IfExists -RelativePath $file -Included $included -Skipped $skipped
}

Add-FilesFromDirectory -RelativeDirectory "src" -Include @("*.py") -Included $included -Skipped $skipped
Add-FilesFromDirectory -RelativeDirectory "scripts" -Include @("*.py") -Included $included -Skipped $skipped
Add-FilesFromDirectory -RelativeDirectory "configs/generated" -Include @("*.json") -Included $included -Skipped $skipped
Add-FilesFromDirectory -RelativeDirectory "results/tables" -Include @("*.csv") -Included $included -Skipped $skipped
Add-FilesFromDirectory -RelativeDirectory "results/figures" -Include @("*.png", "*.jpg", "*.jpeg") -Included $included -Skipped $skipped
Add-FilesFromDirectory -RelativeDirectory "docs/presentation/assets" -Include @("*.png", "*.jpg", "*.jpeg", "*.md") -Included $included -Skipped $skipped
Add-FilesFromDirectory -RelativeDirectory "docs/presentation/scripts/asset_generation" -Include @("*.py") -Included $included -Skipped $skipped

if ($IncludePresentationDecks) {
    Add-FilesFromDirectory -RelativeDirectory "docs/presentation" -Include @("*.pptx") -Included $included -Skipped $skipped
}

$alwaysSkipped = @(
    "data/ - dataset and generated tiles",
    "models/ - trained weights and metadata",
    "results/cache/ - reusable probability-map caches",
    "results/full_predictions/ - large generated inference artifacts",
    "results/qualitative/ - large qualitative prediction grids",
    "docs/presentation/builds/ - generated deck build outputs",
    ".git/ - repository internals",
    "repo_context.txt and docs_context.txt - large previous context dumps"
)

$includedCount = $included.Count
$includedBytes = ($included | Measure-Object -Property SizeBytes -Sum).Sum
if ($null -eq $includedBytes) {
    $includedBytes = 0
}

$treeLines = $included |
    Sort-Object Path |
    ForEach-Object { "{0} ({1:N0} bytes)" -f $_.Path, $_.SizeBytes }

$skippedLines = $skipped |
    Sort-Object Path |
    ForEach-Object { "{0} - {1} ({2:N0} bytes)" -f $_.Path, $_.Reason, $_.SizeBytes }

$manifest = @"
# seg2gis-buildings thesis context export

Created: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Repository: $RepoRoot
Max single file size: $MaxFileMB MB
Included files: $includedCount
Included raw size: $("{0:N0}" -f $includedBytes) bytes

## Intended Use

This zip is a compact project context package for discussing the master's thesis, writing LaTeX text, and asking ChatGPT about methodology, experiments, and results.
It includes thesis result CSVs under results/tables, including the post-processing ablation tables when they have been generated.

## Included

$($treeLines -join "`n")

## Skipped By Rule

$($alwaysSkipped -join "`n")

## Skipped Missing Or Too Large

$($skippedLines -join "`n")
"@

Write-TextFile -RelativePath "EXPORT_MANIFEST.md" -Content $manifest

$tree = ($included | Sort-Object Path | ForEach-Object { $_.Path }) -join "`n"
Write-TextFile -RelativePath "PROJECT_CONTEXT_TREE.txt" -Content $tree

if (Test-Path -LiteralPath $OutputPath) {
    Remove-Item -LiteralPath $OutputPath -Force
}

Compress-Archive -Path (Join-Path $StageRoot "*") -DestinationPath $OutputPath -CompressionLevel Optimal
Remove-Item -LiteralPath $StageRoot -Recurse -Force

$zip = Get-Item -LiteralPath $OutputPath
Write-Host "Created thesis context zip:"
Write-Host $zip.FullName
Write-Host ("Zip size: {0:N2} MB" -f ($zip.Length / 1MB))
Write-Host ("Included files: {0}" -f $includedCount)
Write-Host ("Skipped missing/large files: {0}" -f $skipped.Count)
