# Extra architectures sweep (local, RTX 2000 Ada).
# Runs INDEPENDENTLY from run_all_max.ps1 -- safe to launch in parallel
# in a second PowerShell window, though only one can use the GPU at a
# time (Windows time-slices the GPU, both will run but slower).
# Best used AFTER run_all_max.ps1 finishes, to add architecture breadth.
#
# Idempotent. ~3 hours at 3 seeds x 3 conditions x 4 new architectures = 36 runs.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\run_extra_archs.ps1

$ErrorActionPreference = "Continue"

$python = ".\.venv\Scripts\python.exe"
$ablationSeeds = @(0, 1, 42, 2024, 55, 123, 345, 789, 999, 111)
$conditions    = @("full", "scripted", "qa")

# Models safely fitting in 16GB, ordered fastest-first
$localArchs = @(
    "deberta-v3-small",      # tiny, very fast
    "deberta-v3-base"       # strongest general encoder in base size
)

function Run-One($args_array, $outDir) {
    if (Test-Path "$outDir/test_metrics.csv") {
        Write-Host "  [skip] $outDir"
        return
    }
    Write-Host ""
    Write-Host ">> $outDir"
    & $python -m models.train @args_array --output_dir $outDir
}

foreach ($arch in $localArchs) {
    Write-Host ""
    Write-Host "##############################################"
    Write-Host " Architecture: $arch"
    Write-Host "##############################################"
    foreach ($seed in $ablationSeeds) {
        foreach ($cond in $conditions) {
            Run-One @("--model",$arch,"--condition",$cond,"--seed",$seed) `
                    "runs/${arch}_${cond}_s${seed}"
        }
    }
}

Write-Host ""
Write-Host "##############################################"
Write-Host " Aggregating"
Write-Host "##############################################"
& $python -m models.aggregate