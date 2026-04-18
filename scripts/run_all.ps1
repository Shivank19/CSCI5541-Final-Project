# Full Tier 2 sweep: FinBERT and Longformer x 3 conditions x 3 seeds = 18 runs.
# Idempotent: skips any run whose test_metrics.csv already exists.
#
# Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1

$ErrorActionPreference = "Continue"

$seeds = @(0, 1, 42)
$conditions = @("full", "scripted", "qa")
$models = @("finbert", "longformer")

$python = ".\.venv\Scripts\python.exe"

foreach ($model in $models) {
    Write-Host ""
    Write-Host "=============================="
    Write-Host " Tier 2 sweep: $model"
    Write-Host "=============================="
    foreach ($seed in $seeds) {
        foreach ($cond in $conditions) {
            $out = "runs/${model}_${cond}_s${seed}"
            if (Test-Path "$out/test_metrics.csv") {
                Write-Host "  [skip] $out already complete"
                continue
            }
            Write-Host ""
            Write-Host ">> $model  cond=$cond  seed=$seed"
            & $python -m models.train `
                --model $model `
                --condition $cond `
                --seed $seed `
                --output_dir $out
        }
    }
}

Write-Host ""
Write-Host "=============================="
Write-Host " Aggregating results"
Write-Host "=============================="
& $python -m models.aggregate