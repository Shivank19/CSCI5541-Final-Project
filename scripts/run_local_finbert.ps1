# FinBERT local sweep -- the core Tier 2 model with full ablations.
# Idempotent. Run with:
#   powershell -ExecutionPolicy Bypass -File scripts\run_local_main.ps1
#
# Scope:
#   Phase A: 30-seed default sweep (core robustness)
#   Phase B: truncation ablation (head, middle)
#   Phase C: hyperparameter ablation (lowlr, frozen backbone)
#
# Non-FinBERT architectures live in separate scripts:
#   - scripts/run_local_extra_archs.ps1 : small encoders (bert, distilbert,
#     roberta, deberta-v3-*)
#
# Designed for ~5-7 hours on RTX 2000 Ada. Safe to interrupt anytime.

$ErrorActionPreference = "Continue"

$python = ".\.venv\Scripts\python.exe"

$headerSeeds   = @(0, 1, 2, 7, 13, 42, 100, 2024, 2025, 2026,
                   11, 22, 33, 44, 55, 66, 77, 88, 99, 111,
                   123, 234, 345, 456, 567, 678, 789, 890, 901, 999)  # 30 seeds
$ablationSeeds = @(0, 1, 42, 2024, 55, 123, 345, 789, 999, 111)        # 10 seeds
$conditions    = @("full", "scripted", "qa")

function Run-One($args_array, $outDir) {
    if (Test-Path "$outDir/test_metrics.csv") {
        Write-Host "  [skip] $outDir"
        return
    }
    Write-Host ""
    Write-Host ">> $outDir"
    & $python -m models.train @args_array --output_dir $outDir
}

Write-Host ""
Write-Host "##############################################"
Write-Host " Phase A: FinBERT 30-seed default sweep (core)"
Write-Host "##############################################"
foreach ($seed in $headerSeeds) {
    foreach ($cond in $conditions) {
        Run-One @("--model","finbert","--condition",$cond,"--seed",$seed) `
                "runs/finbert_${cond}_s${seed}"
    }
}

Write-Host ""
Write-Host "##############################################"
Write-Host " Phase B: FinBERT truncation ablation (head, middle)"
Write-Host "##############################################"
foreach ($trunc in @("head", "middle")) {
    foreach ($seed in $ablationSeeds) {
        foreach ($cond in $conditions) {
            Run-One @("--model","finbert","--condition",$cond,"--seed",$seed,
                      "--truncation",$trunc) `
                    "runs/finbert_${cond}_s${seed}_trunc-${trunc}"
        }
    }
}

Write-Host ""
Write-Host "##############################################"
Write-Host " Phase C: FinBERT hyperparameter ablations"
Write-Host "##############################################"
# Low LR
foreach ($seed in $ablationSeeds) {
    foreach ($cond in $conditions) {
        Run-One @("--model","finbert","--condition",$cond,"--seed",$seed,
                  "--lr","5e-6") `
                "runs/finbert_${cond}_s${seed}_lowlr"
    }
}
# Frozen backbone (head-only training, much faster, higher LR)
foreach ($seed in $ablationSeeds) {
    foreach ($cond in $conditions) {
        Run-One @("--model","finbert","--condition",$cond,"--seed",$seed,
                  "--freeze_backbone","--lr","1e-3",
                  "--epochs","30","--patience","5") `
                "runs/finbert_${cond}_s${seed}_frozen"
    }
}

Write-Host ""
Write-Host "##############################################"
Write-Host " Aggregating"
Write-Host "##############################################"
& $python -m models.aggregate