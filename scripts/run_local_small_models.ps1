# Extra architectures sweep (local, RTX 2000 Ada).
# All non-FinBERT encoders that fit in 16GB VRAM.
# Used for the architecture-ablation analysis in the paper.
#
# Idempotent. Run with:
#   powershell -ExecutionPolicy Bypass -File scripts\run_local_extra_archs.ps1
#
# Included architectures:
#   - bert (110M)            : classic BERT-base baseline
#   - distilbert (66M)       : distilled, very fast
#   - roberta (125M)         : general-purpose benchmark encoder
#   - deberta-v3-small (44M) : tiny modern encoder, very fast
#   - deberta-v3-base (183M) : strongest modern encoder in base size
#
# NOT included (see notes):
#   - finbert-tone           : old-style repo, AutoTokenizer doesn't handle it
#                              cleanly with transformers>=4.40
#   - deberta-v3-large       : numerically unstable in fp32 on V100 (MSI);
#                              may work on local Ada but excluded for symmetry
#
# Best run AFTER run_local_main.ps1 finishes, since they share the GPU.
# Estimated wall time: ~4-6 hours for 5 archs x 10 seeds x 3 conditions = 150 runs.
# You can reduce $seeds for a faster pass.

$ErrorActionPreference = "Continue"

$python = ".\.venv\Scripts\python.exe"
$seeds         = @(0, 1, 2, 7, 13, 42, 100, 2024, 2025, 2026,
                   11, 22, 33, 44, 55, 66, 77, 88, 99, 111,
                   123, 234, 345, 456, 567, 678, 789, 890, 901, 999)
$conditions    = @("full", "scripted", "qa")

# Models safely fitting in 16GB, ordered fastest-first
$localArchs = @(
    "distilbert",            # ~2x faster than BERT-base
    "bert",                  # classic BERT baseline
    "roberta"               # general-purpose benchmark
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
    foreach ($seed in $seeds) {
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