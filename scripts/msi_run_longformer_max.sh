#!/bin/bash
#SBATCH --job-name=restate-lf-max
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/msi_%j.log
#SBATCH --error=runs/msi_%j.err
#SBATCH --mail-user=efe00002@umn.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80,TIME_LIMIT

# Maximum-science Longformer sweep on MSI, sized for 12h wall time.
# Idempotent: skips runs whose test_metrics.csv already exists.
#
# Observed timing: ~12 min/run on V100.
# Budget: 11h usable = 660 min = ~55 runs per submission.
#
# Phase ordering is INCREMENTAL -- we run the most important phases first
# so that even a partial completion yields useful results:
#   A1. 10-seed default (already completed from prior submission; mostly skip)
#   A2. Extra 20 seeds on default  -- 60 runs (largest block)
#   B.  Truncation ablation (head, middle) -- 18 runs
#   C.  Hyperparameter ablations (lowlr, frozen) -- 18 runs
#
# Total: ~96 runs. First submission completes ~55. Resubmit for the rest.
#
# Submit with:  sbatch scripts/run_msi_longformer_max.sh
# To continue after first finishes:  sbatch scripts/run_msi_longformer_max.sh

set -uo pipefail

echo "=== Job $SLURM_JOB_ID starting on $(hostname) at $(date) ==="
cd "$SLURM_SUBMIT_DIR"

module purge
module load python3/3.10.9_anaconda2023.03_libmamba
module load cuda/12.1.1
source .venv/bin/activate

python - <<'PY'
import torch, transformers
print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}  vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
print(f"transformers={transformers.__version__}")
PY

export PYTHONUNBUFFERED=1

# 30 seeds for Phase A; 3 seeds for ablations
ALL_SEEDS=(0 1 2 7 13 42 100 2024 2025 2026
           11 22 33 44 55 66 77 88 99 111
           123 234 345 456 567 678 789 890 901 999)
ABLATION_SEEDS=(0 1 42)
CONDITIONS=(full scripted qa)

completed=0; skipped=0; failed=0

run_one() {
    local out="$1"; shift
    if [ -f "$out/test_metrics.csv" ]; then
        skipped=$((skipped + 1))
        return 0
    fi
    echo ""
    echo ">> $out  at $(date +%H:%M:%S)"
    if python -m models.train --output_dir "$out" "$@"; then
        completed=$((completed + 1))
    else
        echo "  [FAIL] $out crashed, continuing"
        failed=$((failed + 1))
    fi
}

echo ""
echo "=============================================="
echo " Phase A: Longformer 30-seed default sweep"
echo "=============================================="
for seed in "${ALL_SEEDS[@]}"; do
    for cond in "${CONDITIONS[@]}"; do
        run_one "runs/longformer_${cond}_s${seed}" \
            --model longformer --condition "$cond" --seed "$seed"
    done
done

echo ""
echo "=============================================="
echo " Phase B: Longformer truncation ablation"
echo "=============================================="
for trunc in head middle; do
    for seed in "${ABLATION_SEEDS[@]}"; do
        for cond in "${CONDITIONS[@]}"; do
            run_one "runs/longformer_${cond}_s${seed}_trunc-${trunc}" \
                --model longformer --condition "$cond" --seed "$seed" \
                --truncation "$trunc"
        done
    done
done

echo ""
echo "=============================================="
echo " Phase C: Longformer hyperparameter ablations"
echo "=============================================="
# Low LR
for seed in "${ABLATION_SEEDS[@]}"; do
    for cond in "${CONDITIONS[@]}"; do
        run_one "runs/longformer_${cond}_s${seed}_lowlr" \
            --model longformer --condition "$cond" --seed "$seed" --lr 5e-6
    done
done
# Frozen backbone
for seed in "${ABLATION_SEEDS[@]}"; do
    for cond in "${CONDITIONS[@]}"; do
        run_one "runs/longformer_${cond}_s${seed}_frozen" \
            --model longformer --condition "$cond" --seed "$seed" \
            --freeze_backbone --lr 1e-3 --epochs 30 --patience 5
    done
done

echo ""
echo "=============================================="
echo " Run summary: completed=$completed skipped=$skipped failed=$failed"
echo "=============================================="

python -m models.aggregate || echo "  (aggregate failed; safe to ignore)"

echo ""
echo "=== Job $SLURM_JOB_ID finished at $(date) ==="