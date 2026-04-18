#!/bin/bash
#SBATCH --job-name=restate-lf
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

# Extended Longformer sweep on MSI (FinBERT already done locally).
# Submit from repo root with:  sbatch scripts/run_msi_longformer.sh
#
# Estimated wall time: ~15-25 hours on V100 for all 30 runs.
# We request 12 hours, meaning the job WILL be killed before completing
# every seed. That's intentional -- each completed run is saved to disk
# independently, and the idempotent loop skips already-complete runs.
# Just re-submit after the first job finishes to continue where it stopped.

set -euo pipefail

echo "=== Job $SLURM_JOB_ID starting on $(hostname) at $(date) ==="
echo "Working dir: $SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

module purge
module load python3/3.10.9_anaconda2023.03_libmamba
module load cuda/12.1.1

# Activate the venv created by setup_msi_env.sh
source .venv/bin/activate

# Sanity check
python - <<'PY'
import torch, transformers
print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}  vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
print(f"transformers={transformers.__version__}")
PY

# Unbuffered output so the log updates in real time
export PYTHONUNBUFFERED=1

# Extended seed list for more stable mean+stderr estimates across the
# small-n sweep. Runs are ordered to prioritize coverage: all 3 conditions
# at each seed before moving to the next seed. That way, if time runs out
# mid-sweep, you still have a valid (partial) run for every condition.
SEEDS=(0 1 42 2 7 13 100 2024 2025 2026)
CONDITIONS=(full scripted qa)

# Track completed/skipped/failed for end-of-job summary
completed=0
skipped=0
failed=0

for seed in "${SEEDS[@]}"; do
    for cond in "${CONDITIONS[@]}"; do
        out="runs/longformer_${cond}_s${seed}"
        if [ -f "$out/test_metrics.csv" ]; then
            echo "  [skip] $out already complete"
            skipped=$((skipped + 1))
            continue
        fi
        echo ""
        echo ">> longformer  cond=$cond  seed=$seed  at $(date +%H:%M:%S)"
        if python -m models.train \
                --model longformer --condition "$cond" --seed "$seed" \
                --output_dir "$out"; then
            completed=$((completed + 1))
        else
            echo "  [FAIL] run $out crashed, continuing to next config"
            failed=$((failed + 1))
        fi
    done
done

echo ""
echo "=============================================="
echo " Run summary: completed=$completed, skipped=$skipped, failed=$failed"
echo "=============================================="

echo ""
echo "=== Aggregating results so far ==="
python -m models.aggregate || echo "  (aggregate failed; can rerun after resubmit)"

echo ""
echo "=== Job $SLURM_JOB_ID finished at $(date) ==="