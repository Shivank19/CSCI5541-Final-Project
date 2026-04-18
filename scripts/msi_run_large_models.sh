#!/bin/bash
#SBATCH --job-name=restate-extra
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=runs/msi_extra_%j.log
#SBATCH --error=runs/msi_extra_%j.err
#SBATCH --mail-user=efe00002@umn.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80,TIME_LIMIT

# Heavy architectures on MSI (too big for RTX 2000 local).
# NOTE: deberta-v3-large removed -- training produces NaN on V100 fp32,
#       and chasing DeBERTa-v3 numerical issues is not worth the compute.
#       Can be reinstated on A100 if time allows. See run_deberta_v3_large.sh
#       if we decide to try it separately.
#
# Submit with:  sbatch scripts/msi_run_large_models.sh

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

ARCHS=(bert-large roberta)
SEEDS=(0 1 42)
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

for arch in "${ARCHS[@]}"; do
    echo ""
    echo "=============================================="
    echo " Architecture: $arch"
    echo "=============================================="
    for seed in "${SEEDS[@]}"; do
        for cond in "${CONDITIONS[@]}"; do
            run_one "runs/${arch}_${cond}_s${seed}" \
                --model "$arch" --condition "$cond" --seed "$seed"
        done
    done
done

echo ""
echo "=============================================="
echo " Run summary: completed=$completed skipped=$skipped failed=$failed"
echo "=============================================="

python -m models.aggregate || echo "  (aggregate failed; safe to ignore)"

echo ""
echo "=== Job $SLURM_JOB_ID finished at $(date) ==="