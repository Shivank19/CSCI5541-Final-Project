#!/usr/bin/env bash
# Full Tier 2 sweep: FinBERT and Longformer x 3 conditions x 3 seeds = 18 runs.
#
# Run FinBERT first (fast, ~15-30 min each). Longformer is much slower
# (~45-90 min each) so is best left overnight.
#
# Edit SEEDS to add more, or run finbert/longformer blocks independently.

set -euo pipefail

cd "$(dirname "$0")/.."

SEEDS=(0 1 42)
CONDITIONS=(full scripted qa)

echo "=============================="
echo " Tier 2 sweep: FinBERT"
echo "=============================="
for seed in "${SEEDS[@]}"; do
    for cond in "${CONDITIONS[@]}"; do
        out="runs/finbert_${cond}_s${seed}"
        if [ -f "$out/test_metrics.csv" ]; then
            echo "  [skip] $out already complete"
            continue
        fi
        echo ""
        echo ">> finbert  cond=$cond  seed=$seed"
        python -m models.train \
            --model finbert --condition "$cond" --seed "$seed" \
            --output_dir "$out"
    done
done

echo ""
echo "=============================="
echo " Tier 2 sweep: Longformer"
echo "=============================="
for seed in "${SEEDS[@]}"; do
    for cond in "${CONDITIONS[@]}"; do
        out="runs/longformer_${cond}_s${seed}"
        if [ -f "$out/test_metrics.csv" ]; then
            echo "  [skip] $out already complete"
            continue
        fi
        echo ""
        echo ">> longformer  cond=$cond  seed=$seed"
        python -m models.train \
            --model longformer --condition "$cond" --seed "$seed" \
            --output_dir "$out"
    done
done

echo ""
echo "=============================="
echo " Aggregating results"
echo "=============================="
python -m models.aggregate
