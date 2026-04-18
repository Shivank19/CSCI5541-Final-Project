# Tier 2 Training: FinBERT + Longformer

Unified fine-tuning pipeline for the two Tier 2 transformer models.
All 18 target runs (2 models × 3 conditions × 3 seeds) are driven by
a single script: `models/train.py`. Metrics are aggregated post-hoc
via `models/aggregate.py`.

---

## 1. Environment Setup (one time)

Install PyTorch with CUDA support for your GPU (Ada generation):

```bash
# Windows or Linux with CUDA 12.1+
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Install the rest:

```bash
pip install -r requirements.txt
pip install -r requirements-ml.txt
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Expected output: `True NVIDIA RTX 2000 Ada Generation` (or similar).

---

## 2. Smoke Test (2 minutes)

Before kicking off real training, confirm the pipeline runs end-to-end
on a tiny subset:

```bash
python -m models.train --model finbert --condition qa --epochs 2 --smoke_test
```

This trains on 16 rows for 2 epochs. If it finishes without errors and
writes `runs/finbert_qa_s42/test_metrics.csv`, the environment is good.

---

## 3. Run a Single Configuration

```bash
# FinBERT on Q&A section, seed 42 (the canonical H2/H4 test)
python -m models.train --model finbert --condition qa --seed 42

# Longformer on full transcript, seed 0
python -m models.train --model longformer --condition full --seed 0
```

Outputs go to `runs/{model}_{condition}_s{seed}/`:
- `config.json` — exact hyperparameters used
- `history.csv` — per-epoch train loss + val metrics
- `test_metrics.csv` — single-row summary (AUROC, AP, F1, confusion matrix)
- `test_predictions.csv` — per-example probabilities, for error analysis

---

## 4. Run the Full Sweep

```bash
# 18 runs total: finbert (fast, ~5h) then longformer (slow, ~12h)
bash scripts/run_all.sh
```

The script is **idempotent** — it skips any `runs/{name}/` directory
that already has a `test_metrics.csv`. Safe to re-run after an interruption.

Recommended schedule for the 2-day sprint:
- **Day 1 morning**: smoke test + launch FinBERT sweep (9 runs, finishes by afternoon)
- **Day 1 evening**: launch Longformer sweep, leave overnight
- **Day 2 morning**: `python -m models.aggregate`, review results, share with team

---

## 5. Aggregate Results

After runs complete:

```bash
python -m models.aggregate
```

Writes:
- `runs/summary_all.csv` — one row per run
- `runs/summary_by_config.csv` — mean ± std per (model, condition)

And prints a console-friendly AUROC table.

---

## 6. Experimental Design Notes

**Conditions tested.** `full` uses `full_text`, `scripted` uses
`scripted_text`, `qa` uses `qa_text`. The `qa` condition drops rows
with empty Q&A segmentation output (7 train, 2 val, 3 test) — n is
logged automatically.

**Tail-biased truncation.** Long transcripts (median ~9800 tokens full,
~3500 tokens Q&A) are truncated by keeping the LAST N tokens, not the
first. Rationale: executive answers toward the end of Q&A carry the
strongest hypothesized evasion signal (H2). FinBERT → last 512 tokens;
Longformer → last 4096 tokens.

**Class weighting.** The dataset is 2:1 control:positive. We weight the
cross-entropy loss inversely so positives are not systematically
down-ranked. Weights are computed per-condition from the training split
after empty-row filtering.

**Early stopping.** Training runs up to 10 epochs (configurable via
`--epochs`). If val AUROC does not improve for 3 consecutive epochs
(configurable via `--patience`), training stops and the best checkpoint
is restored for test-set evaluation.

**Seeds.** Default seed set is `{0, 1, 42}`. With n=139 train / 45
positives, single-seed results will be noisy; always report mean ± std
across seeds in the writeup.

**Bootstrap CIs.** `test_metrics.csv` includes `auroc_ci_lo` and
`auroc_ci_hi` — a 1000-iteration stratified bootstrap 95% CI on the
test-set AUROC. Small-n caveat: these CIs will be wide (often ±0.15).
Report them honestly.

---

## 7. Hardware Notes (RTX 2000 Ada, 16 GB)

FinBERT defaults: batch size 8, no gradient checkpointing, bf16.
Runs at ~15–30 min/epoch × ~5 epochs with early stopping ≈ 15 min total.
Longformer defaults: batch size 1 × grad_accum 8 (effective 8),
gradient checkpointing on, bf16. Runs at ~45–90 min per training run.

If you hit CUDA OOM on Longformer with the `full` condition (longest
inputs), drop `--max_length 2048` or `--grad_accum_steps 16` and
retry. bf16 must remain on.

---

## 8. Extending / Handoff

To add another model family (e.g., RoBERTa-large financial variant),
add an entry to `MODEL_REGISTRY` in `models/train.py` with `hf_name`,
`max_length`, `per_device_batch_size`, `grad_accum_steps`, and
`grad_checkpointing`. No other code changes required.

For LLM prompting (Tier 3, Bilal's tier), the per-example probabilities
in `test_predictions.csv` are in the same format — makes AUROC-based
cross-tier comparison trivial.
