# Tier 2 Modeling: Fine-Tuned Transformers

This directory contains the training and aggregation pipeline for the Tier 2 arm of the CSCI 5541 restatement precursor project. Tier 2 evaluates whether fine-tuned transformer models can detect pre-restatement language signals in quarterly earnings call transcripts.

## Contents

- `train.py` — unified training script. Fine-tunes any model in `MODEL_REGISTRY` on any transcript condition (`full`, `scripted`, `qa`) for a given seed. Handles tokenization (with fast/slow fallback), tail/head/middle truncation, bf16 auto-detection, class-weighted cross-entropy for the 2:1 control/positive imbalance, and early stopping on validation AUROC. Writes `config.json`, `history.csv`, `test_metrics.csv`, and `test_predictions.csv` per run.
- `aggregate.py` — variant-aware aggregator. Reads every `runs/<name>/test_metrics.csv`, parses the run name into `(model, condition, seed, variant)`, and produces summary tables separating default sweeps from ablations. Default sweeps and ablations are summarized independently so pooled means aren't contaminated by hyperparameter variants.
- `common/` — shared utilities used by the trainer and aggregator: data loading (`data.py`), metric computation (`metrics.py`), bootstrap confidence intervals.

## Model selection

The final model set used for all reported results:

```
finbert, longformer, bert, bert-large, roberta, distilbert
```

These six architectures were chosen to cover four different axes of variation so the "inverted signal" finding can be attributed to the data, not to any single model family:

- **finbert (`ProsusAI/finbert`, 110M)** — the primary proposal-specified model. BERT-base pretrained on financial text. Direct match for the domain. Headline evidence for the scripted-section finding comes from this model's 30-seed default sweep.
- **longformer (`allenai/longformer-base-4096`, 149M)** — the other proposal-specified model. Sparse-attention variant that fits full earnings transcripts (median ~9,800 tokens) at a 4,096-token context. Tests whether the signal requires long-range context or sits in a short window near the section boundaries.
- **bert (`bert-base-uncased`, 110M)** — classical BERT baseline. Same architecture as FinBERT without the finance pretraining. Isolates how much of the FinBERT result is from architecture vs. domain-adapted pretraining.
- **bert-large (`bert-large-uncased`, 340M)** — a scaled-up BERT. Tests whether a larger general-purpose model changes the signal direction or magnitude.
- **roberta (`roberta-base`, 125M)** — widely used encoder trained with a different objective and corpus than BERT. Cross-family confirmation.
- **distilbert (`distilbert-base-uncased`, 66M)** — distilled BERT, much smaller. Tests whether the inverted signal persists even with less capacity (i.e., whether it is easy or hard to learn).

**Explicitly excluded** (documented in the paper's limitations):

- `deberta-v3-*` (small, base, large): initially included but excluded from final analysis due to training-time NaN instability on the V100/Ada environment. DeBERTa's disentangled self-attention produces non-finite logits under bf16 mixed precision and retains numerical instability even when forced to fp32 on V100. 180 runs were attempted; all yielded AUROC = 0.5 via the NaN-guard substitution. Skipping this family doesn't weaken the central claim — the inverted scripted signal is already confirmed across the six architectures above.
- `finbert-tone` (`yiyanghkust/finbert-tone`): the repository lacks fast-tokenizer metadata (no `tokenizer.json`), and the slow-tokenizer fallback also fails with `transformers >= 4.40`. Excluded for mechanical rather than scientific reasons. FinBERT (ProsusAI) adequately covers the finance-specific BERT comparison.

## Run naming convention

Every run writes to `runs/<name>/` where `<name>` matches:

```
<model>_<condition>_s<seed>[_<variant>]
```

- `<model>` is a key from `MODEL_REGISTRY`
- `<condition>` is one of `full`, `scripted`, `qa`
- `<seed>` is an integer
- `<variant>` (optional) is one of `trunc-head`, `trunc-middle`, `lowlr`, `frozen`

The trainer's `--output_dir` flag controls this, and the sweep scripts set it accordingly. `aggregate.py` uses the same convention to parse metadata from directory names.

**Examples:**

- `finbert_scripted_s42` — default FinBERT run, seed 42, scripted condition
- `longformer_qa_s0_trunc-head` — Longformer on Q&A with head-biased truncation instead of the default tail-biased truncation
- `finbert_full_s1_frozen` — FinBERT with backbone frozen (head-only fine-tuning)

## Training: `train.py`

### Basic usage

```bash
# Defaults: tail truncation, bf16, 10 epochs, patience 3
python -m models.train --model finbert --condition scripted --seed 42

# Truncation ablation
python -m models.train --model finbert --condition scripted --seed 42 --truncation head

# Hyperparameter ablation: lower learning rate
python -m models.train --model finbert --condition scripted --seed 42 --lr 5e-6

# Head-only training (classifier only; backbone frozen)
python -m models.train --model finbert --condition scripted --seed 42 \
    --freeze_backbone --lr 1e-3 --epochs 30 --patience 5

# Smoke test: 2 epochs on 16 training rows, for pipeline validation
python -m models.train --model finbert --condition qa --epochs 2 --smoke_test
```

### Key design choices

- **Tail-biased truncation (default).** For transcripts longer than `max_length`, we keep the LAST N tokens. Rationale: the proposal hypothesizes that Q&A (end of call) and late executive remarks carry the strongest signal; keeping the tail preserves Q&A content in the `full` condition rather than dropping it.
- **Head and middle truncation (ablation).** `--truncation head` keeps the first N tokens, `--truncation middle` centers the window. Used to test whether the signal lives in a specific part of the transcript.
- **Class-weighted cross-entropy.** Train split is 139 examples at roughly 2:1 control:positive. Inverse-frequency weights prevent positives from being ignored.
- **Early stopping on validation AUROC** with patience 3 (default). The best checkpoint by val AUROC is restored before test evaluation.
- **bf16 mixed precision by default**, with per-model override via `supports_bf16` in the registry. DeBERTa-v3 and some other numerically sensitive models force fp32.
- **NaN guard in `eval_loader`.** If the model produces non-finite predictions, they are replaced with 0.5 (chance) before metric computation. Prevents unhelpful sklearn tracebacks and allows the rest of the sweep to continue.
- **Tokenizer fallback.** If `AutoTokenizer.from_pretrained(...)` fails (for models without fast-tokenizer metadata), the script retries with `use_fast=False`.

### Artifacts per run

Every successful run writes four files to `--output_dir`:

- `config.json` — exact hyperparameters used, including resolved bf16, max_length, batch size, etc.
- `history.csv` — per-epoch training loss, validation metrics, and elapsed time
- `test_metrics.csv` — one-row summary with AUROC, AP, F1, precision, recall, accuracy, bootstrap 95% CI on AUROC, confusion matrix counts
- `test_predictions.csv` — per-example positive-class probability on the test set. Used downstream for error analysis (Tier 3, Bilal's arm) and cross-model calibration.

## Aggregation: `aggregate.py`

Run after any subset of training is complete. Walks `runs/` and produces several CSVs distinguishing default sweeps from ablation variants:

```bash
python -m models.aggregate                        # reads from runs/, writes summaries into runs/
python -m models.aggregate --runs-dir runs --output-dir runs
```

### Outputs

- `summary_all.csv` — one row per completed run, with parsed metadata (model, condition, seed, variant) and all metrics.
- `summary_by_config.csv` — **headline table**: default sweeps only, aggregated by (model, condition). Use this for the "mean ± std across N seeds" reported in the paper.
- `summary_default_by_config.csv` — same as `summary_by_config.csv` (explicit naming).
- `summary_by_family.csv` — every (model, condition, variant) combination. Lets you inspect the ablations.
- `summary_ablations_by_config.csv` — ablations only (truncation, lowlr, frozen), aggregated by (model, condition, variant).
- `summary_mixed_by_config.csv` — pooled across all variants. Diagnostic only; don't report these as headline numbers because they mix different hyperparameters.

### Variant taxonomy

`aggregate.py` parses the variant suffix into three groups:

- `default` — no variant suffix (the primary 30-seed sweep)
- `truncation` — `trunc-head`, `trunc-middle` (tail is the default, so doesn't need a suffix)
- `hyperparam` — `lowlr`, `frozen`

Summaries report mean and standard deviation across seeds for each group separately, so the "headline" 30-seed default number isn't pulled toward the mean by a handful of ablation runs.

### Terminal output

On every run, the script prints:

1. Total completed runs and breakdown by variant
2. Any skipped directories with reasons (missing metrics, bad metadata, unreadable files)
3. Three AUROC summary tables: default-only, ablations-only, and mixed (diagnostic)
4. List of CSVs written

## Reproducing the paper's numbers

```bash
# 1. Setup (once):
pip install -r ../requirements.txt
pip install -r ../requirements-ml.txt

# 2. Run a single config to verify the pipeline:
python -m models.train --model finbert --condition scripted --seed 42

# 3. Full sweep: see scripts/run_local_finbert.ps1, scripts/run_local_small_models.ps1,
#    scripts/msi_run_longformer_max.sh, scripts/msi_run_large_models.sh

# 4. Aggregate all results:
python -m models.aggregate

# 5. Headline numbers live in runs/summary_by_config.csv
```

## Notes

- The trainer, aggregator, and audit script (`scripts/audit_runs.py`) were developed together and share assumptions about run-directory naming. Breaking the naming convention will cause `aggregate.py` to fall back on `config.json` for metadata — slower but still correct.
- Per-run `test_predictions.csv` files are the primary handoff to the error-analysis work in Tier 3. Probability format is identical across architectures, enabling cross-model comparison.
- All training and evaluation use company-disjoint splits (see `../DATA_CURATION_README.md`). No transcript appears in more than one split, and no company appears in more than one split. This is enforced at dataset construction time, not by the trainer.