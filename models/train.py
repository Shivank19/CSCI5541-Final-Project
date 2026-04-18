"""
Unified FinBERT + Longformer fine-tuning script for the restatement
precursor classification task (CSCI 5541 Tier 2).

Usage examples:
    # FinBERT on Q&A section only, seed 42
    python -m models.train --model finbert --condition qa --seed 42

    # Longformer on full transcript, seed 0, custom output dir
    python -m models.train --model longformer --condition full --seed 0 \\
        --output_dir runs/lf_full_s0

    # Quick smoke test (2 epochs, tiny batch) to verify env before the real sweep
    python -m models.train --model finbert --condition full --epochs 2 --smoke_test

Design notes:
    - Tail-biased truncation: we keep the LAST N tokens of each transcript
      because the proposal hypothesizes that Q&A (end of call) and late
      executive remarks carry the strongest linguistic-evasion signal.
    - Class-weighted cross-entropy handles the 2:1 control:positive imbalance.
    - Early stopping on val AUROC with patience 3 (prevents overfitting on
      our small n ~ 139 train).
    - bf16 mixed precision by default on CUDA (Ada-native, more stable than fp16).
    - Gradient checkpointing auto-enabled for Longformer (memory).
    - Metrics + per-example predictions are saved to output_dir as CSVs for
      downstream analysis (H2/H3/H4 hypothesis testing).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Make common modules importable when run as `python -m models.train`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.common import (
    bootstrap_auroc_ci,
    compute_class_weights,
    compute_metrics,
    format_metrics,
    load_all_splits,
)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    # --- Core Tier 2 models (project proposal) ---
    "finbert": {
        "hf_name": "ProsusAI/finbert",
        "max_length": 512,
        "per_device_batch_size": 8,
        "grad_accum_steps": 1,
        "grad_checkpointing": False,
    },
    "longformer": {
        "hf_name": "allenai/longformer-base-4096",
        "max_length": 4096,
        "per_device_batch_size": 1,
        "grad_accum_steps": 8,
        "grad_checkpointing": True,
    },
    # --- General-purpose encoders (architecture ablation) ---
    "roberta": {
        "hf_name": "roberta-base",
        "max_length": 512,
        "per_device_batch_size": 8,
        "grad_accum_steps": 1,
        "grad_checkpointing": False,
    },
    "distilbert": {
        "hf_name": "distilbert-base-uncased",
        "max_length": 512,
        "per_device_batch_size": 16,
        "grad_accum_steps": 1,
        "grad_checkpointing": False,
    },
    "bert": {
        "hf_name": "bert-base-uncased",
        "max_length": 512,
        "per_device_batch_size": 8,
        "grad_accum_steps": 1,
        "grad_checkpointing": False,
    },
    # --- Stronger modern encoder (likely strongest-performing general LM) ---
    # NOTE: DeBERTa-v3 family is numerically sensitive. bf16 produces NaN logits
    # on some sequence lengths. Force fp32 with supports_bf16=False.
    "deberta-v3-base": {
        "hf_name": "microsoft/deberta-v3-base",
        "max_length": 512,
        "per_device_batch_size": 8,
        "grad_accum_steps": 1,
        "grad_checkpointing": False,
        "supports_bf16": False,
    },
    # --- Very small / very fast baseline ---
    "deberta-v3-small": {
        "hf_name": "microsoft/deberta-v3-small",
        "max_length": 512,
        "per_device_batch_size": 16,
        "grad_accum_steps": 1,
        "grad_checkpointing": False,
        "supports_bf16": False,
    },
    # --- Alternative finance-specific models (direct FinBERT comparison) ---
    "finbert-tone": {
        "hf_name": "yiyanghkust/finbert-tone",
        "max_length": 512,
        "per_device_batch_size": 8,
        "grad_accum_steps": 1,
        "grad_checkpointing": False,
    },
    # --- Heavier models (MSI recommended) ---
    "deberta-v3-large": {
        "hf_name": "microsoft/deberta-v3-large",
        "max_length": 512,
        "per_device_batch_size": 2,
        "grad_accum_steps": 4,
        "grad_checkpointing": True,
        "supports_bf16": False,
    },
    "bert-large": {
        "hf_name": "bert-large-uncased",
        "max_length": 512,
        "per_device_batch_size": 4,
        "grad_accum_steps": 2,
        "grad_checkpointing": True,
    },
}


@dataclass
class TrainConfig:
    model: str
    condition: str
    seed: int
    epochs: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    max_length: int
    per_device_batch_size: int
    grad_accum_steps: int
    grad_checkpointing: bool
    patience: int
    data_dir: str
    output_dir: str
    bf16: bool
    smoke_test: bool
    freeze_backbone: bool = False
    truncation: str = "tail"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TranscriptDataset:
    """Truncation-aware tokenization dataset.

    Supports three truncation strategies:
      - 'tail'   : keep LAST max_length tokens (default; end of call / Q&A)
      - 'head'   : keep FIRST max_length tokens (opening of call / scripted start)
      - 'middle' : keep middle max_length tokens (centered on the transcript)

    Rationale: tail captures the end of the call (where spontaneous Q&A lives);
    head captures the opening (most-rehearsed prepared remarks); middle captures
    the transition zone between scripted and Q&A.
    """

    def __init__(self, texts, labels, tokenizer, max_length: int,
                 truncation: str = "tail"):
        import torch
        self.torch = torch
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        if truncation not in ("tail", "head", "middle"):
            raise ValueError(f"truncation must be tail|head|middle, got {truncation}")
        self.truncation = truncation

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])

        # Tokenize without truncation so we can apply our own strategy.
        ids = self.tokenizer.encode(text, add_special_tokens=False)

        # Budget: max_length minus [CLS] + [SEP]
        budget = self.max_length - 2
        if len(ids) > budget:
            if self.truncation == "tail":
                ids = ids[-budget:]
            elif self.truncation == "head":
                ids = ids[:budget]
            else:  # middle
                start = (len(ids) - budget) // 2
                ids = ids[start:start + budget]

        # Re-add special tokens
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        ids = [cls_id] + ids + [sep_id]

        # Pad
        attention_mask = [1] * len(ids)
        pad_len = self.max_length - len(ids)
        if pad_len > 0:
            ids = ids + [self.tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": self.torch.tensor(ids, dtype=self.torch.long),
            "attention_mask": self.torch.tensor(attention_mask,
                                                dtype=self.torch.long),
            "labels": self.torch.tensor(label, dtype=self.torch.long),
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(cfg: TrainConfig) -> Dict[str, float]:
    """Fine-tune the configured model and return test-set metrics."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    print(f"\n[1/5] Loading data: condition={cfg.condition}")
    train_df, val_df, test_df = load_all_splits(cfg.condition, cfg.data_dir)

    if cfg.smoke_test:
        train_df = train_df.head(16)
        val_df = val_df.head(8)
        test_df = test_df.head(8)
        print(f"  [smoke_test] truncated to train={len(train_df)} "
              f"val={len(val_df)} test={len(test_df)}")

    print(f"  train: n={len(train_df)}, pos={int(train_df.label.sum())}")
    print(f"  val:   n={len(val_df)}, pos={int(val_df.label.sum())}")
    print(f"  test:  n={len(test_df)}, pos={int(test_df.label.sum())}")

    # ---- Tokenizer + Model ----
    spec = MODEL_REGISTRY[cfg.model]
    print(f"\n[2/5] Loading {cfg.model}: {spec['hf_name']}")
    tokenizer = AutoTokenizer.from_pretrained(spec["hf_name"])
    # FinBERT ships a 3-class sentiment head; we discard it and re-init for
    # binary classification. `ignore_mismatched_sizes=True` is required so
    # the loader doesn't error on the classifier-weight shape mismatch.
    # id2label/label2id are explicitly re-specified so checkpoint metadata
    # reflects our binary task.
    model = AutoModelForSequenceClassification.from_pretrained(
        spec["hf_name"],
        num_labels=2,
        ignore_mismatched_sizes=True,
        id2label={0: "control", 1: "pre_restatement"},
        label2id={"control": 0, "pre_restatement": 1},
    )
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()

    # Optional: freeze all backbone params, train only the classifier head.
    # Useful as an ablation at small n to check whether fine-tuning the
    # backbone actually helps vs. just training the classifier.
    if cfg.freeze_backbone:
        n_frozen = 0
        n_trainable = 0
        for name, p in model.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False
                n_frozen += p.numel()
            else:
                n_trainable += p.numel()
        print(f"  [freeze_backbone] frozen={n_frozen:,} trainable={n_trainable:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  device: {device}")
    if device.type == "cuda":
        print(f"  gpu:    {torch.cuda.get_device_name(0)}")

    # ---- Datasets ----
    train_ds = TranscriptDataset(train_df.text, train_df.label,
                                 tokenizer, cfg.max_length,
                                 truncation=cfg.truncation)
    val_ds = TranscriptDataset(val_df.text, val_df.label,
                               tokenizer, cfg.max_length,
                               truncation=cfg.truncation)
    test_ds = TranscriptDataset(test_df.text, test_df.label,
                                tokenizer, cfg.max_length,
                                truncation=cfg.truncation)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.per_device_batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.per_device_batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.per_device_batch_size)

    # ---- Optimizer / Scheduler / Loss ----
    w0, w1 = compute_class_weights(train_df.label)
    print(f"  class weights: [neg={w0:.3f}, pos={w1:.3f}]")
    class_weights = torch.tensor([w0, w1], dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    no_decay = ["bias", "LayerNorm.weight"]
    optim_params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_params, lr=cfg.lr)

    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    amp_dtype = torch.bfloat16 if (cfg.bf16 and device.type == "cuda") else None

    # ---- Train loop with early stopping on val AUROC ----
    print(f"\n[3/5] Training: epochs={cfg.epochs}, "
          f"bs={cfg.per_device_batch_size}, grad_accum={cfg.grad_accum_steps}, "
          f"lr={cfg.lr}, bf16={amp_dtype is not None}")

    history: List[Dict[str, float]] = []
    best_auroc = -1.0
    best_epoch = -1
    best_state: Optional[Dict] = None
    patience_ctr = 0

    for epoch in range(cfg.epochs):
        # --- train ---
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"])
                    loss = loss_fn(out.logits, batch["labels"])
            else:
                out = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"])
                loss = loss_fn(out.logits, batch["labels"])

            loss = loss / cfg.grad_accum_steps
            loss.backward()
            running_loss += loss.item() * cfg.grad_accum_steps
            n_batches += 1

            if (step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        train_loss = running_loss / max(1, n_batches)

        # --- eval on val ---
        val_metrics, _, _ = eval_loader(model, val_loader, device, amp_dtype)
        elapsed = time.time() - t0
        print(f"  epoch {epoch+1:>2}/{cfg.epochs} "
              f"| train_loss={train_loss:.4f} "
              f"| val_AUROC={val_metrics['auroc']:.4f} "
              f"val_AP={val_metrics['ap']:.4f} "
              f"val_F1={val_metrics['f1']:.4f} "
              f"| {elapsed:.1f}s")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "elapsed_sec": elapsed,
        })

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            best_epoch = epoch + 1
            # keep a cpu copy so we can restore later without extra GPU mem
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                print(f"  [early stop] no val-AUROC improvement for "
                      f"{cfg.patience} epochs; stopping at epoch {epoch+1}")
                break

    # ---- Restore best checkpoint and evaluate on test ----
    print(f"\n[4/5] Restoring best checkpoint (epoch {best_epoch}, "
          f"val_AUROC={best_auroc:.4f})")
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, test_probs, test_labels = eval_loader(
        model, test_loader, device, amp_dtype
    )
    # AUROC with bootstrap CI on the test set
    _, ci_lo, ci_hi = bootstrap_auroc_ci(test_labels, test_probs,
                                          n_boot=1000, seed=cfg.seed)
    test_metrics["auroc_ci_lo"] = ci_lo
    test_metrics["auroc_ci_hi"] = ci_hi

    print(f"\n[5/5] TEST: {format_metrics(test_metrics)}")
    print(f"       AUROC 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    # ---- Save artifacts ----
    save_artifacts(cfg, out_dir, history, test_metrics,
                   test_df, test_probs, best_epoch, best_auroc)
    return test_metrics


def eval_loader(model, loader, device, amp_dtype):
    """Run a loader through the model and return (metrics, probs, labels)."""
    import torch
    model.eval()
    all_probs: List[float] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"])
            else:
                out = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"])
            probs = torch.softmax(out.logits.float(), dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(batch["labels"].cpu().numpy().tolist())
    probs_arr = np.asarray(all_probs)
    labels_arr = np.asarray(all_labels)
    # Guard against NaN in predictions (seen with DeBERTa-v3 + bf16). If the
    # model produced NaNs, we substitute 0.5 so downstream metrics report
    # chance-level performance instead of raising an unhelpful sklearn error.
    nan_mask = np.isnan(probs_arr)
    if nan_mask.any():
        print(f"  [warn] {int(nan_mask.sum())}/{len(probs_arr)} predictions "
              f"were NaN; replacing with 0.5 (chance)")
        probs_arr = np.where(nan_mask, 0.5, probs_arr)
    return compute_metrics(labels_arr, probs_arr), probs_arr, labels_arr


def save_artifacts(cfg, out_dir, history, test_metrics,
                   test_df, test_probs, best_epoch, best_val_auroc) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Config
    with (out_dir / "config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Per-epoch history
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # Test metrics (single row, easy to concat across runs)
    meta = {
        "model": cfg.model, "condition": cfg.condition, "seed": cfg.seed,
        "best_epoch": best_epoch, "best_val_auroc": best_val_auroc,
    }
    pd.DataFrame([{**meta, **test_metrics}]).to_csv(
        out_dir / "test_metrics.csv", index=False
    )

    # Per-example test predictions for later error analysis
    keep = [c for c in ["transcript_id", "ticker", "call_date",
                        "source", "label"] if c in test_df.columns]
    pred_df = test_df[keep].copy().reset_index(drop=True)
    pred_df["prob_positive"] = test_probs
    pred_df["pred_label_0.5"] = (test_probs >= 0.5).astype(int)
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    print(f"\n  artifacts written to: {out_dir}")
    print(f"    - config.json")
    print(f"    - history.csv")
    print(f"    - test_metrics.csv")
    print(f"    - test_predictions.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(
        description="Fine-tune FinBERT or Longformer on the restatement task."
    )
    p.add_argument("--model", choices=list(MODEL_REGISTRY),
                   required=True, help="Which backbone to fine-tune.")
    p.add_argument("--condition", choices=["full", "scripted", "qa"],
                   required=True, help="Which transcript section to use.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=None,
                   help="Override default max_length for the model.")
    p.add_argument("--per_device_batch_size", type=int, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=None)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--data_dir", default="data/splits")
    p.add_argument("--output_dir", default=None,
                   help="Default: runs/{model}_{condition}_s{seed}")
    p.add_argument("--no_bf16", action="store_true",
                   help="Disable bf16 mixed precision.")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze backbone; train only the classifier head.")
    p.add_argument("--truncation", choices=["tail", "head", "middle"],
                   default="tail",
                   help="Truncation strategy for long transcripts.")
    p.add_argument("--smoke_test", action="store_true",
                   help="Use tiny subset (~16 rows) to verify the pipeline.")

    a = p.parse_args()

    spec = MODEL_REGISTRY[a.model]
    if a.max_length is None:
        a.max_length = spec["max_length"]
    if a.per_device_batch_size is None:
        a.per_device_batch_size = spec["per_device_batch_size"]
    if a.grad_accum_steps is None:
        a.grad_accum_steps = spec["grad_accum_steps"]
    if a.output_dir is None:
        a.output_dir = f"runs/{a.model}_{a.condition}_s{a.seed}"

    # Honor per-model bf16 support. DeBERTa-v3 and similar models declare
    # supports_bf16=False because bf16 produces NaN logits on them.
    # User-provided --no_bf16 also disables.
    model_supports_bf16 = spec.get("supports_bf16", True)
    effective_bf16 = (not a.no_bf16) and model_supports_bf16
    if (not a.no_bf16) and (not model_supports_bf16):
        print(f"[info] {a.model} declares supports_bf16=False; forcing fp32.")

    return TrainConfig(
        model=a.model,
        condition=a.condition,
        seed=a.seed,
        epochs=a.epochs,
        lr=a.lr,
        weight_decay=a.weight_decay,
        warmup_ratio=a.warmup_ratio,
        max_length=a.max_length,
        per_device_batch_size=a.per_device_batch_size,
        grad_accum_steps=a.grad_accum_steps,
        grad_checkpointing=spec["grad_checkpointing"],
        patience=a.patience,
        data_dir=a.data_dir,
        output_dir=a.output_dir,
        bf16=effective_bf16,
        smoke_test=a.smoke_test,
        freeze_backbone=a.freeze_backbone,
        truncation=a.truncation,
    )


def main() -> None:
    cfg = parse_args()
    print(f"== CONFIG ==")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    run_training(cfg)


if __name__ == "__main__":
    main()