"""
Shared evaluation metrics for Tier 2.

Primary metric: AUROC (threshold-free, robust to class imbalance).
Secondary: Average Precision (PR-AUC), F1 @ 0.5, accuracy, confusion matrix.
Also provides bootstrap confidence intervals for AUROC.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute the full metric bundle given true labels and positive-class probs.

    Args:
        y_true: (n,) binary labels in {0,1}
        y_prob: (n,) predicted probability of the positive class
        threshold: decision threshold for F1 / precision / recall

    Returns:
        dict with keys: auroc, ap, f1, precision, recall, accuracy,
        tp, fp, tn, fn, n, n_pos
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        # Happens if only one class present (e.g., tiny dev subset)
        auroc = float("nan")

    try:
        ap = float(average_precision_score(y_true, y_prob))
    except ValueError:
        ap = float("nan")

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "auroc": auroc,
        "ap": ap,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n": int(len(y_true)), "n_pos": int(y_true.sum()),
    }


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Stratified bootstrap confidence interval for AUROC.

    Returns:
        (auroc_point, ci_lower, ci_upper) at (1 - alpha) confidence.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    rng = np.random.default_rng(seed)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        point = float("nan")
        return point, float("nan"), float("nan")

    point = float(roc_auc_score(y_true, y_prob))
    boots = []
    for _ in range(n_boot):
        p = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        n = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p, n])
        try:
            boots.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except ValueError:
            continue
    if not boots:
        return point, float("nan"), float("nan")
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi


def format_metrics(m: Dict[str, float]) -> str:
    """One-line metric summary for console logging."""
    return (
        f"AUROC={m['auroc']:.4f}  AP={m['ap']:.4f}  "
        f"F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
        f"Acc={m['accuracy']:.4f}  "
        f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}  "
        f"(n={m['n']}, pos={m['n_pos']})"
    )
