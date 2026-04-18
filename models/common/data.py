"""
Shared dataset loading utilities for Tier 2 transformer training.

Provides:
- load_split(): read train/val/test CSVs with consistent text-column selection
- get_text_column(): map condition name -> CSV column
- filter_empty(): drop rows with empty text in the chosen condition

Used by both FinBERT and Longformer trainers, and can be used by
Tier 1 baseline and Tier 3 LLM prompting too so that n per condition
is identical across tiers.
"""
from __future__ import annotations

import os
from typing import Tuple

import pandas as pd


CONDITION_TO_COLUMN = {
    "full": "full_text",
    "scripted": "scripted_text",
    "qa": "qa_text",
}


def get_text_column(condition: str) -> str:
    if condition not in CONDITION_TO_COLUMN:
        raise ValueError(
            f"Unknown condition '{condition}'. "
            f"Expected one of: {list(CONDITION_TO_COLUMN)}"
        )
    return CONDITION_TO_COLUMN[condition]


def load_split(
    split: str,
    condition: str,
    data_dir: str = "data/splits",
    drop_empty: bool = True,
) -> pd.DataFrame:
    """Load a single split (train/val/test) with the text column for a condition.

    Args:
        split: 'train' | 'val' | 'test'
        condition: 'full' | 'scripted' | 'qa'
        data_dir: directory containing the three split CSVs
        drop_empty: if True, rows with empty/NaN text in the chosen column
            are dropped. Always report the effective n after dropping.

    Returns:
        DataFrame with columns: transcript_id, ticker, label, text, call_date, source.
        (Canonicalized 'text' column regardless of condition.)
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train|val|test, got '{split}'")

    path = os.path.join(data_dir, f"{split}.csv")
    df = pd.read_csv(path)

    col = get_text_column(condition)
    df = df.rename(columns={col: "text"})

    # Drop other text columns we don't need to keep memory down
    for other_col in CONDITION_TO_COLUMN.values():
        if other_col in df.columns and other_col != col:
            df = df.drop(columns=[other_col])

    # Handle missing/empty text
    before = len(df)
    df["text"] = df["text"].fillna("").astype(str)
    if drop_empty:
        mask = df["text"].str.strip() != ""
        dropped = (~mask).sum()
        df = df[mask].reset_index(drop=True)
        if dropped > 0:
            print(
                f"  [load_split] {split}/{condition}: "
                f"dropped {dropped}/{before} empty-text rows "
                f"(remaining: {len(df)}, positives: {int(df['label'].sum())})"
            )

    keep = [c for c in ["transcript_id", "ticker", "label", "text",
                        "call_date", "source"] if c in df.columns]
    return df[keep].copy()


def load_all_splits(
    condition: str,
    data_dir: str = "data/splits",
    drop_empty: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience: load train/val/test for a single condition."""
    train = load_split("train", condition, data_dir, drop_empty)
    val = load_split("val", condition, data_dir, drop_empty)
    test = load_split("test", condition, data_dir, drop_empty)
    return train, val, test


def compute_class_weights(labels) -> Tuple[float, float]:
    """Inverse-frequency class weights for binary classification.

    Returns (weight_for_class_0, weight_for_class_1).
    Used in CrossEntropyLoss to counter the 2:1 control:positive imbalance.
    """
    import numpy as np

    labels = np.asarray(labels)
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return (1.0, 1.0)
    w_neg = n / (2.0 * n_neg)
    w_pos = n / (2.0 * n_pos)
    return (float(w_neg), float(w_pos))
