#!/usr/bin/env python3
"""
Plot default-sweep delta-AUROC summaries as separate paper-ready PDFs.

Inputs:
- runs/summary_default_by_config.csv   (preferred)
- runs/summary_by_config.csv           (fallback)

Outputs (PDF only, by default into ./plots):
- plots/plot_delta_auroc_full.pdf
- plots/plot_delta_auroc_scripted.pdf
- plots/plot_delta_auroc_qa.pdf

Usage:
    python plots/plot_delta_auroc.py
    python plots/plot_delta_auroc.py --summary runs/summary_default_by_config.csv --output-dir plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


CONDITION_ORDER = ["full", "scripted", "qa"]
MODEL_ORDER = ["roberta", "distilbert", "bert", "longformer", "finbert", "bert-large"]
PRETTY_CONDITION = {
    "full": "Full Transcript",
    "scripted": "Scripted Remarks",
    "qa": "Q&A Only",
}
OUTPUT_BASENAME = "plot_delta_auroc"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Path to default-only aggregate summary CSV. "
             "If omitted, tries runs/summary_default_by_config.csv then runs/summary_by_config.csv.",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory for PDF outputs (default: plots)",
    )
    return ap.parse_args()


def resolve_summary_path(user_path: str | None) -> Path:
    if user_path:
        path = Path(user_path)
        if not path.exists():
            raise FileNotFoundError(f"Summary file not found: {path}")
        return path

    candidates = [
        Path("runs/summary_default_by_config.csv"),
        Path("runs/summary_by_config.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path

    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not find a default summary CSV. Tried:\n" + tried
    )


def validate_columns(df: pd.DataFrame) -> None:
    required_any = {"model", "condition"}
    missing_basic = required_any - set(df.columns)
    if missing_basic:
        raise ValueError(
            f"Missing required columns: {sorted(missing_basic)}"
        )

    has_delta = "delta_auroc_mean" in df.columns
    has_auroc = "auroc_mean" in df.columns
    has_std = "auroc_std" in df.columns

    if not has_delta and not has_auroc:
        raise ValueError(
            "CSV must contain either 'delta_auroc_mean' or 'auroc_mean'."
        )
    if not has_std:
        raise ValueError("CSV must contain 'auroc_std' for error bars.")


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_columns(df)

    if "delta_auroc_mean" not in df.columns:
        df["delta_auroc_mean"] = df["auroc_mean"] - 0.5

    if "n_runs" not in df.columns and "count" in df.columns:
        df["n_runs"] = df["count"]

    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=True)

    available_models = [m for m in MODEL_ORDER if m in set(df["model"])]
    remaining = [m for m in df["model"].unique().tolist() if m not in available_models]
    full_order = available_models + sorted(remaining)

    df["model"] = pd.Categorical(df["model"], categories=full_order, ordered=True)
    df = df.sort_values(["condition", "model"]).reset_index(drop=True)
    return df


def print_terminal_summary(df: pd.DataFrame, summary_path: Path) -> None:
    print("=== Delta-AUROC plot input summary ===")
    print(f"   Source:     {summary_path}")
    print(f"   Rows loaded:{len(df):>7d}")
    print(f"   Models:     {df['model'].nunique():>7d}")
    print(f"   Conditions: {df['condition'].nunique():>7d}")
    print()

    cols = ["condition", "model", "delta_auroc_mean", "auroc_mean", "auroc_std"]
    if "n_runs" in df.columns:
        cols.append("n_runs")

    display_df = df[cols].copy()
    display_df["delta_auroc_mean"] = display_df["delta_auroc_mean"].map(lambda x: f"{x:+.4f}")
    display_df["auroc_mean"] = display_df["auroc_mean"].map(lambda x: f"{x:.4f}")
    display_df["auroc_std"] = display_df["auroc_std"].map(lambda x: f"{x:.4f}")

    print("=== Default delta-AUROC summary ===")
    print(display_df.to_string(index=False))
    print()


def iter_conditions(df: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    for cond in CONDITION_ORDER:
        sub = df[df["condition"] == cond].copy()
        if len(sub) > 0:
            yield cond, sub


def add_value_labels(ax: plt.Axes, xs: list[float], ys: list[float]) -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin if ymax > ymin else 1.0
    offset = span * 0.025

    for x, y in zip(xs, ys):
        va = "bottom" if y >= 0 else "top"
        y_text = y + offset if y >= 0 else y - offset
        ax.text(
            x,
            y_text,
            f"{y:+.3f}",
            ha="center",
            va=va,
            fontsize=9,
        )


def make_condition_plot(sub: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")

    fig, ax = plt.subplots(figsize=(8.8, 4.8))

    # Bar plot
    sns.barplot(
        data=sub,
        x="model",
        y="delta_auroc_mean",
        ax=ax,
        hue="model",
        dodge=False,
        legend=False,
        edgecolor="black",
        linewidth=0.8,
    )

    # Error bars: use AUROC std directly, since delta is just AUROC - 0.5.
    xs = list(range(len(sub)))
    ys = sub["delta_auroc_mean"].astype(float).tolist()
    errs = sub["auroc_std"].astype(float).tolist()
    ax.errorbar(
        xs,
        ys,
        yerr=errs,
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=3,
        zorder=5,
    )

    # Reference line at chance.
    ax.axhline(0.0, linestyle="--", linewidth=1.1)

    # Labels and title.
    condition = str(sub["condition"].iloc[0])
    ax.set_title(f"Default Sweep ΔAUROC — {PRETTY_CONDITION.get(condition, condition)}", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("ΔAUROC (AUROC − 0.5)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")

    # Y-limits with padding.
    yvals = [y - e for y, e in zip(ys, errs)] + [y + e for y, e in zip(ys, errs)] + [0.0]
    ymin = min(yvals)
    ymax = max(yvals)
    pad = max(0.03, 0.12 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_ylim(ymin - pad, ymax + pad)

    add_value_labels(ax, xs, ys)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = resolve_summary_path(args.summary)
    output_dir = Path(args.output_dir)

    df = load_summary(summary_path)
    print_terminal_summary(df, summary_path)

    written = []
    for condition, sub in iter_conditions(df):
        output_path = output_dir / f"{OUTPUT_BASENAME}_{condition}.pdf"
        make_condition_plot(sub, output_path)
        written.append(output_path)

    print("=== Wrote files ===")
    for path in written:
        print(f"   {path}")


if __name__ == "__main__":
    main()
