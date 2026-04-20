#!/usr/bin/env python3
from __future__ import annotations

"""
Create paper-ready default-sweep AUROC distribution plots, one PDF per condition.

Inputs:
    runs/summary_all.csv

Outputs (by default):
    plots/plot_default_auroc_full.pdf
    plots/plot_default_auroc_scripted.pdf
    plots/plot_default_auroc_qa.pdf

Usage:
    python plots/plot_default_auroc.py
    python plots/plot_default_auroc.py --summary runs/summary_all.csv --output-dir plots
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RUN_RE = re.compile(
    r"^(?P<model>.+?)_(?P<condition>full|qa|scripted)_s(?P<seed>\d+)(?:_(?P<variant>.+))?$"
)

CONDITION_ORDER = ["full", "scripted", "qa"]
CONDITION_LABELS = {
    "full": "Full transcript",
    "scripted": "Prepared remarks",
    "qa": "Q&A only",
}

MODEL_LABELS = {
    "bert": "BERT",
    "bert-large": "BERT-large",
    "distilbert": "DistilBERT",
    "finbert": "FinBERT",
    "longformer": "Longformer",
    "roberta": "RoBERTa",
}


def infer_default_from_run_name(run_name: str) -> bool:
    m = RUN_RE.match(str(run_name))
    if not m:
        return False
    variant = m.group("variant")
    return variant is None or variant == ""


def load_default_runs(summary_path: Path) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    df = pd.read_csv(summary_path)
    required = {"run_name", "model", "condition", "auroc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {summary_path}: {sorted(missing)}")

    if "run_type" in df.columns:
        out = df[df["run_type"].astype(str) == "default_sweep"].copy()
    elif "variant" in df.columns:
        out = df[df["variant"].fillna("default").astype(str) == "default"].copy()
    else:
        out = df[df["run_name"].map(infer_default_from_run_name)].copy()

    out["auroc"] = pd.to_numeric(out["auroc"], errors="coerce")
    out = out.dropna(subset=["model", "condition", "auroc"]).copy()
    out = out[out["condition"].isin(CONDITION_ORDER)].copy()

    if out.empty:
        raise ValueError("No default-sweep rows found after filtering.")

    return out


def choose_model_order(df: pd.DataFrame) -> list[str]:
    full_means = (
        df[df["condition"] == "full"]
        .groupby("model", as_index=False)["auroc"]
        .mean()
        .rename(columns={"auroc": "full_mean"})
    )
    overall_means = (
        df.groupby("model", as_index=False)["auroc"]
        .mean()
        .rename(columns={"auroc": "overall_mean"})
    )
    merged = overall_means.merge(full_means, on="model", how="left")
    merged["sort_key"] = merged["full_mean"].fillna(merged["overall_mean"])
    merged = merged.sort_values(["sort_key", "model"], ascending=[False, True])
    return merged["model"].tolist()


def prettify_model_labels(model_order: list[str]) -> list[str]:
    return [MODEL_LABELS.get(m, m) for m in model_order]


def print_terminal_summary(df: pd.DataFrame, model_order: list[str]) -> None:
    print("=== Default AUROC plot input summary ===")
    print(f"   Rows loaded: {len(df)}")
    print(f"   Models:      {len(model_order)}")
    print(f"   Conditions:  {len(CONDITION_ORDER)}")
    print()

    counts = (
        df.groupby(["condition", "model"]).size().rename("n_runs").reset_index()
    )
    counts["condition"] = pd.Categorical(counts["condition"], CONDITION_ORDER, ordered=True)
    counts["model"] = pd.Categorical(counts["model"], model_order, ordered=True)
    counts = counts.sort_values(["condition", "model"])
    print("=== Runs per model/condition ===")
    print(counts.to_string(index=False))
    print()

    summary = (
        df.groupby(["condition", "model"])["auroc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["condition"] = pd.Categorical(summary["condition"], CONDITION_ORDER, ordered=True)
    summary["model"] = pd.Categorical(summary["model"], model_order, ordered=True)
    summary = summary.sort_values(["condition", "model"])
    summary["delta_auroc"] = summary["mean"] - 0.5

    printable = summary.copy()
    printable["mean"] = printable["mean"].map(lambda x: f"{x:.4f}")
    printable["std"] = printable["std"].map(lambda x: f"{x:.4f}")
    printable["delta_auroc"] = printable["delta_auroc"].map(lambda x: f"{x:+.4f}")
    printable = printable.rename(columns={"count": "n_runs"})
    print("=== Mean AUROC by model/condition ===")
    print(printable.to_string(index=False))
    print()


def make_one_plot(df: pd.DataFrame, condition: str, model_order: list[str], output_path: Path) -> None:
    sub = df[df["condition"] == condition].copy()
    if sub.empty:
        raise ValueError(f"No rows found for condition={condition}")

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    palette = sns.color_palette("deep", n_colors=len(model_order))
    palette_map = dict(zip(model_order, palette))
    label_map = dict(zip(model_order, prettify_model_labels(model_order)))

    fig, ax = plt.subplots(figsize=(7.8, 4.6))

    sns.boxplot(
        data=sub,
        x="model",
        y="auroc",
        order=model_order,
        color="#e9edf3",
        width=0.56,
        fliersize=0,
        linewidth=1.0,
        ax=ax,
    )

    sns.stripplot(
        data=sub,
        x="model",
        y="auroc",
        order=model_order,
        hue="model",
        hue_order=model_order,
        palette=palette_map,
        dodge=False,
        jitter=0.18,
        alpha=0.8,
        size=4.3,
        edgecolor="white",
        linewidth=0.35,
        legend=False,
        ax=ax,
    )

    stats = (
        sub.groupby("model", as_index=False)["auroc"]
        .agg(mean="mean", std="std", count="count")
    )
    x_lookup = {m: i for i, m in enumerate(model_order)}
    stats["x"] = stats["model"].map(x_lookup)

    ax.errorbar(
        stats["x"],
        stats["mean"],
        yerr=stats["std"],
        fmt="none",
        ecolor="black",
        elinewidth=1.1,
        capsize=3,
        zorder=4,
    )
    ax.scatter(
        stats["x"],
        stats["mean"],
        marker="D",
        s=30,
        color="black",
        zorder=5,
        label="Mean ± SD",
    )

    ax.axhline(0.5, linestyle="--", linewidth=1.1, color="black", alpha=0.8)

    #y_min = max(0.0, min(0.25, float(sub["auroc"].min()) - 0.04))
    y_min = 0
    #y_max = min(1.0, max(0.75, float(sub["auroc"].max()) + 0.04))
    y_max = 1
    ax.set_ylim(y_min, y_max)

    ax.set_title(f"Default sweep AUROC — {CONDITION_LABELS[condition]}")
    ax.set_xlabel("")
    ax.set_ylabel("AUROC")
    ax.set_xticklabels([label_map[m] for m in model_order], rotation=25, ha="right")

    for i, row in stats.iterrows():
        ax.text(
            row["x"],
            y_max - 0.02,
            f"n={int(row['count'])}",
            ha="center",
            va="top",
            fontsize=8,
            color="#4c566a",
        )

    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summary",
        type=Path,
        default=Path("runs") / "summary_all.csv",
        help="Path to summary_all.csv (default: runs/summary_all.csv)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory for PDF outputs (default: plots)",
    )
    ap.add_argument(
        "--stem",
        default="plot_default_auroc",
        help="Output filename stem (default: plot_default_auroc)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    df = load_default_runs(args.summary)
    model_order = choose_model_order(df)
    print_terminal_summary(df, model_order)

    print("=== Writing figures ===")
    for condition in CONDITION_ORDER:
        output_path = args.output_dir / f"{args.stem}_{condition}.pdf"
        make_one_plot(df, condition, model_order, output_path)
        print(f"   {output_path}")


if __name__ == "__main__":
    main()
