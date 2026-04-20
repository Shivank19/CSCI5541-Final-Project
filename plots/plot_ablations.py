#!/usr/bin/env python3
"""
Plot matched-seed ablation runs for FinBERT and Longformer with conditional styling.

Behavior:
- Reads raw run-level data from runs/summary_all.csv
- Keeps all ablation runs
- Filters default runs to the same seed subset used by the ablations for each model
- For each model:
    * if every condition×variant group has n >= --boxplot-threshold:
        draw light boxplots + raw jittered points
    * otherwise:
        draw raw jittered points + mean markers only

Outputs (PDF only, by default into ./plots):
- plots/plot_ablations_finbert.pdf
- plots/plot_ablations_longformer.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns


MODEL_ORDER = ["finbert", "longformer"]
CONDITION_ORDER = ["full", "scripted", "qa"]
VARIANT_ORDER = ["default", "trunc-head", "trunc-middle", "lowlr", "frozen"]

PRETTY_MODEL = {
    "finbert": "FinBERT",
    "longformer": "Longformer",
}
PRETTY_CONDITION = {
    "full": "Full",
    "scripted": "Scripted",
    "qa": "Q&A",
}
OUTPUT_BASENAME = "plot_ablations"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=str, default="runs/summary_all.csv")
    ap.add_argument("--output-dir", type=str, default="plots")
    ap.add_argument("--boxplot-threshold", type=int, default=5)
    ap.add_argument("--ymin", type=float, default=0.0)
    ap.add_argument("--ymax", type=float, default=1.0)
    return ap.parse_args()


def resolve_summary(path_str: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Run-level summary not found: {path}")
    return path


def validate_columns(df: pd.DataFrame) -> None:
    required = {"model", "condition", "variant", "auroc", "seed", "run_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in summary_all.csv: {sorted(missing)}")


def load_data(summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    validate_columns(df)

    df = df[df["model"].isin(MODEL_ORDER)].copy()
    df = df[df["condition"].isin(CONDITION_ORDER)].copy()
    df = df[df["variant"].isin(VARIANT_ORDER)].copy()
    df = df[df["run_type"].isin(["default_sweep", "ablation"])].copy()

    ablations = df[df["run_type"] == "ablation"].copy()
    defaults = df[df["run_type"] == "default_sweep"].copy()

    matched_defaults_parts = []
    for model in MODEL_ORDER:
        model_ablation_seeds = sorted(
            ablations.loc[ablations["model"] == model, "seed"].dropna().unique().tolist()
        )
        if len(model_ablation_seeds) == 0:
            continue
        part = defaults[
            (defaults["model"] == model) &
            (defaults["seed"].isin(model_ablation_seeds))
        ].copy()
        matched_defaults_parts.append(part)

    matched_defaults = (
        pd.concat(matched_defaults_parts, ignore_index=True)
        if matched_defaults_parts else
        defaults.iloc[0:0].copy()
    )

    out = pd.concat([matched_defaults, ablations], ignore_index=True)

    out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
    out["condition"] = pd.Categorical(out["condition"], categories=CONDITION_ORDER, ordered=True)
    out["variant"] = pd.Categorical(out["variant"], categories=VARIANT_ORDER, ordered=True)

    out = out.sort_values(["model", "condition", "variant", "seed"]).reset_index(drop=True)
    return out


def print_terminal_summary(df: pd.DataFrame, summary_path: Path, boxplot_threshold: int) -> None:
    print("=== Conditional ablation plot input summary ===")
    print(f"   Source:             {summary_path}")
    print(f"   Rows loaded:        {len(df)}")
    print(f"   Models:             {df['model'].nunique()}")
    print(f"   Conditions:         {df['condition'].nunique()}")
    print(f"   Variants:           {df['variant'].nunique()}")
    print(f"   Boxplot threshold:  {boxplot_threshold}")
    print()

    counts = (
        df.groupby(["model", "condition", "variant"], observed=True)
          .size()
          .reset_index(name="n_runs")
          .sort_values(["model", "condition", "variant"])
    )
    print("=== Raw runs per model / condition / variant ===")
    print(counts.to_string(index=False))
    print()


def should_use_boxplot(model_df: pd.DataFrame, threshold: int) -> bool:
    counts = model_df.groupby(["condition", "variant"], observed=True).size()
    if len(counts) == 0:
        return False
    return int(counts.min()) >= threshold


def build_mean_markers(model_df: pd.DataFrame) -> pd.DataFrame:
    return (
        model_df.groupby(["condition", "variant"], observed=True)["auroc"]
        .mean()
        .reset_index(name="auroc_mean")
    )


def _scatter_legend_handles(palette):
    handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='',
            markersize=7,
            markerfacecolor=palette[i],
            markeredgecolor='black',
            markeredgewidth=0.6,
            color=palette[i],
        )
        for i in range(len(VARIANT_ORDER))
    ]
    return handles, VARIANT_ORDER


def make_plot(sub: pd.DataFrame, output_path: Path, ymin: float, ymax: float, boxplot_threshold: int) -> None:
    sns.set_theme(style="whitegrid", context="paper")

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    plot_df = sub.copy()
    plot_df["condition_label"] = plot_df["condition"].map(PRETTY_CONDITION)
    palette = sns.color_palette("deep", n_colors=len(VARIANT_ORDER))

    use_box = should_use_boxplot(sub, boxplot_threshold)

    if use_box:
        sns.boxplot(
            data=plot_df,
            x="condition_label",
            y="auroc",
            hue="variant",
            hue_order=VARIANT_ORDER,
            dodge=True,
            width=0.68,
            showcaps=True,
            fliersize=0,
            linewidth=1.0,
            saturation=1,
            boxprops={"zorder": 1},
            whiskerprops={"linewidth": 1.0, "zorder": 1},
            capprops={"linewidth": 1.0, "zorder": 1},
            medianprops={"linewidth": 1.3, "zorder": 2},
            palette=palette,
            ax=ax,
        )

        # Make boxes translucent so points remain visible.
        for patch in ax.artists:
            patch.set_alpha(0.22)
            patch.set_edgecolor("black")
            patch.set_zorder(1)

        # Some seaborn/matplotlib versions store patches in ax.patches instead.
        for patch in ax.patches:
            try:
                patch.set_alpha(0.22)
                patch.set_edgecolor("black")
                patch.set_zorder(1)
            except Exception:
                pass

        if ax.legend_ is not None:
            ax.legend_.remove()

        sns.stripplot(
            data=plot_df,
            x="condition_label",
            y="auroc",
            hue="variant",
            hue_order=VARIANT_ORDER,
            dodge=True,
            jitter=0.16,
            size=5.4,
            alpha=0.88,
            linewidth=0.45,
            edgecolor="black",
            palette=palette,
            ax=ax,
            zorder=10,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        style_text = "Style: light boxplots + raw points"

    else:
        sns.stripplot(
            data=plot_df,
            x="condition_label",
            y="auroc",
            hue="variant",
            hue_order=VARIANT_ORDER,
            dodge=True,
            jitter=0.16,
            size=5.8,
            alpha=0.82,
            linewidth=0.45,
            edgecolor="black",
            palette=palette,
            ax=ax,
            zorder=5,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()

        mean_df = build_mean_markers(sub)
        mean_df["condition_label"] = mean_df["condition"].map(PRETTY_CONDITION)
        sns.pointplot(
            data=mean_df,
            x="condition_label",
            y="auroc_mean",
            hue="variant",
            hue_order=VARIANT_ORDER,
            dodge=0.48,
            join=False,
            markers="D",
            scale=0.9,
            errorbar=None,
            palette=palette,
            ax=ax,
            zorder=8,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        style_text = "Style: raw points + mean diamonds"

    ax.axhline(0.5, linestyle="--", linewidth=1.1)
    ax.set_xlabel("")
    ax.set_ylabel("AUROC")
    ax.set_ylim(ymin, ymax)

    model = str(sub["model"].iloc[0])
    pretty_model = PRETTY_MODEL.get(model, model)
    counts = sub.groupby("variant", observed=True).size().to_dict()
    counts_text = ", ".join(
        f"{variant}={counts[variant]}"
        for variant in VARIANT_ORDER
        if variant in counts
    )

    ax.set_title(f"{pretty_model} Ablations (matched default seeds)", pad=14)
    subtitle = style_text
    if counts_text:
        subtitle += " | Raw runs per variant: " + counts_text
    ax.text(
        0.5, 1.02,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )

    handles, labels = _scatter_legend_handles(palette)
    legend = ax.legend(handles, labels, title="Variant", frameon=True, ncol=2)
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.ymin >= args.ymax:
        raise ValueError("--ymin must be smaller than --ymax")

    summary_path = resolve_summary(args.summary)
    output_dir = Path(args.output_dir)

    df = load_data(summary_path)
    print_terminal_summary(df, summary_path, args.boxplot_threshold)

    written = []
    for model in MODEL_ORDER:
        sub = df[df["model"] == model].copy()
        if len(sub) == 0:
            continue
        output_path = output_dir / f"{OUTPUT_BASENAME}_{model}.pdf"
        make_plot(sub, output_path, ymin=args.ymin, ymax=args.ymax, boxplot_threshold=args.boxplot_threshold)
        written.append(output_path)

    print("=== Wrote files ===")
    for path in written:
        print(f"   {path}")


if __name__ == "__main__":
    main()
