"""
Aggregate test metrics across all runs and produce summary tables.

Reads every `runs/*/test_metrics.csv`, concatenates them, then produces:
  - runs/summary_all.csv         all individual runs
  - runs/summary_by_config.csv   mean ± std AUROC per (model, condition)

Usage:
    python -m models.aggregate
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    runs_dir = Path("runs")
    csvs = sorted(runs_dir.glob("*/test_metrics.csv"))
    if not csvs:
        print("No runs found. Train some models first.")
        return

    frames = []
    for p in csvs:
        try:
            df = pd.read_csv(p)
            df["run_name"] = p.parent.name
            frames.append(df)
        except Exception as e:
            print(f"  skip {p}: {e}")

    all_runs = pd.concat(frames, ignore_index=True)
    cols_front = ["run_name", "model", "condition", "seed",
                  "auroc", "auroc_ci_lo", "auroc_ci_hi",
                  "ap", "f1", "precision", "recall", "accuracy",
                  "best_epoch", "best_val_auroc"]
    cols = [c for c in cols_front if c in all_runs.columns] + \
           [c for c in all_runs.columns if c not in cols_front]
    all_runs = all_runs[cols]
    all_runs.to_csv(runs_dir / "summary_all.csv", index=False)
    print(f"Wrote {runs_dir/'summary_all.csv'} ({len(all_runs)} runs)")

    # Aggregate by (model, condition)
    agg_cols = ["auroc", "ap", "f1", "precision", "recall", "accuracy"]
    agg_cols = [c for c in agg_cols if c in all_runs.columns]
    grouped = (
        all_runs.groupby(["model", "condition"])[agg_cols]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    grouped.to_csv(runs_dir / "summary_by_config.csv")
    print(f"Wrote {runs_dir/'summary_by_config.csv'}")
    print()
    print("=== AUROC summary (mean ± std across seeds) ===")
    auroc_summary = (
        all_runs.groupby(["model", "condition"])["auroc"]
        .agg(["mean", "std", "count"]).round(4)
    )
    print(auroc_summary)


if __name__ == "__main__":
    main()
