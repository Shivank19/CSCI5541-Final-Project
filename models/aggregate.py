from __future__ import annotations

"""
Aggregate run-level metrics into sweep and ablation summaries.

This script is variant-aware:
- Default sweeps are summarized separately from ablations.
- Ablation variants are kept separate by (model, condition, variant).
- A mixed summary across all variants is also written for diagnostics only.

Expected run directory layout:
    runs/<run_name>/test_metrics.csv
where run_name looks like one of:
    finbert_full_s42
    finbert_qa_s42_trunc-head
    longformer_scripted_s1_lowlr

Outputs (written into --output-dir, default same as --runs-dir):
    summary_all.csv                 # one row per completed run
    summary_by_config.csv          # DEFAULT-only summary by (model, condition)
    summary_default_by_config.csv  # same as summary_by_config.csv
    summary_by_family.csv          # ALL variants by (model, condition, variant)
    summary_ablations_by_config.csv# ablations only by (model, condition, variant)
    summary_mixed_by_config.csv    # diagnostic only: pooled across variants

Usage:
    python -m models.aggregate
    python -m models.aggregate --runs-dir runs --output-dir runs
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


CONDITION_ORDER = ["full", "qa", "scripted"]
VARIANT_ORDER = ["default", "trunc-head", "trunc-middle", "lowlr", "frozen"]
PRIMARY_METRICS = ["auroc", "ap", "f1", "precision", "recall", "accuracy", "best_val_auroc"]
COUNT_METRICS = ["tp", "fp", "tn", "fn", "n", "n_pos"]
ALL_METRICS = PRIMARY_METRICS + COUNT_METRICS + ["best_epoch"]

RUN_RE = re.compile(
    r"^(?P<model>.+?)_(?P<condition>full|qa|scripted)_s(?P<seed>\d+)(?:_(?P<variant>.+))?$"
)


def classify_variant(variant: str) -> str:
    if variant == "default":
        return "default"
    if variant.startswith("trunc-"):
        return "truncation"
    if variant in {"lowlr", "frozen"}:
        return "hyperparam"
    return "other"


def parse_run_name(run_name: str) -> dict[str, Any]:
    m = RUN_RE.match(run_name)
    if not m:
        raise ValueError(f"Unrecognized run name format: {run_name}")
    model = m.group("model")
    condition = m.group("condition")
    seed = int(m.group("seed"))
    variant = m.group("variant") or "default"
    is_default = variant == "default"
    return {
        "model": model,
        "condition": condition,
        "seed": seed,
        "variant": variant,
        "variant_group": classify_variant(variant),
        "run_type": "default_sweep" if is_default else "ablation",
        "family": f"{model}_{condition}" if is_default else f"{model}_{condition}_{variant}",
    }



def load_config_fallback(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return {}
    out: dict[str, Any] = {}
    for key in ["model", "condition", "seed", "variant"]:
        if key in cfg:
            out[key] = cfg[key]
    return out



def infer_metadata(run_dir: Path) -> dict[str, Any]:
    run_name = run_dir.name
    try:
        meta = parse_run_name(run_name)
    except ValueError:
        cfg = load_config_fallback(run_dir)
        if not cfg:
            raise
        if not {"model", "condition", "seed"}.issubset(cfg):
            raise ValueError(
                f"Could not infer metadata for {run_name}: bad run name and incomplete config.json"
            )
        variant = str(cfg.get("variant") or "default")
        meta = {
            "model": str(cfg["model"]),
            "condition": str(cfg["condition"]),
            "seed": int(cfg["seed"]),
            "variant": variant,
            "variant_group": classify_variant(variant),
            "run_type": "default_sweep" if variant == "default" else "ablation",
            "family": f"{cfg['model']}_{cfg['condition']}"
            if variant == "default"
            else f"{cfg['model']}_{cfg['condition']}_{variant}",
        }
    meta["run_name"] = run_name
    return meta



def read_first_metrics_row(metrics_path: Path) -> pd.Series:
    df = pd.read_csv(metrics_path)
    if df.empty:
        raise ValueError("metrics CSV is empty")
    return df.iloc[0]



def load_runs(runs_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    issues: list[str] = []

    run_dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir())
    for run_dir in run_dirs:
        metrics_path = run_dir / "test_metrics.csv"
        if not metrics_path.exists():
            issues.append(f"MISSING_METRICS: {run_dir.name}")
            continue
        try:
            meta = infer_metadata(run_dir)
        except Exception as e:
            issues.append(f"BAD_METADATA: {run_dir.name} ({e})")
            continue
        try:
            row = read_first_metrics_row(metrics_path)
        except Exception as e:
            issues.append(f"BAD_METRICS: {run_dir.name} ({e})")
            continue

        record: dict[str, Any] = {**meta}
        for col in ALL_METRICS:
            record[col] = row[col] if col in row else pd.NA
        rows.append(record)

    if not rows:
        raise RuntimeError(f"No completed runs with readable test_metrics.csv found in {runs_dir}")

    df = pd.DataFrame(rows)

    # Numeric coercion
    for col in [c for c in ALL_METRICS if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Stable sort order for readable outputs
    df["condition"] = pd.Categorical(df["condition"], CONDITION_ORDER, ordered=True)
    variant_categories = VARIANT_ORDER + sorted(v for v in df["variant"].dropna().unique() if v not in VARIANT_ORDER)
    df["variant"] = pd.Categorical(df["variant"], variant_categories, ordered=True)
    df = df.sort_values(["model", "condition", "variant", "seed", "run_name"]).reset_index(drop=True)
    return df, issues



def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["n_runs"])

    records: list[dict[str, Any]] = []
    grouped = df.groupby(group_cols, dropna=False, observed=True, sort=True)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        out: dict[str, Any] = {col: value for col, value in zip(group_cols, key)}
        out.update(
            {
                "n_runs": int(group["run_name"].count()),
                "n_seeds": int(group["seed"].nunique()),
                "auroc_above_half": int((group["auroc"] > 0.5).sum()) if "auroc" in group else 0,
                "auroc_below_half": int((group["auroc"] < 0.5).sum()) if "auroc" in group else 0,
                "auroc_equal_half": int((group["auroc"] == 0.5).sum()) if "auroc" in group else 0,
            }
        )
        for metric in PRIMARY_METRICS:
            if metric in group:
                out[f"{metric}_mean"] = group[metric].mean()
                out[f"{metric}_std"] = group[metric].std(ddof=1)
        for metric in COUNT_METRICS:
            if metric in group:
                out[f"{metric}_mean"] = group[metric].mean()
        if "auroc_mean" in out:
            out["delta_auroc_mean"] = out["auroc_mean"] - 0.5
        records.append(out)

    summary = pd.DataFrame(records)
    if "condition" in summary.columns:
        summary["condition"] = pd.Categorical(summary["condition"], CONDITION_ORDER, ordered=True)
    if "variant" in summary.columns:
        variant_categories = VARIANT_ORDER + sorted(v for v in summary["variant"].dropna().astype(str).unique() if v not in VARIANT_ORDER)
        summary["variant"] = pd.Categorical(summary["variant"], variant_categories, ordered=True)
    sort_cols = [c for c in ["model", "condition", "variant"] if c in summary.columns]
    summary = summary.sort_values(sort_cols).reset_index(drop=True)
    return summary



def format_summary_table(df: pd.DataFrame, include_variant: bool = False) -> str:
    if df.empty:
        return "(none)"
    cols = ["model", "condition"] + (["variant"] if include_variant else []) + [
        "auroc_mean",
        "auroc_std",
        "delta_auroc_mean",
        "n_runs",
    ]
    show = df[cols].copy()
    show["auroc_mean"] = show["auroc_mean"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
    show["auroc_std"] = show["auroc_std"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
    show["delta_auroc_mean"] = show["delta_auroc_mean"].map(lambda x: f"{x:+.4f}" if pd.notna(x) else "nan")
    return show.to_string(index=False)



def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="runs", help="Directory containing per-run subdirectories")
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write summary CSVs (default: same as --runs-dir)",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        raise SystemExit(f"No such runs directory: {runs_dir}")
    output_dir = Path(args.output_dir) if args.output_dir else runs_dir

    df, issues = load_runs(runs_dir)

    default_df = df[df["variant"].astype(str) == "default"].copy()
    ablation_df = df[df["variant"].astype(str) != "default"].copy()

    summary_default = summarize(default_df, ["model", "condition"])
    summary_family = summarize(df, ["model", "condition", "variant"])
    summary_ablations = summarize(ablation_df, ["model", "condition", "variant"])
    summary_mixed = summarize(df, ["model", "condition"])

    # Write outputs
    write_csv(df, output_dir / "summary_all.csv")
    write_csv(summary_default, output_dir / "summary_by_config.csv")
    write_csv(summary_default, output_dir / "summary_default_by_config.csv")
    write_csv(summary_family, output_dir / "summary_by_family.csv")
    write_csv(summary_ablations, output_dir / "summary_ablations_by_config.csv")
    write_csv(summary_mixed, output_dir / "summary_mixed_by_config.csv")

    # Terminal report
    print(f"=== Aggregated runs from {runs_dir} ===")
    print(f"   Completed runs loaded: {len(df)}")
    print(f"   Default sweep runs:    {len(default_df)}")
    print(f"   Ablation runs:         {len(ablation_df)}")
    print()

    if issues:
        print("=== Skipped / unreadable run directories ===")
        for msg in issues[:20]:
            print(f"   {msg}")
        if len(issues) > 20:
            print(f"   ... and {len(issues) - 20} more")
        print()

    by_variant = (
        df.groupby(["variant", "run_type"], observed=True)["run_name"]
        .count()
        .reset_index(name="n_runs")
        .sort_values(["run_type", "variant"])
    )
    print("=== Run counts by variant ===")
    print(by_variant.to_string(index=False))
    print()

    print("=== Default sweep AUROC summary (mean ± std across seeds) ===")
    print(format_summary_table(summary_default, include_variant=False))
    print()

    print("=== Ablation AUROC summary (mean ± std across seeds) ===")
    print(format_summary_table(summary_ablations, include_variant=True))
    print()

    print("=== Mixed summary across all variants (diagnostic only) ===")
    print(format_summary_table(summary_mixed, include_variant=False))
    print()

    print("=== Wrote files ===")
    for name in [
        "summary_all.csv",
        "summary_by_config.csv",
        "summary_default_by_config.csv",
        "summary_by_family.csv",
        "summary_ablations_by_config.csv",
        "summary_mixed_by_config.csv",
    ]:
        print(f"   {output_dir / name}")


if __name__ == "__main__":
    main()
