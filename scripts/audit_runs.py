"""
Audit the runs/ directory for corrupted or incomplete training runs.

Flags any run directory that matches these patterns:

  MISSING_METRICS  : directory exists but has no test_metrics.csv
                     -> training crashed before eval; safe to delete + rerun
  SMOKE_TEST       : test_metrics has n < 20 (smoke test accidentally saved)
                     -> delete + rerun with full dataset
  NAN_METRICS      : AUROC or other metrics are NaN
                     -> model diverged; rerun (may need hparam change)
  PARTIAL_PREDS    : test_predictions.csv missing even though metrics exist
                     -> eval completed but save failed; metrics OK, but missing
                        per-example probs for error analysis
  CHANCE_RESULT    : AUROC == 0.5 (likely NaN-guard replacement); worth review
  MISSING_CONFIG   : config.json missing; reproducibility risk

Also reports totals by run "family" (model+condition+variant).

Usage:
    python scripts/audit_runs.py
    python scripts/audit_runs.py --delete-corrupted   (actually remove them)
    python scripts/audit_runs.py --runs-dir path/to/runs
"""
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from collections import defaultdict

import pandas as pd


CORRUPTION_LEVELS = {
    "MISSING_METRICS": "severe",   # must delete + rerun
    "SMOKE_TEST":      "severe",   # must delete + rerun
    "NAN_METRICS":     "severe",   # must delete + rerun
    "PARTIAL_PREDS":   "minor",    # metrics valid; predictions missing
    "CHANCE_RESULT":   "warning",  # metrics == 0.5 may be NaN-guarded
    "MISSING_CONFIG":  "minor",    # reproducibility degraded
}


def audit_run(run_dir: Path) -> list[str]:
    """Return list of issue tags for a single run directory."""
    issues = []

    metrics_path = run_dir / "test_metrics.csv"
    preds_path = run_dir / "test_predictions.csv"
    config_path = run_dir / "config.json"

    # Check metrics CSV
    if not metrics_path.exists():
        issues.append("MISSING_METRICS")
        return issues  # can't check anything else without metrics

    try:
        df = pd.read_csv(metrics_path)
    except Exception:
        issues.append("MISSING_METRICS")
        return issues

    if len(df) == 0:
        issues.append("MISSING_METRICS")
        return issues

    row = df.iloc[0]

    # Smoke test contamination: n too small to be real test set
    if "n" in row and pd.notna(row["n"]) and int(row["n"]) < 20:
        issues.append("SMOKE_TEST")

    # NaN in critical metrics
    nan_metrics = []
    for col in ["auroc", "ap", "f1"]:
        if col in row and pd.isna(row[col]):
            nan_metrics.append(col)
    if nan_metrics:
        issues.append("NAN_METRICS")

    # Chance result — possibly NaN-guarded
    if "auroc" in row and pd.notna(row["auroc"]) and row["auroc"] == 0.5:
        issues.append("CHANCE_RESULT")

    # Predictions file
    if not preds_path.exists():
        issues.append("PARTIAL_PREDS")
    else:
        try:
            pred_df = pd.read_csv(preds_path)
            if len(pred_df) == 0:
                issues.append("PARTIAL_PREDS")
        except Exception:
            issues.append("PARTIAL_PREDS")

    # Config file
    if not config_path.exists():
        issues.append("MISSING_CONFIG")

    return issues


def parse_run_family(run_name: str) -> str:
    """Group runs by (model, condition, variant) ignoring seed.

    e.g., 'finbert_qa_s42' -> 'finbert_qa'
    e.g., 'longformer_full_s0_lowlr' -> 'longformer_full_lowlr'
    """
    parts = run_name.split("_")
    # Drop the seed component (starts with 's' followed by digits)
    kept = [p for p in parts if not (p.startswith("s") and p[1:].isdigit())]
    return "_".join(kept)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="runs",
                    help="Path to runs/ directory (default: runs)")
    ap.add_argument("--delete-corrupted", action="store_true",
                    help="Actually delete run directories with severe issues")
    ap.add_argument("--verbose", action="store_true",
                    help="Show per-run details for clean runs too")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        print(f"No such directory: {runs_dir}")
        return

    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    if not run_dirs:
        print(f"No run directories found in {runs_dir}")
        return

    by_issue = defaultdict(list)
    by_family = defaultdict(lambda: {"total": 0, "clean": 0, "issues": 0})
    clean_runs = []
    flagged_runs = []  # (path, issues)

    for rd in run_dirs:
        issues = audit_run(rd)
        family = parse_run_family(rd.name)
        by_family[family]["total"] += 1
        if not issues:
            clean_runs.append(rd.name)
            by_family[family]["clean"] += 1
        else:
            flagged_runs.append((rd, issues))
            by_family[family]["issues"] += 1
            for tag in issues:
                by_issue[tag].append(rd.name)

    total = len(run_dirs)
    print(f"=== Audited {total} run directories in {runs_dir} ===")
    print(f"   Clean:   {len(clean_runs)}")
    print(f"   Flagged: {len(flagged_runs)}")
    print()

    if by_issue:
        print("=== Issues by type ===")
        # Sort by severity, then by count
        severity_order = {"severe": 0, "minor": 1, "warning": 2}
        for issue, runs in sorted(
            by_issue.items(),
            key=lambda x: (severity_order.get(CORRUPTION_LEVELS.get(x[0], "warning"), 3),
                           -len(x[1]))
        ):
            sev = CORRUPTION_LEVELS.get(issue, "?")
            print(f"  [{sev:7s}] {issue:18s}  {len(runs)} runs")
            for r in runs[:5]:
                print(f"              {r}")
            if len(runs) > 5:
                print(f"              ... and {len(runs) - 5} more")
        print()

    print("=== By run family ===")
    print(f"   {'family':<40s}  {'total':>6s}  {'clean':>6s}  {'issues':>6s}")
    for fam in sorted(by_family):
        d = by_family[fam]
        marker = " (!)" if d["issues"] > 0 else ""
        print(f"   {fam:<40s}  {d['total']:>6d}  {d['clean']:>6d}  {d['issues']:>6d}{marker}")
    print()

    if args.verbose and clean_runs:
        print("=== Clean runs ===")
        for r in clean_runs:
            print(f"   {r}")
        print()

    # Collect severe issues for potential deletion
    severe_dirs = []
    for rd, issues in flagged_runs:
        severe_issues = [i for i in issues if CORRUPTION_LEVELS.get(i) == "severe"]
        if severe_issues:
            severe_dirs.append((rd, severe_issues))

    if severe_dirs:
        print(f"=== {len(severe_dirs)} directories with SEVERE issues ===")
        for rd, issues in severe_dirs:
            print(f"   {rd.name}  [{', '.join(issues)}]")
        print()

        if args.delete_corrupted:
            print(f"Deleting {len(severe_dirs)} directories (--delete-corrupted was set)...")
            for rd, _ in severe_dirs:
                shutil.rmtree(rd)
                print(f"  removed: {rd}")
            print("Done. Resubmit sweep scripts to regenerate these runs.")
        else:
            print("To delete these and allow resubmission to re-run them, use:")
            print(f"  python {Path(__file__).name} --delete-corrupted")
    else:
        print("No severe issues found.")


if __name__ == "__main__":
    main()