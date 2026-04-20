# Plots

This folder contains plotting scripts for the paper figures.

## Current scripts

- `plot_default_auroc.py`  
  Creates default-sweep AUROC distribution plots from `runs/summary_all.csv`.  
  Outputs:
  - `plots/plot_default_auroc_full.pdf`
  - `plots/plot_default_auroc_scripted.pdf`
  - `plots/plot_default_auroc_qa.pdf`

- `plot_delta_auroc.py`  
  Creates default-only delta-AUROC summary plots from `runs/summary_default_by_config.csv`  
  (or `runs/summary_by_config.csv` as fallback).  
  Outputs:
  - `plots/plot_delta_auroc_full.pdf`
  - `plots/plot_delta_auroc_scripted.pdf`
  - `plots/plot_delta_auroc_qa.pdf`

- `plot_ablations_conditional.py`  
  Creates matched-seed ablation plots for FinBERT and Longformer from `runs/summary_all.csv`.  
  The script:
  - keeps all ablation runs
  - filters default runs to the same seed subset used by the ablations
  - uses conditional styling:
    - boxplots + raw points when group size is large enough
    - raw points + mean markers otherwise  
  Outputs:
  - `plots/plot_ablations_finbert.pdf`
  - `plots/plot_ablations_longformer.pdf`

## Recommended workflow

1. Aggregate results first:
```bash
python -m models.aggregate
```

2. Generate the main default AUROC plots:
```bash
python plots/plot_default_auroc.py
```

3. Generate the delta-AUROC summary plots:
```bash
python plots/plot_delta_auroc.py
```

4. Generate the ablation plots:
```bash
python plots/plot_ablations_conditional.py
```

## Notes

- All plot scripts save **PDF only** by default.
- The default output directory is `plots/`.
- The plotting scripts are designed to read the aggregate CSV outputs in `runs/` instead of reparsing the raw run directories.
- For paper reporting:
  - use default-sweep plots as the main result figures
  - use ablation plots separately
  - keep mixed summaries as diagnostic only
