# Training Scripts

This directory contains sweep runners and experiment utilities for the Tier 2 encoder experiments. Scripts are grouped by where they run and what they do.

## Naming convention

- `run_local_*.ps1` — local Windows PowerShell sweeps
- `msi_run_*.sh` — Slurm batch jobs for MSI
- `msi_*.sh` — MSI helper scripts (setup, monitoring)
- `run_all.*` — legacy 3-seed sweeps kept for reproducibility
- `scripts/audit_runs.py` — run-integrity audit utility
- `python -m models.aggregate` — result aggregation utility

All sweeps are **idempotent**: if `runs/<name>/test_metrics.csv` already exists, that configuration is skipped. This makes the sweeps safe to interrupt and resume.

## Local scripts (Windows, RTX 2000 Ada)

### `run_local_main.ps1`
Canonical local sweep for the main FinBERT experiments plus standard ablations and small extra architectures.

- Phase A — FinBERT 30-seed default sweep (90 runs)
- Phase B — FinBERT truncation ablation: `head`, `middle` (18 runs)
- Phase C — FinBERT hyperparameter ablations: `lowlr`, `frozen` (18 runs)
- Phase D — extra architectures: RoBERTa, DistilBERT, BERT (27 runs)

Total: ~153 runs, ~8 hours.

Launch:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_local_main.ps1 *>&1 | Tee-Object -FilePath runs\sweep_main.log
```

### `run_local_deberta.ps1`
Additional encoder architectures that fit on local VRAM.

- deberta-v3-small
- deberta-v3-base
- finbert-tone
- bert-large

3 seeds × 3 conditions × 4 architectures = 36 runs.

Launch:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_local_deberta.ps1 *>&1 | Tee-Object -FilePath runs\sweep_deberta.log
```

### `run_all.ps1` / `run_all.sh` (legacy)
Original 3-seed Tier 2 sweeps for early reproducibility. These are useful for reference, but should not be used as the headline experiment runners now that the larger sweeps and ablations exist.

## MSI scripts (V100 / A100)

### `msi_env_setup.sh`
One-time MSI environment setup. Creates `.venv`, installs PyTorch with CUDA if available, installs the relaxed ML stack needed for training, and verifies the core imports.

Run once from the MSI login node:
```bash
bash scripts/msi_env_setup.sh
```

### `msi_check_gpus.sh`
Quick snapshot of MSI GPU partition availability.

Shows:
- total nodes
- idle / mixed / allocated nodes
- pending jobs
- your own current jobs

Run:
```bash
bash scripts/msi_check_gpus.sh
```

Use this before each submission to choose the best partition.

### `msi_run_longformer.sh` (legacy)
Original 10-seed Longformer sweep. Kept for reproducibility of the earlier MSI runs.

Submit:
```bash
sbatch scripts/msi_run_longformer.sh
```

### `msi_run_longformer_max.sh`
Main Longformer MSI sweep, sized for one 12-hour submission and designed to be resumed by re-submitting.

- Phase A — 30-seed default sweep (90 runs total)
- Phase B — truncation ablation: `head`, `middle` (18 runs)
- Phase C — hyperparameter ablations: `lowlr`, `frozen` (18 runs)

The script is incremental and idempotent, so partial completion is fine; re-submit after timeout to continue from the remaining runs.

Submit:
```bash
sbatch scripts/msi_run_longformer_max.sh
```

### `msi_run_large_models.sh`
Current MSI large-model sweep.

**Current contents:** `bert-large` only, across 30 seeds × 3 conditions = 90 runs.

Notes:
- `deberta-v3-large` was removed from this script because it produced NaNs on V100 fp32.
- RoBERTa is **not** part of this MSI script in the current version.

Submit:
```bash
sbatch scripts/msi_run_large_models.sh
```

## Monitoring

During a local sweep:
```powershell
Get-Content runs\sweep_main.log -Tail 30 -Wait
```

During an MSI sweep:
```bash
squeue -u $USER
tail -f runs/msi_<JOBID>.log
```

## Run audit utility

### `scripts/audit_runs.py`
Audits the `runs/` directory for corrupted or incomplete runs.

Flags include:
- missing metrics
- smoke-test contamination
- NaN metrics
- missing predictions
- chance-result warnings
- missing config files

It also summarizes runs by **family** (`model + condition + variant`).

Usage:
```bash
python scripts/audit_runs.py
python scripts/audit_runs.py --verbose
python scripts/audit_runs.py --delete-corrupted
```

Run this before aggregation if you suspect interrupted or corrupted runs.

## Result aggregation

Aggregate all completed runs with:
```bash
python -m models.aggregate
```

The current aggregate writes:

- `runs/summary_all.csv` — one row per completed run
- `runs/summary_by_config.csv` — default-sweep summary (canonical main table)
- `runs/summary_default_by_config.csv` — same default-only summary
- `runs/summary_by_family.csv` — grouped by `model + condition + variant`
- `runs/summary_ablations_by_config.csv` — ablations only
- `runs/summary_mixed_by_config.csv` — pooled-across-variants diagnostic summary

The terminal output also prints:
- completed/default/ablation run counts
- run counts by variant
- default-sweep AUROC summary
- ablation AUROC summary
- mixed diagnostic summary

The aggregate is safe to run repeatedly as more runs finish.

## Recommended workflow

If starting fresh:

1. On MSI login node:
   ```bash
   bash scripts/msi_env_setup.sh
   ```

2. Check cluster availability:
   ```bash
   bash scripts/msi_check_gpus.sh
   ```

3. Launch the main Longformer MSI sweep:
   ```bash
   sbatch scripts/msi_run_longformer_max.sh
   ```

4. Launch the MSI large-model sweep:
   ```bash
   sbatch scripts/msi_run_large_models.sh
   ```

5. On local machine, launch the main local sweep:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\run_local_main.ps1 *>&1 | Tee-Object -FilePath runs\sweep_main.log
   ```

6. After that completes, launch the extra local architectures:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\run_local_deberta.ps1 *>&1 | Tee-Object -FilePath runs\sweep_deberta.log
   ```

7. Audit runs:
   ```bash
   python scripts/audit_runs.py
   ```

8. Aggregate results:
   ```bash
   python -m models.aggregate
   ```

## Notes on reporting

For paper/reporting purposes:

- Use the **default sweep** summary as the main results table.
- Use the **ablation summary** separately.
- Treat the **mixed summary** as diagnostic only.
- Use `summary_all.csv` for seed-level plots and distribution visualizations.
