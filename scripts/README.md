# Training Scripts

This directory contains all sweep runners. Scripts are grouped by where they run and what they sweep.

## Naming convention

- `run_local_*.ps1` — runs on your local Windows machine (PowerShell)
- `msi_run_*.sh` — Slurm batch jobs for MSI
- `msi_*.sh` — MSI helper scripts (setup, monitoring)
- `run_all.*` — legacy 3-seed sweeps, kept for reproducibility of early results

All sweeps are **idempotent** — if `runs/<name>/test_metrics.csv` already exists, that configuration is skipped. Safe to interrupt and resume.

## Local scripts (Windows, RTX 2000 Ada)

### `run_local_main.ps1`
The main FinBERT sweep plus standard ablations and a few small extra architectures.

- Phase A — FinBERT 30-seed default sweep (90 runs)
- Phase B — FinBERT truncation ablation: head, middle (18 runs)
- Phase C — FinBERT hyperparameter ablations: lowlr, frozen backbone (18 runs)
- Phase D — Extra architectures: RoBERTa, DistilBERT, BERT (27 runs)

Total: ~153 runs, ~8 hours.

Launch:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_local_main.ps1 *>&1 | Tee-Object -FilePath runs\sweep_main.log
```

### `run_local_deberta.ps1`
Additional encoder architectures that fit on 16 GB local VRAM.

- deberta-v3-small, deberta-v3-base, finbert-tone, bert-large
- 3 seeds x 3 conditions x 4 architectures = 36 runs

Best run **after** `run_local_main.ps1` completes (sequential, not parallel — the GPU can only do one thing at a time).

Launch:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_local_deberta.ps1 *>&1 | Tee-Object -FilePath runs\sweep_deberta.log
```

### `run_all.ps1` (legacy)
Original 3-seed sweep of FinBERT and Longformer. Superseded by the scripts above but kept for reference. Do not run.

## MSI scripts (V100 or a100-4)

### `msi_env_setup.sh`
One-time environment setup on MSI. Creates `.venv`, installs torch (cu121 or cu124), transformers, and the relaxed ML stack. Idempotent.

Run once from the MSI login node:
```bash
bash scripts/msi_env_setup.sh
```

### `msi_check_gpus.sh`
Snapshot of MSI GPU partition load: idle/mix/alloc nodes and pending job counts. Shows your own jobs at the bottom.

```bash
bash scripts/msi_check_gpus.sh
```

Use before every submission to pick the partition with the shortest queue. Currently both `run_msi_longformer_max.sh` and `run_msi_large_models.sh` target `v100`; switch to `a100-4` (and `gpu:a100:1`) when its queue is free.

### `msi_run_longformer.sh` (legacy)
Original 10-seed Longformer sweep. Kept for reproducibility.

### `msi_run_longformer_max.sh`
Main Longformer sweep, sized for one 12-hour submission.

- Phase A — 30-seed default sweep (90 runs, ~60 new after initial sub)
- Phase B — truncation ablation (18 runs)
- Phase C — hyperparameter ablations (18 runs)

At ~12 min/run on V100: one submission completes ~55 runs. Resubmit after TIME_LIMIT email to continue — idempotency handles the rest.

Submit:
```bash
sbatch scripts/msi_run_longformer_max.sh
```

### `msi_run_large_models.sh`
Heavy architectures that don't fit on the local GPU.

- deberta-v3-large, bert-large, roberta
- 3 seeds x 3 conditions x 3 architectures = 27 runs

Expected wall time: 2-4 hours. Runs in parallel with `msi_run_longformer_max.sh` — separate job, separate GPU, no conflict.

If the file isn't executable:
```bash
chmod +x scripts/msi_run_large_models.sh
```

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

## After everything runs

Aggregate results into summary tables:
```bash
python -m models.aggregate            # on MSI or local
# produces runs/summary_all.csv and runs/summary_by_config.csv
```

Works on any subset of completed runs — can be called multiple times as more complete.

## Recommended launch order

If you're starting fresh:

1. On MSI login node: `bash scripts/msi_env_setup.sh` (one time, ~5 min)
2. On MSI: `sbatch scripts/msi_run_longformer_max.sh` (backgrounded, ~12h)
3. On MSI: `sbatch scripts/msi_run_large_models.sh` (backgrounded, ~3h)
4. On local: `run_local_main.ps1` (foreground or logged, ~8h)
5. When 4 finishes: `run_local_deberta.ps1` (~3h)

Total: three parallel tracks, ~12h wall-clock until you have everything.