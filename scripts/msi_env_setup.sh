#!/bin/bash
# One-time environment setup for the project on MSI, with fallbacks.
#
# Strategy: try the ideal install first; if it fails, progressively relax.
# At each step, we only care that the final `python -m models.train` import
# succeeds. We don't re-run the data pipeline here, so we don't need every
# package in requirements.txt -- only the ML subset needed for training.
#
# Usage:
#   bash scripts/setup_msi_env.sh

set -uo pipefail   # note: no -e, we handle errors ourselves

echo "=== MSI environment setup (venv-based, with fallbacks) ==="

module purge
module load python3/3.10.9_anaconda2023.03_libmamba
module load cuda/12.1.1

# --- venv ---
if [ -d ".venv" ]; then
    echo ".venv already exists -- will reuse."
else
    echo "Creating .venv ..."
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip --quiet

# ---------------------------------------------------------------------------
# Helper: try a pip install command, return 0 on success.
# ---------------------------------------------------------------------------
try_install() {
    local label="$1"; shift
    echo ""
    echo "--- Attempting: $label ---"
    if pip install "$@"; then
        echo "    SUCCESS: $label"
        return 0
    else
        echo "    FAILED: $label"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Step 1: PyTorch with CUDA
# Priority: cu124 (supports torch 2.6+, unblocks FinBERT CVE issue)
#           fall back to cu121 (torch 2.5.x, Longformer still works)
#           fall back to default PyPI (CPU only, will warn loudly at the end)
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo " Step 1: Installing PyTorch with CUDA support"
echo "=============================================="

if try_install "torch cu124 (preferred)" \
        torch --index-url https://download.pytorch.org/whl/cu124; then
    TORCH_SOURCE="cu124"
elif try_install "torch cu121 (fallback 1)" \
        torch --index-url https://download.pytorch.org/whl/cu121; then
    TORCH_SOURCE="cu121"
elif try_install "torch default PyPI (fallback 2, may be CPU-only!)" \
        torch; then
    TORCH_SOURCE="default-pypi"
else
    echo "ERROR: could not install torch from any source. Aborting."
    exit 1
fi
echo "Using torch from: $TORCH_SOURCE"

# ---------------------------------------------------------------------------
# Step 2: Core ML stack
# Priority: pinned versions from requirements.txt
#           fall back to unpinned (let pip resolve compatible versions)
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo " Step 2: Installing core ML stack"
echo "=============================================="

# Packages actually needed for `python -m models.train`:
#   torch (installed above), transformers, scikit-learn, scipy,
#   numpy, pandas, tqdm, python-dotenv (imported by data.py), requests, huggingface_hub
#
# Packages used only by the data pipeline (data.py), which we do NOT re-run here:
#   aiohttp, datasets, pyarrow, typer, etc. -- safe to skip on MSI.

if try_install "requirements.txt + requirements-ml.txt (pinned)" \
        -r requirements.txt -r requirements-ml.txt; then
    STACK_SOURCE="pinned"
else
    echo ""
    echo "Pinned install failed (likely Python version conflict on pinned numpy)."
    echo "Falling back to relaxed install of just the training subset..."

    if try_install "relaxed ML stack (training subset only)" \
            "numpy<2.3" pandas scikit-learn scipy \
            transformers accelerate \
            sentencepiece protobuf tiktoken \
            tqdm python-dotenv requests huggingface_hub; then
        STACK_SOURCE="relaxed"
    else
        echo "ERROR: could not install core ML stack. Aborting."
        exit 1
    fi
fi
echo "Using ML stack from: $STACK_SOURCE"

# ---------------------------------------------------------------------------
# Step 2.5: Tokenizer backends (idempotent, even if pinned install succeeded)
# Required for DeBERTa-v3, XLM-R, T5, and related architectures.
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo " Step 2.5: Ensuring tokenizer backends"
echo "=============================================="
pip install --quiet sentencepiece protobuf tiktoken || \
    echo "  (warning: tokenizer backend install failed; DeBERTa-v3 will not work)"

# ---------------------------------------------------------------------------
# Step 3: Verification
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo " Step 3: Verifying install"
echo "=============================================="

python - <<'PY'
import sys
failed = []

def check(mod_name, min_version=None):
    try:
        m = __import__(mod_name)
        v = getattr(m, "__version__", "?")
        print(f"  OK  {mod_name:20s} = {v}")
        return True
    except ImportError as e:
        print(f"  !!  {mod_name:20s} = FAILED ({e})")
        failed.append(mod_name)
        return False

check("torch")
check("transformers")
check("sklearn")
check("scipy")
check("numpy")
check("pandas")
check("tqdm")
check("sentencepiece")
check("google.protobuf")
check("tiktoken")

import torch
print()
print(f"  torch.version.cuda   = {torch.version.cuda}")
print(f"  torch.cuda.available = {torch.cuda.is_available()} (expected False on login node)")

if failed:
    print()
    print(f"FAILED IMPORTS: {failed}")
    sys.exit(1)
else:
    print()
    print("All required imports succeeded.")
PY

VERIFY_RC=$?

echo ""
echo "=============================================="
echo " Setup summary"
echo "=============================================="
echo "  torch source:   $TORCH_SOURCE"
echo "  ML stack:       $STACK_SOURCE"
echo "  verification:   $([ $VERIFY_RC -eq 0 ] && echo OK || echo FAILED)"

if [ "$TORCH_SOURCE" = "default-pypi" ]; then
    echo ""
    echo "  WARNING: torch was installed from default PyPI, which usually"
    echo "  means CPU-only. Training will work but will be unusably slow."
    echo "  Check your CUDA module is loaded and rerun this script."
fi

if [ $VERIFY_RC -ne 0 ]; then
    echo ""
    echo "  Setup did NOT complete cleanly. See errors above."
    exit 1
fi

echo ""
echo "=== Done. ==="
echo ""
echo "To use this env in a new shell:"
echo "  module load python3/3.10.9_anaconda2023.03_libmamba cuda/12.1.1"
echo "  source .venv/bin/activate"