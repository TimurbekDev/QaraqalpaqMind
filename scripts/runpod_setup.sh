#!/usr/bin/env bash
#
# One-command setup for a RunPod GPU pod.
#
#   bash scripts/runpod_setup.sh
#
# Assumes the RunPod PyTorch template (torch already installed, CUDA present)
# and a persistent volume at /workspace. Safe to re-run: every step is
# idempotent, which matters because pods get reclaimed mid-setup.
#
# What it does NOT do: start training, or download the corpus. Data transfer is
# a decision with a licence attached - see step 5 below.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"

log()  { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

cd "$REPO_ROOT"

# --- 1. sanity ---------------------------------------------------------------
log "Environment"
python3 --version
if ! command -v nvidia-smi >/dev/null 2>&1; then
  die "nvidia-smi not found. This is not a GPU pod."
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

PY_MINOR="$(python3 -c 'import sys; print(sys.version_info.minor)')"
if [ "$PY_MINOR" -lt 12 ]; then
  die "Python 3.12+ required (found 3.${PY_MINOR}); the codebase uses PEP 695 generics."
fi

# --- 2. persistent caches ----------------------------------------------------
# The default HF cache is under $HOME, which on a container is the ephemeral
# overlay filesystem. A 16 GB model download would be lost on every restart.
if [ -d "$WORKSPACE" ]; then
  log "Pointing caches at the persistent volume: $WORKSPACE"
  export HF_HOME="${HF_HOME:-$WORKSPACE/.cache/huggingface}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORKSPACE/.cache/triton}"
  mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR"

  # Persist for future shells, without duplicating the export if re-run.
  if ! grep -q "QARAQALPAQMIND_ENV" ~/.bashrc 2>/dev/null; then
    {
      echo ""
      echo "# QARAQALPAQMIND_ENV"
      echo "export HF_HOME=$HF_HOME"
      echo "export TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
      echo "export TOKENIZERS_PARALLELISM=false"
    } >> ~/.bashrc
  fi
else
  warn "$WORKSPACE not found; caches stay on ephemeral storage and will be lost on restart."
fi

# Silences a tokenizers fork warning that appears once per dataloader worker.
export TOKENIZERS_PARALLELISM=false

# --- 3. install --------------------------------------------------------------
log "Installing the package"
python3 -m pip install --upgrade pip --quiet

# torch is NOT reinstalled: the template ships a CUDA-matched build that
# satisfies the >=2.5 bound, and replacing it risks a CPU-only or
# CUDA-mismatched wheel.
TORCH_BEFORE="$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo none)"
log "torch before install: $TORCH_BEFORE"

python3 -m pip install -e ".[train,ingest,clean]" --quiet

TORCH_AFTER="$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo none)"
if [ "$TORCH_BEFORE" != "none" ] && [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
  warn "torch changed $TORCH_BEFORE -> $TORCH_AFTER. Verify CUDA still works:"
  warn "  python3 -c 'import torch; print(torch.cuda.is_available())'"
fi

# --- 4. environment file -----------------------------------------------------
if [ ! -f .env ]; then
  log "Creating .env from the template"
  cp .env.example .env
  warn "Set HF_TOKEN in .env if you need gated datasets or want to push/pull data."
fi
if [ -n "${HF_HOME:-}" ] && ! grep -q '^HF_HOME=' .env 2>/dev/null; then
  echo "HF_HOME=$HF_HOME" >> .env
fi

# --- 5. data -----------------------------------------------------------------
PRETRAIN="data/datasets/pretrain/pretrain_v1.jsonl.zst"
if [ -s "$PRETRAIN" ]; then
  log "Training corpus present: $PRETRAIN"
else
  warn "No training corpus. data/ is not in git, so a fresh clone has none."
  warn ""
  warn "Choose one:"
  warn "  a) Pull the exact corpus you built elsewhere (needs HF_TOKEN):"
  warn "       qm ingest pull <your-name>/qaraqalpaqmind-data"
  warn "  b) Rebuild from public sources only - no crawling, ~93% of the corpus:"
  warn "       qm ingest all --bulk-only && qm clean all && qm dedup run"
  warn "  c) Copy the file to $PRETRAIN yourself (runpodctl / scp)."
fi

# --- 6. verify ---------------------------------------------------------------
log "Verifying the install"
qm version
qm doctor || true

echo
log "Preflight"
qm train preflight --config configs/cpt/qwen3_8b_qlora_24gb.yaml || {
  warn "Preflight reported problems. Fix them before training."
  exit 1
}

echo
log "Setup complete. To train:"
echo "    qm train plan --config configs/cpt/qwen3_8b_qlora_24gb.yaml"
echo "    qm train cpt  --config configs/cpt/qwen3_8b_qlora_24gb.yaml"
echo
log "Training resumes automatically after a pod restart (resume_from_checkpoint: auto)."
