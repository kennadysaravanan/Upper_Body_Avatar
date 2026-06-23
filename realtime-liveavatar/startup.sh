#!/usr/bin/env bash
# Container entrypoint. Optionally installs the LiveAvatar renderer, then starts
# the FastAPI app with uvicorn.
set -euo pipefail

echo "[startup] AVATAR_ENGINE=${AVATAR_ENGINE:-mock} GPU_COUNT=${GPU_COUNT:-1}"

if [[ "${AVATAR_ENGINE:-mock}" == "liveavatar" ]]; then
  REPO="${LIVEAVATAR_REPO:-/workspace/LiveAvatar}"
  if [[ ! -d "$REPO" ]]; then
    echo "[startup] cloning LiveAvatar -> $REPO"
    git clone https://github.com/Alibaba-Quark/LiveAvatar "$REPO"
  fi
  if [[ "${INSTALL_RENDERER:-0}" == "1" ]]; then
    echo "[startup] installing torch + renderer deps (pod-specific)…"
    pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
    pip install flash-attn==2.8.3 --no-build-isolation || true
    pip install -r "$REPO/requirements.txt"
  fi
  if [[ ! -d "${LIVEAVATAR_CKPT:-$REPO/ckpt}/Wan2.2-S2V-14B" ]]; then
    echo "[startup] WARNING: checkpoints not found under ${LIVEAVATAR_CKPT:-$REPO/ckpt}."
    echo "          Download Wan-AI/Wan2.2-S2V-14B and Quark-Vision/Live-Avatar (see SETUP.md)."
  fi
fi

WORKERS="${UVICORN_WORKERS:-1}"   # keep 1: each worker holds a single GPU pipeline
exec uvicorn backend.main:app \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" \
  --workers "$WORKERS" --log-config /dev/null
