# ── Realtime LiveAvatar — CUDA runtime image for RunPod GPU pods ─────────────
# For `mock` mode you can swap the base for python:3.10-slim. The CUDA base is
# what the WanS2V-14B renderer needs.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv \
        ffmpeg git curl ca-certificates \
        libsndfile1 libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# Renderer deps are installed at deploy time against the cloned LiveAvatar repo
# (see startup.sh / SETUP.md), because torch + flash-attn are pod-specific.

COPY backend ./backend
COPY frontend ./frontend
COPY startup.sh .
RUN chmod +x startup.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["./startup.sh"]
