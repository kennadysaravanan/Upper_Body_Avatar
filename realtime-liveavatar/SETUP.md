# RunPod Deployment Guide

Two deployment profiles:

| Profile | GPU | Purpose | Realtime? |
|---|---|---|---|
| **Mock** | none / any | full stack, demo, CI, frontend dev | yes (placeholder render) |
| **LiveAvatar realtime** | **5× H800 (80 GB)** or 8× H100 | production avatar | yes, ~45 FPS, 1.2 s TTFF |
| LiveAvatar single-GPU | 1× 80 GB (48 GB w/ FP8) | offline / preview | no (sub-realtime) |

> Reality check (see `REPORT.md` §0): realtime LiveAvatar needs a **multi-GPU
> pod**. A single GPU runs the model but not at realtime. The OpenAI half is
> sub-second on any pod.

---

## 1. Create a RunPod GPU pod

1. RunPod → **Pods** → **Deploy**.
2. GPU: **5× H800 80 GB** (realtime) or **1× H100 80 GB** (offline/dev). For mock,
   any small pod or even CPU works.
3. Template: **RunPod PyTorch 2.x / CUDA 12.4** (or "Custom" with our Dockerfile).
4. Container disk ≥ 60 GB; **Volume ≥ 200 GB** mounted at `/workspace` (the
   WanS2V-14B + LoRA checkpoints are large).
5. Expose **HTTP 8000** and **TCP 22**. For WebRTC media, also expose a **UDP
   port range** (e.g. 40000–40100) or plan to use TURN (step 8).

## 2. Clone the repositories

```bash
cd /workspace
git clone https://github.com/Alibaba-Quark/LiveAvatar          # the model
git clone <your-repo>/realtime-liveavatar                       # this platform
cd realtime-liveavatar
cp .env.example .env
```

## 3. Install dependencies

```bash
# platform deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# renderer deps (GPU pod only — skip for mock)
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
# Hopper (H800/H200):
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280 --extra-index-url https://download.pytorch.org/whl/cu128
# otherwise:
# pip install flash-attn==2.8.3 --no-build-isolation
pip install -r /workspace/LiveAvatar/requirements.txt
apt-get update && apt-get install -y ffmpeg
```

## 4. Download checkpoints (GPU profile)

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir /workspace/LiveAvatar/ckpt/Wan2.2-S2V-14B
huggingface-cli download Quark-Vision/Live-Avatar --local-dir /workspace/LiveAvatar/ckpt/LiveAvatar
```

## 5. Configure OpenAI credentials

OpenAI keys are entered **per session in the browser** — nothing server-side is
required. (Optional: set `OPENAI_HEALTH_KEY` in `.env` for health probes only.)

Edit `.env` for the GPU profile:

```bash
AVATAR_ENGINE=liveavatar
GPU_COUNT=1            # 1 = embedded single-GPU generate() path (runs in this process)
ENABLE_FP8=true        # fits 48GB
OFFLOAD_MODEL=true
TARGET_FPS=16          # MUST match LA_NATIVE_FPS so audio/video stay in sync
LIVEAVATAR_REPO=/workspace/LiveAvatar
LIVEAVATAR_CKPT=/workspace/LiveAvatar/ckpt/Wan2.2-S2V-14B
LA_SIZE=704*384
LA_SAMPLE_STEPS=4
LA_LORA_PATH=ckpt/LiveAvatar
PRELOAD_MODEL=1        # load the 14B weights at startup, keep warm
```

### Single-GPU (embedded) vs multi-GPU TPP (realtime)

`backend/avatar/liveavatar_engine.py` is **already bound** to the real `WanS2V`
API — no code to write. Two modes:

- **`GPU_COUNT=1` (embedded):** the engine calls `WanS2V.generate(...)` per spoken
  sentence inside the FastAPI process. Works after `git clone` + checkpoints on a
  single 80GB (48GB w/ FP8) GPU. Latency is per-sentence (offline-ish), not the
  45 FPS streaming path — correct and runnable, just not the fastest.
- **`GPU_COUNT=5` (TPP realtime):** the 5-GPU pipeline
  (`causal_s2v_pipeline_tpp*`) runs as a `torchrun --nproc_per_node=5` job (see
  the repo's `infinite_inference_multi_gpu.sh` /
  `minimal_inference/s2v_streaming_interact.py`). It does **not** embed in one
  uvicorn process. Run it as a separate worker and bridge frames over a queue;
  the engine docstring marks the seam. Launch uvicorn under torchrun only if you
  understand the rank-0-serves model.

## 6. Build the Docker image (optional but recommended)

```bash
docker build -t realtime-liveavatar:latest .
# or run the whole stack:
docker compose up -d --build
```

## 7. Start the backend (serves the frontend too)

```bash
# bare metal:
PYTHONPATH=. ./startup.sh
# or:
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8000

# verify:
curl -s http://localhost:8000/readyz | python -m json.tool
```

The frontend is served by the same process at `/` (StaticFiles mount). No
separate frontend server is needed.

## 8. Configure TURN (required for browsers behind NAT)

WebRTC media is peer-to-peer UDP and does **not** pass through nginx. On RunPod,
either expose a UDP range and set the pod's public IP, or run coturn:

```bash
docker run -d --network host coturn/coturn \
  -n --listening-port=3478 --realm=avatar \
  --user=avatar:supersecret --no-tls --no-dtls
```

Then in `.env`:

```bash
TURN_URL=turn:<POD_PUBLIC_IP>:3478
TURN_USER=avatar
TURN_PASSWORD=supersecret
```

## 9. Nginx + SSL

WebRTC requires HTTPS (browsers block `getUserMedia`/secure contexts otherwise).

```bash
# put fullchain.pem + privkey.pem in nginx/certs/ (Let's Encrypt or RunPod proxy)
docker compose up -d nginx
```

If you use RunPod's built-in HTTPS proxy for port 8000, you can skip nginx and
point the browser at the RunPod-provided `https://...proxy.runpod.net` URL.

## 10. Verify deployment

```bash
curl -s https://<your-host>/healthz        # {"status":"ok"}
curl -s https://<your-host>/readyz         # engine, gpu_count, sessions
```

Open `https://<your-host>/` in a browser and follow `SMOKE_TEST.md`.
