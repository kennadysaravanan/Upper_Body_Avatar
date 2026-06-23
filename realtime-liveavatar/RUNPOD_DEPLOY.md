# RunPod Deployment — REALTIME talking avatar (H100 SXM)

Your exact workflow:

1. **Phase 1** — H100 SXM, **GPU count 1**, attach a Network Volume, download
   EVERYTHING (repos, venv, model weights) onto the volume. Then **terminate**.
2. **Phase 2** — H100 SXM, **GPU count 5**, attach the SAME volume, launch the
   **realtime** avatar (5-GPU TPP). Stop when idle. Repeat Phase 2 whenever you
   need it — no re-download.

> **Realtime, not offline.** Realtime LiveAvatar = **5 GPUs in ONE pod** running
> the TPP pipeline (`torchrun --nproc_per_node=5`). The 5 GPUs make per-sentence
> `generate()` run at ~45 FPS throughput, so each spoken reply is produced
> near-instantly and streamed. It is **not** 5 separate pods, and it is **not**
> the single-GPU offline path.

Volume mounts at `/workspace` everywhere below.

═══════════════════════════════════════════════════════════════════════════════
## PHASE 0 — Create the Network Volume (once)
═══════════════════════════════════════════════════════════════════════════════

1. RunPod → **Storage → Network Volumes → New**.
2. **Region:** pick a datacenter that has **H100 SXM** availability (volume is
   region-locked; both pods must be in this region).
3. **Size: 300 GB.**
4. Create. ✅

═══════════════════════════════════════════════════════════════════════════════
## PHASE 1 — Setup pod: H100 SXM × 1 (download everything, then terminate)
═══════════════════════════════════════════════════════════════════════════════

Use **H100 SXM ×1** for setup (same Hopper arch as the 5-GPU pod) so the compiled
`flash-attn` wheel matches and is reused in Phase 2.

### 1.1 Deploy the pod
- RunPod → **Pods → Deploy**.
- GPU: **H100 SXM**, **Count = 1**.
- **Network Volume:** attach the Phase-0 volume → mount `/workspace`.
- Template: **RunPod PyTorch 2.4+ / CUDA 12.4** (Ubuntu 22.04).
- Container disk: 30 GB. Start → open **Web Terminal**.

### 1.2 Clone repos onto the volume
```bash
cd /workspace
git clone https://github.com/Alibaba-Quark/LiveAvatar
git clone https://github.com/kennadysaravanan/Upper_Body_Avatar
```

### 1.3 Create the venv ON THE VOLUME (so it persists)
```bash
# python3.10-dev + build-essential are REQUIRED: some LiveAvatar deps (pyworld,
# etc.) compile C/C++ and need Python.h + g++. Without them you get
# "fatal error: Python.h: No such file or directory".
apt-get update && apt-get install -y \
  python3.10 python3.10-venv python3.10-dev build-essential ffmpeg git
python3.10 -m venv /workspace/venv
source /workspace/venv/bin/activate
python -m pip install --upgrade pip
```

### 1.4 Install ALL dependencies (incl. flash-attn — H100 is Hopper)
```bash
# torch
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# flash-attention 3 for Hopper (prebuilt wheel)
pip install flash_attn_3 \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280 \
  --extra-index-url https://download.pytorch.org/whl/cu128

# platform app deps + model repo deps
pip install -r /workspace/Upper_Body_Avatar/realtime-liveavatar/requirements.txt
pip install -r /workspace/LiveAvatar/requirements.txt
pip install "huggingface_hub[cli]"
```

### 1.5 Download the model weights to the volume
```bash
cd /workspace/LiveAvatar && mkdir -p ckpt
huggingface-cli download Wan-AI/Wan2.2-S2V-14B   --local-dir ckpt/Wan2.2-S2V-14B
huggingface-cli download Quark-Vision/Live-Avatar --local-dir ckpt/LiveAvatar
# if gated: huggingface-cli login   (paste your HF token) then re-run
```

### 1.6 (Recommended) verify the model loads on 1 GPU before paying for 5
```bash
cd /workspace/Upper_Body_Avatar/realtime-liveavatar
python -c "import sys; sys.path.insert(0,'/workspace/LiveAvatar'); \
import liveavatar; print('liveavatar import OK')"
du -sh /workspace/LiveAvatar/ckpt/*
ls /workspace/venv/bin/python   # venv persisted
```

### 1.7 TERMINATE the setup pod
RunPod console → **Terminate**. The Network Volume keeps repos + venv + weights.
You now pay only ~$0.07/GB/mo for storage. ✅

═══════════════════════════════════════════════════════════════════════════════
## PHASE 2 — Serving pod: H100 SXM × 5 (REALTIME), stop when idle
═══════════════════════════════════════════════════════════════════════════════

### 2.1 Deploy the pod
- RunPod → **Pods → Deploy**, **same region** as the volume.
- GPU: **H100 SXM**, **Count = 5**.
- **Network Volume:** attach the same one → `/workspace`.
- Container disk: 30 GB.
- **Expose HTTP port 8000.** (RunPod gives you `https://<id>-8000.proxy.runpod.net`.)
- Start → open terminal.

### 2.2 Reactivate the environment (NO re-download)
```bash
source /workspace/venv/bin/activate
cd /workspace/Upper_Body_Avatar/realtime-liveavatar
git pull        # pick up latest code (needs your token or a public repo)
```

### 2.3 Configure `.env` for realtime
```bash
cp .env.example .env
```
Edit `.env`:
```bash
AVATAR_ENGINE=liveavatar
GPU_COUNT=5
LIVEAVATAR_REPO=/workspace/LiveAvatar
LIVEAVATAR_CKPT=/workspace/LiveAvatar/ckpt/Wan2.2-S2V-14B
LA_LORA_PATH=ckpt/LiveAvatar
LA_SIZE=720*400          # the multi-GPU realtime size from gradio_multi_gpu.sh
LA_NUM_GPUS_DIT=4        # 4 DiT + 1 VAE across the 5 GPUs
LA_SAMPLE_STEPS=4
TARGET_FPS=16            # match LiveAvatar native fps so audio/video stay in sync
ENABLE_FP8=true
ENABLE_COMPILE=true      # ~2.5-3x throughput after a one-time warmup compile
```

### 2.4 Launch the realtime server (5-GPU TPP under torchrun)
```bash
cd /workspace/Upper_Body_Avatar/realtime-liveavatar
export AVATAR_ENGINE=liveavatar GPU_COUNT=5
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=86400
export PYTHONPATH=.

torchrun --nproc_per_node=5 --master_port=29502 -m backend.main_tpp
```
- All 5 ranks load WanS2V and join the TPP collective.
- Rank 0 serves the OpenAI + WebRTC app on port 8000.
- First model load + compile takes a few minutes (warm after that).

### 2.5 Use it
Open `https://<pod-id>-8000.proxy.runpod.net/` → upload a portrait → paste your
OpenAI key → pick `gpt-4o-mini` → **Connect** → type → the avatar talks in
realtime. (WebRTC media also needs STUN/TURN for NAT — see SETUP.md §8.)

### 2.6 Smoke check (second terminal on the pod)
```bash
source /workspace/venv/bin/activate
curl -s http://localhost:8000/readyz | python -m json.tool   # engine=liveavatar, gpu_count=5
```

### 2.7 STOP the pod when idle
RunPod console → **Stop**. Volume persists. Next time: repeat 2.1–2.5 (no
download, no reinstall, flash-attn already in the venv).

═══════════════════════════════════════════════════════════════════════════════
## Fallback: the repo's native realtime demo (guaranteed to run)
═══════════════════════════════════════════════════════════════════════════════
If you want the authors' own realtime multi-GPU UI (Gradio, no OpenAI), on the
5-GPU pod:
```bash
cd /workspace/LiveAvatar
export CUDA_VISIBLE_DEVICES=0,1,2,3,4 ENABLE_COMPILE=true
bash gradio_multi_gpu.sh      # serves on :7860 (expose that port)
```
This proves the 5-GPU TPP renderer works on your pod independent of the OpenAI app.

═══════════════════════════════════════════════════════════════════════════════
## Cost
═══════════════════════════════════════════════════════════════════════════════
| Item | Billed when |
|---|---|
| Network Volume (300 GB) | always (~$21/mo) |
| Setup pod H100 SXM ×1 | only during Phase 1 (download, ~1–2 h) |
| Serving pod H100 SXM ×5 | only while running Phase 2 — **stop when idle** |

═══════════════════════════════════════════════════════════════════════════════
## Gotchas
═══════════════════════════════════════════════════════════════════════════════
- Volume **region must match** the H100 SXM region; a volume attaches to one pod
  at a time.
- Keep the **venv on `/workspace`** or installs won't persist.
- `git pull` on the serving pod needs your token (or make the repo public).
- The 5-GPU NCCL dispatch in `backend/main_tpp.py` should be validated on the pod;
  if it errors, use the native `gradio_multi_gpu.sh` path above while debugging.
- WebRTC needs HTTPS (the RunPod proxy URL is HTTPS) + STUN/TURN for media.
- Realtime ≠ instant: expect ~1.2 s to first frame (model TTFF), then smooth 30+
  FPS — true sub-second end-to-end is not possible with a 14B diffusion model.
