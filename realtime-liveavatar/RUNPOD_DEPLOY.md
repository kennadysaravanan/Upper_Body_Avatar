# RunPod Deployment — two-phase, cost-saving workflow

Goal: download the heavy stuff **once** to a persistent **Network Volume** on a
cheap pod, terminate it, then attach the same volume to an expensive multi-GPU
pod only when serving. You pay GPU rates only while a pod runs; the volume
persists at ~$0.07/GB/mo.

> **Important correction:** realtime LiveAvatar = **5 GPUs in ONE pod** (the TPP
> pipeline runs as `torchrun --nproc_per_node=5` and passes latents between GPUs
> inside one machine). It is **not** 5 separate pods. So "serving" below is a
> single multi-GPU pod (5×H800 or 8×H100).
>
> If you only want the simpler **single-GPU embedded** path (my FastAPI app calls
> `WanS2V.generate()` per sentence — runnable, not 45 FPS), serving is just a
> 1×H100/H800 pod and you skip the torchrun bits.

Everything below assumes the volume mounts at `/workspace`.

---

## PHASE 0 — Create the Network Volume (once)

1. RunPod → **Storage** → **Network Volumes** → **New**.
2. **Region/Datacenter:** pick one that has **H800/H100 availability** (the volume
   is region-locked; the serving pod must be in the same region).
3. **Size:** **250 GB** (Wan2.2-S2V-14B + T5 + VAE + audio encoder + LoRA ≈ 60–90 GB;
   leave room for the venv and temp files).
4. Create it. Note the region.

---

## PHASE 1 — Setup pod (cheap, download everything)

You do NOT need a GPU to download models. Use the cheapest pod **in the volume's
region**. (A cheap GPU like 1×A40/L40 is fine too — just don't install flash-attn
here.)

### 1.1 Create the pod
- RunPod → **Pods** → **Deploy**.
- GPU: cheapest available (or a CPU pod if offered) in the volume's region.
- **Network Volume:** attach the one from Phase 0 → mount path `/workspace`.
- Template: **RunPod PyTorch 2.x / CUDA 12.4** (or Ubuntu 22.04).
- Container disk: 20 GB is enough (everything heavy goes on the volume).
- Start it, open the **Web Terminal** (or SSH).

### 1.2 Clone repos onto the volume
```bash
cd /workspace
git clone https://github.com/Alibaba-Quark/LiveAvatar
git clone https://github.com/kennadysaravanan/Upper_Body_Avatar
```

### 1.3 Create the venv ON THE VOLUME (so it persists)
```bash
apt-get update && apt-get install -y python3.10 python3.10-venv ffmpeg git
python3.10 -m venv /workspace/venv
source /workspace/venv/bin/activate
python -m pip install --upgrade pip
```

### 1.4 Install dependencies (skip flash-attn here)
```bash
# torch (cu128 wheel works on any NVIDIA; arch-independent)
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# the platform app deps
pip install -r /workspace/Upper_Body_Avatar/realtime-liveavatar/requirements.txt

# the model repo deps (do NOT install flash-attn now — it's Hopper-specific,
# install it on the H800 pod in Phase 2)
pip install -r /workspace/LiveAvatar/requirements.txt
pip install "huggingface_hub[cli]"
```

### 1.5 Download the checkpoints to the volume
```bash
cd /workspace/LiveAvatar
mkdir -p ckpt
huggingface-cli download Wan-AI/Wan2.2-S2V-14B   --local-dir ckpt/Wan2.2-S2V-14B
huggingface-cli download Quark-Vision/Live-Avatar --local-dir ckpt/LiveAvatar
# (if a model is gated, run `huggingface-cli login` with your HF token first)
```

### 1.6 Verify the download, then STOP the pod
```bash
du -sh /workspace/LiveAvatar/ckpt/*          # confirm sizes look right
ls /workspace/venv/bin/python                # venv persisted on the volume
```
- In the RunPod console: **Stop** (or **Terminate**) the setup pod.
- The Network Volume keeps everything. ✅ You now pay only volume storage.

---

## PHASE 2 — Serving pod (multi-GPU, only when you need it)

### 2.1 Create the pod
- RunPod → **Pods** → **Deploy**, **same region** as the volume.
- GPU: **5× H800 80GB** (realtime TPP) — or **1× H100/H800** for the single-GPU
  embedded path.
- **Network Volume:** attach the same one → `/workspace`.
- **Expose ports:** HTTP **8000**. For WebRTC media also expose a **UDP range**
  (e.g. 40000–40100) or plan to use TURN (see SETUP.md §8).
- Start, open terminal.

### 2.2 Reactivate the environment (no re-download!)
```bash
source /workspace/venv/bin/activate
cd /workspace/Upper_Body_Avatar/realtime-liveavatar
git pull            # grab any code updates
```

### 2.3 Install flash-attn FOR HOPPER (fast, prebuilt wheel)
```bash
pip install flash_attn_3 \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280 \
  --extra-index-url https://download.pytorch.org/whl/cu128
# (1×non-Hopper GPU instead? use: pip install flash-attn==2.8.3 --no-build-isolation)
```

### 2.4 Configure `.env`
```bash
cp .env.example .env
```
Edit `.env`:
```bash
AVATAR_ENGINE=liveavatar
LIVEAVATAR_REPO=/workspace/LiveAvatar
LIVEAVATAR_CKPT=/workspace/LiveAvatar/ckpt/Wan2.2-S2V-14B
LA_LORA_PATH=ckpt/LiveAvatar
TARGET_FPS=16              # match LiveAvatar native fps for A/V sync
PRELOAD_MODEL=1
# single-GPU embedded path:
GPU_COUNT=1
ENABLE_FP8=true
```

### 2.5 Launch

**Option A — single-GPU embedded (my FastAPI app, simplest, runs end-to-end):**
```bash
cd /workspace/Upper_Body_Avatar/realtime-liveavatar
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Open the RunPod-provided `https://<pod-id>-8000.proxy.runpod.net/`, upload a
portrait, paste your OpenAI key, Connect, chat.

**Option B — 5-GPU realtime TPP (native repo demo, true 45 FPS):**
The 5-GPU pipeline runs as a torchrun multi-process job and is driven by the
repo's own launcher, not embedded in a single uvicorn process:
```bash
cd /workspace/LiveAvatar
# realtime gradio demo over 5 GPUs (edit the .sh for image/size if needed):
export ENABLE_COMPILE=true
bash gradio_multi_gpu.sh
```
To put the OpenAI brain+voice in front of the 5-GPU renderer, run that TPP job as
a **separate worker** and bridge frames to my FastAPI app over a queue — see
`backend/avatar/liveavatar_engine.py` (module docstring marks the seam). Trying to
run uvicorn itself under `torchrun --nproc_per_node=5` requires the rank-0-serves
model and is the advanced path.

### 2.6 Smoke test
Follow `SMOKE_TEST.md`. Quick checks:
```bash
curl -s http://localhost:8000/readyz | python -m json.tool   # engine=liveavatar
```

### 2.7 STOP the pod when done
RunPod console → **Stop**. Volume persists; spin up again later by repeating 2.1–2.5
(flash-attn from 2.3 persists in the venv, so that step is then a no-op/instant).

---

## Cost summary

| Item | When billed |
|---|---|
| Network Volume (250 GB) | always (~$17/mo) — cheap |
| Setup pod (cheap GPU/CPU) | only during Phase 1 download (an hour or two) |
| Serving pod (5×H800) | only while running in Phase 2 — **stop it when idle** |

The expensive 5×H800 time is now minutes-to-hours of actual serving, not hours of
downloading. That's the saving.

## Gotchas
- Volume **region must match** the serving GPU's region.
- A network volume attaches to **one pod at a time**.
- Keep the **venv on `/workspace`** or installs won't persist.
- Don't install **flash-attn** on a non-Hopper setup pod.
- WebRTC needs **HTTPS** + **STUN/TURN**; the RunPod `…proxy.runpod.net` URL is
  HTTPS but media still needs UDP/TURN (SETUP.md §8).
- `git pull` on the serving pod needs your token again — or make the repo public.
