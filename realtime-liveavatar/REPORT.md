# Realtime LiveAvatar + OpenAI — Technical Research Report

> Target: turn **Alibaba‑Quark/LiveAvatar** (offline / batch audio‑driven avatar
> generator) into a **realtime, browser‑based, OpenAI‑powered talking‑avatar
> platform** on RunPod GPUs.

This document is the engineering analysis. It is deliberately blunt about what is
and is not physically achievable with this specific model, because the rest of the
architecture only makes sense once that is clear.

---

## 0. Executive summary (read this first)

LiveAvatar is **not** a lightweight 3DMM / blendshape talking head (SadTalker,
Wav2Lip, GeneFace). It is **WanS2V‑14B**, a 14‑billion‑parameter Diffusion
Transformer (DiT) that generates video directly in a causal‑3D‑VAE latent space.
That single fact drives every decision below.

Published numbers from the authors (ECCV 2026 paper, arXiv 2512.04677):

| Metric | Value | Hardware |
|---|---|---|
| Sustained framerate | **45.2 FPS** | 5× H800 (80 GB) |
| Time‑to‑first‑frame (TTFF) | **1.21 s** | 5× H800 |
| Denoising steps | 4 (distilled from 80) | — |
| Latent block size | 3 latent frames / block | — |
| Max stream length | 10,000+ s | — |
| Single‑GPU 80 GB | offline only (not realtime) | 1× H800/H100/A100‑80 |
| FP8 quantized | fits 48 GB | 1× (still not realtime) |

**Consequences for the requested targets:**

| Requested target | Verdict | Reality |
|---|---|---|
| First LLM token < 500 ms | ✅ achievable | OpenAI streaming first token ~200–400 ms |
| Speech start < 300 ms | ✅ achievable | OpenAI TTS streaming first audio ~150–350 ms |
| Avatar reaction < 500 ms | ⚠️ partial | model TTFF is ~1.2 s on a 5‑GPU node |
| **End‑to‑end < 1 s** | ❌ **not with this model** | renderer alone ≈ 1.2 s TTFF |
| 30 FPS | ✅ on multi‑GPU | 45 FPS on 5×H800; <5 FPS single‑GPU |
| Single RunPod GPU realtime | ❌ | needs a **multi‑GPU pod** (5×H800 / 8×H100) |

The honest end‑to‑end budget we design to is **≈ 1.5–2.0 s perceived
first‑response latency** on a 5×H800 pod, then sustained 30–45 FPS streaming.
Everything in the codebase is built so that the OpenAI half is sub‑second and the
*only* irreducible cost is the diffusion renderer — and so that you can run the
**entire stack end‑to‑end today in a mock renderer** without owning 5 H800s.

---

## 1. Existing architecture (repository analysis)

Source: <https://github.com/Alibaba-Quark/LiveAvatar> · paper
<https://huggingface.co/papers/2512.04677> · base model `Wan-AI/Wan2.2-S2V-14B`
· LoRA `Quark-Vision/Live-Avatar`.

### 1.1 Repository layout (observed)

```
LiveAvatar/
├── ckpt/
│   ├── Wan2.2-S2V-14B/          # base diffusion weights (download separately)
│   └── LiveAvatar/              # LiveAvatar LoRA + distilled scheduler weights
├── liveavatar/                  # core package
│   ├── configs/                 # model / pipeline configs
│   ├── models/                  # DiT, VAE, audio conditioning modules
│   ├── utils/
│   ├── scheduler.py             # 4-step distilled diffusion scheduler
│   └── util.py
├── minimal_inference/           # lightweight single-file inference path
├── examples/
├── infinite_inference_multi_gpu.sh   # 5-GPU realtime (TPP) entrypoint
├── infinite_inference_single_gpu.sh  # single-GPU offline entrypoint
├── gradio_multi_gpu.sh               # realtime Gradio demo
├── gradio_single_gpu.sh              # offline Gradio demo
└── requirements.txt
```

### 1.2 Model architecture

- **Backbone:** WanS2V DiT (Diffusion Transformer), 14B params, operating on a
  **compressed latent space** produced by a **causal 3D VAE** (spatial + temporal
  compression). Video is never generated pixel‑by‑pixel; the DiT denoises latents
  and the VAE decoder lifts them back to RGB.
- **Conditioning inputs:**
  1. **Reference image** → appearance / identity (the uploaded avatar photo).
  2. **Audio embedding** `aⁱ` → drives lip + facial motion. The encoder is the
     Wan2.2‑S2V audio stack (wav2vec2‑class features at 16 kHz; **confirm the
     exact symbol in `liveavatar/models/`** — this is the integration point).
  3. **Text prompt** → coarse character description via a pretrained text encoder.
- **Distillation:** original 80‑step diffusion is distilled to **4 steps** via
  Distribution‑Matching Distillation (DMD) + Self‑Forcing, which is what makes any
  realtime path possible.

### 1.3 Audio pipeline

```
raw speech (wav/pcm) ─► resample 16 kHz mono ─► audio encoder (wav2vec2-class)
                     ─► per-frame audio embeddings aⁱ ─► cross-attention into DiT
```

The audio embedding is aligned to the latent **temporal** axis: each latent frame
(which the VAE later expands to N RGB frames) consumes a window of audio features.

### 1.4 Video / frame generation pipeline (offline, as shipped)

```
ref image ──► VAE encode ──► z_ref
text ───────► text encoder ─► c_text
audio ──────► audio encoder ► a[0..T]
                              │
        for each block b (3 latent frames):
            z_b = randn
            for step in 4 distilled steps:
                z_b = DiT(z_b, t_step, z_ref, c_text, a_block, kv_cache)
            kv_cache ◄─ append(z_b)        # block-wise autoregression
            frames_b = VAE.decode(z_b)     # streaming-VAE in v1.1
        ──► all blocks ──► ffmpeg mux with audio ──► output.mp4
```

### 1.5 Inference workflow (as shipped)

- **Single‑GPU** (`infinite_inference_single_gpu.sh`): sequential — runs all 4
  denoise steps for block *b*, decodes, moves to *b+1*. Correct but slow
  (well under realtime). Used for offline render. `--num_clip` previews,
  `ENABLE_FP8=true` fits 48 GB, `--enable_online_decode` trims VRAM.
- **Multi‑GPU** (`infinite_inference_multi_gpu.sh`): **Timestep‑forcing Pipeline
  Parallelism (TPP)** — each of 5 GPUs is pinned to *one* denoising timestep.
  Block *b* flows GPU0→GPU1→…→GPU4 like a CPU instruction pipeline; while GPU4
  finishes block *b*’s last step, GPU0 is already on block *b+1*’s first step.
  Steady‑state throughput = 1 block per slowest‑stage time → 45 FPS. Only latent
  tensors cross GPUs; each GPU keeps a **local KV cache**.

### 1.6 Long‑horizon stability (why it doesn’t drift over 10,000 s)

- **History Corrupt** — KV cache stores *noisy* latents, not clean ones.
- **Adaptive Attention Sink (AAS)** — the reference "sink" frame is swapped for
  the model’s own first generated latent after warmup.
- **Rolling RoPE** — positional embeddings slide so relative distances stay
  constant as the stream grows.

### 1.7 Existing limitations (for our use case)

1. **Batch‑oriented I/O.** Entry points take an audio *file* + image *file* and
   write an *mp4*. There is no socket, no frame callback, no partial flush.
2. **Audio must exist up front.** Offline path assumes the whole utterance is
   known; our TTS produces audio *incrementally*.
3. **Gradio‑only frontend.** No WebRTC, no signaling, no multi‑session server.
4. **Realtime ⇒ 5 GPUs.** TPP needs the multi‑GPU node; otherwise sub‑realtime.
5. **Cold start is heavy.** 14B weights + LoRA + VAE + compile + FP8 = minutes to
   load; must be a long‑lived warm worker, never per‑request.
6. **One stream per pipeline.** The TPP pipeline is single‑tenant; concurrency =
   more pods, not more threads.

---

## 2. Realtime conversion plan

### 2.1 Retain (use as‑is)

- The **model weights** (WanS2V‑14B + LiveAvatar LoRA + causal 3D VAE).
- The **4‑step distilled scheduler** (`liveavatar/scheduler.py`).
- The **TPP multi‑GPU engine** — this is the crown jewel; we wrap it, not rewrite.
- The **audio encoder** and conditioning math.
- Stability tricks (History Corrupt, AAS, Rolling RoPE) — unchanged.

### 2.2 Rewrite (adapt the boundaries)

- **Driver loop → streaming session object.** Replace "read file, loop, write
  mp4" with a long‑lived `AvatarSession` exposing `push_audio(pcm)` and an async
  `frames()` generator. Same internal autoregressive loop; new I/O surface.
- **Audio ingestion → incremental.** Feed wav2vec2 features in rolling windows as
  TTS PCM arrives, instead of encoding one complete file.
- **Output → frame queue, not ffmpeg mux.** `VAE.decode(block)` pushes RGB frames
  onto an `asyncio.Queue` consumed by a WebRTC `VideoStreamTrack`.

### 2.3 Replace (drop entirely)

- **Gradio UI** → custom HTML/CSS/JS + WebRTC.
- **mp4 file output / ffmpeg muxing** → live WebRTC media tracks.
- **CLI argument plumbing** → FastAPI + WebSocket session API.
- **Any built‑in TTS/ASR** → **OpenAI only** (GPT, TTS, STT, Realtime).

### 2.4 Migration path (offline → realtime), concretely

```
STEP 0  Wrap the shipped pipeline behind AvatarEngine (base.py).         [adapter]
STEP 1  Stand up the whole stack with MockAvatarEngine (runs anywhere).  [done]
STEP 2  Implement LiveAvatarEngine.start_session(): load weights once,
        build the TPP pipeline, keep it warm.                            [GPU pod]
STEP 3  Convert the block loop into an async generator that yields frames
        instead of accumulating to disk.                                 [GPU pod]
STEP 4  Wire incremental wav2vec2 feature extraction to push_audio().    [GPU pod]
STEP 5  Pipe OpenAI TTS PCM → push_audio(); pipe frames() → WebRTC.      [glue]
STEP 6  Tune block size / FP8 / compile for the chosen pod.              [perf]
```

The codebase ships **STEP 0–1 and 5 fully working**, with STEP 2–4 implemented as
a clearly‑marked `LiveAvatarEngine` wrapper whose integration points map 1:1 to
the repo’s documented modules.

---

## 3. Realtime architecture

### 3.1 Component diagram

```
┌──────────────────────────────────────── BROWSER (1 HTML page) ───────────────────────────────────────┐
│  index.html ─ styles.css                                                                              │
│  app.js  ──► upload image, API key, model name, "Connect", text box                                  │
│  websocket.js ──► control channel (JSON)        webrtc.js ──► media (audio+video)                     │
│        │                                                │                                              │
└────────┼────────────────────────────────────────────────┼────────────────────────────────────────────┘
         │ wss:// (signaling + control)                     │ SRTP/DTLS media (RTP)
┌────────┼────────────────────────────────────────────────┼────────────────────────────────────────────┐
│        ▼                       FASTAPI BACKEND (RunPod GPU pod)        ▼                                │
│  websocket/signaling.py  ◄──► session_manager.py  ◄──► streaming/frame_track.py + audio_track.py       │
│        │                            │                          ▲                                       │
│        ▼                            ▼                          │ frames + pcm                          │
│  openai/orchestrator.py ──► llm.py ──► tts.py ──► [PCM chunks] ─┼─► avatar/audio_features.py            │
│        ▲          (gpt-4o / 4.1 / 5)   (gpt-4o-mini-tts)        │           │                          │
│        │ stt.py (gpt-4o-transcribe, optional voice-in)          │           ▼                          │
│        │                                                        │   avatar/liveavatar_engine.py        │
│        │                                                        └── (TPP pipeline over 5 GPUs) ─frames─┘
│  api/health.py · utils/metrics.py (Prometheus) · config/settings.py                                    │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
         │                                   │
   Redis (session state / pub-sub)     PostgreSQL (sessions, transcripts, usage)
```

### 3.2 Layer responsibilities

| Layer | Module | Responsibility |
|---|---|---|
| Frontend | `frontend/*` | capture input, render WebRTC `<video>`, control UI |
| Signaling | `websocket/signaling.py` | WebSocket: SDP/ICE exchange + control msgs |
| Sessions | `websocket/session_manager.py` | one `ConversationSession` per peer |
| OpenAI | `openai/{llm,tts,stt,orchestrator}.py` | brain + voice, streaming only |
| Avatar | `avatar/{base,mock,liveavatar}_engine.py` | renderer behind one interface |
| Streaming | `streaming/{frame_track,audio_track,media_clock}.py` | aiortc tracks + A/V sync |
| Monitoring | `utils/{logging,metrics}.py`, `api/health.py` | logs, Prometheus, health |

---

## 4. Realtime avatar pipeline (stage by stage)

```
User text ─► OpenAI GPT ─► token stream ─► sentence aggregator
          ─► OpenAI TTS (stream, pcm 24k) ─► PCM chunks (20–40 ms)
          ─► resample 16k + wav2vec2 features ─► push_audio()
          ─► LiveAvatar TPP: 4-step denoise per block ─► latent blocks
          ─► streaming-VAE decode ─► RGB frames (480p)
          ─► frame queue ─► aiortc VideoStreamTrack (VP8/H264)
          ─► WebRTC ─► browser <video>   (audio on a parallel WebRTC track)
```

1. **User text** arrives over the WebSocket control channel (or STT transcript).
2. **GPT** (`chat.completions`/Responses, `stream=True`) emits tokens. We
   aggregate to sentence/clause boundaries to start TTS as early as possible.
3. **TTS** (`gpt-4o-mini-tts`, `response_format="pcm"`, streamed) yields 24 kHz
   16‑bit mono PCM as the model speaks — no waiting for the full reply.
4. **Audio features**: PCM is resampled to 16 kHz and run through the wav2vec2‑class
   encoder in rolling windows aligned to latent blocks.
5. **Motion / frame generation**: features `push_audio()` into the warm
   `AvatarSession`; the TPP loop denoises 3‑latent‑frame blocks in 4 steps.
6. **Decode**: streaming‑VAE turns each latent block into RGB frames.
7. **Transport**: frames → `asyncio.Queue` → `VideoStreamTrack` (encoded VP8/H264).
   The same TTS PCM → `AudioStreamTrack`. A shared **media clock** timestamps both
   so lips match voice.
8. **Browser** plays the WebRTC stream in a `<video>` element.

---

## 5. Low‑latency design

Budget (5×H800 pod), measured from user pressing Send:

| Stage | Target | Technique |
|---|---|---|
| GPT first token | ≤ 400 ms | streaming, low `max_tokens` warmup, no system reasoning bloat |
| First sentence ready | ≤ 600 ms | clause‑boundary flush, don’t wait for full reply |
| TTS first PCM | +150–350 ms | `gpt-4o-mini-tts` streaming, pcm format (no mp3 decode) |
| Audio→features | +20–40 ms | rolling wav2vec2, GPU, no disk |
| Renderer TTFF | +1.2 s | **irreducible** — TPP warm pipeline, 4 steps, FP8, compile |
| Frame→browser | +50–150 ms | VP8 low‑latency, jitter buffer min, same‑region pod |
| **Perceived first frame** | **≈1.5–2.0 s** | then sustained 30–45 FPS |

Key techniques (implemented or documented):

- **Keep the GPU pipeline warm** — model load + compile once at startup; sessions
  reuse it. Cold start is minutes; never per request.
- **Sentence‑level pipelining** — TTS starts on the first clause, not the full
  GPT reply; avatar starts on the first PCM chunk.
- **PCM end‑to‑end** — avoid mp3/opus decode hops between TTS and features.
- **Block size tuning** — smaller latent blocks → lower TTFF, slightly lower peak
  FPS. Tune for the pod.
- **FP8 + `torch.compile` + cuDNN/FlashAttn‑3** — the repo’s `ENABLE_FP8` /
  `ENABLE_COMPILE` flags; ~2.5–3× throughput.
- **Backpressure** — bounded frame queue; if the network stalls, drop oldest
  frames, never block the GPU loop.
- **Co‑locate** browser‑facing TURN and pod in one region; trim jitter buffer.

**Where < 1 s is impossible:** the 1.2 s renderer TTFF. To go lower you must
change models (e.g. a 3DMM/Gaussian‑avatar renderer) — out of scope, since the
brief is specifically LiveAvatar. We optimize everything *around* it and make the
1.2 s the only wall.

---

## 6. Deliverables map

| Deliverable | Where |
|---|---|
| Research report / repo analysis | this file |
| Realtime architecture + diagrams | this file §3–4, `ARCHITECTURE.md` |
| Folder structure | `ARCHITECTURE.md`, repo tree |
| Full source (backend + frontend) | `backend/`, `frontend/` |
| Docker deployment | `Dockerfile`, `docker-compose.yml`, `startup.sh` |
| RunPod guide | `SETUP.md` |
| Smoke testing | `SMOKE_TEST.md` |
| Performance guide | `PERFORMANCE.md` |
| Production checklist | `PRODUCTION_CHECKLIST.md` |

---

## Sources

- [Alibaba-Quark/LiveAvatar (GitHub)](https://github.com/Alibaba-Quark/LiveAvatar)
- [README.md](https://github.com/Alibaba-Quark/LiveAvatar/blob/main/README.md)
- [Paper: Live Avatar — Streaming Real‑time Audio‑Driven Avatar Generation with Infinite Length (arXiv 2512.04677)](https://huggingface.co/papers/2512.04677)
- [Wan-AI/Wan2.2-S2V-14B (base model)](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)
- [Quark-Vision/Live-Avatar (LoRA)](https://huggingface.co/Quark-Vision/Live-Avatar)
