# Performance Optimization Guide

## Latency budget (5× H800 pod), user-Send → first avatar frame

| Stage | Target | Lever |
|---|---|---|
| GPT first token | ≤ 400 ms | `stream=True`, low `max_tokens`, short system prompt, warm conn |
| First clause ready | ≤ 600 ms | `SentenceAggregator` flushes at `.!?;:,` — don't wait for full reply |
| TTS first PCM | +150–350 ms | `gpt-4o-mini-tts`, `response_format="pcm"` (no mp3 decode) |
| Audio → wav2vec2 features | +20–40 ms | rolling window on GPU, no disk |
| **Renderer TTFF** | **+~1.2 s** | **irreducible** for WanS2V-14B; the wall we design around |
| Encode + network | +50–150 ms | VP8 low-latency, min jitter buffer, same-region TURN |
| **Perceived first frame** | **≈1.5–2.0 s** | then sustained 30–45 FPS |

> `< 1 s end-to-end` is impossible with this model: its own TTFF is 1.21 s on 5
> H800s. Everything else is already sub-second. To break 1 s you must change the
> renderer (3DMM / Gaussian avatar) — outside this brief.

## Renderer (the dominant cost)

1. **Keep it warm.** Load WanS2V-14B + LoRA + VAE once at startup
   (`PRELOAD_MODEL=1`); never per request. Cold start is minutes.
2. **TPP multi-GPU** (`GPU_COUNT=5`). Each GPU pinned to one of the 4 distilled
   denoise steps → asynchronous pipeline → ~45 FPS. Single-GPU = sub-realtime.
3. **`ENABLE_COMPILE=true`** — `torch.compile` + cuDNN/FlashAttention-3. ~2.5–3×
   throughput after a one-time warmup compile.
4. **`ENABLE_FP8=true`** — fits 48 GB and speeds matmuls on Hopper.
5. **Block size.** Smaller latent blocks → lower TTFF, marginally lower peak FPS.
   Tune in the repo config for your pod.
6. **Resolution.** 480×480 is the latency sweet spot; 720p roughly doubles cost.

## OpenAI layer

- **Pipeline, don't serialize.** TTS starts on clause 1 while GPT writes clause 2;
  avatar starts on the first PCM chunk. Implemented in `orchestrator.py`.
- **PCM everywhere.** No opus/mp3 hops between TTS and avatar features.
- **Bounded history.** `LLMClient` caps context to last ~20 turns to keep TTFT low.
- **Reuse the `AsyncOpenAI` client** (HTTP/2 keep-alive) per session.

## Transport

- **Backpressure, never block the GPU.** `AvatarVideoTrack.put()` drops the oldest
  frame when the queue is full; the renderer is never stalled by a slow client.
- **Silence-fill audio.** `AvatarAudioTrack` emits continuous 20 ms frames so the
  timeline never gaps and A/V stays aligned.
- **Co-locate.** TURN + pod + (ideally) user in one region. Trim the jitter buffer.
- **VP8** for lowest CPU-encode latency; H264 if clients require it.

## Scaling

- **One stream per pipeline.** The TPP pipeline is single-tenant. Concurrency =
  more pods behind a load balancer, not more threads. `MAX_SESSIONS` guards a pod.
- Use Redis pub/sub for cross-pod session routing; Postgres for transcripts/usage.
- Autoscale on `avatar_active_sessions` and GPU utilization.

## What to measure

`/metrics` exposes: `avatar_llm_first_token_seconds`,
`avatar_tts_first_chunk_seconds`, `avatar_first_frame_seconds`,
`avatar_frames_total`, `avatar_active_sessions`. Alert if first-frame p95 > 2.5 s
or sustained FPS < 24.
