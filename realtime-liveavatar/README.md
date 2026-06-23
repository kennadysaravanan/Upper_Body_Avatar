# Realtime LiveAvatar + OpenAI

A production-grade, browser-based **realtime conversational avatar** platform:
OpenAI provides the brain (GPT) and voice (TTS/STT); **Alibaba‑Quark LiveAvatar**
(WanS2V‑14B) is the streaming video renderer; FastAPI + WebRTC deliver it to the
browser. Deployable on RunPod GPUs.

> **Read `REPORT.md` first.** It is the technical analysis and is blunt about one
> thing: LiveAvatar is a **14B diffusion model**, not a lightweight avatar. Its
> own published numbers are 45 FPS / 1.21 s time-to-first-frame on **5× H800**.
> So realtime needs a multi-GPU pod, and true `<1 s` end-to-end is not possible
> with this model — the renderer alone is ~1.2 s. We optimize everything around
> that and design to ~1.5–2.0 s perceived first response, then sustained 30–45 FPS.

## What makes this runnable today

The whole stack works **right now with no GPU** via a built-in `mock` renderer
that animates your uploaded photo from the TTS audio (mouth, blink, head bob).
Swapping to the real 14B engine is one config change — `AVATAR_ENGINE=liveavatar`
— and binding one function (`avatar/liveavatar_engine.py::_build_pipeline`) to the
cloned repo. Everything else (OpenAI, WebRTC, sessions, transport) is identical.

## Quickstart (mock mode, laptop)

```bash
cd realtime-liveavatar
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                      # AVATAR_ENGINE=mock by default
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000, upload a portrait, paste your OpenAI key, pick a model,
click **Connect**, and chat. (WebRTC needs a secure context: localhost is treated
as secure; for remote hosts use HTTPS — see `SETUP.md`.)

## OpenAI-only

LLM: GPT‑4o / GPT‑4.1 / GPT‑5 · TTS: `gpt-4o-mini-tts` (streaming PCM) · STT:
`gpt-4o-transcribe` · optional Realtime API. No Anthropic / Gemini / DeepSeek /
Ollama / vLLM — by design.

## User flow

upload image → enter key + model → **Connect** → session + warm avatar →
type text → GPT (stream) → TTS (stream PCM) → wav2vec2 features → LiveAvatar
motion + frame generation → WebRTC → browser. Audio and video are separate WebRTC
tracks on a shared clock for lip-sync.

## Docs

| File | Contents |
|---|---|
| `REPORT.md` | Research report, repo analysis, conversion plan, architecture, diagrams |
| `ARCHITECTURE.md` | Tree, sequence diagrams, data plane |
| `SETUP.md` | RunPod deployment (10 steps) |
| `SMOKE_TEST.md` | Exact verification commands |
| `PERFORMANCE.md` | Latency budget + optimization |
| `PRODUCTION_CHECKLIST.md` | Go-live checklist |

## Tests

```bash
PYTHONPATH=. pytest backend/tests -q
```

## License

Platform code: MIT. LiveAvatar / WanS2V: Apache‑2.0 (see upstream).
