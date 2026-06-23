# Architecture

See `REPORT.md` for the full research analysis and rationale. This file is the
quick structural reference.

## Repository tree

```
realtime-liveavatar/
├── backend/
│   ├── main.py                     # FastAPI app + lifespan + static mount
│   ├── config/
│   │   └── settings.py             # pydantic settings (env-driven)
│   ├── api/
│   │   └── health.py               # /healthz /readyz /metrics
│   ├── websocket/
│   │   ├── signaling.py            # /ws control + SDP/ICE signaling
│   │   └── session_manager.py      # ConversationSession + registry
│   ├── openai/
│   │   ├── llm.py                  # GPT streaming (4o / 4.1 / 5)
│   │   ├── tts.py                  # streaming TTS -> PCM
│   │   ├── stt.py                  # speech-to-text (voice-in)
│   │   └── orchestrator.py         # GPT -> TTS -> audio+avatar pipeline
│   ├── avatar/
│   │   ├── base.py                 # AvatarEngine / AvatarSession interface
│   │   ├── mock_engine.py          # runnable no-GPU renderer
│   │   ├── liveavatar_engine.py    # real WanS2V-14B TPP wrapper
│   │   ├── audio_features.py       # pcm/resample/RMS/sentence-aggregator
│   │   └── __init__.py             # engine factory
│   ├── streaming/
│   │   ├── frame_track.py          # aiortc video track (frame queue)
│   │   └── audio_track.py          # aiortc audio track (PCM + silence-fill)
│   ├── models/schemas.py           # WS protocol models
│   ├── utils/{logging,metrics}.py  # JSON logs + Prometheus
│   └── tests/                      # pytest (core pipeline + openai stub)
├── frontend/
│   ├── index.html  styles.css
│   ├── app.js  websocket.js  webrtc.js  avatar.js
├── nginx/nginx.conf
├── Dockerfile  docker-compose.yml  startup.sh
├── requirements.txt  .env.example
├── REPORT.md  ARCHITECTURE.md  SETUP.md  SMOKE_TEST.md
├── PERFORMANCE.md  PRODUCTION_CHECKLIST.md  README.md
```

## Control-plane sequence (connect + one turn)

```
Browser                       FastAPI /ws                    Engine / OpenAI
   │  ws connect ───────────────►│
   │  hello{img,key,model} ─────►│  decode img, start_session()  ──► engine warm
   │                             │  build RTCPeerConnection(+tracks)
   │  ◄──────── ready{ice} ──────│
   │  offer(SDP recvonly) ──────►│  setRemote, createAnswer
   │  ◄──────── answer(SDP) ─────│
   │  ice ⇄ ice  (trickle) ─────►│  addIceCandidate
   │  ═══ WebRTC media (audio+video RTP) established ═══
   │  user_text "hi" ──────────►│  orchestrator.handle_user_text()
   │                             │   ├─ GPT stream ─► sentence flush
   │  ◄── assistant_text deltas ─│   ├─ TTS stream ─► PCM
   │                             │   ├─ PCM ─► audio_track (playback)
   │                             │   └─ PCM(16k) ─► avatar.push_audio()
   │  ◄═══ avatar video+audio frames over WebRTC ═══
```

## Data-plane: two clocks, one timeline

- **Video**: `engine.frames()` → `AvatarVideoTrack` queue → VP8/H264 @ 90 kHz clock.
- **Audio**: TTS PCM → `AvatarAudioTrack` 20 ms frames @ 24 kHz clock, silence-filled.
- Both tracks are paced to wall-clock realtime; the browser's WebRTC stack does
  final lip-sync alignment via RTP timestamps.

## Swapping the renderer

`AVATAR_ENGINE=mock` → `liveavatar`. The only code that differs is
`avatar/liveavatar_engine.py::_build_pipeline()` (bind to the cloned repo). All
transport, OpenAI, and session code is engine-agnostic.
