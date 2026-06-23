# Production Checklist

## Functional
- [ ] `pytest backend/tests` green
- [ ] `/healthz`, `/readyz`, `/metrics` reachable
- [ ] Frontend served at `/`, loads over HTTPS
- [ ] Mock mode: full conversation works end-to-end
- [ ] LiveAvatar mode: repo cloned to `LIVEAVATAR_REPO`, checkpoints present, `import liveavatar` works
- [ ] `TARGET_FPS` set to `LA_NATIVE_FPS` (16) so audio/video stay in sync
- [ ] First `generate()` succeeds and returns frames (verify video tensor layout in `_video_to_frames`)
- [ ] First-frame p95 measured and within budget (see PERFORMANCE.md)
- [ ] Interrupt ("Stop") cancels mid-utterance

## Renderer / GPU
- [ ] Multi-GPU pod (5× H800 or 8× H100) for realtime; single-GPU only for offline
- [ ] `PRELOAD_MODEL=1` — weights loaded once at startup, kept warm
- [ ] `ENABLE_COMPILE=true`, `ENABLE_FP8=true` validated for the pod
- [ ] `GPU_COUNT` matches the pod; TPP pipeline confirmed running
- [ ] VRAM headroom verified (no OOM under sustained load)

## OpenAI
- [ ] OpenAI-only — no Anthropic/Gemini/DeepSeek/Ollama/vLLM in the dependency tree
- [ ] Keys are per-session from the browser; never logged, never persisted
- [ ] Model names validated against the user's account access (4o / 4.1 / 5)
- [ ] Rate-limit / 429 handling and user-visible error surfacing
- [ ] Per-session usage metered (tokens, TTS chars) if billing users

## Transport / WebRTC
- [ ] HTTPS/WSS enforced (HTTP→HTTPS redirect)
- [ ] STUN configured; **TURN** configured for NAT'd clients
- [ ] UDP media port range exposed on the pod
- [ ] `MAX_SESSIONS` set to safe per-pod concurrency
- [ ] nginx `proxy_read_timeout` long enough for long sessions

## Security
- [ ] CORS restricted to known origins (not `*`) in prod
- [ ] API keys redacted from logs (JSON logger never dumps message bodies)
- [ ] Upload size capped (`client_max_body_size`); image type validated
- [ ] Container runs non-root where possible; secrets via env, not baked images
- [ ] TURN credentials rotated; not committed

## Reliability / Ops
- [ ] Structured JSON logs shipped to a log store
- [ ] Prometheus scraping `/metrics`; dashboards + alerts (first-frame p95, FPS, errors)
- [ ] Graceful shutdown closes all sessions (`SessionManager.shutdown`)
- [ ] Health-check-driven restart policy (`restart: unless-stopped`)
- [ ] Redis/Postgres provisioned if multi-pod (session routing, transcripts)
- [ ] Load test at target concurrency; capacity plan = pods × MAX_SESSIONS
- [ ] Runbook for OOM, OpenAI outage, TURN failure

## Cost
- [ ] GPU cost modeled (5× H800 is the realtime floor)
- [ ] Idle pods scaled to zero / hibernated
- [ ] OpenAI spend alerting per key/tenant
