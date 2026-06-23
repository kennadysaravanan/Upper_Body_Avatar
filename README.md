# Upper_Body_Avatar

Realtime, browser-based, OpenAI-powered talking-avatar platform built on
**Alibaba-Quark/LiveAvatar (WanS2V-14B)**. FastAPI + WebRTC backend, single-page
frontend, RunPod GPU deployment.

The full project lives in [`realtime-liveavatar/`](realtime-liveavatar/). Start
with [`realtime-liveavatar/README.md`](realtime-liveavatar/README.md) and
[`realtime-liveavatar/REPORT.md`](realtime-liveavatar/REPORT.md).

## Quickstart (mock mode, no GPU)

```bash
cd realtime-liveavatar
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000, upload a portrait, paste your OpenAI key, Connect, chat.
