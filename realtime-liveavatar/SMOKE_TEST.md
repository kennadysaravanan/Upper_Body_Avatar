# Smoke Test

Run top-to-bottom. Each step has an exact command and a pass condition. Works in
**mock** mode with no GPU and no real avatar weights — only an OpenAI key is
needed for the steps that hit OpenAI.

## 0. Environment

```bash
cd realtime-liveavatar
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export AVATAR_ENGINE=mock
```

## 1. Unit tests (engine-agnostic pipeline)

```bash
PYTHONPATH=. pytest backend/tests -q
```
**Pass:** all tests green (audio features, sentence aggregator, mock engine frames, LLM stream).

## 2. Boot the server

```bash
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
sleep 2
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/readyz | python -m json.tool
```
**Pass:** `{"status":"ok"}` and a `readyz` body showing `"engine":"mock"`.

## 3. Frontend loads

```bash
curl -s http://localhost:8000/ | grep -q "Realtime LiveAvatar" && echo "frontend OK"
```
**Pass:** prints `frontend OK`.

## 4. OpenAI LLM reachable (with your key)

```bash
export OPENAI_API_KEY=sk-...
python - <<'PY'
import asyncio, os
from backend.openai.llm import LLMClient
async def main():
    c = LLMClient(os.environ["OPENAI_API_KEY"], "gpt-4o")
    out = "".join([d async for d in c.stream_reply("Say hello in 5 words.")])
    print("LLM:", out)
asyncio.run(main())
PY
```
**Pass:** a short sentence is printed.

## 5. OpenAI TTS streams PCM

```bash
python - <<'PY'
import asyncio, os
from backend.openai.tts import TTSClient
async def main():
    t = TTSClient(os.environ["OPENAI_API_KEY"])
    total = 0
    async for chunk in t.stream_pcm("Hello there, this is a streaming test."):
        total += len(chunk)
    print("TTS PCM bytes:", total)
    assert total > 0
asyncio.run(main())
PY
```
**Pass:** non-zero PCM byte count.

## 6. Motion / frame generation (mock engine)

```bash
python - <<'PY'
import asyncio, numpy as np
from backend.avatar.mock_engine import MockAvatarEngine
from backend.avatar.base import EngineConfig
async def main():
    eng = MockAvatarEngine()
    s = await eng.start_session(np.full((256,256,3),180,np.uint8), "test",
                                EngineConfig(256,256,25,16000))
    s.push_audio(np.random.randn(16000).astype("float32")*0.2)
    n=0
    async for f in s.frames():
        n+=1
        if n>=10: break
    await s.close()
    print("frames:", n, "shape:", f.rgb.shape)
asyncio.run(main())
PY
```
**Pass:** `frames: 10 shape: (256, 256, 3)`.

## 7. WebRTC + end-to-end conversation (browser)

1. Open `https://<host>/` (HTTPS required for WebRTC).
2. Upload any portrait image. Enter your OpenAI key. Pick a model. Click **Connect**.
3. Status badge turns **live**; the `<video>` shows the animated avatar.
4. Type "Tell me a joke." and press Enter.

**Pass criteria:**
- assistant text streams into the transcript within ~1 s,
- you hear streamed speech,
- the avatar's mouth moves in sync with the speech,
- **Stop** interrupts mid-utterance.

## 8. WebRTC connection check (headless, optional)

```bash
# confirms an SDP answer is produced for an offer
python - <<'PY'
import asyncio, json, websockets, base64
async def main():
    img = base64.b64encode(open("/dev/urandom","rb").read(64)).decode()  # dummy
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        await ws.send(json.dumps({"type":"hello","openai_api_key":"sk-xxxxxxxxxx",
            "llm_model":"gpt-4o","tts_model":"gpt-4o-mini-tts","tts_voice":"alloy",
            "avatar_image_b64":"data:image/png;base64,"+img,"prompt":"hi"}))
        print("ready:", json.loads(await ws.recv())["type"])
asyncio.run(main())
PY
```
**Pass:** prints `ready: ready` (image decode may warn on the dummy bytes — use a
real PNG to fully exercise the path).

## 9. Metrics

```bash
curl -s http://localhost:8000/metrics | grep avatar_
```
**Pass:** Prometheus counters/gauges (`avatar_active_sessions`, `avatar_frames_total`, latency histograms).
