# WebSocket streams (live video + events)

## Description

These routes use the **WebSocket** protocol (not shown as separate operations in OpenAPI). They require **`REDIS_URL`** on the server for live fan-out from the frame bus.

Clients should **reconnect** after disconnect (the server does not auto-reconnect for you).

## Endpoints

| Protocol | Path | Payload |
|----------|------|---------|
| WebSocket | `/cameras/{camera_id}/live` | Binary: one **JPEG** frame per message (annotated) |
| WebSocket | `/cameras/{camera_id}/events` | Text JSON: same family of objects as `GET /detection/stream` (including V2 `evidence` on task events — [ml_image_v2.md](./ml_image_v2.md)) |
| WebSocket | `/tasks/{task_name}/live` | Binary JPEG for the camera bound to that **unique** task name |

### Close codes (`/tasks/{task_name}/live`)

| Code | Meaning |
|------|---------|
| 4004 | Task name not found |
| 4009 | Ambiguous task name (multiple tasks share `taskName`) |
| 1011 | Redis unavailable |

## curl / CLI testing

Standard `curl` does not speak WebSocket. Use a dedicated client.

### wscat (Node)

```bash
npx wscat -c "ws://localhost:9000/cameras/1/events"
```

### websocat (Rust binary)

```bash
websocat "ws://localhost:9000/cameras/1/events"
```

### Python quick test (events)

```bash
python3 - <<'PY'
import asyncio, json
import websockets

async def main():
    uri = "ws://localhost:9000/cameras/1/events"
    async with websockets.connect(uri) as ws:
        msg = await asyncio.wait_for(ws.recv(), timeout=30)
        print(json.loads(msg))

asyncio.run(main())
PY
```

Install dependency: `pip install websockets`.

### Live JPEG binary stream

Binary frames are not practical to print in curl; use a small script or browser:

```javascript
const ws = new WebSocket("ws://ML_SERVER_HOST:9000/cameras/1/live");
ws.binaryType = "arraybuffer";
ws.onmessage = (e) => { /* e.data is JPEG bytes */ };
ws.onclose = () => setTimeout(() => location.reload(), 2000);
```

## Edge device note

- Use **`ws://` or `wss://`** to match your TLS termination (reverse proxy often terminates `wss` and speaks `ws` upstream).
- Slow clients may **drop frames** when send buffer exceeds timeout (see `app.py` comments: `WS_SEND_TIMEOUT_MS`).
- Default live publish rate is controlled by **`REDIS_LIVE_FPS`** (documented in code, default around 13 fps).
