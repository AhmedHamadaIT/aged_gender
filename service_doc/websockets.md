# WebSocket streams (live video + events)

## Description

These routes use the **WebSocket** protocol (not shown as separate operations in OpenAPI). They require **`REDIS_URL`** on the server for live fan-out from the frame bus.

Clients should **reconnect** after disconnect. The server **does** retry the Redis subscription from inside an open WebSocket session (`WS_REDIS_MAX_RETRIES`, `WS_REDIS_RECONNECT_DELAY_MS`); if Redis stays down, the socket closes with **1011**.

For **live JPEG**, optional query **`last_seq`**: `ws://host/cameras/1/live?last_seq=123` replays a short server-side ring of frames with sequence greater than `123` (best-effort; same ring cap as `SSE_REPLAY_BUFFER` / `WS_FRAME_REPLAY_BUFFER`). Frames on the wire remain **raw JPEG bytes** after decode.

**Frontend implementation** (blob URLs, rAF, `ImageBitmap`, reconnect, optional seq header): see [`docs/frontend-live-stream-guide.md`](../docs/frontend-live-stream-guide.md).

Uvicorn (see [`docker-compose.yml`](../docker-compose.yml)) is configured with **`--ws-ping-interval`** and **`--ws-ping-timeout`** for protocol-level WebSocket pings.

### Camera-ID validation

`camera_id` path parameters are validated by `apis.ws_live.validate_camera_id()` before the socket is accepted.  Invalid IDs cause an immediate close:

| Rejected value | Reason |
|---|---|
| `null`, `undefined`, `none`, `nan` | Reserved JavaScript / JSON sentinel values |
| Any ID containing characters outside `[A-Za-z0-9_\-]` | Invalid character set (e.g. spaces, `!`, `@`) |

Valid examples: `cam1`, `cam_01`, `entrance-left`, `42`.

### Redis connection efficiency (optional)

When `WS_MUX_ENABLED=true`, live frame WebSockets use [`apis/_redis_fanout.py`](../apis/_redis_fanout.py): **one Redis pubsub subscription per camera** shared across all clients on that camera (instead of one subscription per client). Default is `false` (legacy per-client pubsub).

When `LIVE_PUBLISH_REQUIRE_SUBSCRIBER=true`, FrameBus skips publishing to `live:frame:{camera_id}` if the subscriber counter is zero (saves CPU/Redis when no one is watching).

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
| 1011 | Redis unavailable, or Redis reconnect attempts exhausted while the socket was open |

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
