# Detection service (pipeline control + SSE)

## Description

Controls **FrameBus** (per-camera frame grab + fan-out) and **task worker** processes. After cameras and tasks are configured, `POST /detection/start` spawns workers. Results are pushed to an internal queue consumed by **`GET /detection/stream`** (Server-Sent Events).

Crossing, PPE, phone, and cashier task events share the SSE channel. Payload shapes include `eventType`, task metadata, and **`evidence`** with V2 **`captureImage`** / **`sceneImage`** objects. See [ml_image_v2.md](./ml_image_v2.md) and `app.py` docstring.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/detection/start` | Start pipelines for all cameras with enabled tasks, or one camera if `?camera_id=` is set |
| POST | `/detection/stop` | Stop one camera (`?camera_id=`) or all running cameras if param omitted |
| POST | `/detection/stop/all` | Alias for stop-all |
| GET | `/detection/status` | Per-camera FPS, frame counts, errors |
| GET | `/detection/stream` | SSE stream of JSON events (optional query filters) |

### SSE query parameters (`GET /detection/stream`)

All filters combine with **AND** semantics:

| Query | Description |
|-------|-------------|
| `taskId` | Integer task id |
| `taskName` | Exact task name (not unique across tasks) |
| `eventType` | e.g. `CROSS_LINE` |
| `channelId` | Camera channel id string |

Idle connections receive SSE comment keepalives (`: ping`) about every 30 seconds.

## curl — start all

```bash
export BASE="http://localhost:9000"
curl -sS -X POST "${BASE}/detection/start"
```

## curl — start one camera

```bash
curl -sS -X POST "${BASE}/detection/start?camera_id=1"
```

## curl — status

```bash
curl -sS "${BASE}/detection/status"
```

## curl — SSE stream (raw)

```bash
curl -sS -N "${BASE}/detection/stream"
```

`curl -N` disables buffering so lines arrive live.

## curl — SSE with filters

```bash
curl -sS -N "${BASE}/detection/stream?taskId=10&channelId=1&eventType=CROSS_LINE"
```

## curl — stop one camera

```bash
curl -sS -X POST "${BASE}/detection/stop?camera_id=1"
```

## curl — stop all

```bash
curl -sS -X POST "${BASE}/detection/stop"
```

or

```bash
curl -sS -X POST "${BASE}/detection/stop/all"
```

## Common HTTP errors

| Code | Typical cause |
|------|----------------|
| 400 | No cameras or no enabled tasks before start |
| 404 | `camera_id` on start/stop does not match any task/camera |
| 409 | Start called while that camera’s bus is already running; or stop when nothing running |

## Edge device note

For **reliable long-lived SSE**, implement client reconnect with `Last-Event-ID` only if you add that server-side; the current server does not resume history. Poll `GET /detection/status` alongside SSE to detect stalled RTSP or process death.
