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
| GET | `/stream/metrics` | Per-camera pipeline stats (fps, drop rate, task-queue depth, frame count, annotation mode) |
| GET | `/stream/health` | Same as `/stream/metrics` wrapped under `{"cameras": [...]}` |
| GET | `/stream/resilience-stats` | Reconnect counters, circuit-breaker state, DLQ pending count |
| GET | `/stream/debug/annotation-state` | FrameBus annotation + Redis publish diagnostics per camera |

### SSE query parameters (`GET /detection/stream`)

All filters combine with **AND** semantics:

| Query | Description |
|-------|-------------|
| `taskId` | Integer task id |
| `taskName` | Exact task name (not unique across tasks) |
| `eventType` | e.g. `CROSS_LINE` |
| `channelId` | Camera channel id string |

Idle connections receive SSE comment keepalives (`: ping`) about every 30 seconds.

Optional header **`Last-Event-ID`** (last seen **`_seq`** on events) requests replay after reconnect:

1. In-memory ring (`SSE_REPLAY_BUFFER`, default 200).
2. When `REDIS_STREAMS_ENABLED=true`, also reads `live:events:{camera_id}` via Redis `XRANGE` (shadow written by `task_worker` alongside Pub/Sub).

See [API_USAGE.md](../docs/API_USAGE.md) §5 and [OPTIMIZATION_REFERENCE.md](../docs/OPTIMIZATION_REFERENCE.md). Ops metrics: **`GET /stream/resilience-stats`**, optional **`GET /metrics`** (`PROMETHEUS_ENABLED`).

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
| 409 | Start called while that camera's bus is already running; or stop when nothing running (including when a local video file has been exhausted and the process exited naturally) |

## Worker process lifecycle

`DetectionResource` manages FrameBus and task-worker `multiprocessing.Process` objects.

- **Start** (`POST /detection/start`): spawns one FrameBus + N task-worker processes per camera channel. Calls `_reap_finished_processes()` first to join any naturally-exited processes before the new ones are created.
- **Stop** (`POST /detection/stop`): sets the per-camera `stop_event`, drains task queues (up to `STOP_DRAIN_TIMEOUT_SEC`, default `3`), then `join(timeout=5)` on all workers.
- **Zombie prevention**: `_reap_finished_processes()` is called at the start of every `_start()` and `_stop()`. It calls `join(timeout=0)` on any process whose `exitcode is not None` — this prevents OS zombies from accumulating when a local-file FrameBus finishes before an explicit stop is issued.
- **Watchdog** (optional, `WATCHDOG_ENABLED=true`): detects unexpectedly dead FrameBus processes and calls `_restart_channel()`. Off by default to avoid unintentional restarts on edge devices where stream exhaustion is expected.

## Stream metrics curl

```bash
export BASE="http://localhost:9000"

# Per-camera pipeline stats (fps, drop_rate, frame_count, annotation_mode, etc.)
curl -sS "${BASE}/stream/metrics"

# Annotation + Redis publish diagnostics
curl -sS "${BASE}/stream/debug/annotation-state"

# Resilience counters + DLQ size
curl -sS "${BASE}/stream/resilience-stats"
```

## Edge device note

For **reliable long-lived SSE**, implement client reconnect with `Last-Event-ID` only if you add that server-side; the current server does not resume history. Poll `GET /detection/status` alongside SSE to detect stalled RTSP or process death.

When running a **local video file** (non-RTSP URL), the FrameBus process exits when the file ends. `GET /detection/status` will show `running: false` and `stopped_reason: "stream_exhausted"`. A subsequent `POST /detection/start` will restart the pipeline from the beginning of the file.
