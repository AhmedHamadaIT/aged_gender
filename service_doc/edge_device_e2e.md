# Edge device / integrator — end-to-end flow

This guide is for a **small computer, NVR, or gateway** that orchestrates the ML server (it does **not** run the heavy models; it calls the API). The **RTSP URL** you register must be reachable **from the ML server**, not necessarily from the edge box (unless they are the same host).

## Prerequisites

1. ML server process is running (`uvicorn app:app --host 0.0.0.0 --port 9000` or your deployment).
2. You know `ML_SERVER_HOST` and port (default `9000`).
3. For live WebSocket video/events from this API, configure **`REDIS_URL`** on the server.
4. Cameras are on the network; **the server can open** the RTSP stream you pass in `POST /cameras`.

## Recommended call order

| Step | Action | Endpoint |
|------|--------|----------|
| 1 | Verify API is up | `GET /` |
| 2 | Register each camera (id ↔ RTSP) | `POST /cameras` |
| 3 | Confirm cameras | `GET /cameras` |
| 4 | Register each analytics task (`channelId` = camera `id`) | `POST /api/tasks` |
| 5 | Confirm tasks | `GET /api/tasks` |
| 6 | Start processing | `POST /detection/start` (optional `?camera_id=` for one camera) |
| 7 | Monitor | `GET /detection/status`, `GET /stream/metrics`, `GET /stream/resilience-stats`, `GET /detection/stream` (SSE), and/or WebSockets |
| 8 | Stop | `POST /detection/stop` or `POST /detection/stop/all` |

**Resilience:** `GET /stream/resilience-stats` surfaces circuit breaker state, buffered events, respawn counts, and embedding DLQ depth. Docker-specific failures (NVIDIA runtime, image arch, healthcheck) are covered in [../docs/stream_test_runbook.md](../docs/stream_test_runbook.md#docker-troubleshooting).

## Step 1 — Health

```bash
export BASE="http://ML_SERVER_HOST:9000"
curl -sS "${BASE}/"
```

Expect JSON with `service` and `version`.

## Step 2 — Register cameras

Use the same string for `id` that you will use as `channelId` in tasks (often `"1"`, `"2"`, or a device serial).

```bash
curl -sS -X POST "${BASE}/cameras" \
  -H "Content-Type: application/json" \
  -d '{
    "cameras": [
      {"id": "1", "url": "rtsp://192.168.1.50:554/stream1"}
    ]
  }'
```

## Step 3 — List cameras

```bash
curl -sS "${BASE}/cameras"
```

## Step 4 — Register a task

`algorithmType` must be one of: `CROSS_LINE`, `MASK_HAIRNET_CHEF_HAT`, `CASHIER_BOX_OPEN`, `PHONE_USAGE`.

Per-algorithm docs (config, events, curl): [cross_line.md](./cross_line.md), [mask_hairnet_chef_hat.md](./mask_hairnet_chef_hat.md), [phone_usage.md](./phone_usage.md), [cashier.md](./cashier.md). **Evidence JSON shape, disk paths, env, SSH / JSONL tail:** [ml_image_v2.md](./ml_image_v2.md).

Example cross-line task (minimal; adjust `areaPosition` to your resolution/line):

```bash
curl -sS -X POST "${BASE}/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 10,
    "taskName": "entrance_line",
    "algorithmType": "CROSS_LINE",
    "channelId": "1",
    "enable": true,
    "threshold": 60,
    "areaPosition": "[{\"line_id\":\"1\",\"line_name\":\"Entrance\",\"point\":[{\"x\":100,\"y\":400},{\"x\":900,\"y\":400}],\"direction\":1}]",
    "detailConfig": {"enableAttrDetect": false, "enableReid": false, "alarmType": []},
    "validWeekday": ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"],
    "validStartTime": 0,
    "validEndTime": 86400000
  }'
```

For **cashier** monitoring, use `algorithmType`: `CASHIER_BOX_OPEN` and consume **`/cashier/*`** as well as or instead of generic crossing SSE — see [cashier.md](./cashier.md).

## Step 5 — List tasks

```bash
curl -sS "${BASE}/api/tasks"
```

## Step 6 — Start detection

Start all cameras that have enabled tasks and registered RTSP URLs:

```bash
curl -sS -X POST "${BASE}/detection/start"
```

Start only camera `"1"`:

```bash
curl -sS -X POST "${BASE}/detection/start?camera_id=1"
```

## Step 7 — Monitor

**HTTP status (polling):**

```bash
curl -sS "${BASE}/detection/status"
```

**SSE (server pushes one JSON event per line crossing / task event):**

```bash
curl -sS -N "${BASE}/detection/stream"
```

Filtered SSE (example: one task, one channel):

```bash
curl -sS -N "${BASE}/detection/stream?taskId=10&channelId=1"
```

**WebSocket** (browser or `wscat`): see [websockets.md](./websockets.md).

## Step 8 — Stop

Stop one camera pipeline:

```bash
curl -sS -X POST "${BASE}/detection/stop?camera_id=1"
```

Stop everything:

```bash
curl -sS -X POST "${BASE}/detection/stop"
# or
curl -sS -X POST "${BASE}/detection/stop/all"
```

## Operational notes for edge integrators

- **409 on start**: Camera pipeline already running for that `channelId` — call stop first or use a fresh server state.
- **400 on start**: No tasks or no cameras — complete steps 2 and 4 first.
- **Single worker vs SSE**: `GET /detection/stream` broadcasts in-process; multiple Uvicorn workers need an external broker for shared SSE (documented in code comments).
- **Person / semantic search**: Separate optional services; see [person_search.md](./person_search.md) and [semantic_search.md](./semantic_search.md). They do not require `/detection/start`.
- **Debug on the ML host over SSH** (JSONL, logs, `jq`): [ml_image_v2.md](./ml_image_v2.md#ssh--watch-task-jsonl-all-cases).
