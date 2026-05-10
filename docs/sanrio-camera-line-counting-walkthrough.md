# Sanrio walkthrough — camera, line counting, and live annotation

This document is a **step-by-step user journey** from zero to a running **line-crossing / people counting** setup on **ml-server** (FastAPI + FrameBus + Redis). It is written for **backend** and **frontend** teams implementing the same flow in production dashboards.

**Codename:** *Sanrio* (replace with your release name if you prefer.)

**What you get at the end**

- A registered RTSP (or file) **camera**
- A **CROSS_LINE** task with one or more virtual **lines** (entrance counting, A→B direction, etc.)
- **`POST /detection/start`** spinning up capture + tracking + task workers
- **Live annotated video** over WebSocket (boxes + track IDs)
- **Crossing events** on **SSE** (and optional WebSocket), for you to **aggregate into counts** per line in your app

---

## 0. What I assume before I start

- The **ml-server** API is reachable (example: `http://localhost:9000` or `http://<jetson-ip>:9000`).
- I have a valid **stream URL** (RTSP or local path the server can read).
- For **live WebSocket** preview, **Redis** must be configured (`REDIS_URL`). The repo’s **[`docker-compose.yml`](../docker-compose.yml)** sets **`REDIS_URL=redis://redis:6379/0`** on the `yolo-detect` service and runs a **`redis`** service with **AOF** on a named volume (`redis_data`). Without Redis, HTTP task registration still works, but **`WS /cameras/{id}/live`** and per-camera **event** WebSockets are not available as designed.

I skim **`README.md`** and my **`.env`** for env vars such as `WIDTH`, `HEIGHT`, `YOLO_MODEL`, `CONF_THRESHOLD`, `REDIS_URL`, `REDIS_LIVE_FPS`, `LIVE_ANNOTATION_MODE`. **Important:** [`docker-compose.yml`](../docker-compose.yml) also injects defaults (see §3) — they override `.env` for keys listed under `environment:` unless your Compose setup changes that.

---

## 1. I bring the server up

**Option A — Docker Compose (recommended for Jetson + GPU stack)**

From the repo root:

```bash
docker compose up -d --build
```

The **`yolo-detect`** service uses **`runtime: nvidia`**, mounts **`./models`** and **`./outputs`**, exposes **`9000`**, and depends on **`redis`** and **`qdrant`**. Uvicorn runs with **`--ws-ping-interval`** / **`--ws-ping-timeout`** for WebSocket keepalive. A **healthcheck** hits **`GET /health`** inside the container (allow ~3 minutes **`start_period`** on first boot while models load).

If Compose fails with **unknown runtime `nvidia`**, the image **`apt-get`** step fails on the wrong CPU architecture, or **`yolo-detect`** stays **unhealthy**, see **[Stream / Docker troubleshooting](stream_test_runbook.md#docker-troubleshooting)**.

**Option B — Local uvicorn (no Compose)**

```bash
uvicorn app:app --host 0.0.0.0 --port 9000
```

I still need **`REDIS_URL`** pointing at a reachable Redis if I want live WebSockets.

I open **`http://localhost:9000/docs`** to confirm routes exist: `POST /cameras`, `POST /api/tasks`, `POST /detection/start`, `GET /detection/stream`, WebSocket **`/cameras/{camera_id}/live`**.

---

## 2. I register my camera

**As a user**, I register an **id → url** pair. The **`id`** is what the rest of the system calls **`channelId`** on tasks.

```bash
export BASE=http://localhost:9000

curl -s -X POST "$BASE/cameras" \
  -H "Content-Type: application/json" \
  -d '{
    "cameras": [
      {
        "id": "store_entrance_1",
        "url": "rtsp://user:pass@192.168.1.50/stream1"
      }
    ]
  }'
```

**Flat body** is also accepted (dashboards often send this):

```json
{ "id": "store_entrance_1", "url": "rtsp://..." }
```

I verify:

```bash
curl -s "$BASE/cameras"
```

Response includes each camera’s **`snapshot`** URL when a JPEG could be grabbed — useful for drawing lines in the UI.

**Backend team:** persist `id` and `url` in our device/channel service; the ML service keeps them in memory until restart unless we re-`POST /cameras`.

---

## 3. I prepare line geometry (critical detail)

**Crossing logic runs on frames after resize inside FrameBus** (see `frame_bus.py` + `vision_utils.resize`). The **effective** inference size is whatever **`WIDTH`** / **`HEIGHT`** the running process sees:

- **Code default** (no Compose): often **`WIDTH=1280`**; if **`HEIGHT`** is `0`, height is **proportional** (aspect preserved).
- **[`docker-compose.yml`](../docker-compose.yml)** currently sets **`WIDTH=480`** and **`HEIGHT=360`** for the `yolo-detect` service — line coordinates in `areaPosition` must match **that** 480×360 inference frame, not 1280-wide assumptions.

**`GET /cameras` snapshots** are taken from the **raw stream** (not necessarily the same width/height as inference). So **as a user** I do one of:

1. **Resize the snapshot in my UI** to exactly **`WIDTH × H_infer`** (with the same aspect rule as the server), draw lines there, and send those pixel coordinates in `areaPosition`; or  
2. Draw on the full snapshot, then **scale** \((x, y)\): \(x' = x \cdot W_{infer}/W_{snap}\), \(y' = y \cdot H_{infer}/H_{snap}\) when building `areaPosition`.

If I skip this, lines will **not** line up with people and crossings will be wrong or missing.

**Frontend team:** expose **WIDTH/HEIGHT** (or inferred H) from backend config so the annotator canvas matches inference space.

---

## 4. I define my counting lines in `areaPosition`

For **`algorithmType`: `CROSS_LINE`**, `areaPosition` is a **JSON string** — an array of line objects. Each line has **two points** in **pixel coordinates** (inference frame space).

From `services/cross_line.py`, each element looks like:

| Field | Meaning |
|--------|---------|
| `line_id` | Stable id (string) — I use this when aggregating counts |
| `line_name` | Human label (e.g. `Entrance`) |
| `point` | Exactly **two** objects `{ "x", "y" }` — segment endpoints |
| `direction` | `0` = both directions count; `1` / `2` = only one crossing direction |

**Direction semantics (practical):** The service detects **side flips** relative to the directed segment from point[0] → point[1]. Use **`direction`** to **ignore** crossings that go the “wrong” way for my counting story (e.g. only **in** through the door).

Example **string** I store in `areaPosition` (escaped for JSON). Coordinates below fit **`WIDTH=480`, `HEIGHT=360`** (current [`docker-compose.yml`](../docker-compose.yml) defaults); if you run **`WIDTH=1280`**, scale these points to your inference frame (§3).

```json
"[{\"line_id\":\"1\",\"line_name\":\"Main entrance\",\"point\":[{\"x\":240,\"y\":80},{\"x\":240,\"y\":300}],\"direction\":1}]"
```

I can register **multiple lines** (multiple entries in the array) for zones with several virtual thresholds.

---

## 5. I create the task (`POST /api/tasks`)

**As a user**, I upsert a task. **`channelId`** must equal my camera **`id`** (`store_entrance_1`).

```bash
curl -s -X POST "$BASE/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 101,
    "taskName": "customer_walkin_main",
    "algorithmType": "CROSS_LINE",
    "channelId": "store_entrance_1",
    "enable": true,
    "threshold": 50,
    "areaPosition": "[{\"line_id\":\"1\",\"line_name\":\"Main entrance\",\"point\":[{\"x\":240,\"y\":80},{\"x\":240,\"y\":300}],\"direction\":1}]",
    "detailConfig": {
      "enableAttrDetect": false,
      "enableReid": false
    },
    "validWeekday": ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"],
    "validStartTime": 0,
    "validEndTime": 86400000
  }'
```

**Optional:** If I created the task with a placeholder `channelId`, I can **re-point** it with:

`POST /cameras/store_entrance_1/tasks` body: `{ "taskId": 101 }` (or `task_id`).

**Backend team:** our orchestration service should **idempotently** upsert tasks when configs change; `taskId` is our stable primary key in the ML API.

---

## 6. I start detection

**Single camera** (typical when testing): query only — **no JSON body** is required.

```bash
curl -s -X POST "$BASE/detection/start?camera_id=store_entrance_1"
```

If **more than one** camera has enabled tasks, starting **without** `camera_id` returns **400** unless I allow all:

```bash
curl -s -X POST "$BASE/detection/start?all_channels=true"
```

I check health:

```bash
curl -s "$BASE/detection/status"
```

Optional — pipeline / decoder stats (works the same in Docker):

```bash
curl -s "$BASE/stream/metrics"
```

---

## 7. I consume crossing events (for “line counting”)

The server emits **one event per crossing** (`CROSS_LINE`), not a running total. **My application** increments counters per `line.id` / `taskId` / `channelId`.

**SSE (good for web and simple integrations):**

```bash
curl -N "$BASE/detection/stream?eventType=CROSS_LINE&channelId=store_entrance_1"
```

Each `data:` line is a JSON object shaped like (see `app.py` docstring):

- `eventType`: `CROSS_LINE`
- `taskId`, `taskName`, `channelId`
- `line`: `{ id, name, direction }` — **direction here reflects the crossing direction that fired**
- `person`: `trackingId`, `boundingBox`, `attributes`, …
- `evidence`: image hints when enabled

**Filters** can reduce noise: `taskId`, `taskName`, `eventType`, `channelId` (combined with AND semantics as implemented).

**WebSocket (same payloads, per camera):**

- `WS /cameras/store_entrance_1/events` — JSON text messages

**On disk:** events append to **`EVENTS_DIR/task_<taskId>.jsonl`** (default under `/local/storage/events` unless overridden) — useful for audits and replay.

**Backend team:** subscribe once per deployment; with **`REDIS_URL`** set, multiple uvicorn workers can share the same Redis-backed SSE bridge (see `README.md` / API docs).

**Reconnect / resilience (optional):** events may include **`_seq`**. On SSE reconnect, send header **`Last-Event-ID: <last_seq>`** for a short replay of buffered events (bounded; see [`API_USAGE.md`](API_USAGE.md) §5). For ops, **`GET /stream/resilience-stats`** and **`GET /stream/metrics`** expose circuit state, buffers, and respawn counters when those features are enabled (Compose sets **`WATCHDOG_ENABLED`**, **`EVENT_BUFFER_MAX`**, **`SSE_REPLAY_BUFFER`** on `yolo-detect`).

---

## 8. I show live annotated video to operators

**WebSocket (binary JPEG per message, no Base64):**

```text
ws://localhost:9000/cameras/store_entrance_1/live
```

Requirements:

- **`REDIS_URL`** set so FrameBus can publish and the API can subscribe (Compose: `redis://redis:6379/0`).
- Client handles **backpressure** (slow clients may drop frames — see **`WS_SEND_TIMEOUT_MS`**).
- The server **retries the Redis subscription** inside an open socket up to **`WS_REDIS_MAX_RETRIES`**; if Redis stays down, the socket closes with **1011**.

**Optional — replay after reconnect:** `ws://.../cameras/store_entrance_1/live?last_seq=<n>` asks for a best-effort replay of recent frames with sequence **>** `n` (server-side ring buffer; same family of limits as **`SSE_REPLAY_BUFFER`** / **`WS_FRAME_REPLAY_BUFFER`**).

**Convenience alias** by task name:

```text
ws://localhost:9000/tasks/customer_walkin_main/live
```

**Frontend team:** render JPEG blobs with `Blob` + `URL.createObjectURL` or `createImageBitmap`; reconnect on disconnect; consider showing FPS and last error from **`/detection/status`**. With Docker Compose, expect WebSocket **ping/pong** from uvicorn (see `yolo-detect` `command` in [`docker-compose.yml`](../docker-compose.yml)).

---

## 9. I draw lines on the stream in the UI (optional overlay)

The **server-side** live JPEG includes **YOLO boxes + track IDs** (per `LIVE_ANNOTATION_MODE`). **Virtual lines from `areaPosition` are not automatically drawn** on that feed today — the canonical geometry lives in the task config.

**As a frontend developer**, I:

1. Load the same line endpoints I sent in `areaPosition` (in inference pixel space).
2. If my preview is scaled on screen, I **scale** line coordinates for **display only**; the server still uses the stored pixel values.
3. Optionally snap lines using a still frame: **`GET /cameras`** → `snapshot` URL, then scale as in section 3.

---

## 10. I stop or change configuration

**Stop processing:**

```bash
curl -s -X POST "$BASE/detection/stop?camera_id=store_entrance_1"
```

To stop **everything**: `POST "$BASE/detection/stop"` with no `camera_id` (or `POST /detection/stop/all`).

**Change lines:** `PUT /api/tasks/{task_id}` with updated `areaPosition`, then restart detection so workers pick up config (follow your operational playbook — in-memory task registry updates may require stop/start depending on how you manage processes).

**Remove task:** `DELETE /api/tasks/{task_id}`.

---

## 11. Checklist I give both teams

| Step | Backend | Frontend |
|------|---------|----------|
| Camera CRUD | Call `POST/GET` `/cameras`; store ids | Show list; use `snapshot` for annotator if available |
| Line editor | Validate `areaPosition` JSON against `apis/tasks.py` rules | Canvas in **inference resolution**; scale from snapshot if needed |
| Task CRUD | `POST/PUT /api/tasks`; unique `taskId` | Form + preview |
| Start/stop | `POST /detection/start|stop`, poll `/detection/status` | Control panel + error toasts |
| Counting | Subscribe SSE or WS events; aggregate by `line.id` | Live counters, charts, per-line filters |
| Live view | Ensure `REDIS_URL` in deploy; Redis reachable from API container | WebSocket JPEG loop |
| Docker / GPU | `docker compose up -d --build`; NVIDIA runtime; `./models` mounted | N/A (ops) |
| Deep health | Poll **`/stream/metrics`**, **`/stream/resilience-stats`** if debugging drops / Redis | Optional status widgets |

---

## 12. Common mistakes I avoid

- **`channelId` ≠ camera `id`** → no frames for task / start returns 404-style errors.
- **Line coordinates in wrong resolution** → crossings never trigger or trigger in the wrong place. **Double-check `WIDTH`/`HEIGHT`** for your deploy (Compose defaults to **480×360** in [`docker-compose.yml`](../docker-compose.yml)).
- **Expecting built-in totals** → the service emits **events**; **counts are application state**.
- **Ignoring `track_id`** — crossings require stable tracking; very poor RTSP or tiny people may yield `-1` tracks that are skipped.
- **Missing Redis** → live WebSocket preview may be broken even if detection runs.
- **Assuming `localhost:6379` inside Docker** → from **`yolo-detect`**, Redis host is the service name **`redis`**, not `localhost` (use **`REDIS_URL=redis://redis:6379/0`** as in Compose).

---

## 13. Reference implementation in this repo

| Concern | Location |
|---------|----------|
| Compose stack (Redis, Qdrant, GPU, env defaults) | [`docker-compose.yml`](../docker-compose.yml) |
| Docker / stream ops | [`stream_test_runbook.md`](stream_test_runbook.md) |
| Task schema + validation | `apis/tasks.py` |
| CROSS_LINE logic + event shape | `services/cross_line.py` |
| Start/stop + workers | `apis/detection.py`, `frame_bus.py`, `task_worker.py` |
| SSE + replay ring | `GET /detection/stream` in `app.py`, `apis/detection_stream.py` |
| Live WebSocket | `apis/ws_live.py`, routes in `app.py` |
| Stream metrics / resilience API | `apis/stream_metrics.py` (`/stream/metrics`, `/stream/resilience-stats`, …) |
| Overview | `README.md` |

---

*End of walkthrough — Sanrio scenario from camera registration through live annotation and event-driven line counting.*
