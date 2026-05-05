# Vision Pipeline API

**Version 2.0.0** — FastAPI server for multi-camera computer vision.

The runtime path is **task-based**: you register RTSP cameras (`POST /cameras`), define **tasks** per channel (`POST /api/tasks` with an `algorithmType` such as `CROSS_LINE`, `MASK_HAIRNET_CHEF_HAT`, or `CASHIER_BOX_OPEN`), then start workers (`POST /detection/start`). **FrameBus** captures each stream once, runs a shared YOLO+tracker, fans frames to **task workers** (`task_worker.py`), and publishes annotated JPEG frames to **Redis** for the live WebSocket stream. Task results (line crossings, PPE violations, cashier alerts) are broadcast on **`GET /detection/stream`** (SSE). **Live annotated video** streams via **`WS /cameras/{id}/live`** (binary JPEG over WebSocket — no Base64, 50 ms backpressure drop).

**Cashier** is integrated as API task type **`CASHIER_BOX_OPEN`** — implemented by class **`CashierDrawerTask`** in [`services/cashier.py`](services/cashier.py) (same module as **`CashierService`**), registered in [`services/__init__.py`](services/__init__.py) `TASK_REGISTRY`. Each processed frame emits one structured task event on **`GET /detection/stream`** (`eventType`: **`CASHIER_BOX_OPEN`**, Eyego-style payload under **`data`**), is appended to **`EVENTS_DIR/task_<taskId>.jsonl`** (same pattern as `CROSS_LINE`), and updates **`/cashier/status`** and **`/cashier/events`**. Zones, evidence JPEGs/GIFs, and optional **`/cashier/stream/{camera_id}`** remain under **`/cashier/*`**. Configure **`YOLO_MODEL`** so FrameBus outputs person / drawer / cash on that channel; see [Cashier + FrameBus model](#cashier--framebus-model).

The older **per-process pipeline** (`pipeline.py` + `services.REGISTRY`: `detector`, `age_gender`, `mood`, `ppe`, `cashier`) is still in the repo for scripts and batch tooling; the **HTTP app does not** expose `POST /detection/setup`.

---

## Contents

| Section | What it covers |
|--------|----------------|
| [Setup & run](#setup--run) | Local `uvicorn`, Docker Compose (GPU), base URL, `/docs` |
| [Features (complete inventory)](#features-complete-inventory) | Architecture, every task type, SSE, cashier, legacy pipeline, models |
| [Sample `cashier_zones.yaml`](#sample-configuration-file-configcashier_zonesyaml) | Zone/threshold layout |
| [cURL reference](#complete-curl-reference-all-http-routes) | All HTTP examples |
| [API reference](#api-endpoints-reference) | Lifecycle and route descriptions |
| [Testing & examples](#testing--real-result-examples) | Image tests, JSON samples |
| [Cashier validation & cases](#cashier-full-dataset-validation-report) | Batch metrics, N1–A7 tables |
| [Models & structure](#-models--services) | Weights, ONNX, repo layout |
| [Configuration](#-configuration) | Env vars, thresholds |
| [Endpoint summary table](#-api-endpoints-summary) | Quick lookup |

---

## Setup & Run

### Prerequisites

- Python 3.9+
- CUDA/cuDNN enabled environment (optional but recommended for GPU acceleration)

### Installation (local)

### Installation

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the FastAPI server:

   ```bash
   uvicorn app:app --host 0.0.0.0 --port 9000
   ```

### Docker Compose (GPU)
The repo includes [`docker-compose.yml`](docker-compose.yml): NVIDIA runtime, project bind-mount, `models/` and `outputs/` volumes, port **9000**, optional [`.env`](.env) (copy from [`.env.example`](.env.example) for RTSP/Jetson hints).

```bash
docker compose up -d --build
docker compose logs -f yolo-detect
# shell inside container:
docker compose exec yolo-detect bash
```

If your host only supports Compose v1, run the same commands with `docker-compose`
from the directory that contains `docker-compose.yml`.

**Default base URL:** `http://localhost:9000` — use `http://<host>:9000` on a Jetson or remote machine. Interactive OpenAPI: `/docs`, `/redoc`.

---

## Features (complete inventory)

### Architecture (v2 runtime)
- **FrameBus** ([`frame_bus.py`](frame_bus.py)) — One process per active camera: RTSP capture, **YOLO** detection, **BoT-SORT** tracking, fan-out of `{frame + tracks}` to task queues, and **Redis publish** of annotated JPEG frames to `live:frame:{camera_id}` for the live WebSocket stream. Env: `YOLO_MODEL`, `CONF_THRESHOLD`, `DEVICE`, `FILTER_CLASSES`, `WIDTH`, `HEIGHT`, `SAVE_OUTPUT`, `OUTPUT_DIR`, `REDIS_URL`, `REDIS_LIVE_FPS`, `LIVE_ANNOTATION_MODE` (`ultralytics` / `opencv` / `none`). Ingest path: `RTSP_BACKEND` (`auto` / `gstreamer` / `ffmpeg` / `opencv`) — see [`stream.py`](stream.py) / [`stream_gstreamer.py`](stream_gstreamer.py) on Jetson.
- **Task workers** ([`task_worker.py`](task_worker.py)) — One worker process per enabled task; looks up `algorithmType` in [`services/__init__.py`](services/__init__.py) `TASK_REGISTRY`, emits events to the shared result queue **and** publishes them to Redis `live:event:{camera_id}`.
- **FastAPI lifespan** ([`app.py`](app.py)) — Starts `DetectionSSEBridge` (subscribes from both the multiprocessing queue and Redis `live:event:*` so multiple uvicorn workers can all serve SSE). Serves **WebSocket** live frame stream at `WS /cameras/{id}/live` via [`apis/ws_live.py`](apis/ws_live.py).
- **Redis** ([`docker-compose.yml`](docker-compose.yml) `redis:7-alpine`) — Fire-and-forget Pub/Sub broker. FrameBus publishes binary JPEG frames; FastAPI WebSocket handlers subscribe. No persistence needed (`--appendonly no`).

### Cameras

- **Multi-camera RTSP** — `POST /cameras`, `GET /cameras`, `DELETE /cameras/{cam_id}`. Many responses use the standardized envelope pattern (`error_codes/response.py`).

### Tasks (`/api/tasks` CRUD)

- Register work per **`channelId`** (must match a camera **`id`** from `POST /cameras`).
- **`algorithmType`** must be one of: **`CROSS_LINE`**, **`MASK_HAIRNET_CHEF_HAT`**, **`CASHIER_BOX_OPEN`** ([`apis/tasks.py`](apis/tasks.py) `TaskRegistry.SUPPORTED`).

| `algorithmType` | Implementation | What it does | SSE `eventType` on `/detection/stream` |
|-----------------|----------------|--------------|----------------------------------------|
| `CROSS_LINE` | [`CrossLineTask`](services/cross_line.py) | Person crosses configured lines (`areaPosition` JSON); optional age/gender on crossing via `detailConfig.enableAttrDetect`; `enableReid` reserved | `CROSS_LINE` |
| `MASK_HAIRNET_CHEF_HAT` | [`MaskHairnetChefHatTask`](services/mask_hairnet_chef_hat.py) | PPE compliance inside polygon zones; violations from `detailConfig.alarmType` (`no_mask`, `no_hat`, `no_chef_hat`, …) | `MASK_HAIRNET_CHEF_HAT` |
| `CASHIER_BOX_OPEN` | [`CashierDrawerTask`](services/cashier.py) | Zones, drawer/cash/person logic **N1–N6** / **A1–A7**; structured JSON per frame on detection SSE + `/cashier/*` | `CASHIER_BOX_OPEN` |

**Shared task fields** (see `TaskConfig` in [`apis/tasks.py`](apis/tasks.py)): `taskId`, `taskName`, `channelId`, `enable`, `threshold` (0–100 confidence), `areaPosition` (JSON string: lines or polygons per algorithm), `detailConfig`, `validWeekday`, `validStartTime`, `validEndTime` (ms).

**`detailConfig` per algorithm**

- **CROSS_LINE:** `enableAttrDetect`, `enableReid`
- **MASK_HAIRNET_CHEF_HAT:** `alarmType` (list of violation keys)
- **CASHIER_BOX_OPEN:** `drawerOpenLimit`, `serviceWaitLimit`, `enableStaffList`, `staffIds`

### Detection control, SSE events & live WebSocket stream

- **Runtime** — `POST /detection/start` / `POST /detection/stop` (optional query `camera_id`); `GET /detection/status` (FPS, frame counts, errors per camera).
- **`GET /detection/stream`** — Server-Sent Events: one JSON **task event** per message (crossings, PPE violations, etc.). Optional filters (AND): `taskId`, `taskName`, `eventType`, `channelId`. Idle `: ping` keepalives; overridable with **`DETECTION_SSE_KEEPALIVE_SEC`**. With `REDIS_URL` set, the bridge also reads from Redis `live:event:*` enabling multi-worker SSE.
- **`WS /cameras/{camera_id}/live`** — Binary WebSocket stream of **annotated JPEG frames** (YOLO boxes + track IDs drawn by FrameBus). Raw bytes per message — no Base64 overhead. Frames dropped with 50 ms timeout when clients are slow (`WS_SEND_TIMEOUT_MS`). Published at `REDIS_LIVE_FPS` (default 13 fps). See [`apis/ws_live.py`](apis/ws_live.py).
- **`WS /cameras/{camera_id}/events`** — JSON WebSocket stream of detection events per camera (same shape as SSE events).

### Cashier monitor (`/cashier/*`)

- Enable with **`POST /api/tasks`** using `algorithmType: "CASHIER_BOX_OPEN"` and matching `channelId`.
- **HTTP:** `GET /cashier/status`; `GET` / `POST /cashier/zones` (optional body keys: `detail_config`, `task`, `detection_threshold` per [`apis/cashier.py`](apis/cashier.py)); `POST /cashier/zones/reset`; `GET` / `DELETE /cashier/events` (filters: `severity`, `case_id`, `camera_id`, `limit`, `offset`); `GET /cashier/evidence` + download path; **media** routes for latest/event JPG/GIF; `GET .../drawer_count`.
- **Cashier SSE:** `GET /cashier/stream/{camera_id}` (all events), `GET .../only` (alerts-oriented; skips `frame`-type events). Publisher emits typed SSE (`event:` lines); connect handshake includes `connected`.
- **On disk / env:** `CASHIER_CONFIG` (default `./config/cashier_zones.yaml`), `CASHIER_EVIDENCE_DIR` (default `./evidence/cashier`), **`EVENTS_DIR`** (default `/local/storage/events` — per-task JSONL `task_<taskId>.jsonl` for cashier frames), **`CASHIER_LOG_MAX`** (default 5000 — in-memory event deque cap).

### Legacy batch / scripts

- **`pipeline.py` + `REGISTRY`** — Per-frame services: `detector`, `age_gender`, `mood`, `ppe`, `cashier` for offline JSONL, tests, and tooling (not the primary HTTP v2 path).

### Developer surface

- **OpenAPI** — `/docs`, `/redoc`
- **Tests** — `tests/` (e.g. `test_detection_stream.py`, `test_cashier_api.py`); run `pytest` from repo root when present
- **Adding tasks** — See [`services/__init__.py`](services/__init__.py) docstring and [`docs/ADDING_A_SERVICE.md`](docs/ADDING_A_SERVICE.md)

### Cashier + FrameBus model

FrameBus uses **one** YOLO model per running camera process. **`CASHIER_BOX_OPEN`** expects detections with cashier classes (e.g. person / drawer / cash). Point **`YOLO_MODEL`** at weights that expose those classes for channels where cashier runs. Tasks that assume COCO **`person`** (e.g. `CROSS_LINE`) need a compatible model if combined on the same channel—often you use **separate channels** or **separate deployments** for cashier vs. generic crossing.

---

## Sample configuration file (`config/cashier_zones.yaml`)

The cashier service loads **`CASHIER_CONFIG`** (default `./config/cashier_zones.yaml`). On disk, each zone uses **`points` as `[[x, y], …]`** with normalized coordinates in `[0, 1]`. **`POST /cashier/zones`** must send **`points` as `[{"x":…,"y":…}, …]`**; the API writes the nested-list form back to the file. `GET /cashier/zones` returns the merged file (zones, thresholds, and any extra keys such as `buffer`, `debounce`, `evidence`, `gif`, `meta`).

Abbreviated example (see repository file for full values):

```yaml
thresholds:
  config_reload_interval: 60
  customer_wait_max_seconds: 30
  drawer_open_max_seconds: 30
  proximity_iou: 0.05
zones:
  ROI_CASHIER:
    active: true
    shape: polygon
    points:
      - [0.32969, 0.54861]
      - [0.65000, 0.52778]
      - [0.69297, 0.99444]
      - [0.33516, 0.98750]
  ROI_CUSTOMER:
    active: true
    shape: polygon
    points:
      - [0.32031, 0.28472]
      - [0.64297, 0.25556]
      - [0.63516, 0.00000]
      - [0.31172, 0.00000]
buffer:
  jpeg_quality: 75
  size: 100
debounce:
  default: 3
  A3: 1
  A4: 1
evidence:
  save_gif: true
  save_thumbnail: true
  log_rotate_mb: 100
gif:
  fps: 10
  quality: 85
meta:
  version: 1.2.0
```

---

## Complete cURL reference (all HTTP routes)

Set `BASE` to your server root, for example:

```bash
export BASE=http://localhost:9000
# or: export BASE=http://<jetson-ip>:9000
```

### `GET /`

```bash
curl -s "$BASE/"
```

### Cameras — `POST` / `GET` / `DELETE /cameras/{cam_id}`

```bash
curl -s -X POST "$BASE/cameras" \
  -H "Content-Type: application/json" \
  -d '{"cameras":[{"id":"cam1","url":"rtsp://user:pass@192.168.1.10/stream"}]}'

curl -s "$BASE/cameras"

curl -s -X DELETE "$BASE/cameras/cam1"
```

### Tasks — `POST` / `GET` / `PUT` / `DELETE /api/tasks/{task_id}`

Register work per `channelId` (must match a camera `id` from `POST /cameras`). Example: cashier drawer monitor on channel `4`:

```bash
curl -s -X POST "$BASE/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 101,
    "taskName": "cashier_drawer_monitor",
    "algorithmType": "CASHIER_BOX_OPEN",
    "channelId": 4,
    "enable": true,
    "threshold": 50,
    "areaPosition": "[]",
    "detailConfig": {},
    "validWeekday": ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"],
    "validStartTime": 0,
    "validEndTime": 86400000
  }'

curl -s "$BASE/api/tasks"
curl -s "$BASE/api/tasks/101"
```

`algorithmType` must be one of: `CROSS_LINE`, `MASK_HAIRNET_CHEF_HAT`, `CASHIER_BOX_OPEN` (see [`apis/tasks.py`](apis/tasks.py)).

### Detection — `start`, `stop`, `status`, `stream`

There is **no** `POST /detection/setup` in the v2 HTTP API.

```bash
curl -s -X POST "$BASE/detection/start?camera_id=4"
curl -s -X POST "$BASE/detection/start"
curl -s -X POST "$BASE/detection/stop?camera_id=4"
curl -s -X POST "$BASE/detection/stop"

curl -s "$BASE/detection/status"

# SSE: task events (e.g. crossings, PPE violations, cashier frames)
curl -N "$BASE/detection/stream"
# Filtered SSE
curl -N "$BASE/detection/stream?eventType=CROSS_LINE&channelId=1"
```

### Live annotated frame stream — WebSocket

Binary WebSocket stream of JPEG frames with YOLO bounding boxes drawn by FrameBus.
Requires Redis (`REDIS_URL`). One channel per camera.

```bash
# Verify WebSocket handshake (expect HTTP 101 Switching Protocols)
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  -H "Sec-WebSocket-Version: 13" \
  "$BASE/cameras/cam1/live"

# Connect with websocat CLI and see frame sizes
websocat --binary "ws://localhost:9000/cameras/cam1/live" \
  | while IFS= read -r -d '' chunk; do echo "frame ${#chunk} bytes"; done

# JSON event stream over WebSocket
websocat "ws://localhost:9000/cameras/cam1/events"

# Verify Redis is publishing frames
docker exec -it redis redis-cli SUBSCRIBE live:frame:cam1

# Check FrameBus connected
docker compose logs yolo-detect | grep "FrameBus: Redis"
```

**Browser quickstart** (save as `stream.html`, open in any browser):

```html
<!DOCTYPE html><html><body style="background:#111;color:#0f0;text-align:center">
<h3>Live Annotated Stream</h3>
<div id="st">Connecting…</div>
<img id="img" style="max-width:95%;border:2px solid #0f0">
<script>
function connect(id) {
  const ws = new WebSocket(`ws://localhost:9000/cameras/${id}/live`);
  ws.binaryType = "arraybuffer";
  ws.onopen  = () => document.getElementById("st").textContent = "Connected ✓";
  ws.onclose = () => { document.getElementById("st").textContent = "Reconnecting…";
                       setTimeout(() => connect(id), 2000); };
  ws.onerror = () => ws.close();
  ws.onmessage = e => {
    const img = document.getElementById("img");
    URL.revokeObjectURL(img.src);
    img.src = URL.createObjectURL(new Blob([e.data], {type:"image/jpeg"}));
  };
}
connect("cam1");   // change to your camera id
</script></body></html>
```

**Live stream environment variables:**

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection URL |
| `REDIS_LIVE_FPS` | `13` | Target publish rate (fps); FrameBus auto-adjusts per measured camera FPS |
| `LIVE_ANNOTATION_MODE` | `ultralytics` | `ultralytics` (full `plot()`), `opencv` (fast boxes), or `none` (no overlay) for live Redis JPEGs; still uses full plot when `SAVE_OUTPUT` is on |
| `RTSP_BACKEND` | `auto` | `auto` (GStreamer NVDEC on Jetson if available, else Adaptive FFmpeg, else OpenCV), or force `gstreamer` / `ffmpeg` / `opencv` |
| `WS_SEND_TIMEOUT_MS` | `50` | Drop frame if client cannot receive within this many ms |

### Cashier — zones (`GET` / `POST` / `POST …/reset`)

```bash
curl -s "$BASE/cashier/zones"

curl -s "$BASE/cashier/zones" | jq '.thresholds'

curl -s -X POST "$BASE/cashier/zones" \
  -H "Content-Type: application/json" \
  -d '{
    "thresholds": {
      "drawer_open_max_seconds": 45,
      "customer_wait_max_seconds": 45,
      "proximity_iou": 0.06,
      "config_reload_interval": 60
    }
  }'

curl -s -X POST "$BASE/cashier/zones" \
  -H "Content-Type: application/json" \
  -d '{
    "ROI_CASHIER": {
      "shape": "rectangle",
      "points": [{"x": 0.0, "y": 0.0}, {"x": 0.5, "y": 1.0}],
      "active": true
    },
    "ROI_CUSTOMER": {
      "shape": "polygon",
      "points": [
        {"x": 0.5, "y": 0.0},
        {"x": 1.0, "y": 0.0},
        {"x": 1.0, "y": 1.0},
        {"x": 0.5, "y": 1.0}
      ],
      "active": true
    },
    "thresholds": {
      "drawer_open_max_seconds": 30,
      "customer_wait_max_seconds": 30
    }
  }'

curl -s -X POST "$BASE/cashier/zones/reset"
```

Successful `POST /cashier/zones` returns `{"status":"updated","config":{...}}` with the full merged config (including `zones` as `[[x,y],…]`).

### Cashier — `status`, `events`, `evidence`

```bash
curl -s "$BASE/cashier/status"

curl -s "$BASE/cashier/events?severity=ALERT&case_id=A5&camera_id=cam1&limit=50&offset=0"

curl -s -X DELETE "$BASE/cashier/events"

curl -s "$BASE/cashier/evidence?severity=alert&case_id=A5&limit=20"

curl -s -o evidence.jpg "$BASE/cashier/evidence/<relative/path/from/list>"
```

### Cashier — per-camera SSE and media

```bash
curl -N "$BASE/cashier/stream/cam1"
curl -N "$BASE/cashier/stream/cam1/only"

curl -s -o latest.jpg "$BASE/cashier/media/cam1/latest/jpg"
curl -s -o latest.gif "$BASE/cashier/media/cam1/latest/gif"
curl -s -o event.jpg "$BASE/cashier/media/cam1/event/<event_id>/jpg"
curl -s -o event.gif "$BASE/cashier/media/cam1/event/<event_id>/gif"

# Logged "triggered" event count — NOT total frames with drawer open (see docs/CASHIER_BOX_OPEN.md Part III — Drawer metrics)
curl -s "$BASE/cashier/media/cam1/drawer_count"
```

### Cashier — thresholds & timers (A5 / A6)

- **`customer_wait_max_seconds`** → **A5** (customer in customer zone, no cashier, wait exceeds limit).  
- **`drawer_open_max_seconds`** → **A6** (drawer open duration exceeds limit).  

Full **thresholds → logic → output** table: [`docs/CASHIER_BOX_OPEN.md`](docs/CASHIER_BOX_OPEN.md) Part III (*A5 / A6: thresholds → logic → output*).

```bash
curl -s "$BASE/cashier/zones" | jq '.thresholds'
curl -s "$BASE/cashier/zones" | jq '.zones'
```

### After `POST /detection/start` (runtime checks)

```bash
curl -s "$BASE/detection/status"
curl -s "$BASE/cashier/status"
curl -s "$BASE/cashier/events?limit=20"
```

### jq examples (streams)

**`GET /detection/stream`** carries **task events** (e.g. `eventType`, `taskId`, `person`, `line`). Example: print `eventType` and `taskId`:

```bash
curl -sN "$BASE/detection/stream" \
  | grep "^data:" | sed 's/^data: //' \
  | jq '{eventType, taskId, channelId}'
```

**Cashier** — structured frames also appear on **`/detection/stream`** (`eventType=CASHIER_BOX_OPEN`, `data.personStructural`, …). For HTTP status, zones, and per-camera SSE:

```bash
curl -sN "$BASE/detection/stream?eventType=CASHIER_BOX_OPEN" \
  | grep "^data:" | head -1 | sed 's/^data: //' | jq '{eventType, taskId, data}'
curl -s "$BASE/cashier/status" | jq .
curl -s "$BASE/cashier/zones" | jq '.thresholds'
```

For batch / **`pipeline.py`** JSONL that still embeds `data.use_case.cashier`, keep using your offline `jq` filters against those files; the live v2 detection SSE shape differs.

### SSH — tail batch JSONL, evidence log, or app log

Replace `<user>`, `<jetson-ip>`, and the repo path on the device.

**Batch JSONL** (same logical shape as live SSE bodies, without the `data:` prefix):

```bash
# Full-dataset run (includes summary.json in repo when present)
ssh <user>@<jetson-ip> "tail -f /path/to/ml-server/outputs/cashier_test/20260329T060741_7464/stream.jsonl"

# Local cashier-YOLO batch (multiple_persons frames)
ssh <user>@<jetson-ip> "tail -f /path/to/ml-server/outputs/cashier_test/20260329T135320_10106/stream.jsonl"

# First-70-frames batch with updated zones
ssh <user>@<jetson-ip> "tail -f /path/to/ml-server/outputs/test_70/*.json"
```

**Cashier structured task log** (one JSON line per processed frame; set `EVENTS_DIR` if not using the default):

```bash
ssh <user>@<jetson-ip> "tail -f /local/storage/events/task_101.jsonl"
```

**Application log** (if your deployment writes here):

```bash
ssh <user>@<jetson-ip> "tail -f /path/to/ml-server/logger/app.log"
```

More detail: [`docs/CASHIER_BOX_OPEN.md`](docs/CASHIER_BOX_OPEN.md) (Part III — cashier cURL), SSE examples: [`sse_cashier.md`](sse_cashier.md).

---

## API Endpoints Reference

Typical **v2** lifecycle:

1. **`POST /cameras`** — register RTSP URLs (camera `id` strings must match task `channelId` values).
2. **`POST /api/tasks`** — register enabled tasks per channel (`algorithmType` + config).
3. *(Optional)* **`POST /cashier/zones`** — tune cashier ROIs/thresholds (`CASHIER_CONFIG`).
4. **`POST /detection/start`** — spawn FrameBus + task workers for enabled tasks.
5. **`GET /detection/stream`** — SSE for task events (e.g. crossings). **`GET /cashier/stream/{camera_id}`** — SSE for cashier frames/alerts.
*(Alternatively, you can run it via Docker Compose if standard deployment is configured).*

---

## 📖 API Endpoints Reference

The application lifecycle works as follows:

1. Register Cameras.
2. Setup the Pipeline Services.
3. Start the Pipeline processing.
4. Consume the Detection Stream (SSE).

---

### **Health Check**

#### `GET /`

Returns basic service information.

**Sample Response**

```json
{
  "service": "Vision Pipeline API",
  "version": "2.0.0"
}
```

---

### **Camera Management Routes**

#### `POST /cameras`

Configure one or multiple cameras.

**Sample Request**

```json
{
  "cameras": [
    {
      "id": "cam1",
      "url": "rtsp://username:password@10.0.0.5:554/stream1"
    }
  ]
}
```

**Sample Response** (envelope may include `"error": null`)

**Sample Response**

```json
{
  "status": "configured",
  "cameras": {
    "cam1": "rtsp://username:password@10.0.0.5:554/stream1"
  },
  "error": null
}
```

#### `GET /cameras`

List all currently configured cameras.

**Sample Response**

```json
{
  "count": 1,
  "cameras": [
    {
      "id": "cam1",
      "url": "rtsp://username:password@10.0.0.5:554/stream1"
    }
  ],
  "error": null
}
```

#### `DELETE /cameras/{cam_id}`

Delete a configured camera.

**Sample Response**

```json
{
  "status": "removed",
  "camera_id": "cam1",
  "remaining": []
}
```

---

### **Task routes** (`/api/tasks`)

Tasks are stored in memory and drive which algorithms run after `POST /detection/start`. Each task includes `taskId`, `taskName`, `algorithmType`, `channelId`, `enable`, `threshold`, `areaPosition` (JSON string; lines or polygons depending on algorithm), `detailConfig`, and schedule fields.

Supported `algorithmType` values: **`CROSS_LINE`**, **`MASK_HAIRNET_CHEF_HAT`**, **`CASHIER_BOX_OPEN`**.

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/tasks` | Create or replace a task |
| GET | `/api/tasks` | List all tasks |
| GET | `/api/tasks/{task_id}` | Get one task |
| PUT | `/api/tasks/{task_id}` | Update a task |
| DELETE | `/api/tasks/{task_id}` | Remove a task |

Implementation: [`apis/tasks.py`](apis/tasks.py). Worker dispatch: [`task_worker.py`](task_worker.py) + [`services/__init__.py`](services/__init__.py) `TASK_REGISTRY`.

---

### **Detection routes** (runtime + SSE)

### **Detection Pipeline Routes**

#### `POST /detection/setup`

Configure which models/services will run in the pipeline.

**Sample Request**

```json
{
  "pipeline": ["detector", "age_gender", "mood","ppe"]
}
```

**Sample Response**

```json
{
  "status": "configured",
  "pipeline": ["detector", "age_gender", "mood","ppe"]
}
```

#### `POST /detection/start`

Starts **FrameBus** processes (one per channel with enabled tasks) and **task worker** processes. Optional query `camera_id` filters to tasks for that channel only.

Start processing cameras using the configured pipeline. You can optionally specify a `camera_id` as a query parameter string. If not specified, it starts all configured cameras.

**Sample Request**

`POST /detection/start?camera_id=4`

**Sample Response**

```json
{
  "status": "started",
  "cameras": ["4"],
  "tasks": ["101"]
}
```

#### `POST /detection/stop`

Stops running buses/workers. Without `camera_id`, stops all. With `camera_id`, stops that channel only.

Stop camera streams. You can optionally specify a `camera_id` as a query parameter string. If not specified, it stops all running cameras.

**Sample Request**

`POST /detection/stop?camera_id=4`

**Sample Response**

```json
{
  "status": "stopped",
  "cameras": ["4"]
}
```

#### `GET /detection/status`

Operational status per camera (FPS, frame counts, errors) from shared state updated by FrameBus.

Returns the operational status, FPS, and detection counts for all cameras.

**Sample Response**

```json
{
  "cameras": {
    "4": {
      "camera_id": "4",
      "rtsp_url": "rtsp://10.0.0.5:554/stream1",
      "running": true,
      "frame_count": 420,
      "fps": 28.5,
      "last_detections": 3,
      "total_detections": 1200,
      "uptime_seconds": 14.7,
      "error": null
    }
  }
}
```

#### `GET /detection/stream`

An SSE (Server-Sent Events) endpoint that yields one JSON payload per frame, combining inferences from all running cameras.

SSE endpoint: each `data:` line is one **task-produced event** (for example a `CROSS_LINE` crossing). Cashier-specific live output is on **`GET /cashier/stream/{camera_id}`**.

**Example `data:` line (illustrative crossing event)**

**Sample Response stream**

```json
{
  "eventId": "…",
  "eventType": "CROSS_LINE",
  "timestamp": 1774310401528,
  "taskId": 13,
  "taskName": "customer_walkin_main",
  "channelId": 4,
  "line": {"id": "1", "name": "Entrance", "direction": 1},
  "person": {"trackingId": 1, "boundingBox": {}, "attributes": {}, "confidence": 0.9},
  "evidence": {
    "captureImage": { "url": "…", "path": "2026-04-22/cam-1_…_….jpg", "type": "capture", "format": "image/jpeg", "timestamp": "2026-04-22T12:00:00.000Z" },
    "sceneImage": { "url": "…", "path": "2026-04-22/cam-1_…_….jpg", "type": "scene", "format": "image/jpeg", "timestamp": "2026-04-22T12:00:00.000Z" }
  }
}
```

Task JSONL on disk: `$EVENTS_DIR/task_<taskId>.jsonl`. Full **ML Image V2** + SSH: [service_doc/ml_image_v2.md](service_doc/ml_image_v2.md).

---

### **Legacy pipeline JSON** (`pipeline.py` / batch)

The following structure applies to **offline or script-driven** runs that use `CameraPipeline` and `services.REGISTRY` (not the v2 `/detection/stream` SSE). Each result may include a legacy base64 field **`frame`**, plus optional **`images`** (V2) when `PIPELINE_IMAGE_MODE` is not `none` (`full` = include `base64` under `raw` / `annotated`; `light` = `type` only; `none` = omit `images`):

```json
{
  "camera_id": "main_room",
  "frame_count": 5,
  "timestamp": "2026-03-17T12:50:23+00:00",
  "frame": "<base64 JPEG raw pre-annotation>",
  "images": {
    "raw": { "type": "raw", "base64": "..." },
    "annotated": { "type": "annotated", "base64": "..." }
  },
  "data": {
    "detection": {
      "count": 1,
      "items": [{
        "bbox": [120, 80, 340, 420],
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.91
      }]
    },
    "use_case": {
      "age_gender": [],
      "mood": [],
      "ppe": []
    }
  }
}
```

### POST `/person_search/search`

Upload an image to find the top **K** similar identities using image-based person re-identification (OSNet) from the Qdrant vector database.  
This is used to track a specific person across different cameras and frames.

#### Parameters (Multipart Form Data)

- `file`: Image file containing the person to search for (`UploadFile`).
- `top_k` *(optional)*: Integer — number of top matches to return. Default is `10`.

#### Sample Response

```json
{
  "status": "success",
  "count": 3,
  "results": [
    {
      "score": 0.6129,
      "metadata": {
        "type": "person_search",
        "camera_id": "test_cam",
        "track_id": 13,
        "image_path": "/local/storage/gallery/crops/test_cam/track_13.jpg",
        "frame_id": 13,
        "confidence": 0.7773560881614685,
        "timestamp": "2026-04-14T11:23:05.384983"
      }
    },
    {
      "score": 0.5816,
      "metadata": {
        "type": "person_search",
        "camera_id": "test_cam",
        "track_id": 37,
        "image_path": "/local/storage/gallery/crops/test_cam/track_37.jpg",
        "frame_id": 15,
        "confidence": 0.496114581823349,
        "timestamp": "2026-04-14T11:23:05.684256"
      }
    },
    {
      "score": 0.5759,
      "metadata": {
        "type": "person_search",
        "camera_id": "test_cam",
        "track_id": 5,
        "image_path": "/local/storage/gallery/crops/test_cam/track_5.jpg",
        "frame_id": 1,
        "confidence": 0.7277824878692627,
        "timestamp": "2026-04-14T11:22:59.393466"
      }
    }
  ]
}
```

### POST `/semantic_search/search`

Upload an image to retrieve the top **K** semantically similar objects or scenes using image embeddings (mobileclip2-s0) from the Qdrant vector database.

#### Parameters (Multipart Form Data)

- `file`: Image file to search with (`UploadFile`).
- `top_k` *(optional)*: Integer — number of top matches to return. Default is `10`.

#### Sample Response

```json
{
  "status": "success",
  "count": 3,
  "results": [
    {
      "score": 0.2387,
      "metadata": {
        "type": "semantic_search",
        "camera_id": "test_cam",
        "track_id": 5,
        "image_path": "/local/storage/gallery/crops/test_cam/track_5.jpg",
        "frame_id": 1,
        "confidence": 0.7277824878692627,
        "timestamp": "2026-04-14T11:22:59.393466"
      }
    },
    {
    "score": 0.2387,
      "metadata": {
        "type": "semantic_search",
        "camera_id": "test_cam",
        "track_id": 5,
        "image_path": "/local/storage/gallery/crops/test_cam/track_5.jpg",
        "frame_id": 1,
        "confidence": 0.7277824878692627,
        "timestamp": "2026-04-14T11:22:59.393466"
      }
    },
    {
     "score": 0.2387,
      "metadata": {
        "type": "semantic_search",
        "camera_id": "test_cam",
        "track_id": 5,
        "image_path": "/local/storage/gallery/crops/test_cam/track_5.jpg",
        "frame_id": 1,
        "confidence": 0.7277824878692627,
        "timestamp": "2026-04-14T11:22:59.393466"
    }
  ]
}
      "age_gender": [{
        "bbox": [120, 80, 340, 420],
        "gender": "Female",
        "age_group": "MiddleAged",
        "confidence": 0.87
      }],
      "mood": [{
        "bbox": [120, 80, 340, 420],
        "mood": "Happy",
        "confidence": 0.94
      }],
      "ppe": [
      {
        "person_bbox": [
          677,
          305,
          850,
          523
        ],
        "count": 2,
        "items": [
          {
            "class_id": 2,
            "class_name": "gloves",
            "confidence": 0.9647,
            "x1": 742,
            "y1": 480,
            "x2": 805,
            "y2": 527
          },
          {
            "class_id": 1,
            "class_name": "hairnet",
            "confidence": 0.8999,
            "x1": 769,
            "y1": 312,
            "x2": 863,
            "y2": 374
          }
        ]
    }]
      }
    }
  }
```

**Sample Response stream (Face Recognition Task Event)**

```json
data: {
  "eventId": "a1b2c3d4e5f6...",
  "eventType": "FACE",
  "timestamp": 1774310401528,
  "timestampUTC": "2026-04-02T10:00:01Z",
  "taskId": 30,
  "taskName": "attendance",
  "channelId": 1,
  "person": {
    "id": 101,
    "name": "John Doe",
    "isStranger": false,
    "trackingId": "15",
    "confidence": 88
  },
  "face": {
    "quality": 85.2,
    "yaw": -5.1,
    "score": 82.5,
    "failCount": 0
  },
  "evidence": {
    "captureImage": "/local/storage/captures/2026/04/02/a1b2_crop.jpg",
    "sceneImage": "/local/storage/scenes/2026/04/02/a1b2_scene.jpg"
  }
}
```

---

## ✅ Testing & Real Result Examples

### Test Results Overview

#### Test 1: Sample Image (zidane.jpg)

- **Detections**: 2 Persons
- **Inference Time**: ~2 seconds
- **Confidence Scores**: 81.9% - 83.6%

#### Test 2: F1 Image (f1.webp) - Complete Test Results

**Input Image**: `f1.webp` (WebP format, 33 KB)

- **Local annotated version (provided)**: `/home/a7med/Downloads/f1a.png`

**Annotated Output Image with All Detections:**

![F1 Annotated Image with Detections - 500x281 pixels](outputs/f1_test/annotated.jpg)

**Visual Elements in Image:**

- Green bounding boxes around detected persons
- Blue bounding box around detected chair
- Labels showing: Class, Confidence %, Gender | Age Group, and Mood/Emotion
- All coordinates and detection metadata embedded

**Detection Summary:**

```
Total Objects Detected: 4

Person 1 (Top-Left Area)
├─ Bounding Box: [289, 71, 458, 257]
├─ Confidence: 89.01%
├─ Gender: Male
├─ Age Group: MiddleAged
├─ Mood: Angry
└─ Mood Confidence: 51.92%

Person 2 (Left Side)
├─ Bounding Box: [125, 70, 268, 220]
├─ Confidence: 84.15%
├─ Gender: Female
├─ Age Group: Senior
├─ Mood: Angry
└─ Mood Confidence: 45.12%

Person 3 (Center)
├─ Bounding Box: [209, 73, 301, 217]
├─ Confidence: 83.13%
├─ Gender: Male
├─ Age Group: Senior
├─ Mood: Happy
└─ Mood Confidence: 47.54%

Chair (Right Side)
├─ Bounding Box: [353, 136, 458, 259]
├─ Confidence: 52.47%
└─ Class ID: 56
```

**Performance Metrics:**

| Metric | Value |
|--------|-------|
| Total Processing Time | ~3 seconds |
| Image Resolution | 500×281 pixels |
| Number of Detections | 4 |
| Average Confidence | 82.9% |
| Age/Gender Confidence | 56-75% |
| Mood Detection Confidence | 45-52% |
| Annotated Image Size | 85 KB |
| Output JSON Size | 64 KB |

---

### Complete JSON Response Example (F1 Test)

**Raw API Response Structure:**

```json
{
  "camera_id": "f1_image",
  "frame_count": 1,
  "timestamp": "2026-03-24T09:29:31.395013Z",
  "frame": "<base64_encoded_jpeg_string_containing_annotated_image>",
  "data": {
    "detection": {
      "count": 4,
      "items": [
        {
          "bbox": [289, 71, 458, 257],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.8901,
          "center": [373, 164],
          "width": 169,
          "height": 186
        },
        {
          "bbox": [125, 70, 268, 220],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.8415,
          "center": [196, 145],
          "width": 143,
          "height": 150
        },
        {
          "bbox": [209, 73, 301, 217],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.8313,
          "center": [255, 145],
          "width": 92,
          "height": 144
        },
        {
          "bbox": [353, 136, 458, 259],
          "class_id": 56,
          "class_name": "chair",
          "confidence": 0.5247,
          "center": [405, 197],
          "width": 105,
          "height": 123
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {
          "bbox": [289, 71, 458, 257],
          "gender": "Male",
          "age_group": "MiddleAged",
          "confidence": 0.5693
        },
        {
          "bbox": [125, 70, 268, 220],
          "gender": "Female",
          "age_group": "Senior",
          "confidence": 0.5635
        },
        {
          "bbox": [209, 73, 301, 217],
          "gender": "Male",
          "age_group": "Senior",
          "confidence": 0.7525
        },
        {
          "bbox": [353, 136, 458, 259],
          "gender": "Male",
          "age_group": "Senior",
          "confidence": 0.5758
        }
      ],
      "mood": [
        {
          "bbox": [289, 71, 458, 257],
          "mood": "Angry",
          "confidence": 0.5192
        },
        {
          "bbox": [125, 70, 268, 220],
          "mood": "Angry",
          "confidence": 0.4512
        },
        {
          "bbox": [209, 73, 301, 217],
          "mood": "Happy",
          "confidence": 0.4754
        },
        {
          "bbox": [353, 136, 458, 259],
          "mood": "Angry",
          "confidence": 0.4556
        }
      ]
    }
  }
}
```

**JSON Field Reference:**

- **camera_id**: Identifier of the source camera or image
- **frame_count**: Frame number in stream (1 for single image)
- **timestamp**: ISO 8601 timestamp of processing
- **frame**: Base64-encoded JPEG containing annotated image (with all boxes, labels, confidence scores)
- **detection.count**: Total number of detected objects
- **bbox**: `[x1, y1, x2, y2]` - top-left (x1,y1) and bottom-right (x2,y2) pixel coordinates
- **center**: `[cx, cy]` - center point of bounding box
- **class_id**: COCO dataset class ID (0 = person, 56 = chair, etc.)
- **confidence**: Detection confidence score (0.0-1.0)
- **gender**: Predicted gender ("Male" or "Female")
- **age_group**: Predicted age category ("Young", "MiddleAged", "Senior", "Elderly")
- **mood**: Detected emotion ("Angry", "Happy", "Neutral")

---

HTTP examples for every route are in **[Complete cURL reference (all HTTP routes)](#complete-curl-reference-all-http-routes)** above. This repository’s running app exposes **no** `/process`, `/stream` (POST), `/mood`, `/age-gender`, `/health`, or **`POST /detection/setup`** — use **`/cameras`**, **`/api/tasks`**, **`/detection/*`**, and **`/cashier/*`** as documented.

---

## Cashier full-dataset validation report

**Run ID:** `20260329T060741_7464`  
**Authoritative metrics:** [`outputs/cashier_test/20260329T060741_7464/summary.json`](outputs/cashier_test/20260329T060741_7464/summary.json) (aggregated from [`stream.jsonl`](outputs/cashier_test/20260329T060741_7464/stream.jsonl)).

### Executive summary

| Metric | Value |
|--------|------:|
| Total frames processed | 22,537 |
| Wall-clock span (first → last record timestamp) | 4,798.03 s (**1h 19m 58s**) |
| Average throughput | **4.6971 FPS** (frames ÷ duration) |
| **Total times drawer open (frame count)** | **0** |

The **drawer-open frame count** is the number of frames whose `case_id` is one of **N3, N6, A1, A3, A4, A6** (cases where the cashier-zone drawer is treated as open in the business logic). For this dataset run, no frames fell into those classes, so the aggregate is zero.

### Severity breakdown (observed)

| Severity | Frames | Share of total |
|----------|-------:|---------------:|
| NORMAL | 20,626 | 91.52% |
| ALERT | 1,911 | 8.48% |
| CRITICAL | 0 | 0.00% |

### Case distribution (N1–N6, A1–A7)

Counts are **per-frame classifications** over the full run. Cases not observed are listed as **0**.

| case_id | Severity (typical) | Frames |
|---------|-------------------|-------:|
| N1 | NORMAL | 18,058 |
| N2 | NORMAL | 2,568 |
| N3 | NORMAL | 0 |
| N4 | NORMAL | 0 |
| N5 | NORMAL | 0 |
| N6 | NORMAL | 0 |
| A1 | ALERT / CRITICAL | 0 |
| A2 | ALERT | 68 |
| A3 | CRITICAL | 0 |
| A4 | CRITICAL | 0 |
| A5 | ALERT | 1,843 |
| A6 | ALERT | 0 |
| A7 | ALERT | 0 |

### Visual evidence

**Animated summary (GIF)** — when generated by your batch pipeline, place it next to the run artifacts:

![Cashier test run animation](outputs/cashier_test/20260329T060741_7464/cashier_test.gif)

*If the file is not present in the repository yet, add `cashier_test.gif` under the run directory after export from annotated frames or your tooling.*

**Representative stills from this run** (annotated / evidence paths under `outputs/cashier_test/20260329T060741_7464/`):

| Case | Role | Example path |
|------|------|----------------|
| N1 | Idle | [`evidence/normal/N1/frame_000000_time_00m00s_1of3.jpg`](outputs/cashier_test/20260329T060741_7464/evidence/normal/N1/frame_000000_time_00m00s_1of3.jpg) |
| N2 | Cashier only | [`evidence/normal/N2/frame_000465_time_02m35s_1of3.jpg`](outputs/cashier_test/20260329T060741_7464/evidence/normal/N2/frame_000465_time_02m35s_1of3.jpg) |
| A5 | Customer waiting | [`evidence/alert/A5/frame_000386_time_02m08s_3of3.jpg`](outputs/cashier_test/20260329T060741_7464/evidence/alert/A5/frame_000386_time_02m08s_3of3.jpg) |
| A2 | Unexpected persons in cashier zone | [`evidence/alert/A2/frame_000404_time_02m14s_3of3.jpg`](outputs/cashier_test/20260329T060741_7464/evidence/alert/A2/frame_000404_time_02m14s_3of3.jpg) |

**Critical cases (A3, A4, …)** — no CRITICAL frames in this run; when they occur, evidence follows the same layout, e.g. `evidence/critical/A3/…`, `evidence/critical/A4/…`.

![N1 idle](outputs/cashier_test/20260329T060741_7464/evidence/normal/N1/frame_000000_time_00m00s_1of3.jpg)

![N2 cashier only](outputs/cashier_test/20260329T060741_7464/evidence/normal/N2/frame_000465_time_02m35s_1of3.jpg)

![A5 customer waiting](outputs/cashier_test/20260329T060741_7464/evidence/alert/A5/frame_000386_time_02m08s_3of3.jpg)

![A2 unexpected in cashier zone](outputs/cashier_test/20260329T060741_7464/evidence/alert/A2/frame_000404_time_02m14s_3of3.jpg)

### Artifacts for this run

- [`stream.jsonl`](outputs/cashier_test/20260329T060741_7464/stream.jsonl) — one JSON record per frame  
- [`events/`](outputs/cashier_test/20260329T060741_7464/events/) — per-frame JSON mirrors  
- [`annotated/`](outputs/cashier_test/20260329T060741_7464/annotated/) — annotated JPEGs  
- [`evidence/`](outputs/cashier_test/20260329T060741_7464/evidence/) — normal / alert / critical evidence by case  
- [`summary.json`](outputs/cashier_test/20260329T060741_7464/summary.json) — aggregates documented above  

---

## Complete Cashier Pipeline Lifecycle (v2)

Recommended order:

1. **`POST /cameras`** — register RTSP sources; use camera `id` values that match task `channelId` (e.g. `"4"` ↔ `channelId: 4`).  
2. **`POST /api/tasks`** — add an enabled task with `"algorithmType": "CASHIER_BOX_OPEN"` for that `channelId`.  
3. **`POST /cashier/zones`** (optional) — zones/thresholds; defaults from `CASHIER_CONFIG`.  
4. **`GET /cashier/zones`** (optional) — verify merged YAML on disk.  
5. Set **`YOLO_MODEL`** (and env) so FrameBus outputs cashier classes on that deployment.  
6. **`POST /detection/start`** — start FrameBus + workers.  
7. **`GET /cashier/status`**, **`GET /cashier/stream/{camera_id}`**, **`GET /cashier/events`** — cashier HTTP/SSE. **`GET /detection/stream`** carries **all** task events, including **`CASHIER_BOX_OPEN`** structured frames (`eventType`, `data`, …) alongside crossings and PPE.

Copy-paste **`curl`**: [Complete cURL reference (all HTTP routes)](#complete-curl-reference-all-http-routes). SSH tail examples are in that section.

---

## SSE / SSH Stream Logs (Cashier + Mood + Age/Gender)

### Stream event sources

**Full dataset batch (2026-03-29)** — Line-delimited JSON is archived as [`outputs/cashier_test/20260329T060741_7464/stream.jsonl`](outputs/cashier_test/20260329T060741_7464/stream.jsonl). **Additional batches:** [`outputs/cashier_test/20260329T135320_10106/stream.jsonl`](outputs/cashier_test/20260329T135320_10106/stream.jsonl) (local cashier-YOLO run) and [`outputs/test_70`](outputs/test_70) (first 70 `multiple_persons` frames with updated ROI polygons).

## 🛠️ cURL Testing Examples

### Face Recognition Workflow (v2 API)

#### 1. Setup Face Library and Add Persons

```bash
# Create Library
curl -X POST "http://localhost:9000/api/face/lib" -F "lib_id=1" -F "name=Office"

# Register an Employee
curl -X POST "http://localhost:9000/api/face/lib/1/persons" \
  -F "person_id=101" -F "name=John Doe" \
  -F "images=@/path/to/john_face.jpg"
```

#### 2. Register Face Task

```bash
curl -X POST "http://localhost:9000/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 30,
    "taskName": "attendance",
    "algorithmType": "FACE",
    "channelId": 1,
    "threshold": 70,
    "libIds": "1",
    "enableStranger": true,
    "detailConfig": {
      "facePixelSize": 60,
      "yawThreshold": 35,
      "pitchThreshold": 25,
      "failCount": 2
    }
  }'
```

#### 3. Quick Single-Image Testing (No Pipeline)

```bash
curl -X POST "http://localhost:9000/api/face/recognize" \
  -F "image=@employee_test.jpg" \
  -F "lib_ids=1" \
  -F "threshold=70"
```

### Quick Start - Test with Image File (Legacy Pipeline)

#### Test Single Image

```bash
ssh <user>@<jetson-ip> "tail -f /path/to/ml-server/outputs/cashier_test/20260329T060741_7464/stream.jsonl"
ssh <user>@<jetson-ip> "tail -f /path/to/ml-server/outputs/cashier_test/20260329T135320_10106/stream.jsonl"
```

**Earlier simulation (reference still)** — Annotated frame from the production simulation run:

- Image: `outputs/cashier_production_sim/20260326T050929Z/annotated/frame_096720.jpg`

![Annotated Frame Used For Stream Example](outputs/cashier_production_sim/20260326T050929Z/annotated/frame_096720.jpg)

### SSE event shape with `cashier` in `use_case`

The `cashier` block inside `use_case` follows this structure:

```json
{
  "input": {
    "path": "/home/a7med/Documents/all_original_frames/Cashier Drawer/others/frame_097440.jpg",
    "name": "frame_097440.jpg"
  },
  "outputs": {
    "annotated_image": "outputs/cashier_aged_gender_single/20260326T064528Z/annotated/frame_097440.jpg"
  },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": {
      "count": 2,
      "items": [
        {
          "bbox": [
            684,
            0,
            1169,
            275
          ],
          "class_id": 0,
          "class_name": "Person",
          "confidence": 0.8949,
          "center": [
            926,
            137
          ],
          "width": 485,
          "height": 275
        },
        {
          "bbox": [
            811,
            874,
            1264,
            1080
          ],
          "class_id": 0,
          "class_name": "Person",
          "confidence": 0.8725,
          "center": [
            1037,
            977
          ],
          "width": 453,
          "height": 206
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {
          "bbox": [
            684,
            0,
            1169,
            275
          ],
          "gender": "Female",
          "age_group": "Senior",
          "confidence": 0.5035
        },
        {
          "bbox": [
            811,
            874,
            1264,
            1080
          ],
          "gender": "Male",
          "age_group": "Senior",
          "confidence": 0.742
        }
      ],
      "cashier": {
        "persons": [
          {
            "person_bbox": [
              811,
              874,
              1264,
              1080
            ],
            "confidence": 0.8725,
            "zone": "ROI_CASHIER",
            "transaction": true,
            "items": {
              "drawers": [
                {
                  "bbox": [
                    845,
                    692,
                    1186,
                    922
                  ],
                  "confidence": 0.87
                }
              ],
              "cash": [
                {
                  "bbox": [
                    858,
                    727,
                    934,
                    832
                  ],
                  "confidence": 0.85
                },
                {
                  "bbox": [
                    1038,
                    697,
                    1107,
                    804
                  ],
                  "confidence": 0.85
                },
                {
                  "bbox": [
                    973,
                    705,
                    1045,
                    811
                  ],
                  "confidence": 0.85
                },
                {
                  "bbox": [
                    1100,
                    690,
                    1168,
                    800
                  ],
                  "confidence": 0.85
                }
              ]
            }
          },
          {
            "person_bbox": [
              684,
              0,
              1169,
              275
            ],
            "confidence": 0.8949,
            "zone": "ROI_CUSTOMER",
            "transaction": false,
            "items": {
              "drawers": [],
              "cash": []
            }
          }
        ],
        "summary": {
          "cashier_zone": {
            "persons": 1,
            "drawers": 1,
            "cash": 4
          },
          "customer_zone": {
            "persons": 1,
            "drawers": 0,
            "cash": 0
          },
          "case_id": "N3",
          "severity": "NORMAL",
          "alerts": [
            "N3 EVENT: Transaction in progress"
          ],
          "transaction": true,
          "frame_saved": true,
          "evidence_path": "/home/a7med/ml-server/outputs/cashier_aged_gender_single/20260326T064528Z/evidence/normal/N3/cam_20260326T064542_598744.jpg",
          "frame_id": 1,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": [
            {
              "bbox": [
                811,
                874,
                1264,
                1080
              ],
              "confidence": 0.8725,
              "zone": "ROI_CASHIER"
            }
          ]
        }
      }
    }
  }
}
```

**Cashier Field Reference:**

- **persons**: Array of all detected persons with their zone assignment and associated items
  - **person_bbox**: `[x1, y1, x2, y2]` bounding box of the person
  - **zone**: Either `ROI_CASHIER` (behind the register) or `ROI_CUSTOMER` (customer-facing side)
  - **transaction**: `true` if this person is involved in an active transaction
  - **items.drawers**: List of open drawer detections associated with this person
  - **items.cash**: List of cash detections associated with this person
- **summary**: Aggregated zone counts and case classification
  - **cashier_zone / customer_zone**: Count of `persons`, `drawers`, and `cash` per zone
  - **case_id**: Classified scenario (see case table below)
  - **severity**: `NORMAL`, `ALERT`, or `CRITICAL`
  - **alerts**: List of human-readable alert messages for this frame
  - **transaction**: `true` if an active transaction is in progress
  - **frame_saved**: `true` if an evidence frame was written to disk
  - **evidence_path**: Absolute path to the saved evidence image (null if not saved)
  - **cashier_persons**: Subset list of persons confirmed to be in the cashier zone

---

### Cashier Case IDs (All Cases)

#### Normal Cases

| Case | Severity | Condition |
|------|----------|-----------|
| **N1** | NORMAL | Idle register — nothing happening |
| **N2** | NORMAL | Cashier on duty, no customer |
| **N3** | NORMAL | Active transaction (cashier + drawer + customer + cash, nearby) |
| **N4** | NORMAL | Staff handover / supervisor at register (2 persons near drawer) |
| **N5** | NORMAL | Customer waiting (no transaction started) |
| **N6** | NORMAL | Drawer open, no cash (card transaction / float check) |

#### Alert Cases

| Case | Severity | Condition |
|------|----------|-----------|
| **A1** | ALERT / CRITICAL | Unattended open drawer (CRITICAL if customer present) |
| **A2** | ALERT | Unexpected person in cashier zone |
| **A3** | CRITICAL | Cash + open drawer, no cashier (theft signature) |
| **A4** | CRITICAL | Unauthorised person at open register with cash |
| **A5** | ALERT | Customer waiting too long (>30s) without cashier |
| **A6** | ALERT | Drawer open too long (>30s) |
| **A7** | ALERT | Cash in customer zone, no cashier present |

---

### Target Format Reference: Cashier Test Cases (Normal & Alert)

All cashier test cases (both normal operations and anomalies) yield an annotated image mapped to a detailed `result.json` log. The structure below demonstrates exactly how the response block changes for every single case:

#### N1 (Idle register — nothing happening)

```json
{
  "input": { "path": "...", "name": "frame_N1.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_N1.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 0, "items": [] },
    "use_case": {
      "cashier": {
        "persons": [],
        "summary": {
          "cashier_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "customer_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "case_id": "N1",
          "severity": "NORMAL",
          "alerts": [],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/normal/N1/cam_N1.jpg",
          "frame_id": 1,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": []
        }
      }
    }
  }
}
```

#### N2 (Cashier on duty, no customer)

```json
{
  "input": { "path": "...", "name": "frame_N2.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_N2.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 1, "items": [{ "class_name": "Person" }] },
    "use_case": {
      "cashier": {
        "persons": [{ "zone": "ROI_CASHIER", "transaction": false, "items": { "drawers": [], "cash": [] } }],
        "summary": {
          "cashier_zone": { "persons": 1, "drawers": 0, "cash": 0 },
          "customer_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "case_id": "N2",
          "severity": "NORMAL",
          "alerts": [],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/normal/N2/cam_N2.jpg",
          "frame_id": 2,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": [{ "zone": "ROI_CASHIER" }]
        }
      }
    }
  }
}
```

#### N3 (Active transaction — e.g. frame_097440.jpg)

```json
{
  "input": { "path": "...", "name": "frame_097440.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_097440.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 6, "items": [{ "class_name": "Person" }, { "class_name": "Person" }, "..."] },
    "use_case": {
      "cashier": {
        "persons": [
          { "zone": "ROI_CASHIER", "transaction": true, "items": { "drawers": [{...}], "cash": [{...}] } },
          { "zone": "ROI_CUSTOMER", "transaction": false, "items": { "drawers": [], "cash": [] } }
        ],
        "summary": {
          "cashier_zone": { "persons": 1, "drawers": 1, "cash": 4 },
          "customer_zone": { "persons": 1, "drawers": 0, "cash": 0 },
          "case_id": "N3",
          "severity": "NORMAL",
          "alerts": ["N3 EVENT: Transaction in progress"],
          "transaction": true,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/normal/N3/cam_N3.jpg",
          "frame_id": 3,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": [{ "zone": "ROI_CASHIER" }]
        }
      }
    }
  }
}
```

#### N4 (Staff handover / supervisor at register)

```json
{
  "input": { "path": "...", "name": "frame_N4.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_N4.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 2, "items": [{ "class_name": "Person" }, { "class_name": "Person" }] },
    "use_case": {
      "cashier": {
        "persons": [
          { "zone": "ROI_CASHIER", "transaction": false, "items": { "drawers": [], "cash": [] } },
          { "zone": "ROI_CASHIER", "transaction": false, "items": { "drawers": [], "cash": [] } }
        ],
        "summary": {
          "cashier_zone": { "persons": 2, "drawers": 0, "cash": 0 },
          "customer_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "case_id": "N4",
          "severity": "NORMAL",
          "alerts": ["N4 EVENT: Multiple staff at register"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/normal/N4/cam_N4.jpg",
          "frame_id": 4,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": [{ "zone": "ROI_CASHIER" }, { "zone": "ROI_CASHIER" }]
        }
      }
    }
  }
}
```

#### N5 (Customer waiting without transaction)

```json
{
  "input": { "path": "...", "name": "frame_N5.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_N5.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 1, "items": [{ "class_name": "Person" }] },
    "use_case": {
      "cashier": {
        "persons": [{ "zone": "ROI_CUSTOMER", "transaction": false, "items": { "drawers": [], "cash": [] } }],
        "summary": {
          "cashier_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "customer_zone": { "persons": 1, "drawers": 0, "cash": 0 },
          "case_id": "N5",
          "severity": "NORMAL",
          "alerts": ["N5 EVENT: Customer waiting"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/normal/N5/cam_N5.jpg",
          "frame_id": 5,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": []
        }
      }
    }
  }
}
```

#### N6 (Drawer open, no cash — card transaction)

```json
{
  "input": { "path": "...", "name": "frame_N6.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_N6.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 2, "items": [{ "class_name": "Person" }, { "class_name": "drawer_open" }] },
    "use_case": {
      "cashier": {
        "persons": [{ "zone": "ROI_CASHIER", "transaction": true, "items": { "drawers": [{...}], "cash": [] } }],
        "summary": {
          "cashier_zone": { "persons": 1, "drawers": 1, "cash": 0 },
          "customer_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "case_id": "N6",
          "severity": "NORMAL",
          "alerts": [],
          "transaction": true,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/normal/N6/cam_N6.jpg",
          "frame_id": 6,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": [{ "zone": "ROI_CASHIER" }]
        }
      }
    }
  }
}
```

#### A1 (Unattended open drawer)

```json
{
  "input": { "path": "...", "name": "frame_A1.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_A1.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 1, "items": [{ "class_name": "drawer_open" }] },
    "use_case": {
      "cashier": {
        "persons": [],
        "summary": {
          "cashier_zone": { "persons": 0, "drawers": 1, "cash": 0 },
          "customer_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "case_id": "A1",
          "severity": "ALERT",
          "alerts": ["A1 WARNING: Unattended open drawer"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/alert/A1/cam_A1.jpg",
          "frame_id": 7,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": []
        }
      }
    }
  }
}
```

#### A2 (Unexpected person in cashier zone)

```json
{
  "input": { "path": "...", "name": "frame_A2.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_A2.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 2, "items": [{ "class_name": "Person" }, { "class_name": "Person" }] },
    "use_case": {
      "cashier": {
        "persons": [
          { "zone": "ROI_CASHIER", "transaction": false, "items": { "drawers": [], "cash": [] } },
          { "zone": "ROI_CASHIER", "transaction": false, "items": { "drawers": [], "cash": [] } }
        ],
        "summary": {
          "cashier_zone": { "persons": 2, "drawers": 0, "cash": 0 },
          "customer_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "case_id": "A2",
          "severity": "ALERT",
          "alerts": ["A2 WARNING: Unexpected person in cashier zone"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/alert/A2/cam_A2.jpg",
          "frame_id": 8,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": [{ "zone": "ROI_CASHIER" }, { "zone": "ROI_CASHIER" }]
        }
      }
    }
  }
}
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg" \
  -F "camera_id=test_camera_1"
```

#### Test with Result File Output

```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@/home/a7med/Downloads/f1.webp" \
  -F "camera_id=f1_test" \
  --output results.json

# View prettified response
cat results.json | jq '.'
```

#### Extract & Save Annotated Image from Response

```bash
# Save full response
response=$(curl -X POST "http://localhost:8000/process" \
  -F "file=@image.jpg" \
  -F "camera_id=test")

# Extract base64 frame and decode
echo "$response" | jq -r '.frame' | base64 -d > annotated_image.jpg

# View detections summary
echo "$response" | jq '.data.detection'
```

### Advanced Testing

#### Batch Process Multiple Images

```bash
mkdir -p results
for image in *.jpg *.png *.webp; do
  [ -f "$image" ] || continue
  
  echo "Processing: $image"
  curl -X POST "http://localhost:8000/process" \
    -F "file=@$image" \
    -F "camera_id=batch_$(date +%s%N)" \
    --output "results/$image.json"
done
```

#### A3 (Cash + open drawer, no cashier - theft signature)

```json
{
  "input": { "path": "...", "name": "frame_A3.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_A3.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 2, "items": [{ "class_name": "drawer_open" }, { "class_name": "cash" }] },
    "use_case": {
      "cashier": {
        "persons": [],
        "summary": {
          "cashier_zone": { "persons": 0, "drawers": 1, "cash": 1 },
          "customer_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "case_id": "A3",
          "severity": "CRITICAL",
          "alerts": ["A3 CRITICAL: Cash + open drawer, no cashier"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/alert/A3/cam_A3.jpg",
          "frame_id": 9,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": []
        }
      }
    }
  }
}
```

#### A4 (Unauthorised person at open register with cash)

```json
{
  "input": { "path": "...", "name": "frame_A4.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_A4.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 3, "items": [{ "class_name": "Person" }, { "class_name": "drawer_open" }, { "class_name": "cash" }] },
    "use_case": {
      "cashier": {
        "persons": [{ "zone": "ROI_CASHIER", "transaction": true, "items": { "drawers": [{...}], "cash": [{...}] } }],
        "summary": {
          "cashier_zone": { "persons": 1, "drawers": 1, "cash": 1 },
          "customer_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "case_id": "A4",
          "severity": "CRITICAL",
          "alerts": ["A4 CRITICAL: Unauthorised person at open register with cash"],
          "transaction": true,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/alert/A4/cam_A4.jpg",
          "frame_id": 10,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": [{ "zone": "ROI_CASHIER" }]
        }
      }
    }
  }
}
#### Stream Processing

```bash
curl -X POST "http://localhost:8000/stream" \
  -F "file=@video.mp4" \
  -F "camera_id=video_stream" \
  -N  # No buffering, stream output
```

#### A5 (Customer waiting too long >30s)

```json
{
  "input": { "path": "...", "name": "frame_A5.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_A5.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 1, "items": [{ "class_name": "Person" }] },
    "use_case": {
      "cashier": {
        "persons": [{ "zone": "ROI_CUSTOMER", "transaction": false, "items": { "drawers": [], "cash": [] } }],
        "summary": {
          "cashier_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "customer_zone": { "persons": 1, "drawers": 0, "cash": 0 },
          "case_id": "A5",
          "severity": "ALERT",
          "alerts": ["A5 WARNING: Customer waiting too long"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/alert/A5/cam_A5.jpg",
          "frame_id": 11,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": []
        }
      }
    }
  }
}
```

#### A6 (Drawer open too long >30s)

```json
{
  "input": { "path": "...", "name": "frame_A6.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_A6.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 2, "items": [{ "class_name": "Person" }, { "class_name": "drawer_open" }] },
    "use_case": {
      "cashier": {
        "persons": [{ "zone": "ROI_CASHIER", "transaction": false, "items": { "drawers": [{...}], "cash": [] } }],
        "summary": {
          "cashier_zone": { "persons": 1, "drawers": 1, "cash": 0 },
          "customer_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "case_id": "A6",
          "severity": "ALERT",
          "alerts": ["A6 WARNING: Drawer open too long"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/alert/A6/cam_A6.jpg",
          "frame_id": 12,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": [{ "zone": "ROI_CASHIER" }]
        }
      }
    }
  }
}
#### Test Mood Detection Only

```bash
curl -X POST "http://localhost:8000/mood" \
  -F "file=@face_image.jpg" | jq '.data.use_case.mood'
```

#### A7 (Cash in customer zone, no cashier present)

```json
{
  "input": { "path": "...", "name": "frame_A7.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_A7.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 2, "items": [{ "class_name": "Person" }, { "class_name": "cash" }] },
    "use_case": {
      "cashier": {
        "persons": [{ "zone": "ROI_CUSTOMER", "transaction": false, "items": { "drawers": [], "cash": [{...}] } }],
        "summary": {
          "cashier_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "customer_zone": { "persons": 1, "drawers": 0, "cash": 1 },
          "case_id": "A7",
          "severity": "ALERT",
          "alerts": ["A7 WARNING: Cash in customer zone, no cashier present"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/alert/A7/cam_A7.jpg",
          "frame_id": 13,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": []
        }
      }
    }
  }
}
```

### Full-dataset cashier test (2026-03-29)

- `outputs/cashier_test/20260329T060741_7464/summary.json` — metrics and case table (source of truth for the validation report above)
- `outputs/cashier_test/20260329T060741_7464/stream.jsonl` — per-frame pipeline JSON
- `outputs/cashier_test/20260329T060741_7464/annotated/`, `evidence/`, `events/`
- `outputs/cashier_test/20260329T060741_7464/cashier_test.gif` — optional animated export

### Production simulation artifacts (earlier reference run)

- `outputs/cashier_production_sim/20260326T050929Z/annotated_sequence.gif`
- `outputs/cashier_production_sim/20260326T050929Z/stream.jsonl`
- `outputs/cashier_production_sim/20260326T050929Z/summary.json`
- `outputs/cashier_production_sim/20260326T050929Z/annotated/`

### Single-frame test output (cashier + aged-gender)

For the one-off test on `frame_097440`, outputs are saved to:

- Annotated frame: `outputs/cashier_aged_gender_single/20260326T064528Z/annotated/frame_097440.jpg`
- JSON results: `outputs/cashier_aged_gender_single/20260326T064528Z/result.json`

### Response processing examples (SSE + jq)

Use `export BASE=http://localhost:9000`. Strip the SSE `data:` prefix before `jq` (see [Complete cURL reference](#complete-curl-reference-all-http-routes)).

**v2 `GET /detection/stream`** — task events (example: crossing metadata):

#### Test Age/Gender Detection Only

```bash
curl -X POST "http://localhost:8000/age-gender" \
  -F "file=@face_image.jpg" | jq '.data.use_case.age_gender'
```

#### Health Check

```bash
curl -sN "$BASE/detection/stream" \
  | grep "^data:" | head -1 | sed 's/^data: //' \
  | jq '{eventType, taskId, channelId, line}'
```

**Cashier** — same structured events as **`GET /detection/stream?eventType=CASHIER_BOX_OPEN`**, plus `/cashier/stream` and `/cashier/status`:

#### Extract Detections with jq

```bash
curl -sN "$BASE/detection/stream?eventType=CASHIER_BOX_OPEN" | head -n 5
curl -sN "$BASE/cashier/stream/cam1" | head -n 20
curl -s "$BASE/cashier/status" | jq .
```

**Legacy / batch** — if you have JSONL from `CameraPipeline` with `data.use_case.cashier`, you can still `jq` that file:

```bash
jq '.data.use_case.cashier.summary | {case_id, severity, alerts}' < stream.jsonl
```

---

## 🎯 Models & Services

### 1. YOLO v8 Nano (Object Detection)

- **Model**: `models/yolov8n.pt` (or `models/yolov8n.engine`)
- **Framework**: PyTorch/TensorRT
- **Purpose**: Person & object detection with bounding boxes
- **Input**: RGB images (any size)
- **Output**: Bounding boxes, class IDs, confidence scores
- **Supported Classes**: 80 COCO classes (persons, chairs, tables, etc.)

### 2. Age/Gender Classification (ONNX)

- **Model**: `models/best_aged_gender_6.onnx`
- **Framework**: ONNX Runtime
- **Gender Classes**: `[Female, Male]`
- **Age Groups**: `[Young, MiddleAged, Senior, Elderly]`
- **Input**: Face crops (224×224 pixels)
- **Output**: Class predictions + confidence scores

### 3. Mood/Emotion Detection (ONNX)

- **Model**: `models/best_mood.onnx`
- **Framework**: ONNX Runtime
- **Classes**: `[Angry, Happy, Neutral]`
- **Input**: Face crops (128×128 pixels)
- **Output**: Mood class + confidence score

### 4. PPE Detection (ONNX)

- **Model**: `models/best_ppe.onnx`
- **Framework**: ONNX Runtime
- **Classes**: `[mask, hairnet, gloves]`
- **Input**: person crops (224×224 pixels)
- **Output**: PPE class + confidence score

### 5. Cashier monitor (YOLO + zone logic)

- **Purpose**: Assign persons to `ROI_CASHIER` / `ROI_CUSTOMER`, detect drawers and cash, classify **N1–N6** / **A1–A7**, optional evidence GIFs and logs under `CASHIER_EVIDENCE_DIR`.
- **Config**: `CASHIER_CONFIG` (default `./config/cashier_zones.yaml`); see [Sample configuration file](#sample-configuration-file-configcashier_zonesyaml).

---

### 6. Person Search

- **Model**: `models/os_net.pt`
- **Input**: person crops (224×224 pixels)
- **Output**: embeddings

### 7. Semantic Search

- **Model**: `models/mobile-clip2-s0.pt`
- **Framework**: ONNX Runtime
- **Input**: person crops
- **Output**: embeddings

---

### 5. Face Recognition (InsightFace)

- **Model**: `buffalo_l` (or configure via `FACE_MODEL`)
- **Framework**: InsightFace / ONNX Runtime / FAISS
- **Purpose**: Face detection, pose estimation, and 512-d embeddings for matching
- **Input**: Person crops
- **Output**: Bounding boxes, quality/pose metrics, matching scores, and JSONL events

---

## 📁 Project Structure

```
.
├── app.py                      # FastAPI main application (v2.0.0)
├── frame_bus.py                # Per-camera capture + YOLO track; fans frames to task queues
├── task_worker.py              # One process per task; dispatches TASK_REGISTRY
├── pipeline.py                 # Optional per-camera CameraPipeline (REGISTRY; scripts/batch)
├── apis/                       # Route handlers
│   ├── cameras.py
│   ├── tasks.py                # /api/tasks CRUD
│   ├── detection.py            # start/stop/status + result queue for SSE
│   ├── detection_stream.py     # SSE bridge + stream filters
│   └── cashier.py              # /cashier/* (zones, SSE, evidence, media)
├── docker-compose.yml          # GPU stack (NVIDIA runtime, port 9000)
├── Dockerfile                  # Container image for the API
├── error_codes/                # Standardized success/error envelopes
├── config/                     # Default cashier zones YAML
│   └── cashier_zones.yaml
├── requirements.txt            # Python dependencies
├── README.md                   # This file (complete documentation)
├── .gitignore                  # Git ignore rules (models/, outputs/, local QA *.md, root test.py)
├── models/                     # Weight files: zero-byte placeholders in git; copy real blobs on edge (see models/README.md)
│   ├── README.md               # Expected filenames + Drive / download_models.py
│   ├── yolov8n.pt              # YOLO (replace on device)
│   ├── best_ppe.onnx
│   ├── best_aged_gender_6.onnx
│   ├── best_mood.onnx
│   ├── best_cashier.onnx       # Cashier ONNX
│   ├── osnet_x1_0.pt           # Person search (ReID)
│   ├── text_encoder.onnx       # Semantic search
│   └── image_encoder.onnx
├── services/                   # Service modules
│   ├── detector.py             # YOLO detection service (REGISTRY / pipeline)
│   ├── age_gender.py           # Age/Gender classification
│   ├── ppe.py                  # PPE detection
│   ├── mood.py                 # Mood/Emotion detection
│   ├── cross_line.py           # CROSS_LINE task
│   ├── mask_hairnet_chef_hat.py
│   ├── cashier.py              # CashierService + CashierDrawerTask (TASK key CASHIER_BOX_OPEN)
│   ├── person_search.py
│   └── semantic_search.py
├── store/
│   └── Identity_manger.py      #handling qdrant embeddings

|   ├── ppe.py                  # PPE detection
│   ├── mood.py                 # Mood/Emotion detection
│   ├── face_engine.py          # InsightFace wrapper
│   ├── face_store.py           # FAISS-backed face embedding store
│   └── face_recognition.py     # Face Recognition task class
├── apis/                       # API routers
│   ├── face_lib.py             # Face library management API endpoints
├── scripts/                    # Testing and utility scripts
│   └── test_image_pipeline.py  # Image inference testing script
├── tests/                      # pytest modules when present (`pytest` from repo root)
├── logger/                     # Logging configuration
│   └── logger_config.py        # Logger setup
└── outputs/                    # Test results directory
    ├── test_image/             # Sample image test results
    │   ├── annotated.jpg       # Annotated image with boxes
    │   ├── stream_data.json    # Full inference results
    │   └── run.log             # Execution logs
    └── f1_test/                # F1.webp test results
        ├── annotated.jpg       # 500×281px annotated JPEG
        ├── stream_data.json    # Complete detection/classification data
        └── run.log             # Pipeline execution log
```

---

## 🔄 Git Workflow

### Current Branch: `mood`

```bash
git log --oneline -10
git status
git add .
git commit -m "Description of changes"
git push origin <your-branch>
```

### Model Files

Models are excluded from git tracking to reduce repository size:

- `best_aged_gender_6.onnx` (~85 MB)
- `best_mood.onnx` (~15 MB)
- `best_ppe.onnx` (~38 MB)
- `yolov8n.pt` (~25 MB)

Download separately or configure via environment variables.

---

## ⚙️ Configuration

### Model Paths

```bash
# Environment variables
export YOLO_MODEL="./models/yolov8n.pt"
export AGE_GENDER_MODEL="./models/best_aged_gender_6.onnx"
export MOOD_MODEL="./models/best_mood.onnx"
export PPE_MODEL="./models/best_PPE.onnx"
```

### FrameBus, streams, and cashier (env)

| Variable | Default | Role |
|----------|---------|------|
| `YOLO_MODEL` | `yolov8n.pt` | YOLO weights path in each FrameBus process |
| `CONF_THRESHOLD` | `0.35` | YOLO confidence |
| `DEVICE` | `0` | CUDA device index or device string |
| `FILTER_CLASSES` | *(empty)* | Comma-separated class IDs to restrict detection |
| `WIDTH` / `HEIGHT` | `1280` / `0` | Capture resize |
| `SAVE_OUTPUT` | `True` | Persist frames under `OUTPUT_DIR` when true |
| `OUTPUT_DIR` | `./outputs` | Base directory for saved runs |
| `DETECTION_SSE_KEEPALIVE_SEC` | `30` | Idle ping interval for `GET /detection/stream` |
| `CASHIER_CONFIG` | `./config/cashier_zones.yaml` | Cashier YAML path |
| `CASHIER_EVIDENCE_DIR` | `./evidence/cashier` | Evidence storage |
| `CASHIER_LOG_MAX` | `5000` | Max in-memory cashier events |

### RTSP stability (H.264 / H.265) and snapshots
| Variable | Default | Role |
|----------|---------|------|
| `RTSP_ENABLE_GPU_DECODE` | `auto` in Compose | Jetson hardware decode for Adaptive FFmpeg ([`stream_adapter.py`](stream_adapter.py)): `auto` enables `*_v4l2m2m` when Jetson is detected; `true` forces that path; `false` uses CPU. Set `true` if `auto` skips HW inside Docker. |
| `RTSP_JETSON_HEVC_DECODER` / `RTSP_JETSON_H264_DECODER` | *(unset)* | Optional ffmpeg decoder names for HEVC/H.265 and H.264 (e.g. `hevc_nvmpi` on some Jetson ffmpeg builds). |
| `RTSP_HWDECODER` | *(unset)* | Legacy fallback decoder when per-codec vars are unset (same name is passed for the probed codec — use only if you know your stream matches). |
| `STREAM_ADAPTER_FFMPEG_STDERR_MAX` | `8192` | Max bytes of ffmpeg stderr retained for logging on RTSP reconnect ([`stream_adapter.py`](stream_adapter.py)). |
| `RTSP_ADAPTIVE_FAILOVER_AFTER` | `4` | If Adaptive FFmpeg reconnect fails this many cycles consecutively, `stream.py` switches that camera process to OpenCV RTSP fallback. |
| `RTSP_FFMPEG_EXTRA_OPTIONS` | *(see `utils/rtsp_ffmpeg.py`)* | Extra OpenCV-FFmpeg options (pipe-separated `key;value` segments); `rtsp_transport;tcp` is always applied |
| `RTSP_MAX_CONSECUTIVE_READ_FAILS` | `10` | Fails before stream reconnect in `stream.py` |
| `STREAM_RECONNECT_BASE_SEC` / `STREAM_RECONNECT_MAX_SEC` | `2` / `30` | Exponential reconnect backoff (capped) |
| `RTSP_OPENCV_BUFFER_SIZE` / `RTSP_OPEN_TIMEOUT_MSEC` / `RTSP_READ_TIMEOUT_MSEC` | `1` / `15000` / `0` | OpenCV capture tuning |
| `CAMERA_SNAPSHOT_CACHE_TTL_SEC` | `5` | `GET /cameras` reuses a recent JPEG per camera instead of opening RTSP on every request |

**Substream (e.g. Hikvision):** for a lighter H.264 feed, point `POST /cameras` at a substream URL such as `.../Streaming/channels/102` or `.../h264/ch1/sub/av_stream` instead of the 4K main path. See [`.env.example`](.env.example).

### ONNX Runtime and GPU policy
| Variable | Default | Role |
|----------|---------|------|
| `ONNX_ALLOW_TENSORRT` | `1` | Allow TensorRT execution provider when installed |
| `ONNX_TENSORRT_CACHE_PATH` | `./trt_cache` | TensorRT engine cache directory |
| `ML_REQUIRE_INFERENCE_GPU` | *(unset)* | If `1`/`true`, YOLO/Ultralytics paths require `torch.cuda` when `DEVICE` is a CUDA index or `cuda` |
| `ML_REQUIRE_ONNX_GPU` | *(unset)* | If `1`/`true`, `onnxruntime` sessions must use CUDA/TensorRT (fails on CPU-only) |
| `REID_MODEL_ONNX` | *(unset)* | Optional OSNet-compatible ONNX for Re-ID (`PersonSearchService`); otherwise `REID_MODEL_PATH` (TorchScript) is used |

### Detection Thresholds
- **YOLO Confidence**: 0.35 (configurable via `CONF_THRESHOLD`)

- **YOLO Confidence**: 0.35 (configurable)
- **Face Detection Minimum Size**: 10×10 pixels
- **Mood Classification**: All 3 classes enabled

---

## 📋 API Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Service info (`version`: **2.0.0**) |
| `/cameras` | POST | Configure cameras |
| `/cameras` | GET | List configured cameras |
| `/cameras/{cam_id}` | DELETE | Remove camera |
| `/api/tasks` | POST | Create/upsert task |
| `/api/tasks` | GET | List tasks |
| `/api/tasks/{task_id}` | GET | Get one task |
| `/api/tasks/{task_id}` | PUT | Update task |
| `/api/tasks/{task_id}` | DELETE | Delete task |
| `/detection/start` | POST | Start FrameBus + task workers (optional `camera_id`) |
| `/detection/stop` | POST | Stop processing (optional `camera_id`; omit = all) |
| `/detection/status` | GET | Operational status per camera |
| `/detection/stream` | GET | SSE: task events (e.g. crossings, PPE); optional `taskId`, `taskName`, `eventType`, `channelId` |
| **`/cameras/{camera_id}/live`** | **WebSocket** | **Binary JPEG frame stream — live annotated video with bounding boxes** |
| **`/cameras/{camera_id}/events`** | **WebSocket** | **JSON detection event stream per camera** |
| `/cashier/status` | GET | Latest cashier summary per camera |
| `/cashier/events` | GET | Paginated structured cashier event log (`severity`, `case_id`, `camera_id`, `limit`, `offset`) |
| `/cashier/events` | DELETE | Clear in-memory event log |
| `/cashier/evidence` | GET | List evidence JPEGs (`severity`, `case_id`, `limit`) |
| `/cashier/evidence/{path}` | GET | Download one evidence JPEG |
| `/cashier/zones` | GET | Read `CASHIER_CONFIG` |
| `/cashier/zones` | POST | Update zones and/or thresholds |
| `/cashier/zones/reset` | POST | Restore default zones to config file |
| `/cashier/stream/{camera_id}` | GET | Per-camera SSE (JSON events only — for live video use WebSocket above) |
| `/cashier/stream/{camera_id}/only` | GET | Per-camera SSE (alerts-oriented) |
| `/cashier/media/{camera_id}/latest/jpg` | GET | Latest evidence JPG |
| `/cashier/media/{camera_id}/latest/gif` | GET | Latest evidence GIF |
| `/cashier/media/{camera_id}/event/{event_id}/jpg` | GET | JPG for event |
| `/cashier/media/{camera_id}/event/{event_id}/gif` | GET | GIF for event |
| `/cashier/media/{camera_id}/drawer_count` | GET | Cumulative drawer open-edge count from `cashier_drawer_open_totals.json` |
| `/person_search/search` | POST | ReID-based person retrieval using OSNet + Qdrant |
| `/semantic_search/search` | POST | Image-based semantic search using MobileCLIP2-S0 + Qdrant |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc |
| `/detection/setup` | POST | Configure pipeline services |
| `/detection/start` | POST | Start processing |
| `/detection/stop` | POST | Stop processing |
| `/detection/status` | GET | Get operational status |
| `/detection/stream` | GET | SSE stream endpoint |
| `/api/face/lib` | POST | Create face library |
| `/api/face/lib` | GET | List face libraries |
| `/api/face/lib/{libId}` | GET/DELETE | Get or delete face library |
| `/api/face/lib/{libId}/persons` | POST/GET | Add or list persons in a library |
| `/api/face/lib/{libId}/persons/{pid}`| DELETE | Delete a person from library |
| `/api/face/recognize` | POST | Single-image face recognition |
| `/api/face/strangers` | GET/DELETE| List or clear stranger records |
| `/api/face/strangers/search` | POST | Search strangers by face |
| `/docs` | GET | Interactive API documentation |

---

## 📞 Support & Documentation

After starting the server (`uvicorn` on port **9000** by default):

- Swagger UI: `http://localhost:9000/docs`
- ReDoc: `http://localhost:9000/redoc`

Supplementary docs (in git): [`docs/API_USAGE.md`](docs/API_USAGE.md) (full API walkthrough including live WebSocket), [`docs/VISION_PIPELINE_README.md`](docs/VISION_PIPELINE_README.md) (pytest, cURL, SSH, cashier `data`/cases/evidence), [`docs/ADDING_A_SERVICE.md`](docs/ADDING_A_SERVICE.md) (new FrameBus tasks), [`docs/CASHIER_BOX_OPEN.md`](docs/CASHIER_BOX_OPEN.md) (Eyego + cashier cURL + mocks + JSON), [`sse_cashier.md`](sse_cashier.md). Ad-hoc run notes (`docs/logs.md`, edge/framing/service test write-ups, `BUG_REPORT.md`) are listed in `.gitignore` so they stay local-only.
For detailed API documentation and interactive testing:

```bash
# Access Swagger UI after starting server
http://localhost:8000/docs

# Access ReDoc documentation
http://localhost:8000/redoc
```

## Models file
<https://drive.google.com/drive/folders/1oAROlqkBo8C3rzTe4hAcS7abaIKC_Ugq?usp=drive_link>

## Models file

<https://drive.google.com/drive/folders/1oAROlqkBo8C3rzTe4hAcS7abaIKC_Ugq?usp=drive_link>
