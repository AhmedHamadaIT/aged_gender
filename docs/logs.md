# Logs, tests, and streaming reference

This page summarizes **where outputs go**, how to **run the test suite**, **curl** patterns for HTTP + SSE after the **cashier structured event** refactor (`CASHIER_BOX_OPEN` on `GET /detection/stream`, task JSONL under `EVENTS_DIR`), the **cashier `data` envelope** (Eyego §4: `id`↔`captureId`, pretty `personStructural`, URL / `deviceSN` env vars), **sample full JSONL** shapes for cashier vs `CROSS_LINE`, and a **copy-paste end-to-end curl test** (cashier zones → all task types → start → logs → stop → cleanup).

**Related:** [API_USAGE.md](./API_USAGE.md) (full HTTP walkthrough), [CASHIER_BOX_OPEN.md](./CASHIER_BOX_OPEN.md) (cashier schema), [VISION_PIPELINE_README.md](./VISION_PIPELINE_README.md).

---

## Contents

1. [Test suite](#test-suite)
2. [Where logs and artifacts are written](#where-logs-and-artifacts-are-written)
3. [Cashier `data` envelope & sample JSONL](#cashier-data-envelope--integration-env)
4. [cURL — global detection SSE](#curl--global-detection-sse-all-task-types)
5. [cURL — cashier HTTP + per-camera SSE](#curl--cashier-http--per-camera-sse)
6. [Live stream WebSocket](#live-stream-websocket)
7. [Tail task JSONL](#tail-task-jsonl-on-a-server)
8. [Endpoints quick list](#endpoints-quick-list)
9. [**End-to-end curl test (start to finish)**](#end-to-end-curl-test-start-to-finish)

---

## Test suite

From the repo root (with the project virtualenv):

```bash
cd /home/a7med/ml-server
.venv/bin/python -m pytest tests/ -q --tb=short
```

**Last documented run:** all tests in `tests/` passed (`test_cashier_api`, `test_cashier_structured_events`, `test_detection_stream`, etc.). Re-run the command above after any change; exit code **0** means success.

Collect tests only:

```bash
.venv/bin/python -m pytest tests/ --co -q
```

---

## Where logs and artifacts are written

| Output | Default location | Env override |
|--------|------------------|--------------|
| **Task events (JSONL)** — one line per frame per task | `/local/storage/events/task_<taskId>.jsonl` | `EVENTS_DIR` |
| **CROSS_LINE** evidence (crop/scene JPEG) | `/local/storage/captures`, `/local/storage/scenes` | `CAPTURE_DIR`, `SCENE_DIR` |
| **CROSS_LINE** event JSONL | `/local/storage/events/task_<taskId>.jsonl` | `EVENTS_DIR` |
| **Cashier** evidence (JPEG/GIF by case) | `./evidence/cashier` (under case folders) | `CASHIER_EVIDENCE_DIR` |
| **Cashier** drawer totals (cumulative open edges + duration) | `<CASHIER_EVIDENCE_DIR>/logs/cashier_drawer_open_totals.json` | `CASHIER_EVIDENCE_DIR`, `CASHIER_DISABLE_DRAWER_TOTAL_PERSIST` |
| **In-memory cashier event deque** (HTTP `GET /cashier/events`) | process RAM | `CASHIER_LOG_MAX` |

**Note:** Legacy `evidence/cashier/logs/events.jsonl` is **no longer** the canonical cashier business log; use **`EVENTS_DIR/task_<taskId>.jsonl`** or **`GET /detection/stream`** / **`GET /cashier/events`**.

---

## Cashier `data` envelope & integration env

The nested **`data`** object on each **`CASHIER_BOX_OPEN`** event is built by `build_cashier_spec_data` in [`services/cashier.py`](../services/cashier.py) (same keys as Eyego §4 and as `GET /cashier/status` payloads that include `data`).

| Field | Notes |
|-------|--------|
| `id` | 32-character hex string — same UUID as in `captureId`, without dashes. |
| `captureId` / `sceneId` | `CASHIER_BOX_OPEN_<uuid4>.jpg` per frame (two independent UUIDs). |
| `personStructural` | JSON as a **string**. **Default:** pretty multi-line (`indent=2`, newlines show as `\n` inside the JSON line). **`CASHIER_COMPACT_PERSON_STRUCTURAL=1`:** single-line compact JSON. |
| `captureUrl` / `sceneUrl` | `{base}/{captureId}` and `{base}/{sceneId}` when a base is configured (see table below). |

**Environment variables (URLs, device, formatting)**

| Variable | Role |
|----------|------|
| `CASHIER_CLOUD_IMAGE_BASE` | Primary base: if set, used for **both** capture and scene URL prefixes. |
| `CASHIER_CAPTURE_URL_BASE` | Capture URL base when `CASHIER_CLOUD_IMAGE_BASE` is unset. |
| `CASHIER_SCENE_URL_BASE` | Scene URL base when `CASHIER_CLOUD_IMAGE_BASE` is unset. |
| `CASHIER_FORCE_LOCAL_URLS` | If truthy (`1`, `true`, `yes`, `on`) and a side still has no base, that side uses `file:///local/storage/images`. |
| `CASHIER_COMPACT_PERSON_STRUCTURAL` | If truthy, `data.personStructural` is minified one-line JSON. |
| `deviceSN` in `data` | Task / zone `deviceSN` → `CASHIER_DEVICE_SN` → `DEVICE_SN` → `HOSTNAME` → `"UNKNOWN"`. |

**Process logs:** building `data` may emit **`DEBUG`** lines (final `deviceSN`, `captureUrl` / `sceneUrl`, whether `personStructural` is compact). If URLs stay empty, **`WARNING`** logs suggest setting the bases above.

**CrossLine vs cashier JSONL:** both append **one JSON object per line** to `EVENTS_DIR/task_<taskId>.jsonl` and appear on **`GET /detection/stream`**. `CROSS_LINE` uses top-level `line`, `person`, `evidence` (local crop/scene paths). `CASHIER_BOX_OPEN` shares the outer envelope (`eventId`, `eventType`, `timestamp`, `taskId`, `channelId`, …) and adds `camera_id`, `case_id`, `severity`, nested **`data`**, and optional `evidence` / `transaction`. Full field lists: [API_USAGE.md](./API_USAGE.md) §5.

### Sample full events (illustrative)

`CROSS_LINE` (one crossing):

```json
{
  "eventId": "87e3e56b6ea6d8c879d8279899cf4a14",
  "eventType": "CROSS_LINE",
  "timestamp": 1775995452345,
  "timestampUTC": "2026-04-12T12:04:12.345000Z",
  "taskId": 10,
  "taskName": "entrance_line",
  "channelId": 1,
  "line": { "id": "1", "name": "Entrance", "direction": 1 },
  "person": {
    "trackingId": "42",
    "reidFeature": [],
    "boundingBox": { "x": 80, "y": 20, "width": 40, "height": 60 },
    "attributes": { "gender": "Unknown", "age": "Unknown" },
    "confidence": 91
  },
  "evidence": {
    "captureImage": "/local/storage/captures/2026/04/12/87e3e56b6ea6d8c879d8279899cf4a14_crop.jpg",
    "sceneImage": "/local/storage/scenes/2026/04/12/87e3e56b6ea6d8c879d8279899cf4a14_scene.jpg"
  }
}
```

`CASHIER_BOX_OPEN` (one frame; `data.personStructural` shortened — real strings include full pretty JSON with `\n`):

```json
{
  "eventId": "a1b2c3d4e5f6789012345678abcdef01",
  "eventType": "CASHIER_BOX_OPEN",
  "timestamp": 1775995452345,
  "timestampUTC": "2026-04-12T12:04:12.345000Z",
  "taskId": 101,
  "taskName": "cashier_drawer_monitor",
  "channelId": 2,
  "camera_id": "2",
  "case_id": "A3",
  "severity": "CRITICAL",
  "data": {
    "algorithmType": "CASHIER_BOX_OPEN",
    "captureId": "CASHIER_BOX_OPEN_775f5537-85fc-4e67-a2ab-946cb45ae272.jpg",
    "sceneId": "CASHIER_BOX_OPEN_f72e460d-2cb7-4fd5-b775-e20ffe80c665.jpg",
    "channelId": 2,
    "channelName": "CAM-02-MAIN",
    "deviceSN": "HQDZW1SBCABAH0205",
    "id": "775f553785fc4e67a2ab946cb45ae272",
    "taskId": 101,
    "taskName": "cashier_drawer_monitor",
    "recordTime": 1775998808673,
    "dateUTC": "2026-04-12T13:00:08.673Z",
    "total_open_count": 9,
    "total_open_duration_ms": 1100000,
    "current_open_duration_ms": 3000,
    "personStructural": "{\n  \"case_matched\": \"A3\",\n  \"case_level\": \"CRITICAL\",\n  \"...\": \"...\"\n}",
    "captureUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_775f5537-85fc-4e67-a2ab-946cb45ae272.jpg",
    "sceneUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_f72e460d-2cb7-4fd5-b775-e20ffe80c665.jpg"
  },
  "evidence": {
    "captureImage": "/path/to/evidence.jpg",
    "sceneImage": "/path/to/evidence.jpg"
  }
}
```

---

## cURL — global detection SSE (all task types)

```bash
export BASE=http://localhost:9000

# All task events (CROSS_LINE, MASK_HAIRNET_CHEF_HAT, CASHIER_BOX_OPEN, …)
curl -sN "$BASE/detection/stream"

# Cashier only (structured `data` block + top-level filters)
curl -sN "$BASE/detection/stream?eventType=CASHIER_BOX_OPEN"

# Cashier for one registered task and channel
curl -sN "$BASE/detection/stream?eventType=CASHIER_BOX_OPEN&taskId=101&channelId=1"
```

Parse first `data:` line with `jq` (strip the `data: ` prefix if piping raw SSE):

```bash
curl -sN --max-time 5 "$BASE/detection/stream?eventType=CASHIER_BOX_OPEN" \
  | grep '^data:' | head -1 | sed 's/^data: //' | jq '{eventType, taskId, case_id, severity, data}'

# Parse nested pretty personStructural (string → JSON)
curl -sN --max-time 5 "$BASE/detection/stream?eventType=CASHIER_BOX_OPEN" \
  | grep '^data:' | head -1 | sed 's/^data: //' | jq -r '.data.personStructural | fromjson'
```

---

## cURL — cashier HTTP + per-camera SSE

```bash
curl -s "$BASE/cashier/status"
curl -s "$BASE/cashier/events?limit=20&camera_id=1&severity=CRITICAL&case_id=A3"
curl -sN "$BASE/cashier/stream/1"
curl -sN "$BASE/cashier/stream/1/only"
curl -s "$BASE/cashier/media/1/drawer_count"
```

**`drawer_open_count`** reads **`cashier_drawer_open_totals.json`** (`by_camera`), i.e. cumulative **closed→open** drawer edges per camera — not a count of JSONL “triggered” lines.

---

## Live stream WebSocket

Binary WebSocket stream of annotated JPEG frames. Requires Redis (`REDIS_URL`).

```bash
export BASE=http://localhost:9000
export WS_BASE=ws://localhost:9000

# Verify WebSocket handshake (expects HTTP 101 Switching Protocols)
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  -H "Sec-WebSocket-Version: 13" \
  "$BASE/cameras/1/live"

# Connect with websocat CLI and print binary frame sizes
websocat --binary "$WS_BASE/cameras/1/live" | while IFS= read -r -d '' chunk; do
    echo "frame: ${#chunk} bytes"
done

# Connect to the event WebSocket and print JSON events
websocat "$WS_BASE/cameras/1/events"

# Verify Redis is publishing frames (inside the redis container)
docker exec -it redis redis-cli SUBSCRIBE live:frame:1
# Should see: message  live:frame:1  <binary JPEG data>  every ~80ms

# Check FrameBus connected
docker compose logs yolo-detect | grep "FrameBus: Redis"
# Expect: [1] FrameBus: Redis connected (redis://redis:6379/0)
```

**What you see on the stream:** YOLO bounding boxes + BoT-SORT track IDs drawn on every frame by FrameBus. For service-level overlays (cashier zones, line drawings) those are drawn by each task worker on their service-specific annotated copy.

See [API_USAGE.md §11](./API_USAGE.md#11-live-stream-websocket-camerasidlive) for the complete browser HTML example and all configuration env vars.

---

## Tail task JSONL on a server

```bash
# Replace task id and path if you set EVENTS_DIR
tail -f /local/storage/events/task_101.jsonl
```

---

## Endpoints quick list

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Health |
| POST/GET/DELETE | `/cameras`, `/cameras/{id}` | Camera registry |
| POST/GET/PUT/DELETE | `/api/tasks`, `/api/tasks/{id}` | Task CRUD |
| POST | `/detection/start`, `/detection/stop` | Run/stop FrameBus + workers |
| GET | `/detection/status` | Per-camera FPS/errors |
| GET | `/detection/stream` | SSE: all task events (`taskId`, `eventType`, `channelId`, …) |
| GET/POST | `/cashier/zones`, `/cashier/zones/reset` | Zone config |
| GET | `/cashier/status` | Latest structured event per camera |
| GET/DELETE | `/cashier/events` | Paginated in-memory event history |
| GET | `/cashier/evidence`, `/cashier/evidence/{path}` | Evidence files |
| GET | `/cashier/stream/{camera_id}`, `.../only` | Per-camera SSE (JSON events) |
| GET | `/cashier/media/...` | Latest/event media, `drawer_count` |
| **WebSocket** | **`/cameras/{camera_id}/live`** | **Binary JPEG frame stream — live annotated video** |
| **WebSocket** | **`/cameras/{camera_id}/events`** | **JSON detection event stream per camera** |

---

## End-to-end curl test (start to finish)

Run this **after** starting the API (example):

```bash
cd /home/a7med/ml-server
YOLO_MODEL="/home/a7med/ml-server/yolov8n.pt" \
  .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 9000
```

**Model caveat:** `CROSS_LINE` / `MASK_HAIRNET_CHEF_HAT` expect a **COCO-style** person model. `CASHIER_BOX_OPEN` needs **cashier** weights (person / drawer / cash) on that channel. For a **single demo camera** you typically either use **one** compatible model and **one** task type, or **separate cameras** per model. The script below registers **all three** on **channel `1`** for API wiring smoke tests; replace `YOLO_MODEL` per your deployment.

### Step 1 — Base URL and health

```bash
export BASE=http://127.0.0.1:9000

curl -s "$BASE/"
# expect: {"service":"Vision Pipeline API","version":"2.0.0"}
```

### Step 2 — Register a camera

Use an **RTSP URL** or a **local file path** (absolute path on the server):

```bash
curl -s -X POST "$BASE/cameras" \
  -H "Content-Type: application/json" \
  -d '{
    "cameras": [
      {"id": "1", "url": "rtsp://user:pass@192.168.1.100/stream"}
    ]
  }'
```

**Local file example** (if the file exists on the machine running uvicorn):

```bash
curl -s -X POST "$BASE/cameras" \
  -H "Content-Type: application/json" \
  -d "{
    \"cameras\": [
      {\"id\": \"1\", \"url\": \"/home/a7med/ml-server/videos/cashier_demo.mp4\"}
    ]
  }"
```

```bash
curl -s "$BASE/cameras"
# expect: {"count":1,"cameras":[{"id":"1","url":"..."}]}
```

### Step 3 — Cashier zones (normalized coordinates `0..1`)

`POST /cashier/zones` merges into `CASHIER_CONFIG` (default `./config/cashier_zones.yaml`). Points are **`{"x","y"}`** in the JSON body.

```bash
curl -s -X POST "$BASE/cashier/zones" \
  -H "Content-Type: application/json" \
  -d '{
    "ROI_CASHIER": {
      "shape": "rectangle",
      "points": [{"x": 0.0, "y": 0.0}, {"x": 0.45, "y": 1.0}],
      "active": true
    },
    "ROI_CUSTOMER": {
      "shape": "rectangle",
      "points": [{"x": 0.45, "y": 0.0}, {"x": 1.0, "y": 1.0}],
      "active": true
    },
    "thresholds": {
      "drawer_open_max_seconds": 30,
      "customer_wait_max_seconds": 30,
      "proximity_iou": 0.05,
      "config_reload_interval": 60
    },
    "detail_config": {
      "drawerOpenLimit": 30,
      "serviceWaitLimit": 30,
      "enableStaffList": false,
      "staffIds": []
    },
    "task": {
      "taskId": 30,
      "taskName": "cashier_drawer_monitor",
      "channelId": 1,
      "channelName": "CAM-01-MAIN",
      "deviceSN": "DEMO-DEVICE-001"
    },
    "detection_threshold": 50
  }'
```

```bash
curl -s "$BASE/cashier/zones" | head -c 2000
# expect: merged zones, thresholds, task metadata
```

### Step 4 — Register tasks (CrossLine + PPE + Cashier)

**CrossLine** — `areaPosition` is a **JSON string**. Each line uses two **pixel** points `[{x,y},{x,y}]` in frame coordinates.

```bash
curl -s -X POST "$BASE/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 10,
    "taskName": "entrance_line",
    "algorithmType": "CROSS_LINE",
    "channelId": 1,
    "enable": true,
    "threshold": 60,
    "areaPosition": "[{\"line_id\":\"1\",\"line_name\":\"Entrance\",\"point\":[{\"x\":100,\"y\":400},{\"x\":900,\"y\":400}],\"direction\":1}]",
    "detailConfig": {"enableAttrDetect": false}
  }'
```

**PPE (`MASK_HAIRNET_CHEF_HAT`)** — polygon **pixel** points (at least three corners):

```bash
curl -s -X POST "$BASE/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 20,
    "taskName": "kitchen_ppe_check",
    "algorithmType": "MASK_HAIRNET_CHEF_HAT",
    "channelId": 1,
    "enable": true,
    "threshold": 70,
    "areaPosition": "[{\"line_id\":\"zone1\",\"point\":[{\"x\":50,\"y\":50},{\"x\":600,\"y\":50},{\"x\":600,\"y\":500},{\"x\":50,\"y\":500}],\"direction\":0}]",
    "detailConfig": {"alarmType": ["no_mask", "no_chef_hat", "no_hat"]}
  }'
```

**Cashier (`CASHIER_BOX_OPEN`)** — zones come from `/cashier/zones`; `areaPosition` is often `"[]"`.

```bash
curl -s -X POST "$BASE/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 30,
    "taskName": "cashier_drawer_monitor",
    "algorithmType": "CASHIER_BOX_OPEN",
    "channelId": 1,
    "enable": true,
    "threshold": 50,
    "areaPosition": "[]",
    "detailConfig": {
      "drawerOpenLimit": 30,
      "serviceWaitLimit": 30,
      "enableStaffList": false,
      "staffIds": []
    }
  }'
```

```bash
curl -s "$BASE/api/tasks"
# expect: count 3, tasks 10 / 20 / 30
```

### Step 5 — Start detection and read status

```bash
curl -s -X POST "$BASE/detection/start?camera_id=1"
# expect: {"status":"started","cameras":["1"],"tasks":["10","20","30"]} (order may vary)

curl -s "$BASE/detection/status"
# expect: per-camera fps, frame_count, errors (empty if stream opens)
```

### Step 6 — Log samples (HTTP, not SSE)

**Cashier structured state** (latest frame event per camera):

```bash
curl -s "$BASE/cashier/status" | jq .
```

**Cashier event history** (one structured event per processed frame while running):

```bash
curl -s "$BASE/cashier/events?limit=5" | jq .
```

**Illustrative `jq` filters** on a saved SSE line (after stripping the `data: ` prefix):

```bash
# CROSS_LINE
echo 'data: {"eventType":"CROSS_LINE","taskId":10,...}' | sed 's/^data: //' | jq '{eventType, taskId, line, person}'

# CASHIER_BOX_OPEN
echo 'data: {"eventType":"CASHIER_BOX_OPEN","case_id":"N1","data":{...}}' | sed 's/^data: //' | jq '{eventType, taskId, case_id, severity, data}'

# Parse cashier data.personStructural (pretty or compact string → JSON)
echo 'data: {"eventType":"CASHIER_BOX_OPEN","data":{"personStructural":"{\"a\":1}"}}' | sed 's/^data: //' | jq -r '.data.personStructural | fromjson'
```

### Step 7 — Live WebSocket stream (annotated frames)

Open in a browser after starting detection (save as `stream.html`):

```html
<!DOCTYPE html><html><body style="background:#111;color:#0f0;text-align:center">
<h3>Live Stream — Camera 1</h3>
<div id="s">Connecting…</div><img id="img" style="max-width:100%">
<script>
function c(id){
  const ws=new WebSocket(`ws://localhost:9000/cameras/${id}/live`);
  ws.binaryType="arraybuffer";
  ws.onopen=()=>document.getElementById("s").textContent="Connected";
  ws.onclose=()=>{document.getElementById("s").textContent="Reconnecting…";setTimeout(()=>c(id),2000)};
  ws.onmessage=e=>{
    const img=document.getElementById("img");
    URL.revokeObjectURL(img.src);
    img.src=URL.createObjectURL(new Blob([e.data],{type:"image/jpeg"}));
  };
}c("1");
</script></body></html>
```

Verify Redis is publishing:

```bash
docker exec -it redis redis-cli SUBSCRIBE live:frame:1
# Should see messages arriving every ~80ms
```

### Step 7c — SSE (short capture; press Ctrl+C if you omit `max-time`)

**All task types:**

```bash
curl -sN --max-time 8 "$BASE/detection/stream" | head -n 30
```

**Filtered:**

```bash
curl -sN --max-time 8 "$BASE/detection/stream?eventType=CROSS_LINE&channelId=1" | head -n 15
curl -sN --max-time 8 "$BASE/detection/stream?eventType=MASK_HAIRNET_CHEF_HAT&channelId=1" | head -n 15
curl -sN --max-time 8 "$BASE/detection/stream?eventType=CASHIER_BOX_OPEN&taskId=30" | head -n 15
```

**Cashier per-camera SSE** (requires an active subscriber so the event loop is captured):

```bash
curl -sN --max-time 8 "$BASE/cashier/stream/1" | head -n 20
```

### Step 8 — On-disk logs (optional)

```bash
# Task-scoped JSONL (default directory; override with EVENTS_DIR)
ls -la /local/storage/events/task_*.jsonl 2>/dev/null || true
tail -n 2 /local/storage/events/task_10.jsonl 2>/dev/null || true
tail -n 2 /local/storage/events/task_30.jsonl 2>/dev/null || true
```

### Step 9 — Stop detection

```bash
curl -s -X POST "$BASE/detection/stop?camera_id=1"
# or stop all cameras:
# curl -s -X POST "$BASE/detection/stop"
```

### Step 10 — Cleanup (tasks + camera)

```bash
curl -s -X DELETE "$BASE/api/tasks/10"
curl -s -X DELETE "$BASE/api/tasks/20"
curl -s -X DELETE "$BASE/api/tasks/30"
curl -s -X DELETE "$BASE/cameras/1"
```

### Expected log patterns (cheat sheet)

| Source | What you see |
|--------|----------------|
| `GET /detection/status` | `running`, `fps`, `error` if URL/file invalid |
| `GET /detection/stream` | `data: {JSON}` lines; `: ping` keepalives |
| `GET /cashier/status` | Latest `CASHIER_BOX_OPEN` structured object for that camera |
| `GET /cashier/events` | Newest-first list of the same shape |
| `task_<id>.jsonl` | One compact JSON object per line per frame (when workers ran); cashier `data.personStructural` may contain escaped `\n` (pretty JSON string) |
| `services.cashier` (process log) | `DEBUG` from `build_cashier_spec_data`; `WARNING` if `captureUrl` / `sceneUrl` end up empty |

For full request/response schemas and more examples, see [API_USAGE.md](./API_USAGE.md).
