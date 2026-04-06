# CASHIER_BOX_OPEN — reference (Eyego cURL, mocks, evidence, JSON)

Single document combining the former `docs/cases-and-repo.md` pointers, `CASHIER_BOX_OPEN_CURL.md`, `CASHIER_BOX_OPEN_CASES_MOCK_RESPONSES.md`, machine-readable **`cashier_all_cases_output.json`**, and the former repo-root **`curl_cashier.md`** (removed) as **Part III** (thresholds, zones, SSE examples, case matrix).

**Narrative + pipeline operations:** [VISION_PIPELINE_README.md](./VISION_PIPELINE_README.md) (Part II — CASHIER_BOX_OPEN).

---

## 1. Backend task API (Eyego-style)

Base URL: set `TASKS_BASE` (default `https://app.eyego.ai`).

### 1.1 Create task — `POST /api/tasks`

Same body as spec §5.1. Payload file: [`scripts/payloads/eyego_create_task_CASHIER_BOX_OPEN.json`](../scripts/payloads/eyego_create_task_CASHIER_BOX_OPEN.json).

```bash
export TASKS_BASE=https://app.eyego.ai

curl -X POST "${TASKS_BASE}/api/tasks" \
  -H "Content-Type: application/json" \
  -d @scripts/payloads/eyego_create_task_CASHIER_BOX_OPEN.json
```

**Inline version** (equivalent to the spec):

```bash
curl -X POST "${TASKS_BASE}/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "cashier_drawer_monitor",
    "algorithmType": "CASHIER_BOX_OPEN",
    "channelId": 2,
    "enable": true,
    "threshold": 45,
    "areaPosition": "[{\"zone_id\":1,\"zone_name\":\"CASHIER\",\"points\":[{\"x\":0.215,\"y\":0.662},{\"x\":0.349,\"y\":0.562},{\"x\":0.421,\"y\":0.75},{\"x\":0.257,\"y\":0.886}]},{\"zone_id\":2,\"zone_name\":\"CUSTOMER\",\"points\":[{\"x\":0.421,\"y\":0.75},{\"x\":0.6,\"y\":0.6},{\"x\":0.8,\"y\":0.8},{\"x\":0.257,\"y\":0.886}]}]",
    "detailConfig": "{\"drawerOpenLimit\":30,\"serviceWaitLimit\":30,\"enableStaffList\":true,\"staffIds\":[1001,1002,1003]}",
    "pushCapture": true,
    "pushScene": true
  }'
```

### 1.2 Update task — `PUT /api/tasks/{id}`

Spec §5.2. Payload: [`scripts/payloads/eyego_update_task_CASHIER_BOX_OPEN.json`](../scripts/payloads/eyego_update_task_CASHIER_BOX_OPEN.json).

```bash
export TASK_ID=101
curl -X PUT "${TASKS_BASE}/api/tasks/${TASK_ID}" \
  -H "Content-Type: application/json" \
  -d @scripts/payloads/eyego_update_task_CASHIER_BOX_OPEN.json
```

### 1.3 Script wrapper

```bash
chmod +x scripts/curl_eyego_style_tasks.sh
export TASKS_BASE=https://app.eyego.ai
./scripts/curl_eyego_style_tasks.sh create   # or update, or both
```

### 1.4 Auth (401) and fake test

Eyego returns **`401` / "No token provided"** unless you send a valid session/API token (header name depends on their stack — often `Authorization: Bearer <token>`). This repo does not store tokens.

**Offline “pass” test** — same JSON files, POST/PUT to **httpbin.org** so the body is echoed and validated (HTTP **200**, `algorithmType`, string `areaPosition` / `detailConfig`, etc.):

```bash
./scripts/curl_eyego_style_tasks.sh fake
# same as:
EYEGO_FAKE_TEST=1 ./scripts/curl_eyego_style_tasks.sh both
```

Requires **network** to `httpbin.org` and **`jq`** for assertions. Exit code **0** means the Eyego-shaped payloads are well-formed; it does **not** call `app.eyego.ai`.

---

## 2. Field mapping → ml-server (`/cashier`)

| Eyego / spec field | ml-server |
|---------------------|-----------|
| `areaPosition` (string) | Parse JSON → `ROI_CASHIER` / `ROI_CUSTOMER` polygons (`zone_id` 1 → cashier, 2 → customer) |
| `threshold` (0–100) | `detection_threshold` on `POST /cashier/zones` |
| `detailConfig` (string) | Parse JSON → `detail_config` object (same keys: `drawerOpenLimit`, `serviceWaitLimit`, `enableStaffList`, `staffIds`) |
| `channelId`, `name` | Optional `task` blob: e.g. `channelId`, `taskName` |
| `pushCapture` / `pushScene` | Not implemented in ml-server (cloud URLs are integration-layer) |

### 2.1 Equivalent — `POST /cashier/zones`

Example file (expanded polygons + `detail_config` + `task` + threshold):  
[`scripts/curl_cashier_box_open_mock.json`](../scripts/curl_cashier_box_open_mock.json).

```bash
export BASE=http://localhost:9000

curl -sS -X POST "${BASE}/cashier/zones" \
  -H "Content-Type: application/json" \
  -d @scripts/curl_cashier_box_open_mock.json
```

Verify:

```bash
curl -sS "${BASE}/cashier/zones" | jq '.detail_config, .thresholds, .zones'
```

### 2.2 Live pipeline (after zones)

```bash
curl -sS -X POST "${BASE}/cameras" -H "Content-Type: application/json" \
  -d '{"cameras":[{"id":"cam1","url":"rtsp://..."}]}'

curl -sS -X POST "${BASE}/detection/setup" -H "Content-Type: application/json" \
  -d '{"pipeline":["detector","cashier"]}'

curl -sS -X POST "${BASE}/detection/start?camera_id=cam1"
```

---

## 3. Output shape (spec §4 / §6) vs ml-server

| Spec `data.*` | ml-server today |
|---------------|-----------------|
| Full `data` object (§4) | `use_case.cashier.data` — same keys as spec (`captureId`, `sceneId`, `recordTime`, `dateUTC`, `personStructural`, URLs, …) |
| `algorithmType` | Top-level `use_case.cashier.algorithmType` and inside `data` |
| `personStructural` | Duplicated: `use_case.cashier.personStructural` and `use_case.cashier.data.personStructural` (JSON string); includes **`total_open_count`** (cumulative drawer open edges, per camera) |
| `total_open_count`, `total_open_duration_ms`, `current_open_duration_ms` | Also on `data` and `summary` (mirror `personStructural`) |
| `captureUrl` / `sceneUrl` | Built when `CASHIER_CLOUD_IMAGE_BASE` and/or `CASHIER_CAPTURE_URL_BASE` / `CASHIER_SCENE_URL_BASE` are set (else `""`) |
| `captureId` / `sceneId` | New UUID-based filenames per frame (logical names for upload; local disk may still use one JPEG until bridge maps both) |
| `deviceSN` | `task.deviceSN` or env `CASHIER_DEVICE_SN` / `DEVICE_SN` |
| `channelId`, `taskId`, `channelName`, … | Config `task` in `cashier_zones.yaml` or `POST /cashier/zones` |
| SSE / status | `GET /cashier/status`, `GET /detection/stream` |

**Parse spec `data` or `personStructural` from status:**

```bash
curl -sS "${BASE}/cashier/status" | jq '.cam.data'
curl -sS "${BASE}/cashier/status" | jq -r '.cam.data.personStructural | fromjson'
```

---

## 4. Evidence behaviour (spec §4.5)

Implemented in [`services/cashier.py`](../services/cashier.py): N3 keyframe; ALERT/CRITICAL evidence + GIF budgets; A3/A4 unbounded post-buffer with long-duration log warning. See [`reports/CASHIER_BOX_OPEN_REPORT.md`](../reports/CASHIER_BOX_OPEN_REPORT.md).

---

## 5. Related scripts and docs

| Artifact | Purpose |
|----------|---------|
| [`scripts/curl_eyego_style_tasks.sh`](../scripts/curl_eyego_style_tasks.sh) | Eyego `POST/PUT /api/tasks` with string `areaPosition` / `detailConfig` |
| [`scripts/curl_cashier_box_open_mock.sh`](../scripts/curl_cashier_box_open_mock.sh) | Local API health + zones + status (needs `uvicorn` on `BASE`) |
| [`scripts/test_cashier_14_rules.py`](../scripts/test_cashier_14_rules.py) | Offline 14-rule table tests (standalone, no pytest) |
| [`tests/test_cashier_box_open_cases.py`](../tests/test_cashier_box_open_cases.py) | Pytest: all cases + `personStructural` / duration fields |
| [`scripts/mock_cashier_all_cases_md.py`](../scripts/mock_cashier_all_cases_md.py) | Runs pytest and prints this doc path |
| **This document** | §§1–4 Eyego + mapping; **Part III** Vision Pipeline cashier cURL; §§6–11 mocks + tests; appendix JSON |


---

# Part III — Cashier cURL reference

**Part III** is the former repo-root **`curl_cashier.md`** (merged here only). Links use **`../`** for paths outside **`docs/`**.

Production FastAPI service. Replace `<jetson-ip>` with the device hostname or IP.

**Base URL:** `http://<jetson-ip>:9000`

Run locally (default port from app): `uvicorn app:app --host 0.0.0.0 --port 9000`

Interactive schemas: `http://<jetson-ip>:9000/docs`

**Batch logs:** Local cashier-YOLO run [`outputs/cashier_test/20260329T135320_10106/stream.jsonl`](../outputs/cashier_test/20260329T135320_10106/stream.jsonl) (73 frames), plus first-70-frames run [`outputs/test_70`](../outputs/test_70) with updated ROI polygons (see [`sse_cashier.md`](../sse_cashier.md)). Full-dataset aggregates: [`outputs/cashier_test/20260329T060741_7464/summary.json`](../outputs/cashier_test/20260329T060741_7464/summary.json) (see **Cashier full-dataset validation report** in [`README.md`](../README.md)).

---

## SSE event reference (`GET /detection/stream`)

Each SSE line is `data: ` followed by one JSON object (then blank line). Example shape for cashier + age/gender + mood. Each `detection.items[]` entry matches [`Detection.to_dict()`](../services/detector.py): `bbox`, `class_id`, `class_name`, `confidence`, `center`, `width`, `height`.

```json
{
  "camera_id": "main_room",
  "frame_count": 97,
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": {
      "count": 2,
      "items": [
        {
          "bbox": [684, 0, 1169, 275],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.92,
          "center": [926, 137],
          "width": 485,
          "height": 275
        },
        {
          "bbox": [811, 874, 1264, 1080],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.88,
          "center": [1037, 977],
          "width": 453,
          "height": 206
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {"bbox": [684, 0, 1169, 275], "gender": "Female", "age_group": "Senior", "confidence": 0.50},
        {"bbox": [811, 874, 1264, 1080], "gender": "Male", "age_group": "Senior", "confidence": 0.74}
      ],
      "mood": [
        {"bbox": [684, 0, 1169, 275], "mood": "Neutral", "confidence": 0.88},
        {"bbox": [811, 874, 1264, 1080], "mood": "Happy", "confidence": 0.91}
      ],
      "cashier": {
        "persons": [
          {
            "person_bbox": [811, 874, 1264, 1080],
            "confidence": 0.87,
            "zone": "ROI_CASHIER",
            "transaction": true,
            "items": {
              "drawers": [{"bbox": [845, 692, 1186, 922], "confidence": 0.87}],
              "cash": [{"bbox": [858, 727, 934, 832], "confidence": 0.85}]
            }
          },
          {
            "person_bbox": [684, 0, 1169, 275],
            "confidence": 0.89,
            "zone": "ROI_CUSTOMER",
            "transaction": false,
            "items": {"drawers": [], "cash": []}
          }
        ],
        "summary": {
          "cashier_zone": {"persons": 1, "drawers": 1, "cash": 4},
          "customer_zone": {"persons": 1, "drawers": 0, "cash": 0},
          "case_id": "N3",
          "severity": "NORMAL",
          "alerts": ["N3 EVENT: Transaction in progress"],
          "transaction": true,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/normal/N3/cam_xxx.jpg",
          "frame_id": 97,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": [{"bbox": [811, 874, 1264, 1080], "confidence": 0.87, "zone": "ROI_CASHIER"}]
        }
      }
    }
  }
}
```

**Implementation note:** Live events from `/detection/stream` also include a top-level **`frame`** field: a base64-encoded JPEG string. It is omitted above because it is large; strip or ignore it when piping to `jq` for debugging.

---

## Standardized error response (current)

Detection and cashier endpoints now return a unified error payload:

```json
{
  "status": "error",
  "error": {
    "code": "404",
    "message": "Cashier evidence not found.",
    "detail": "cam1"
  }
}
```

Use `-i` to display HTTP status code with the JSON body.

### Cashier error examples

```bash
# Missing evidence file
curl -si "http://<jetson-ip>:9000/cashier/evidence/missing.jpg"

# No latest JPG/GIF for this camera
curl -si "http://<jetson-ip>:9000/cashier/media/cashier_cam_01/latest/jpg"
curl -si "http://<jetson-ip>:9000/cashier/media/cashier_cam_01/latest/gif"

# Missing event media
curl -si "http://<jetson-ip>:9000/cashier/media/cashier_cam_01/event/A2_20260329_135421_001/jpg"
curl -si "http://<jetson-ip>:9000/cashier/media/cashier_cam_01/event/A2_20260329_135421_001/gif"
```

### Detection error examples

```bash
# No cameras running -> code 203 in JSON body
curl -si -X POST "http://<jetson-ip>:9000/detection/stop"

# Unknown stream camera -> code 301 in JSON body
curl -si "http://<jetson-ip>:9000/detection/stream/cam999"
```

---

## Recommended setup order

1. `POST /cameras` — register RTSP sources  
2. `POST /detection/setup` — choose pipeline services (`cashier` must be **last** if you use it)  
3. `POST /cashier/zones` — optional; defaults load from `CASHIER_CONFIG` (often `./config/cashier_zones.yaml`)  
4. `GET /cashier/zones` — optional; inspect merged on-disk config (`zones` as `[[x,y],…]`)  
5. `POST /detection/start` — spawn pipeline process(es)  
6. `GET /detection/stream` or `GET /detection/status` — consume results  
7. After traffic flows: `GET /cashier/status`, `GET /cashier/events`, optional `GET /cashier/stream/{camera_id}` — cashier-specific state and SSE  

---

## Health

### `GET /`

Returns API identity.

```bash
curl -s "http://<jetson-ip>:9000/"
```

**Sample JSON response**

```json
{"service": "Vision Pipeline API", "version": "1.0.0"}
```

**Key fields**

- `service` — Product name string.  
- `version` — API version string.  

---

## Cameras

### `POST /cameras`

Register or update one or more cameras in memory (id → RTSP URL).

```bash
curl -s -X POST "http://<jetson-ip>:9000/cameras" \
  -H "Content-Type: application/json" \
  -d '{
    "cameras": [
      {"id": "cam1", "url": "rtsp://user:pass@192.168.1.10/stream"}
    ]
  }'
```

**Sample JSON request body**

```json
{
  "cameras": [
    {"id": "cam1", "url": "rtsp://user:pass@192.168.1.10/stream"}
  ]
}
```

**Sample JSON response**

```json
{
  "status": "configured",
  "cameras": {
    "cam1": "rtsp://user:pass@192.168.1.10/stream"
  }
}
```

**Key fields**

- `status` — Always `"configured"` on success.  
- `cameras` — Map of camera id to RTSP URL (all cameras currently registered).  

---

### `GET /cameras`

List configured cameras as structured rows.

```bash
curl -s "http://<jetson-ip>:9000/cameras"
```

**Sample JSON response**

```json
{
  "count": 1,
  "cameras": [
    {"id": "cam1", "url": "rtsp://user:pass@192.168.1.10/stream"}
  ]
}
```

**Key fields**

- `count` — Number of configured cameras.  
- `cameras` — Array of `{id, url}` (not the same shape as `POST` response).  

---

### `DELETE /cameras/{cam_id}`

Remove a camera by id.

```bash
curl -s -X DELETE "http://<jetson-ip>:9000/cameras/cam1"
```

**Sample JSON response**

```json
{
  "status": "removed",
  "camera_id": "cam1",
  "remaining": []
}
```

**Key fields**

- `status` — `"removed"` on success.  
- `camera_id` — Id that was deleted.  
- `remaining` — Ids still registered.  

---

## Pipeline

### `POST /detection/setup`

Declare which services run per camera process. Order matters for the pipeline.

```bash
curl -s -X POST "http://<jetson-ip>:9000/detection/setup" \
  -H "Content-Type: application/json" \
  -d '{"pipeline": ["detector", "age_gender", "mood", "cashier"]}'
```

**Sample JSON request body (cashier + age/gender + mood)**

```json
{"pipeline": ["detector", "age_gender", "mood", "cashier"]}
```

**Available service names:** `detector`, `age_gender`, `ppe`, `mood`, `cashier` (see `services/__init__.py`).

**Sample JSON response**

```json
{
  "status": "configured",
  "pipeline": ["detector", "age_gender", "mood", "cashier"]
}
```

**Key fields**

- `status` — `"configured"` on success.  
- `pipeline` — Echo of the active pipeline list.  

---

### `POST /detection/start`

Start the pipeline for one camera or for all configured cameras.

```bash
curl -s -X POST "http://<jetson-ip>:9000/detection/start?camera_id=cam1"
```

Omit `camera_id` to start every configured camera.

**Sample JSON response**

```json
{
  "status": "started",
  "cameras": ["cam1"]
}
```

**Key fields**

- `status` — `"started"` on success.  
- `cameras` — List of camera ids that were started in this call.  

**Typical error body:** standardized JSON (`status=error`, `error.code`, `error.message`, optional `error.detail`).

---

### `POST /detection/stop`

Stop one camera or all running cameras.

```bash
curl -s -X POST "http://<jetson-ip>:9000/detection/stop?camera_id=cam1"
```

Omit `camera_id` to stop **all** running cameras.

**Sample JSON response**

```json
{
  "status": "stopped",
  "cameras": ["cam1"]
}
```

**Key fields**

- `status` — `"stopped"` on success.  
- `cameras` — List of camera ids stopped in this call.  

---

### `GET /detection/status`

Per-camera runtime stats (from shared process state).

```bash
curl -s "http://<jetson-ip>:9000/detection/status"
```

**Sample JSON response (core fields)**

```json
{
  "cameras": {
    "cam1": {
      "running": true,
      "fps": 28.5,
      "frame_count": 420,
      "uptime_seconds": 14.7
    }
  }
}
```

**Additional fields** each camera may include (see `schemas.CameraStatus`): `camera_id`, `rtsp_url`, `last_detections`, `total_detections`, `error`.

**Key fields**

- `cameras` — Map keyed by camera id.  
- `running` — Whether the worker process is active.  
- `fps` — Smoothed frames per second.  
- `frame_count` — Frames processed.  
- `uptime_seconds` — Seconds since start for that worker.  

---

### `GET /detection/stream`

Server-Sent Events: one JSON payload per line prefixed with `data: `. Use **`curl -N`** (no buffer) so events appear in real time.

```bash
curl -N "http://<jetson-ip>:9000/detection/stream"
```

**Wire format:** repeated blocks of `data: <json>\n\n`. See [SSE event reference](#sse-event-reference-get-detectionstream) above; remember the extra **`frame`** (base64 JPEG) on real events.

**Key fields (top level)**

- `camera_id` — Source camera.  
- `frame_count` — Monotonic frame index for that worker.  
- `timestamp` — ISO timestamp when the result was emitted.  
- `frame` — Base64 JPEG (large); optional to decode or omit in logs.  
- `data.detection` — Person/object counts and `items`.  
- `data.use_case` — Service outputs (`age_gender`, `mood`, `cashier`, …).  

---

## Cashier zones

### `GET /cashier/zones`

Read the current zone geometry and **thresholds** from `CASHIER_CONFIG` (default `./config/cashier_zones.yaml`). Response is the raw file: top-level `zones`, `thresholds`, plus any other keys present in the YAML/JSON.

```bash
curl -s "http://<jetson-ip>:9000/cashier/zones"
```

**Inspect only timing / IOU thresholds**

```bash
curl -s "http://<jetson-ip>:9000/cashier/zones" | jq '.thresholds'
```

**Sample JSON fragment** (abridged from [`config/cashier_zones.yaml`](../config/cashier_zones.yaml); a live `GET` may also return `buffer`, `debounce`, `evidence`, `gif`, `meta`, etc.)

```json
{
  "thresholds": {
    "config_reload_interval": 60,
    "customer_wait_max_seconds": 30,
    "drawer_open_max_seconds": 30,
    "proximity_iou": 0.05
  },
  "zones": {
    "ROI_CASHIER": {
      "active": true,
      "shape": "polygon",
      "points": [[0.32969, 0.54861], [0.65000, 0.52778], [0.69297, 0.99444], [0.33516, 0.98750]]
    },
    "ROI_CUSTOMER": {
      "active": true,
      "shape": "polygon",
      "points": [[0.32031, 0.28472], [0.64297, 0.25556], [0.63516, 0.00000], [0.31172, 0.00000]]
    }
  }
}
```

**Key fields**

- `zones` — `ROI_CASHIER` / `ROI_CUSTOMER` with normalised `points` as `[[x,y],…]` on disk.  
- `thresholds` — Tuning used by cashier logic (wait/drawer timers, IoU, config reload interval).  

#### Thresholds reference (what to change and why)

| Key | Default | Role |
|-----|---------|------|
| `drawer_open_max_seconds` | `30` | Timer for **A6** (drawer open longer than this while the case logic applies). |
| `customer_wait_max_seconds` | `30` | Timer for **A5** (customer in zone, no cashier, wait longer than this). |
| `proximity_iou` | `0.05` | Minimum **IoU** between boxes to associate a **person** with a **drawer** or **cash** (and drawer→cash proxy). Higher = stricter overlap required. |
| `config_reload_interval` | `60` | Seconds between **disk reloads** of `CASHIER_CONFIG`. A successful `POST /cashier/zones` writes the file immediately; the running process picks it up on the next reload cycle. |

#### A5 / A6: thresholds → logic → output

Both are **time-based** escalations in [`CashierService._evaluate`](../services/cashier.py). They tie **config** to **rules** to **stream JSON**.

| | **A5** | **A6** |
|---|--------|--------|
| **Threshold** | `customer_wait_max_seconds` | `drawer_open_max_seconds` |
| **Logic (plain language)** | Customer present in **customer zone**, no cashier in **cashier zone**, and cumulative wait exceeds the limit. | Open drawer in **cashier zone** (with the usual case preconditions) stays open longer than the limit. |
| **Output** | `data.use_case.cashier.summary.case_id` = `"A5"`, `severity` = `ALERT`, `alerts[]` text about customer waiting. | `case_id` = `"A6"`, `severity` = `ALERT`, `alerts[]` text about drawer open too long. |

**Do not swap:** **A5** is **customer-wait** (customer zone timer). **A6** is **drawer-open duration** (cashier zone timer).

**How to change thresholds**

1. **HTTP (no file edit):** `POST /cashier/zones` with a JSON body containing only `"thresholds": { … }` (see [thresholds-only update](#post-cashierzones-thresholds-only-update) below). Values merge into the existing file.  
2. **File edit:** Edit `config/cashier_zones.yaml` (or your `CASHIER_CONFIG` path) and save; wait up to `config_reload_interval` seconds, or restart the worker if you need instant pickup.  
3. **Verify:** `curl -s "http://<jetson-ip>:9000/cashier/zones" | jq '.thresholds'`.

---

#### Drawer metrics: per-frame count vs `drawer_open_count` API

Mnemonic (the route name is easy to misread):

- **`drawer_count` ≠** total frames where the drawer is open (or total drawer detections).
- **`drawer_count` =** number of logged **`triggered`** events in `events.jsonl` for that camera (alert/session starts and N3 transaction logs — see implementation note below).

| What you need | Where to get it |
|----------------|-----------------|
| **Open drawers in this frame** (zone tally) | SSE / stream JSON: `data.use_case.cashier.summary.cashier_zone.drawers` (integer count for the current frame). |
| **Lifetime “triggered” events** (log lines) | `GET /cashier/media/{camera_id}/drawer_count` — see below. |

**Important:** The route name says `drawer_count`, but the implementation counts **lines** in `evidence/.../logs/events.jsonl` where `status == "triggered"` for that `camera_id` (see [`apis/cashier.py`](../apis/cashier.py)). A `"triggered"` row is written when an **alert/critical evidence session starts** or when a **transaction (N3)** is logged — it is **not** a raw count of “how many times the drawer was detected open” across all frames. For “how many frames had an open drawer”, aggregate `cashier_zone.drawers > 0` from your stream or batch `stream.jsonl`. For “how many alert/critical sessions were opened”, use this endpoint or parse `events.jsonl`.

```bash
curl -s "http://<jetson-ip>:9000/cashier/media/cashier_cam_01/drawer_count"
```

**Sample JSON**

```json
{"camera_id": "cashier_cam_01", "drawer_open_count": 42}
```

---

### `POST /cashier/zones` (thresholds-only update)

You may send **only** `thresholds` to merge into the existing config (zones unchanged). Omitted keys in other sections are left as-is.

```bash
curl -s -X POST "http://<jetson-ip>:9000/cashier/zones" \
  -H "Content-Type: application/json" \
  -d '{
    "thresholds": {
      "drawer_open_max_seconds": 45,
      "customer_wait_max_seconds": 45,
      "proximity_iou": 0.06,
      "config_reload_interval": 60
    }
  }'
```

**Key fields**

- `thresholds.drawer_open_max_seconds` — Before **A6** escalation.  
- `thresholds.customer_wait_max_seconds` — Before **A5** escalation.  
- `thresholds.proximity_iou` — Person–drawer / person–cash association threshold.  
- `thresholds.config_reload_interval` — Seconds between cashier config reloads from disk.  

---

### `POST /cashier/zones`

Update ROI definitions and optional thresholds. Writes through to `CASHIER_CONFIG` (YAML or JSON). Cashier service reloads config on its reload interval (default 60s via `thresholds.config_reload_interval`).

**Important:** The JSON body must use **`points` as `[{"x":…,"y":…}, …]`**. The `[[x,y],[x,y]]` form is what is **stored** on disk and returned by `GET /cashier/zones`, not what `POST` accepts.

```bash
curl -s -X POST "http://<jetson-ip>:9000/cashier/zones" \
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
```

**Sample JSON request body (API shape — use for curl)**

```json
{
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
  }
}
```

**On-disk / `GET /cashier/zones` shape (reference only — do not send this to `POST` without wrapping as x/y objects)**

Rectangle:

```json
"ROI_CASHIER": {
  "shape": "rectangle",
  "points": [[0.0, 0.0], [0.5, 1.0]],
  "active": true
}
```

Polygon:

```json
"ROI_CUSTOMER": {
  "shape": "polygon",
  "points": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
  "active": true
}
```

**Sample JSON response (actual API)**

```json
{
  "status": "updated",
  "config": {
    "zones": {
      "ROI_CASHIER": {
        "shape": "rectangle",
        "points": [[0.0, 0.0], [0.5, 1.0]],
        "active": true
      },
      "ROI_CUSTOMER": {
        "shape": "polygon",
        "points": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
        "active": true
      }
    },
    "thresholds": {
      "drawer_open_max_seconds": 30,
      "customer_wait_max_seconds": 30
    }
  }
}
```

**Note:** Some docs use `{"status":"configured","zones":{…}}`; the running server returns **`status: "updated"`** and a full merged **`config`** (including `zones` and `thresholds`).

**Key fields**

- `status` — `"updated"` after a successful write.  
- `config` — Full merged configuration as persisted.  
- `config.zones` — Normalised `ROI_CASHIER` / `ROI_CUSTOMER` entries.  
- `config.thresholds` — Numeric tuning (wait/drawer timers, IOU, reload interval, etc.).  

---

## jq examples

**Current cashier thresholds from HTTP config**

```bash
curl -s "http://<jetson-ip>:9000/cashier/zones" | jq '.thresholds'
```

### `/detection/stream`

**Case id, severity, alerts from live stream**

```bash
curl -sN "http://<jetson-ip>:9000/detection/stream" \
  | grep "^data:" | sed 's/^data: //' \
  | jq '.data.use_case.cashier.summary | {case_id, severity, alerts}'
```

**Persons with zone and transaction (mood lives in a parallel array, not inside each person)**

```bash
curl -sN "http://<jetson-ip>:9000/detection/stream" \
  | grep "^data:" | sed 's/^data: //' \
  | jq '.data.use_case.cashier.persons[] | {zone, transaction}'
```

**Join cashier persons with mood and age_gender by matching `bbox` / `person_bbox`**

```bash
curl -sN "http://<jetson-ip>:9000/detection/stream" \
  | grep "^data:" | sed 's/^data: //' \
  | jq '
    .data.use_case as $u
    | ($u.cashier.persons // [])[]
    | . as $p
    | ($u.mood       // [] | map(select(.bbox == $p.person_bbox)) | .[0]) as $m
    | ($u.age_gender // [] | map(select(.bbox == $p.person_bbox)) | .[0]) as $a
    | {zone: $p.zone, transaction: $p.transaction, mood: $m, age_gender: $a}
  '
```

---

## Cashier ROI shapes, thresholds, and end of pipeline (curl)

Use these after the API is up (`uvicorn app:app --host 0.0.0.0 --port 9000`). Replace `<jetson-ip>` and `cam1` / `cashier_cam_01` with your camera id.

### Read zones and thresholds (on-disk / merged)

Full file-shaped JSON (includes `zones`, `thresholds`, `buffer`, `debounce`, …):

```bash
curl -s "http://<jetson-ip>:9000/cashier/zones"
```

Zones only (`ROI_CASHIER` / `ROI_CUSTOMER` as stored: `shape`, `points` as `[[x,y],…]`):

```bash
curl -s "http://<jetson-ip>:9000/cashier/zones" | jq '.zones'
```

Timing and IoU used for **A5** / **A6** / person–drawer linking:

```bash
curl -s "http://<jetson-ip>:9000/cashier/zones" | jq '.thresholds'
```

### POST polygons (matches default [`config/cashier_zones.yaml`](../config/cashier_zones.yaml))

`POST` requires `points` as `[{"x":…,"y":…}, …]` (not the `[[x,y],…]` form returned by `GET`).

```bash
curl -s -X POST "http://<jetson-ip>:9000/cashier/zones" \
  -H "Content-Type: application/json" \
  -d '{
    "ROI_CASHIER": {
      "shape": "polygon",
      "active": true,
      "points": [
        {"x": 0.32969, "y": 0.54861},
        {"x": 0.65000, "y": 0.52778},
        {"x": 0.69297, "y": 0.99444},
        {"x": 0.33516, "y": 0.98750}
      ]
    },
    "ROI_CUSTOMER": {
      "shape": "polygon",
      "active": true,
      "points": [
        {"x": 0.32031, "y": 0.28472},
        {"x": 0.64297, "y": 0.25556},
        {"x": 0.63516, "y": 0.00000},
        {"x": 0.31172, "y": 0.00000}
      ]
    },
    "thresholds": {
      "config_reload_interval": 60,
      "customer_wait_max_seconds": 30,
      "drawer_open_max_seconds": 30,
      "proximity_iou": 0.05
    }
  }'
```

### POST rectangle cashier ROI + polygon customer (alternative)

Two corners define the rectangle (normalised): lower-left and upper-right style pair in any order consistent with your layout.

```bash
curl -s -X POST "http://<jetson-ip>:9000/cashier/zones" \
  -H "Content-Type: application/json" \
  -d '{
    "ROI_CASHIER": {
      "shape": "rectangle",
      "active": true,
      "points": [{"x": 0.26, "y": 0.50}, {"x": 0.74, "y": 1.0}]
    },
    "ROI_CUSTOMER": {
      "shape": "polygon",
      "active": true,
      "points": [
        {"x": 0.32031, "y": 0.28472},
        {"x": 0.64297, "y": 0.25556},
        {"x": 0.63516, "y": 0.00000},
        {"x": 0.31172, "y": 0.00000}
      ]
    }
  }'
```

### POST thresholds only (zones unchanged)

```bash
curl -s -X POST "http://<jetson-ip>:9000/cashier/zones" \
  -H "Content-Type: application/json" \
  -d '{
    "thresholds": {
      "drawer_open_max_seconds": 45,
      "customer_wait_max_seconds": 45,
      "proximity_iou": 0.06,
      "config_reload_interval": 60
    }
  }'
```

### Reset zones file to code defaults

```bash
curl -s -X POST "http://<jetson-ip>:9000/cashier/zones/reset"
```

### End of pipeline (after `POST /detection/start`)

`cashier` last in the pipeline list:

```bash
curl -s -X POST "http://<jetson-ip>:9000/detection/setup" \
  -H "Content-Type: application/json" \
  -d '{"pipeline": ["detector", "age_gender", "mood", "cashier"]}'
```

Runtime health and frame counters:

```bash
curl -s "http://<jetson-ip>:9000/detection/status"
```

Latest cashier summary per camera (from in-memory state updated by `CashierService`):

```bash
curl -s "http://<jetson-ip>:9000/cashier/status"
```

Recent alerts / transactions (newest first):

```bash
curl -s "http://<jetson-ip>:9000/cashier/events?limit=20"
```

Triggered-event tally from `events.jsonl` (not per-frame drawer detections — see **Drawer metrics** in this document):

```bash
curl -s "http://<jetson-ip>:9000/cashier/media/cashier_cam_01/drawer_count"
```

Multiplexed vision SSE (all cameras):

```bash
curl -N "http://<jetson-ip>:9000/detection/stream"
```

Per-camera cashier SSE (`event:` lines: `connected`, `frame`, `alert`, …):

```bash
curl -N "http://<jetson-ip>:9000/cashier/stream/cashier_cam_01"
```

---

## Cashier case reference (N1–N6, A1–A7)

| case_id | severity | condition |
|--------|----------|-----------|
| N1 | NORMAL | Baseline; no other scenario matched (no alerts). |
| N2 | NORMAL | Cashier zone occupied; drawer closed; no customer in customer zone. |
| N3 | NORMAL | Transaction in progress (cashier + drawer + customer + cash/nearby rules satisfied). |
| N4 | NORMAL | Staff handover / supervisor at register (multiple staff + drawer context). |
| N5 | NORMAL | Customer in customer zone; no open drawer in cashier zone. |
| N6 | NORMAL | Drawer open; no cash visible (card / float check). |
| A1 | ALERT or CRITICAL | Unattended open drawer; **CRITICAL** if customer is present. |
| A2 | ALERT | Unexpected person in cashier zone (fallback after other rules). |
| A3 | CRITICAL | Cash + open drawer; cashier zone unoccupied. |
| A4 | CRITICAL | Open register with cash; person not near drawer (unauthorised pattern). |
| A5 | ALERT | Customer waiting longer than threshold; no cashier present. |
| A6 | ALERT | Drawer open longer than threshold. |
| A7 | ALERT | Cash in customer zone; no cashier; drawer closed. |

---

## Further reading

Additional cashier HTTP routes (status, events, evidence, per-camera SSE under `/cashier/stream/...`) are documented in OpenAPI at `/docs`.

**CASHIER_BOX_OPEN (Eyego, mocks, appendix JSON):** §§1–5 and the appendix in **this** document; scripts `scripts/curl_eyego_style_tasks.sh`, `scripts/curl_cashier_box_open_mock.sh`.

**JPEG vs GIF (which case saves what, and cURL for `/cashier/media/...`):** see **Event-level media (GIF vs JPEG)** in [`sse_cashier.md`](../sse_cashier.md).

---

# Mock `personStructural`, evidence, offline QA

The following sections (§6–§11) supplement §§1–4 above: quick ml-server curl, evidence table, parsed **`personStructural`** per case (N1–N6, A1–A7), synthetic `data` envelope, pytest commands, and severity cheat sheet.

**Cumulative drawer metrics:** `personStructural` (and `data`) include **`total_open_count`**, **`total_open_duration_ms`**, and **`current_open_duration_ms`**. Persisted under `evidence/cashier/logs/cashier_drawer_open_totals.json`. Disable with `CASHIER_DISABLE_DRAWER_TOTAL_PERSIST=1`. Optional `CASHIER_DRAWER_DURATION_PERSIST_SEC` (default 10) flushes duration while the drawer remains open.

**Offline QA:** fake detection counts map to **`personStructural`** (JSON string); shapes match **`pytest`** (`tests/test_cashier_box_open_cases.py`).

**Backend envelope** wraps these fields under `data`: `algorithmType`, `captureId`, `sceneId`, `channelId`, …, `personStructural`, `captureUrl`, `sceneUrl`. Live pipeline exposes `personStructural` on **`use_case.cashier`** ([`services/cashier.py`](../services/cashier.py)); cloud URLs are added by your integration layer.

---

## 6. Quick curl — ml-server after zones

```bash
export BASE=http://localhost:9000

# Load zones + task meta (same geometry as spec §3)
curl -sS -X POST "${BASE}/cashier/zones" \
  -H "Content-Type: application/json" \
  -d @scripts/curl_cashier_box_open_mock.json

# Parse latest personStructural from status (camera id = CASHIER_CAMERA_ID, default "cam")
curl -sS "${BASE}/cashier/status" | jq -r 'to_entries[0].value.personStructural | fromjson'
```

**Eyego-style task create** (string `areaPosition` / `detailConfig`): see [§1 — Backend task API (Eyego-style)](#1-backend-task-api-eyego-style).

---

## 7. Evidence behaviour (images / GIF)

| Level | Cases | What gets saved |
|--------|--------|------------------|
| NORMAL (no file) | N1, N2, N4, N5, N6 | State only — no JPEG |
| NORMAL (audit) | N3 | One annotated keyframe + JSONL `triggered` |
| ALERT | A1, A2, A5, A6, A7 | JPEG + GIF budgets in [`services/cashier.py`](../services/cashier.py) (`_GIF_BUDGET`) |
| CRITICAL | A3, A4 | Uncapped post-buffer until case resolves; GIF when event ends |

**Fetch evidence (running server):**

```bash
curl -sS -o /tmp/latest.jpg "${BASE}/cashier/media/cam/latest/jpg"
curl -sS -o /tmp/latest.gif "${BASE}/cashier/media/cam/latest/gif"
```

Replace `cam` with `CASHIER_CAMERA_ID`.

---

## 8. All 13 cases — mock `personStructural` (parsed)

Values below are **illustrative**; your engine fills `confidence`, `detections[]`, and durations from real frames.

### N1 — Idle register

```json
{
  "case_matched": "N1",
  "case_level": "NORMAL",
  "alert_triggered": false,
  "critical_triggered": false,
  "zones": {
    "cashier": {
      "persons_count": 0,
      "drawers_count": 0,
      "cash_count": 0,
      "unauthorized_present": false
    },
    "customer": { "persons_count": 0, "cash_count": 0 }
  },
  "detections": []
}
```

### N2 — Cashier on duty, no customer

```json
{
  "case_matched": "N2",
  "case_level": "NORMAL",
  "alert_triggered": false,
  "critical_triggered": false,
  "zones": {
    "cashier": {
      "persons_count": 1,
      "drawers_count": 0,
      "cash_count": 0,
      "unauthorized_present": false
    },
    "customer": { "persons_count": 0, "cash_count": 0 }
  },
  "detections": [
    {"class": "Person", "confidence": 0.91, "zone_id": 1}
  ]
}
```

### N3 — Active transaction (audit keyframe)

```json
{
  "case_matched": "N3",
  "case_level": "NORMAL",
  "alert_triggered": false,
  "critical_triggered": false,
  "zones": {
    "cashier": {
      "persons_count": 1,
      "drawers_count": 1,
      "cash_count": 1,
      "unauthorized_present": false
    },
    "customer": { "persons_count": 1, "cash_count": 1 }
  },
  "detections": [
    {"class": "Person", "confidence": 0.94, "zone_id": 1},
    {"class": "Drawer_Open", "confidence": 0.87, "zone_id": 1},
    {"class": "Cash", "confidence": 0.76, "zone_id": 1},
    {"class": "Person", "confidence": 0.89, "zone_id": 2},
    {"class": "Cash", "confidence": 0.72, "zone_id": 2}
  ]
}
```

### N4 — Two staff handover

```json
{
  "case_matched": "N4",
  "case_level": "NORMAL",
  "alert_triggered": false,
  "critical_triggered": false,
  "zones": {
    "cashier": {
      "persons_count": 2,
      "drawers_count": 1,
      "cash_count": 0,
      "unauthorized_present": false
    },
    "customer": { "persons_count": 0, "cash_count": 0 }
  },
  "detections": []
}
```

### N5 — Customer waiting (under wait limit)

```json
{
  "case_matched": "N5",
  "case_level": "NORMAL",
  "alert_triggered": false,
  "critical_triggered": false,
  "zones": {
    "cashier": {
      "persons_count": 0,
      "drawers_count": 0,
      "cash_count": 0,
      "unauthorized_present": false
    },
    "customer": { "persons_count": 1, "cash_count": 0 }
  },
  "detections": [
    {"class": "Person", "confidence": 0.88, "zone_id": 2}
  ]
}
```

### N6 — Card / no cash

```json
{
  "case_matched": "N6",
  "case_level": "NORMAL",
  "alert_triggered": false,
  "critical_triggered": false,
  "zones": {
    "cashier": {
      "persons_count": 1,
      "drawers_count": 1,
      "cash_count": 0,
      "unauthorized_present": false
    },
    "customer": { "persons_count": 0, "cash_count": 0 }
  },
  "detections": [
    {"class": "Person", "confidence": 0.9, "zone_id": 1},
    {"class": "Drawer_Open", "confidence": 0.85, "zone_id": 1}
  ]
}
```

### A1 — Unattended open drawer (ALERT or CRITICAL)

**CRITICAL** variant (customer in zone, drawer open, no cash) — row 3:

```json
{
  "case_matched": "A1",
  "case_level": "CRITICAL",
  "alert_triggered": true,
  "critical_triggered": true,
  "zones": {
    "cashier": {
      "persons_count": 0,
      "drawers_count": 1,
      "cash_count": 0,
      "unauthorized_present": false
    },
    "customer": { "persons_count": 1, "cash_count": 0 }
  },
  "detections": [
    {"class": "Drawer_Open", "confidence": 0.9, "zone_id": 1},
    {"class": "Person", "confidence": 0.85, "zone_id": 2}
  ],
  "drawer_open_duration_ms": 12000
}
```

**ALERT** variant (no customer, no cash) — row 4: same fields with `"case_level": "ALERT"` and `"critical_triggered": false`.

### A2 — Unauthorized person in cashier zone

When `enableStaffList` is true and bindings mark a cashier-zone person as non-staff, `unauthorized_present` is true and Person rows may include `is_authorized`:

```json
{
  "case_matched": "A2",
  "case_level": "ALERT",
  "alert_triggered": true,
  "critical_triggered": false,
  "zones": {
    "cashier": {
      "persons_count": 1,
      "drawers_count": 0,
      "cash_count": 0,
      "unauthorized_present": true
    },
    "customer": { "persons_count": 0, "cash_count": 0 }
  },
  "detections": [
    {"class": "Person", "confidence": 0.88, "zone_id": 1, "is_authorized": false}
  ]
}
```

### A3 — Cash + open drawer, no cashier (CRITICAL)

```json
{
  "case_matched": "A3",
  "case_level": "CRITICAL",
  "alert_triggered": true,
  "critical_triggered": true,
  "zones": {
    "cashier": {
      "persons_count": 0,
      "drawers_count": 1,
      "cash_count": 1,
      "unauthorized_present": false
    },
    "customer": { "persons_count": 0, "cash_count": 0 }
  },
  "detections": [
    {"class": "Drawer_Open", "confidence": 0.92, "zone_id": 1},
    {"class": "Cash", "confidence": 0.81, "zone_id": 1}
  ],
  "drawer_open_duration_ms": 3000
}
```

### A4 — Non-staff + open drawer + cash (CRITICAL)

```json
{
  "case_matched": "A4",
  "case_level": "CRITICAL",
  "alert_triggered": true,
  "critical_triggered": true,
  "zones": {
    "cashier": {
      "persons_count": 1,
      "drawers_count": 1,
      "cash_count": 1,
      "unauthorized_present": true
    },
    "customer": { "persons_count": 0, "cash_count": 0 }
  },
  "detections": [
    {"class": "Person", "confidence": 0.9, "zone_id": 1, "is_authorized": false},
    {"class": "Drawer_Open", "confidence": 0.88, "zone_id": 1},
    {"class": "Cash", "confidence": 0.82, "zone_id": 1}
  ],
  "drawer_open_duration_ms": 8000
}
```

### A5 — Customer waiting too long (ALERT)

```json
{
  "case_matched": "A5",
  "case_level": "ALERT",
  "alert_triggered": true,
  "critical_triggered": false,
  "zones": {
    "cashier": {
      "persons_count": 0,
      "drawers_count": 0,
      "cash_count": 0,
      "unauthorized_present": false
    },
    "customer": { "persons_count": 1, "cash_count": 0 }
  },
  "detections": [
    {"class": "Person", "confidence": 0.87, "zone_id": 2}
  ],
  "wait_duration_ms": 35000
}
```

### A6 — Drawer open too long (ALERT)

```json
{
  "case_matched": "A6",
  "case_level": "ALERT",
  "alert_triggered": true,
  "critical_triggered": false,
  "zones": {
    "cashier": {
      "persons_count": 1,
      "drawers_count": 1,
      "cash_count": 0,
      "unauthorized_present": false
    },
    "customer": { "persons_count": 0, "cash_count": 0 }
  },
  "detections": [
    {"class": "Person", "confidence": 0.91, "zone_id": 1},
    {"class": "Drawer_Open", "confidence": 0.86, "zone_id": 1}
  ],
  "drawer_open_duration_ms": 45000
}
```

### A7 — Cash left in customer zone (ALERT)

```json
{
  "case_matched": "A7",
  "case_level": "ALERT",
  "alert_triggered": true,
  "critical_triggered": false,
  "zones": {
    "cashier": {
      "persons_count": 0,
      "drawers_count": 0,
      "cash_count": 0,
      "unauthorized_present": false
    },
    "customer": { "persons_count": 0, "cash_count": 1 }
  },
  "detections": [
    {"class": "Cash", "confidence": 0.79, "zone_id": 2}
  ]
}
```

---

## 9. Full synthetic `data` envelope (spec §6 shape)

Example **N3** (pretty-printed `personStructural` string in real payloads is one line):

```json
{
  "data": {
    "algorithmType": "CASHIER_BOX_OPEN",
    "captureId": "CASHIER_BOX_OPEN_a3f9c2d1-e8b7-4a65-9c2d-1e8b7a654123.jpg",
    "sceneId": "CASHIER_BOX_OPEN_b4e0d3f2-c9a8-4b76-8e5f-2c9a8b765234.jpg",
    "channelId": 2,
    "channelName": "CAM-02-MAIN",
    "deviceSN": "HQDZW1SBCABAH0205",
    "id": "a3f9c2d1e8b74a659c2d1e8b7a654123",
    "taskId": 101,
    "taskName": "cashier_drawer_monitor",
    "recordTime": 1774463039587,
    "dateUTC": "2026-04-02T10:23:59.587Z",
    "personStructural": "{\"case_matched\":\"N3\",\"case_level\":\"NORMAL\",\"alert_triggered\":false,\"critical_triggered\":false,\"zones\":{\"cashier\":{\"persons_count\":1,\"drawers_count\":1,\"cash_count\":1,\"unauthorized_present\":false},\"customer\":{\"persons_count\":1,\"cash_count\":1}},\"detections\":[{\"class\":\"Person\",\"confidence\":0.94,\"zone_id\":1}]}",
    "captureUrl": "https://storage.example.com/.../capture.jpg",
    "sceneUrl": "https://storage.example.com/.../scene.jpg"
  }
}
```

---

## 10. Automated tests

```bash
# 14-rule table + personStructural helpers + evidence mapping check
python3 -m pytest tests/test_cashier_box_open_cases.py -v

# Legacy standalone script (same rules, no pytest)
python3 scripts/test_cashier_14_rules.py
```

---

## 11. Case → severity cheat sheet

| Code | Severity (typical) | Notes |
|------|-------------------|--------|
| N1–N6 | NORMAL | N3 is the “EVENT” audit case |
| A1 | ALERT or CRITICAL | CRITICAL when customer present, no cash (row 3) |
| A2, A5, A6, A7 | ALERT | |
| A3, A4 | CRITICAL | |

First matching row in the 14-rule chain wins — see [`services/cashier.py`](../services/cashier.py) `_evaluate`.
---

## Appendix — `cashier_all_cases_output.json` (machine-readable)

Regenerate (updates the fenced block below):

```bash
python3 scripts/generate_cashier_all_cases_output.py
```

<!-- BEGIN cashier_all_cases_json -->

```json
{
  "meta": {
    "algorithm": "CASHIER_BOX_OPEN",
    "note": "Illustrative values for ids, times, totals, and detections. Wire payloads may wrap as { \"data\": <object> }. ml-server also mirrors these fields on use_case.cashier.summary.",
    "case_codes_13": [
      "N1",
      "N2",
      "N3",
      "N4",
      "N5",
      "N6",
      "A1",
      "A2",
      "A3",
      "A4",
      "A5",
      "A6",
      "A7"
    ],
    "extra_keys": "A1 has two examples: A1_ALERT and A1_CRITICAL (same case_matched code A1)."
  },
  "cases": {
    "N1": {
      "description": "Idle register — no persons, drawer closed",
      "data": {
        "algorithmType": "CASHIER_BOX_OPEN",
        "captureId": "CASHIER_BOX_OPEN_n1_cap.jpg",
        "sceneId": "CASHIER_BOX_OPEN_n1_scene.jpg",
        "channelId": 2,
        "channelName": "CAM-02-MAIN",
        "deviceSN": "HQDZW1SBCABAH0205",
        "id": "n10000000000000000000000000000001",
        "taskId": 101,
        "taskName": "cashier_drawer_monitor",
        "recordTime": 1774463039587,
        "dateUTC": "2026-04-02T10:23:59.587Z",
        "total_open_count": 0,
        "total_open_duration_ms": 0,
        "current_open_duration_ms": 0,
        "personStructural": "{\"case_matched\":\"N1\",\"case_level\":\"NORMAL\",\"alert_triggered\":false,\"critical_triggered\":false,\"total_open_count\":0,\"total_open_duration_ms\":0,\"current_open_duration_ms\":0,\"zones\":{\"cashier\":{\"persons_count\":0,\"drawers_count\":0,\"cash_count\":0,\"unauthorized_present\":false},\"customer\":{\"persons_count\":0,\"cash_count\":0}},\"detections\":[]}",
        "captureUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_n1_cap.jpg",
        "sceneUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_n1_scene.jpg"
      }
    },
    "N2": {
      "description": "Cashier on duty, drawer closed",
      "data": {
        "algorithmType": "CASHIER_BOX_OPEN",
        "captureId": "CASHIER_BOX_OPEN_n2_cap.jpg",
        "sceneId": "CASHIER_BOX_OPEN_n2_scene.jpg",
        "channelId": 2,
        "channelName": "CAM-02-MAIN",
        "deviceSN": "HQDZW1SBCABAH0205",
        "id": "n20000000000000000000000000000002",
        "taskId": 101,
        "taskName": "cashier_drawer_monitor",
        "recordTime": 1774463040587,
        "dateUTC": "2026-04-02T10:23:59.587Z",
        "total_open_count": 2,
        "total_open_duration_ms": 480000,
        "current_open_duration_ms": 0,
        "personStructural": "{\"case_matched\":\"N2\",\"case_level\":\"NORMAL\",\"alert_triggered\":false,\"critical_triggered\":false,\"total_open_count\":2,\"total_open_duration_ms\":480000,\"current_open_duration_ms\":0,\"zones\":{\"cashier\":{\"persons_count\":1,\"drawers_count\":0,\"cash_count\":0,\"unauthorized_present\":false},\"customer\":{\"persons_count\":0,\"cash_count\":0}},\"detections\":[{\"class\":\"Person\",\"confidence\":0.91,\"zone_id\":1}]}",
        "captureUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_n2_cap.jpg",
        "sceneUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_n2_scene.jpg"
      }
    },
    "N3": {
      "description": "Active transaction",
      "data": {
        "algorithmType": "CASHIER_BOX_OPEN",
        "captureId": "CASHIER_BOX_OPEN_n3_cap.jpg",
        "sceneId": "CASHIER_BOX_OPEN_n3_scene.jpg",
        "channelId": 2,
        "channelName": "CAM-02-MAIN",
        "deviceSN": "HQDZW1SBCABAH0205",
        "id": "n30000000000000000000000000000003",
        "taskId": 101,
        "taskName": "cashier_drawer_monitor",
        "recordTime": 1774463041587,
        "dateUTC": "2026-04-02T10:23:59.587Z",
        "total_open_count": 5,
        "total_open_duration_ms": 890000,
        "current_open_duration_ms": 12000,
        "personStructural": "{\"case_matched\":\"N3\",\"case_level\":\"NORMAL\",\"alert_triggered\":false,\"critical_triggered\":false,\"total_open_count\":5,\"total_open_duration_ms\":890000,\"current_open_duration_ms\":12000,\"zones\":{\"cashier\":{\"persons_count\":1,\"drawers_count\":1,\"cash_count\":1,\"unauthorized_present\":false},\"customer\":{\"persons_count\":1,\"cash_count\":1}},\"detections\":[{\"class\":\"Person\",\"confidence\":0.94,\"zone_id\":1},{\"class\":\"Drawer_Open\",\"confidence\":0.87,\"zone_id\":1},{\"class\":\"Cash\",\"confidence\":0.76,\"zone_id\":1},{\"class\":\"Person\",\"confidence\":0.89,\"zone_id\":2},{\"class\":\"Cash\",\"confidence\":0.72,\"zone_id\":2}]}",
        "captureUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_n3_cap.jpg",
        "sceneUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_n3_scene.jpg"
      }
    },
    "N4": {
      "description": "Two staff handover — all near drawer",
      "data": {
        "algorithmType": "CASHIER_BOX_OPEN",
        "captureId": "CASHIER_BOX_OPEN_n4_cap.jpg",
        "sceneId": "CASHIER_BOX_OPEN_n4_scene.jpg",
        "channelId": 2,
        "channelName": "CAM-02-MAIN",
        "deviceSN": "HQDZW1SBCABAH0205",
        "id": "n40000000000000000000000000000004",
        "taskId": 101,
        "taskName": "cashier_drawer_monitor",
        "recordTime": 1774463042587,
        "dateUTC": "2026-04-02T10:23:59.587Z",
        "total_open_count": 6,
        "total_open_duration_ms": 920000,
        "current_open_duration_ms": 5000,
        "personStructural": "{\"case_matched\":\"N4\",\"case_level\":\"NORMAL\",\"alert_triggered\":false,\"critical_triggered\":false,\"total_open_count\":6,\"total_open_duration_ms\":920000,\"current_open_duration_ms\":5000,\"zones\":{\"cashier\":{\"persons_count\":2,\"drawers_count\":1,\"cash_count\":0,\"unauthorized_present\":false},\"customer\":{\"persons_count\":0,\"cash_count\":0}},\"detections\":[{\"class\":\"Person\",\"confidence\":0.92,\"zone_id\":1},{\"class\":\"Person\",\"confidence\":0.9,\"zone_id\":1},{\"class\":\"Drawer_Open\",\"confidence\":0.88,\"zone_id\":1}]}",
        "captureUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_n4_cap.jpg",
        "sceneUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_n4_scene.jpg"
      }
    },
    "N5": {
      "description": "Customer waiting, under wait limit, drawer closed",
      "data": {
        "algorithmType": "CASHIER_BOX_OPEN",
        "captureId": "CASHIER_BOX_OPEN_n5_cap.jpg",
        "sceneId": "CASHIER_BOX_OPEN_n5_scene.jpg",
        "channelId": 2,
        "channelName": "CAM-02-MAIN",
        "deviceSN": "HQDZW1SBCABAH0205",
        "id": "n50000000000000000000000000000005",
        "taskId": 101,
        "taskName": "cashier_drawer_monitor",
        "recordTime": 1774463043587,
        "dateUTC": "2026-04-02T10:23:59.587Z",
        "total_open_count": 3,
        "total_open_duration_ms": 125000,
        "current_open_duration_ms": 0,
        "personStructural": "{\"case_matched\":\"N5\",\"case_level\":\"NORMAL\",\"alert_triggered\":false,\"critical_triggered\":false,\"total_open_count\":3,\"total_open_duration_ms\":125000,\"current_open_duration_ms\":0,\"zones\":{\"cashier\":{\"persons_count\":0,\"drawers_count\":0,\"cash_count\":0,\"unauthorized_present\":false},\"customer\":{\"persons_count\":1,\"cash_count\":0}},\"detections\":[{\"class\":\"Person\",\"confidence\":0.88,\"zone_id\":2}]}",
        "captureUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_n5_cap.jpg",
        "sceneUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_n5_scene.jpg"
      }
    },
    "N6": {
      "description": "Drawer open, no cash (card / float)",
      "data": {
        "algorithmType": "CASHIER_BOX_OPEN",
        "captureId": "CASHIER_BOX_OPEN_n6_cap.jpg",
        "sceneId": "CASHIER_BOX_OPEN_n6_scene.jpg",
        "channelId": 2,
        "channelName": "CAM-02-MAIN",
        "deviceSN": "HQDZW1SBCABAH0205",
        "id": "n60000000000000000000000000000006",
        "taskId": 101,
        "taskName": "cashier_drawer_monitor",
        "recordTime": 1774463044587,
        "dateUTC": "2026-04-02T10:23:59.587Z",
        "total_open_count": 7,
        "total_open_duration_ms": 950000,
        "current_open_duration_ms": 8000,
        "personStructural": "{\"case_matched\":\"N6\",\"case_level\":\"NORMAL\",\"alert_triggered\":false,\"critical_triggered\":false,\"total_open_count\":7,\"total_open_duration_ms\":950000,\"current_open_duration_ms\":8000,\"zones\":{\"cashier\":{\"persons_count\":1,\"drawers_count\":1,\"cash_count\":0,\"unauthorized_present\":false},\"customer\":{\"persons_count\":0,\"cash_count\":0}},\"detections\":[{\"class\":\"Person\",\"confidence\":0.9,\"zone_id\":1},{\"class\":\"Drawer_Open\",\"confidence\":0.85,\"zone_id\":1}]}",
        "captureUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_n6_cap.jpg",
        "sceneUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_n6_scene.jpg"
      }
    },
    "A1_ALERT": {
      "description": "A1 ALERT — unattended open drawer, no customer, no cash",
      "data": {
        "algorithmType": "CASHIER_BOX_OPEN",
        "captureId": "CASHIER_BOX_OPEN_a1a_cap.jpg",
        "sceneId": "CASHIER_BOX_OPEN_a1a_scene.jpg",
        "channelId": 2,
        "channelName": "CAM-02-MAIN",
        "deviceSN": "HQDZW1SBCABAH0205",
        "id": "a1a0000000000000000000000000001",
        "taskId": 101,
        "taskName": "cashier_drawer_monitor",
        "recordTime": 1774463045587,
        "dateUTC": "2026-04-02T10:23:59.587Z",
        "total_open_count": 8,
        "total_open_duration_ms": 1000000,
        "current_open_duration_ms": 15000,
        "personStructural": "{\"case_matched\":\"A1\",\"case_level\":\"ALERT\",\"alert_triggered\":true,\"critical_triggered\":false,\"total_open_count\":8,\"total_open_duration_ms\":1000000,\"current_open_duration_ms\":15000,\"zones\":{\"cashier\":{\"persons_count\":0,\"drawers_count\":1,\"cash_count\":0,\"unauthorized_present\":false},\"customer\":{\"persons_count\":0,\"cash_count\":0}},\"detections\":[{\"class\":\"Drawer_Open\",\"confidence\":0.9,\"zone_id\":1}],\"drawer_open_duration_ms\":15000}",
        "captureUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_a1a_cap.jpg",
        "sceneUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_a1a_scene.jpg"
      }
    },
    "A1_CRITICAL": {
      "description": "A1 CRITICAL — customer in zone, drawer open, no cash",
      "data": {
        "algorithmType": "CASHIER_BOX_OPEN",
        "captureId": "CASHIER_BOX_OPEN_a1c_cap.jpg",
        "sceneId": "CASHIER_BOX_OPEN_a1c_scene.jpg",
        "channelId": 2,
        "channelName": "CAM-02-MAIN",
        "deviceSN": "HQDZW1SBCABAH0205",
        "id": "a1c0000000000000000000000000001",
        "taskId": 101,
        "taskName": "cashier_drawer_monitor",
        "recordTime": 1774463046587,
        "dateUTC": "2026-04-02T10:23:59.587Z",
        "total_open_count": 8,
        "total_open_duration_ms": 1015000,
        "current_open_duration_ms": 12000,
        "personStructural": "{\"case_matched\":\"A1\",\"case_level\":\"CRITICAL\",\"alert_triggered\":true,\"critical_triggered\":true,\"total_open_count\":8,\"total_open_duration_ms\":1015000,\"current_open_duration_ms\":12000,\"zones\":{\"cashier\":{\"persons_count\":0,\"drawers_count\":1,\"cash_count\":0,\"unauthorized_present\":false},\"customer\":{\"persons_count\":1,\"cash_count\":0}},\"detections\":[{\"class\":\"Drawer_Open\",\"confidence\":0.9,\"zone_id\":1},{\"class\":\"Person\",\"confidence\":0.85,\"zone_id\":2}],\"drawer_open_duration_ms\":12000}",
        "captureUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_a1c_cap.jpg",
        "sceneUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_a1c_scene.jpg"
      }
    },
    "A2": {
      "description": "Unauthorized person in cashier zone",
      "data": {
        "algorithmType": "CASHIER_BOX_OPEN",
        "captureId": "CASHIER_BOX_OPEN_a2_cap.jpg",
        "sceneId": "CASHIER_BOX_OPEN_a2_scene.jpg",
        "channelId": 2,
        "channelName": "CAM-02-MAIN",
        "deviceSN": "HQDZW1SBCABAH0205",
        "id": "a2000000000000000000000000000002",
        "taskId": 101,
        "taskName": "cashier_drawer_monitor",
        "recordTime": 1774463047587,
        "dateUTC": "2026-04-02T10:23:59.587Z",
        "total_open_count": 4,
        "total_open_duration_ms": 600000,
        "current_open_duration_ms": 0,
        "personStructural": "{\"case_matched\":\"A2\",\"case_level\":\"ALERT\",\"alert_triggered\":true,\"critical_triggered\":false,\"total_open_count\":4,\"total_open_duration_ms\":600000,\"current_open_duration_ms\":0,\"zones\":{\"cashier\":{\"persons_count\":1,\"drawers_count\":0,\"cash_count\":0,\"unauthorized_present\":true},\"customer\":{\"persons_count\":0,\"cash_count\":0}},\"detections\":[{\"class\":\"Person\",\"confidence\":0.88,\"zone_id\":1,\"is_authorized\":false}]}",
        "captureUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_a2_cap.jpg",
        "sceneUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_a2_scene.jpg"
      }
    },
    "A3": {
      "description": "Cash + open drawer, no cashier",
      "data": {
        "algorithmType": "CASHIER_BOX_OPEN",
        "captureId": "CASHIER_BOX_OPEN_a3_cap.jpg",
        "sceneId": "CASHIER_BOX_OPEN_a3_scene.jpg",
        "channelId": 2,
        "channelName": "CAM-02-MAIN",
        "deviceSN": "HQDZW1SBCABAH0205",
        "id": "a3000000000000000000000000000003",
        "taskId": 101,
        "taskName": "cashier_drawer_monitor",
        "recordTime": 1774463048587,
        "dateUTC": "2026-04-02T10:23:59.587Z",
        "total_open_count": 9,
        "total_open_duration_ms": 1100000,
        "current_open_duration_ms": 3000,
        "personStructural": "{\"case_matched\":\"A3\",\"case_level\":\"CRITICAL\",\"alert_triggered\":true,\"critical_triggered\":true,\"total_open_count\":9,\"total_open_duration_ms\":1100000,\"current_open_duration_ms\":3000,\"zones\":{\"cashier\":{\"persons_count\":0,\"drawers_count\":1,\"cash_count\":1,\"unauthorized_present\":false},\"customer\":{\"persons_count\":0,\"cash_count\":0}},\"detections\":[{\"class\":\"Drawer_Open\",\"confidence\":0.92,\"zone_id\":1},{\"class\":\"Cash\",\"confidence\":0.81,\"zone_id\":1}],\"drawer_open_duration_ms\":3000}",
        "captureUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_a3_cap.jpg",
        "sceneUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_a3_scene.jpg"
      }
    },
    "A4": {
      "description": "Non-staff + open drawer + cash",
      "data": {
        "algorithmType": "CASHIER_BOX_OPEN",
        "captureId": "CASHIER_BOX_OPEN_a4_cap.jpg",
        "sceneId": "CASHIER_BOX_OPEN_a4_scene.jpg",
        "channelId": 2,
        "channelName": "CAM-02-MAIN",
        "deviceSN": "HQDZW1SBCABAH0205",
        "id": "a4000000000000000000000000000004",
        "taskId": 101,
        "taskName": "cashier_drawer_monitor",
        "recordTime": 1774463049587,
        "dateUTC": "2026-04-02T10:23:59.587Z",
        "total_open_count": 10,
        "total_open_duration_ms": 1150000,
        "current_open_duration_ms": 8000,
        "personStructural": "{\"case_matched\":\"A4\",\"case_level\":\"CRITICAL\",\"alert_triggered\":true,\"critical_triggered\":true,\"total_open_count\":10,\"total_open_duration_ms\":1150000,\"current_open_duration_ms\":8000,\"zones\":{\"cashier\":{\"persons_count\":1,\"drawers_count\":1,\"cash_count\":1,\"unauthorized_present\":true},\"customer\":{\"persons_count\":0,\"cash_count\":0}},\"detections\":[{\"class\":\"Person\",\"confidence\":0.9,\"zone_id\":1,\"is_authorized\":false},{\"class\":\"Drawer_Open\",\"confidence\":0.88,\"zone_id\":1},{\"class\":\"Cash\",\"confidence\":0.82,\"zone_id\":1}],\"drawer_open_duration_ms\":8000}",
        "captureUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_a4_cap.jpg",
        "sceneUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_a4_scene.jpg"
      }
    },
    "A5": {
      "description": "Customer waiting too long",
      "data": {
        "algorithmType": "CASHIER_BOX_OPEN",
        "captureId": "CASHIER_BOX_OPEN_a5_cap.jpg",
        "sceneId": "CASHIER_BOX_OPEN_a5_scene.jpg",
        "channelId": 2,
        "channelName": "CAM-02-MAIN",
        "deviceSN": "HQDZW1SBCABAH0205",
        "id": "a5000000000000000000000000000005",
        "taskId": 101,
        "taskName": "cashier_drawer_monitor",
        "recordTime": 1774463050587,
        "dateUTC": "2026-04-02T10:23:59.587Z",
        "total_open_count": 3,
        "total_open_duration_ms": 125000,
        "current_open_duration_ms": 0,
        "personStructural": "{\"case_matched\":\"A5\",\"case_level\":\"ALERT\",\"alert_triggered\":true,\"critical_triggered\":false,\"total_open_count\":3,\"total_open_duration_ms\":125000,\"current_open_duration_ms\":0,\"zones\":{\"cashier\":{\"persons_count\":0,\"drawers_count\":0,\"cash_count\":0,\"unauthorized_present\":false},\"customer\":{\"persons_count\":1,\"cash_count\":0}},\"detections\":[{\"class\":\"Person\",\"confidence\":0.87,\"zone_id\":2}],\"wait_duration_ms\":35000}",
        "captureUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_a5_cap.jpg",
        "sceneUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_a5_scene.jpg"
      }
    },
    "A6": {
      "description": "Drawer open too long",
      "data": {
        "algorithmType": "CASHIER_BOX_OPEN",
        "captureId": "CASHIER_BOX_OPEN_a6_cap.jpg",
        "sceneId": "CASHIER_BOX_OPEN_a6_scene.jpg",
        "channelId": 2,
        "channelName": "CAM-02-MAIN",
        "deviceSN": "HQDZW1SBCABAH0205",
        "id": "a6000000000000000000000000000006",
        "taskId": 101,
        "taskName": "cashier_drawer_monitor",
        "recordTime": 1774463051587,
        "dateUTC": "2026-04-02T10:23:59.587Z",
        "total_open_count": 11,
        "total_open_duration_ms": 1200000,
        "current_open_duration_ms": 45000,
        "personStructural": "{\"case_matched\":\"A6\",\"case_level\":\"ALERT\",\"alert_triggered\":true,\"critical_triggered\":false,\"total_open_count\":11,\"total_open_duration_ms\":1200000,\"current_open_duration_ms\":45000,\"zones\":{\"cashier\":{\"persons_count\":1,\"drawers_count\":1,\"cash_count\":0,\"unauthorized_present\":false},\"customer\":{\"persons_count\":0,\"cash_count\":0}},\"detections\":[{\"class\":\"Person\",\"confidence\":0.91,\"zone_id\":1},{\"class\":\"Drawer_Open\",\"confidence\":0.86,\"zone_id\":1}],\"drawer_open_duration_ms\":45000}",
        "captureUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_a6_cap.jpg",
        "sceneUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_a6_scene.jpg"
      }
    },
    "A7": {
      "description": "Cash in customer zone, no cashier",
      "data": {
        "algorithmType": "CASHIER_BOX_OPEN",
        "captureId": "CASHIER_BOX_OPEN_a7_cap.jpg",
        "sceneId": "CASHIER_BOX_OPEN_a7_scene.jpg",
        "channelId": 2,
        "channelName": "CAM-02-MAIN",
        "deviceSN": "HQDZW1SBCABAH0205",
        "id": "a7000000000000000000000000000007",
        "taskId": 101,
        "taskName": "cashier_drawer_monitor",
        "recordTime": 1774463052587,
        "dateUTC": "2026-04-02T10:23:59.587Z",
        "total_open_count": 3,
        "total_open_duration_ms": 125000,
        "current_open_duration_ms": 0,
        "personStructural": "{\"case_matched\":\"A7\",\"case_level\":\"ALERT\",\"alert_triggered\":true,\"critical_triggered\":false,\"total_open_count\":3,\"total_open_duration_ms\":125000,\"current_open_duration_ms\":0,\"zones\":{\"cashier\":{\"persons_count\":0,\"drawers_count\":0,\"cash_count\":0,\"unauthorized_present\":false},\"customer\":{\"persons_count\":0,\"cash_count\":1}},\"detections\":[{\"class\":\"Cash\",\"confidence\":0.79,\"zone_id\":2}]}",
        "captureUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_a7_cap.jpg",
        "sceneUrl": "https://storage.googleapis.com/logs-data-images/CASHIER_BOX_OPEN_a7_scene.jpg"
      }
    }
  }
}
```

<!-- END cashier_all_cases_json -->
