# Vision Pipeline — combined README (API, tests, cashier cases)

Single reference that merges:

- **Service testing** — cURL, SSH, and optional **pytest** under `tests/` for **v2** (`TASK_REGISTRY`, `/cashier/*`). Part I below is the canonical place for commands and paths.
- **CASHIER_BOX_OPEN / cashier integration** — where frame **`data`** lives, schema, GIF/evidence (summary in Part II below); **Eyego cURL, Part III (Vision Pipeline cashier cURL / thresholds / stream), mock `personStructural`, appendix JSON:** [`CASHIER_BOX_OPEN.md`](./CASHIER_BOX_OPEN.md). Former **`cases-and-repo.md`** material is summarized here.

**API version:** `2.0.0` (`GET /`).

| More detail | Doc |
|-------------|-----|
| Docker / Compose failures (NVIDIA runtime, image arch, healthcheck, Redis volume) | [stream_test_runbook.md](./stream_test_runbook.md#docker-troubleshooting) |
| Tests, log paths, curl cheat sheet | Part I of this file; [API_USAGE.md](./API_USAGE.md) for HTTP walkthrough |
| Optimization env vars, CI, security flags | [OPTIMIZATION_REFERENCE.md](./OPTIMIZATION_REFERENCE.md) |
| ML Image V2 — evidence, env, disk, SSH / JSONL | [../service_doc/ml_image_v2.md](../service_doc/ml_image_v2.md) |
| Add a FrameBus task | [ADDING_A_SERVICE.md](./ADDING_A_SERVICE.md) |
| API walkthrough | [API_USAGE.md](./API_USAGE.md) |
| Cashier (Eyego, `/cashier` cURL Part III, mocks, appendix JSON) | [CASHIER_BOX_OPEN.md](./CASHIER_BOX_OPEN.md) |
| Cashier SSE wire format | [sse_cashier.md](../sse_cashier.md) |

---

## Table of contents

**Part I — Operations & service tests**

1. [Run automated tests](#part-i--run-automated-tests-all-services)
2. [REGISTRY vs TASK_REGISTRY](#part-i--two-kinds-of-services)
3. [Environment](#part-i--environment)
4. [Health](#part-i--1-health)
5. [Cameras](#part-i--2-cameras)
6. [Tasks — CROSS_LINE](#part-i--3-tasks--cross_line)
7. [Tasks — MASK_HAIRNET_CHEF_HAT](#part-i--4-tasks--mask_hairnet_chef_hat-ppe-zone)
8. [Tasks — CASHIER_BOX_OPEN](#part-i--5-tasks--cashier_box_open-cashier-monitor)
9. [List / get / delete tasks](#part-i--6-list--get--delete-tasks)
10. [Detection — start, status, stop, stream](#part-i--7-detection--start-status-stop-stream)
11. [Cashier HTTP](#part-i--8-cashier-http-all-services-on-cashier)
12. [Error samples](#part-i--9-error-samples)
13. [Full local checklist](#part-i--10-full-local-checklist-copy-paste)
14. [Reference — task `payload`](#part-i-reference--payload-shape-from-adding_a_service)

**Part II — CASHIER_BOX_OPEN (frame data, cases, GIF, evidence)** *(formerly cases-and-repo.md)*

15. [Where this payload appears](#part-ii--where-this-payload-appears)
16. [Common `data` schema](#part-ii--common-data-schema-all-cases)
17. [Drawer metrics & persistence](#part-ii--drawer-metrics--persistence)
18. [GIF & evidence by case](#part-ii--gif--evidence-by-case)
19. [Evidence repository layout](#part-ii--evidence-repository-layout)
20. [Full examples (machine-readable)](#part-ii--full-per-case-json-machine-readable)
21. [Quick curl — zones + status](#part-ii--quick-curl--ml-server-after-zones)
22. [Related files](#part-ii--related-files)

---

# Part I — Operations & service tests

## Part I — Run automated tests (all services)

From the repo root:

```bash
# Fast CI-style run (no ML models, no Docker)
pip install -r requirements-ci.txt
python3 -m pytest tests/unit/ tests/contract/ tests/security/ \
  -m "not models and not integration and not e2e and not soak" -v --tb=short

# Full tree (may skip faiss / ONNX placeholders)
python3 -m pytest tests/ -v --tb=short
```

| Test area | Examples |
|-----------|----------|
| Unit | `test_cashier_parametrized.py`, `test_geometry_properties.py`, `test_sse_schema_snapshot.py`, `test_multi_camera_throughput.py`, `test_cross_line_line_patch.py` |
| Contract | `tests/contract/test_sse_filters.py`, OpenAPI snapshot |
| Security | `tests/security/test_path_traversal.py` (auth, upload cap, path traversal) |
| Integration | `tests/integration/` — use `docker compose -f docker-compose.test.yml up -d` |
| E2E | `tests/e2e/test_final_system_validation.py` |

CI: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). See [OPTIMIZATION_REFERENCE.md](./OPTIMIZATION_REFERENCE.md) for env flags used in production tuning.

---

## Part I — Two kinds of “services”

| Kind | Where | HTTP? |
|------|--------|--------|
| **`REGISTRY`** | `services/__init__.py` | No direct routes — used by **`pipeline.py`** / `CameraPipeline` (batch or scripts): `detector`, `age_gender`, `ppe`, `mood`, `cashier` |
| **`TASK_REGISTRY`** | Same file | Driven by **`POST /api/tasks`** + **`POST /detection/start`**: `CROSS_LINE`, `MASK_HAIRNET_CHEF_HAT`, `CASHIER_BOX_OPEN`, `PHONE_USAGE` |

Cashier appears in both: **`CashierService`** and **`CashierDrawerTask`** (registered as **`CASHIER_BOX_OPEN`**) in **`services/cashier.py`**.

---

## Part I — Environment

```bash
export BASE=http://localhost:9000
# Remote box:
export BASE=http://192.168.1.50:9000
# Optional — full URLs in V2 evidence:
export PUBLIC_ML_BASE_URL=https://ml.example.com
# Optional — batch `CameraPipeline` only (`full` | `light` | `none`):
export PIPELINE_IMAGE_MODE=full
```

```bash
uvicorn app:app --host 0.0.0.0 --port 9000
```

Image contract + disk + SSH: [../service_doc/ml_image_v2.md](../service_doc/ml_image_v2.md).

---

## Part I — 1. Health

### cURL

```bash
curl -s "$BASE/"
```

### Example response (200)

```json
{
  "service": "Vision Pipeline API",
  "version": "2.0.0"
}
```

### SSH

```bash
ssh user@192.168.1.50 'curl -s http://127.0.0.1:9000/'
```

---

## Part I — 2. Cameras

Camera **`id`** (string) must match **`channelId`** from tasks when you call `POST /detection/start` (both compared as strings).

### Register

```bash
curl -s -X POST "$BASE/cameras" \
  -H "Content-Type: application/json" \
  -d '{"cameras":[{"id":"1","url":"rtsp://192.168.1.10/stream"}]}'
```

### Example response (200)

```json
{
  "status": "configured",
  "cameras": {
    "1": "rtsp://192.168.1.10/stream"
  },
  "error": null
}
```

### List / delete

```bash
curl -s "$BASE/cameras"
curl -s -X DELETE "$BASE/cameras/1"
```

### SSH

```bash
ssh user@jetson 'curl -s http://127.0.0.1:9000/cameras'
```

---

## Part I — 3. Tasks — `CROSS_LINE`

FrameBus must produce **tracked persons** (`class_name == "person"`, `track_id != -1`) for crossings on **`GET /detection/stream`**.

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
    "detailConfig": { "enableAttrDetect": false },
    "validWeekday": ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"],
    "validStartTime": 0,
    "validEndTime": 86400000
  }'
```

### Example response (200)

```json
{
  "status": "created",
  "task": {
    "taskId": 10,
    "taskName": "entrance_line",
    "algorithmType": "CROSS_LINE",
    "channelId": 1,
    "enable": true,
    "threshold": 60,
    "areaPosition": "[{\"line_id\":\"1\",\"line_name\":\"Entrance\",\"point\":[{\"x\":100,\"y\":400},{\"x\":900,\"y\":400}],\"direction\":1}]",
    "detailConfig": {
      "enableAttrDetect": false,
      "enableReid": false,
      "alarmType": []
    },
    "validWeekday": ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"],
    "validStartTime": 0,
    "validEndTime": 86400000
  }
}
```

---

## Part I — 4. Tasks — `MASK_HAIRNET_CHEF_HAT` (PPE polygon zone)

PPE uses a **polygon** in pixel coordinates — not a line. Events use the Eyego **`data`** envelope with **`personStructural`**. Guide: [../service_doc/mask_hairnet_chef_hat.md](../service_doc/mask_hairnet_chef_hat.md).

```bash
curl -s -X POST "$BASE/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 8,
    "taskName": "staff_safety_bar_area",
    "algorithmType": "MASK_HAIRNET_CHEF_HAT",
    "channelId": 7,
    "enable": true,
    "threshold": 70,
    "areaPosition": "[{\"x\":1004,\"y\":56},{\"x\":1831,\"y\":89},{\"x\":2013,\"y\":1831},{\"x\":308,\"y\":1876},{\"x\":304,\"y\":1872}]",
    "detailConfig": {
      "alarmType": ["no_mask", "no_hat"],
      "channelName": "7",
      "deviceSN": "HQDZW1SBCABAH0235"
    }
  }'
```

### Example response (200)

```json
{
  "status": "created",
  "task": {
    "taskId": 8,
    "taskName": "staff_safety_bar_area",
    "algorithmType": "MASK_HAIRNET_CHEF_HAT",
    "channelId": 7,
    "enable": true,
    "threshold": 70,
    "areaPosition": "[{\"x\":1004,\"y\":56},{\"x\":1831,\"y\":89},{\"x\":2013,\"y\":1831},{\"x\":308,\"y\":1876},{\"x\":304,\"y\":1872}]",
    "detailConfig": {
      "enableAttrDetect": false,
      "enableReid": false,
      "alarmType": ["no_mask", "no_hat"]
    },
    "validWeekday": ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"],
    "validStartTime": 0,
    "validEndTime": 86400000
  }
}
```

### Example SSE event (illustrative)

```json
{
  "eventType": "MASK_HAIRNET_CHEF_HAT",
  "taskId": 8,
  "taskName": "staff_safety_bar_area",
  "channelId": "7",
  "data": {
    "algorithmType": "MASK_HAIRNET_CHEF_HAT",
    "channelId": 7,
    "channelName": "7",
    "deviceSN": "HQDZW1SBCABAH0235",
    "personStructural": "{\"alarmType\":\"no_chef_hat\",\"areaPoints\":\"[{\\\"x\\\":1004,\\\"y\\\":56}]\",\"objectX\":1417,\"objectY\":115,\"objectWidth\":117,\"objectHeight\":144,\"score\":79}",
    "captureUrl": "https://storage.googleapis.com/logs-data-images/MASK_HAIRNET_CHEF_HAT_….jpg….jpg",
    "sceneUrl": "https://storage.googleapis.com/logs-data-images/MASK_HAIRNET_CHEF_HAT_….jpg….jpg"
  },
  "evidence": {
    "captureImage": { "type": "capture", "path": "2026-03-24/MASK_HAIRNET_CHEF_HAT_….jpg" },
    "sceneImage":   { "type": "scene",   "path": "2026-03-24/MASK_HAIRNET_CHEF_HAT_….jpg" }
  }
}
```

Filter SSE: `curl -sN "$BASE/detection/stream?eventType=MASK_HAIRNET_CHEF_HAT&taskId=8"`. URL bases: `PPE_CLOUD_IMAGE_BASE` / `PPE_CAPTURE_URL_BASE` / `PPE_SCENE_URL_BASE`.

---

## Part I — 5. Tasks — `CASHIER_BOX_OPEN` (cashier monitor)

Set **`YOLO_MODEL`** to cashier weights so FrameBus emits person/drawer/cash on that channel. Use **`/cashier/*`** for zones, status, and per-camera SSE; use **`GET /detection/stream?eventType=CASHIER_BOX_OPEN`** for multiplexed structured frames and **`$EVENTS_DIR/task_<taskId>.jsonl`** for durable JSONL (see [Run automated tests](#part-i--run-automated-tests-all-services) above). Each cashier line matches the **task-event** pattern used by `CROSS_LINE` (one JSON object per line, shared top-level keys) but carries nested Eyego **`data`** (`personStructural` as a string, **pretty JSON by default**; **`id`** ties to **`captureId`** UUID; URL bases via **`CASHIER_CLOUD_IMAGE_BASE`** / per-side bases / optional **`CASHIER_FORCE_LOCAL_URLS`**).

```bash
curl -s -X POST "$BASE/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 30,
    "taskName": "cashier_mon",
    "algorithmType": "CASHIER_BOX_OPEN",
    "channelId": 1,
    "enable": true,
    "threshold": 50,
    "areaPosition": "[]",
    "detailConfig": {}
  }'
```

### Example response (200)

```json
{
  "status": "created",
  "task": {
    "taskId": 30,
    "taskName": "cashier_mon",
    "algorithmType": "CASHIER_BOX_OPEN",
    "channelId": 1,
    "enable": true,
    "threshold": 50,
    "areaPosition": "[]",
    "detailConfig": {
      "enableAttrDetect": false,
      "enableReid": false,
      "alarmType": []
    },
    "validWeekday": ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"],
    "validStartTime": 0,
    "validEndTime": 86400000
  }
}
```

---

## Part I — 6. List / get / delete tasks

```bash
curl -s "$BASE/api/tasks"
curl -s "$BASE/api/tasks/10"
curl -s -X DELETE "$BASE/api/tasks/30"
```

### Example `GET /api/tasks` (200)

```json
{
  "count": 3,
  "tasks": [
    {
      "taskId": 10,
      "taskName": "entrance_line",
      "algorithmType": "CROSS_LINE",
      "channelId": 1,
      "enable": true,
      "threshold": 60,
      "areaPosition": "...",
      "detailConfig": {
        "enableAttrDetect": false,
        "enableReid": false,
        "alarmType": []
      },
      "validWeekday": ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"],
      "validStartTime": 0,
      "validEndTime": 86400000
    }
  ]
}
```

---

## Part I — 7. Detection — start, status, stop, stream

There is **no** `POST /detection/setup` in the v2 HTTP API.

### Start

```bash
curl -s -X POST "$BASE/detection/start"
curl -s -X POST "$BASE/detection/start?camera_id=1"
```

### Example success (200)

```json
{
  "status": "started",
  "cameras": ["1"],
  "tasks": ["10", "20", "30"]
}
```

### Status / stop

```bash
curl -s "$BASE/detection/status"
curl -s -X POST "$BASE/detection/stop"
curl -s -X POST "$BASE/detection/stop?camera_id=1"
```

### Example 409 (nothing running)

```json
{
  "detail": "No cameras are currently running."
}
```

### SSE — task events

```bash
curl -sN --max-time 15 "$BASE/detection/stream" | head -n 20
```

### Example `data:` line (illustrative `CROSS_LINE`)

```json
{
  "eventId": "…",
  "eventType": "CROSS_LINE",
  "timestamp": 1774310401528,
  "timestampUTC": "2026-04-02T12:00:00.000Z",
  "taskId": 10,
  "taskName": "entrance_line",
  "channelId": 1,
  "line": { "id": "1", "name": "Entrance", "direction": 1 },
  "person": {
    "trackingId": "42",
    "boundingBox": {},
    "attributes": {},
    "confidence": 91
  },
  "evidence": {
    "captureImage": { "url": "…", "path": "2026-04-22/cam-1_…_….jpg", "type": "capture", "format": "image/jpeg", "timestamp": "2026-04-22T12:00:00.000Z" },
    "sceneImage": { "url": "…", "path": "2026-04-22/cam-1_…_….jpg", "type": "scene", "format": "image/jpeg", "timestamp": "2026-04-22T12:00:00.000Z" }
  }
}
```

`captureImage` / `sceneImage` are **ML Image Contract V2** objects, not raw paths. Full contract: [../service_doc/ml_image_v2.md](../service_doc/ml_image_v2.md).

### Example structured `CASHIER_BOX_OPEN` (illustrative)

Same outer envelope as above; body under **`data`**. `personStructural` is abbreviated — in real output it is a longer pretty-printed JSON string.

```json
{
  "eventId": "…",
  "eventType": "CASHIER_BOX_OPEN",
  "timestamp": 1774310401528,
  "timestampUTC": "2026-04-02T12:00:00.000Z",
  "taskId": 101,
  "taskName": "cashier_drawer_monitor",
  "channelId": 1,
  "camera_id": "1",
  "case_id": "N3",
  "severity": "NORMAL",
  "data": {
    "algorithmType": "CASHIER_BOX_OPEN",
    "captureId": "CASHIER_BOX_OPEN_550e8400-e29b-41d4-a716-446655440000.jpg",
    "sceneId": "CASHIER_BOX_OPEN_6ba7b810-9dad-11d1-80b4-00c04fd430c8.jpg",
    "id": "550e8400e29b41d4a716446655440000",
    "personStructural": "{\n  \"case_matched\": \"N3\",\n  \"case_level\": \"INFO\"\n}",
    "captureUrl": "",
    "sceneUrl": "",
    "evidence": {
      "captureImage": { "url": "…", "path": "2026-04-22/CASHIER_BOX_OPEN_….jpg", "type": "capture", "format": "image/jpeg", "timestamp": "2026-04-22T12:00:00.000Z" },
      "sceneImage": { "url": "…", "path": "2026-04-22/CASHIER_BOX_OPEN_….jpg", "type": "scene", "format": "image/jpeg", "timestamp": "2026-04-22T12:00:00.000Z" }
    }
  }
}
```

### SSH — task JSONL (per algorithm)

Each enabled task appends one JSON line per event to **`$EVENTS_DIR/task_<taskId>.jsonl`** (default `EVENTS_DIR=/local/storage/events`). Replace `10` with your task id from `GET /api/tasks`.

```bash
# Cross-line, PPE, phone, or cashier — same file naming pattern
ssh user@jetson 'tail -f /local/storage/events/task_10.jsonl'
ssh user@jetson 'tail -f /local/storage/events/task_20.jsonl'
# Pretty-print last line evidence
ssh user@jetson 'tail -n1 /local/storage/events/task_10.jsonl' | python3 -m json.tool
```

Reference: [../service_doc/ml_image_v2.md](../service_doc/ml_image_v2.md).

---

## Part I — 8. Cashier HTTP (all services on `/cashier`)

### Zones

```bash
curl -s "$BASE/cashier/zones"
curl -s -X POST "$BASE/cashier/zones" \
  -H "Content-Type: application/json" \
  -d '{"thresholds":{"drawer_open_max_seconds": 45}}'
curl -s -X POST "$BASE/cashier/zones/reset"
```

### Status, events, evidence

```bash
curl -s "$BASE/cashier/status"
curl -s "$BASE/cashier/events?limit=10&severity=ALERT"
curl -s -X DELETE "$BASE/cashier/events"
curl -s "$BASE/cashier/evidence?limit=5"
```

### SSE

```bash
curl -sN --max-time 10 "$BASE/cashier/stream/1" | head -n 15
curl -sN --max-time 10 "$BASE/cashier/stream/1/only" | head -n 15
```

### Media

```bash
curl -s "$BASE/cashier/media/1/drawer_count"
```

### SSH — structured cashier task JSONL

```bash
ssh user@jetson 'tail -f /local/storage/events/task_101.jsonl'
```

(`EVENTS_DIR` defaults to `/local/storage/events`; filename is `task_<taskId>.jsonl`.)

---

## Part I — 9. Error samples

```bash
curl -s -X POST "$BASE/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{"taskId":99,"taskName":"x","algorithmType":"UNKNOWN","channelId":1}'
```

```json
{
  "detail": "Unsupported algorithmType 'UNKNOWN'. Supported: ['CASHIER_BOX_OPEN', 'CROSS_LINE', 'MASK_HAIRNET_CHEF_HAT']"
}
```

---

## Part I — 10. Full local checklist (copy-paste)

```bash
export BASE=http://127.0.0.1:9000
curl -s "$BASE/"
curl -s -X POST "$BASE/cameras" -H "Content-Type: application/json" \
  -d '{"cameras":[{"id":"1","url":"rtsp://127.0.0.1/test"}]}'
curl -s -X POST "$BASE/api/tasks" -H "Content-Type: application/json" \
  -d '{"taskId":10,"taskName":"line","algorithmType":"CROSS_LINE","channelId":1,"threshold":50,"areaPosition":"[]","detailConfig":{}}'
curl -s "$BASE/api/tasks"
curl -s "$BASE/detection/status"
curl -s -X POST "$BASE/detection/stop"
curl -s "$BASE/cashier/zones"
```

---

## Part I — Reference — `payload` shape (from ADDING_A_SERVICE)

```json
{
  "camera_id": "1",
  "frame_id": 42,
  "timestamp": "2026-04-06T12:00:01.123456",
  "frame_b64": "<jpeg base64>",
  "frame": "<ndarray in process>",
  "detection": {
    "count": 2,
    "items": ["<Detection objects>"]
  }
}
```

See [ADDING_A_SERVICE.md](./ADDING_A_SERVICE.md) for the `Detection` dataclass fields.

---

# Part II — CASHIER_BOX_OPEN (frame data, cases, GIF, evidence)

This section replaces the former root file **`cases-and-repo.md`**: where integration **`data`** appears, field meanings, and artifact layout. Full JSON for every case is in **[`docs/CASHIER_BOX_OPEN.md`](./CASHIER_BOX_OPEN.md)** (appendix — fenced JSON) — not duplicated inline here.

---

## Part II — Where this payload appears

| Surface | Location |
|---------|----------|
| **Pipeline / internal context** | `context["data"]["use_case"]["cashier"]` includes a top-level **`data`** object (Eyego-style keys), plus `persons`, `summary`, `personStructural`, etc. |
| **HTTP** | `GET /cashier/status` → per-camera object includes **`data`** when populated. |
| **Wire / cloud bridge** | Your backend may wrap the object as **`{ "data": { … } }`** exactly as in the integration spec. |

**Dynamic fields (typically each frame):** `captureId`, `sceneId`, `id`, `recordTime`, `dateUTC`. **`captureUrl` / `sceneUrl`** are empty unless `CASHIER_CLOUD_IMAGE_BASE` or `CASHIER_CAPTURE_URL_BASE` / `CASHIER_SCENE_URL_BASE` are set.

**Stable / config-backed:** `channelId`, `channelName`, `taskId`, `taskName`, `deviceSN`, `algorithmType` — from `config/cashier_zones.yaml` → `task` block and env (`CASHIER_DEVICE_SN`, `DEVICE_SN`, `CASHIER_CHANNEL_NAME`).

---

## Part II — Common `data` schema (all cases)

Typical keys on the **`data`** object (Eyego / spec §4 / §6):

| Field | Type | Notes |
|-------|------|--------|
| `algorithmType` | string | e.g. `CASHIER_BOX_OPEN` |
| `captureId` | string | Logical capture JPEG name |
| `sceneId` | string | Logical scene JPEG name |
| `channelId` | int | Camera / channel |
| `channelName` | string | Display name |
| `taskId` | int | Task id |
| `taskName` | string | e.g. `cashier_drawer_monitor` |
| `deviceSN` | string | From env / config |
| `id` | string | Unique record id |
| `recordTime` | int | Epoch ms |
| `dateUTC` | string | ISO Z |
| `total_open_count` | int | Cumulative drawer open edges (cashier ROI) |
| `total_open_duration_ms` | int | Cumulative ms drawer open |
| `current_open_duration_ms` | int | Current open streak (0 if closed) |
| `personStructural` | string | JSON string: zones, case, detections, flags |
| `captureUrl` | string | Optional cloud URL |
| `sceneUrl` | string | Optional cloud URL |

The **`personStructural`** string parses to an object with `case_matched`, `case_level`, `alert_triggered`, `critical_triggered`, `zones` (cashier/customer counts, `unauthorized_present`), `detections`, optional `drawer_open_duration_ms` / `wait_duration_ms`, and the drawer total fields above. See [CASHIER_BOX_OPEN.md](./CASHIER_BOX_OPEN.md) §8 for parsed examples per case **N1–N6**, **A1–A7**.

---

## Part II — Drawer metrics & persistence

- **`total_open_count`** — rising edge when drawer goes closed → open in cashier ROI (per camera).
- **`total_open_duration_ms`** — cumulative open time (inter-frame sampling while previously open).
- **Persistence:** `evidence/cashier/logs/cashier_drawer_open_totals.json` (`by_camera`, `total_open_duration_ms_by_camera`). Disable with **`CASHIER_DISABLE_DRAWER_TOTAL_PERSIST=1`**. Optional **`CASHIER_DRAWER_DURATION_PERSIST_SEC`** (default 10) flushes duration while the drawer stays open.

---

## Part II — GIF & evidence by case

| Level | Cases | Behaviour |
|-------|-------|-----------|
| NORMAL (no file) | N1, N2, N4, N5, N6 | State only — no JPEG |
| NORMAL (audit) | N3 | Annotated keyframe (no legacy `events.jsonl` row) |
| ALERT | A1, A2, A5, A6, A7 | JPEG + GIF budgets (`_GIF_BUDGET` in `services/cashier.py`) |
| CRITICAL | A3, A4 | Large post-buffer until case resolves; GIF when event ends |

**Fetch from running server:**

```bash
curl -sS -o /tmp/latest.jpg "${BASE}/cashier/media/cam/latest/jpg"
curl -sS -o /tmp/latest.gif "${BASE}/cashier/media/cam/latest/gif"
```

Replace `cam` with your `CASHIER_CAMERA_ID` (default `cam`).

---

## Part II — Evidence repository layout

Under **`CASHIER_EVIDENCE_DIR`** (default `./evidence/cashier`):

- **`$EVENTS_DIR/task_<taskId>.jsonl`** — one JSON line per processed cashier frame (structured `CASHIER_BOX_OPEN` event), same pattern as `CROSS_LINE` task logs.
- **`logs/cashier_drawer_open_totals.json`** — drawer open-edge count + duration aggregates (optional).
- **Case folders** — e.g. `normal/N3/`, `alert/A5/`, `critical/A3/` with JPEG + sidecar JSON (see `services/cashier.py` → `_EvidenceWriter`).

---

## Part II — Full per-case JSON (machine-readable)

All cases **N1–N6**, **A1–A7** (plus extra variants where applicable) as complete **`data`** objects are in **[`docs/CASHIER_BOX_OPEN.md`](./CASHIER_BOX_OPEN.md)** (appendix — fenced JSON).

Regenerate that block:

```bash
python3 scripts/generate_cashier_all_cases_output.py
```

---

## Part II — Quick curl — ml-server after zones

```bash
export BASE=http://localhost:9000

curl -sS -X POST "${BASE}/cashier/zones" \
  -H "Content-Type: application/json" \
  -d @scripts/curl_cashier_box_open_mock.json

curl -sS "${BASE}/cashier/status" | jq -r 'to_entries[0].value.personStructural | fromjson'
```

**Eyego-style task create/update** (external API): [CASHIER_BOX_OPEN.md](./CASHIER_BOX_OPEN.md) §1.

---

## Part II — Related files

| Path | Role |
|------|------|
| [CASHIER_BOX_OPEN.md](./CASHIER_BOX_OPEN.md) | Eyego `POST/PUT` + Part III (cashier cURL) + mocks (§§6–11) + appendix JSON |
| [`services/cashier.py`](../services/cashier.py) | 14-rule evaluation, GIF budgets, persistence |
| `tests/test_cashier_api.py`, `tests/test_cashier_structured_events.py` | Cashier HTTP + structured event envelope |

---

## CI

```bash
python3 -m pytest tests/ -v --tb=short
```

If nothing is collected under `tests/`, pytest exits with code **5** (no tests); that is normal until you add or restore test modules.
