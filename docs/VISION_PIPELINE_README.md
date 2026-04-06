# Vision Pipeline — combined README (API, tests, cashier cases)

Single reference that merges:

- **Service testing** — cURL, SSH, pytest for **v2** (`TASK_REGISTRY`, `/cashier/*`) — previously split across [`SERVICE_TEST.md`](./SERVICE_TEST.md) (stub) and [`tests/pipeline_test/README.md`](../tests/pipeline_test/README.md) (stub).
- **CASHIER_BOX_OPEN / cashier integration** — where frame **`data`** lives, schema, GIF/evidence (summary in Part II below); **Eyego cURL, Part III (Vision Pipeline cashier cURL / thresholds / stream), mock `personStructural`, appendix JSON:** [`CASHIER_BOX_OPEN.md`](./CASHIER_BOX_OPEN.md). Former **`cases-and-repo.md`** material is summarized here.

**API version:** `2.0.0` (`GET /`).

| More detail | Doc |
|-------------|-----|
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
8. [Tasks — CASHIER_DRAWER](#part-i--5-tasks--cashier_drawer-cashier-monitor)
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
python3 -m pytest tests/pipeline_test/ tests/test_task_services_smoke.py tests/test_cashier_box_open_cases.py -v --tb=short
```

| Test module | What it covers |
|-------------|----------------|
| `tests/pipeline_test/test_pipeline_api.py` | `REGISTRY` / `TASK_REGISTRY` keys, OpenAPI paths, cameras, detection stop (409), cashier REST |
| `tests/test_task_services_smoke.py` | `CrossLineTask`, `MaskHairnetChefHatTask` (empty frame), `CashierDrawerTask` (mocked ONNX service) |
| `tests/test_cashier_box_open_cases.py` | Cashier rule table / `CashierService` offline logic |

**Single-file API smoke test:**

```bash
python3 -m pytest tests/pipeline_test/test_pipeline_api.py -v
```

---

## Part I — Two kinds of “services”

| Kind | Where | HTTP? |
|------|--------|--------|
| **`REGISTRY`** | `services/__init__.py` | No direct routes — used by **`pipeline.py`** / `CameraPipeline` (batch or scripts): `detector`, `age_gender`, `ppe`, `mood`, `cashier` |
| **`TASK_REGISTRY`** | Same file | Driven by **`POST /api/tasks`** + **`POST /detection/start`**: `CROSS_LINE`, `MASK_HAIRNET_CHEF_HAT`, `CASHIER_DRAWER` |

Cashier appears in both: **`CashierService`** and **`CashierDrawerTask`** in **`services/cashier.py`**.

---

## Part I — Environment

```bash
export BASE=http://localhost:9000
# Remote box:
export BASE=http://192.168.1.50:9000
```

```bash
uvicorn app:app --host 0.0.0.0 --port 9000
```

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

## Part I — 4. Tasks — `MASK_HAIRNET_CHEF_HAT` (PPE zone)

```bash
curl -s -X POST "$BASE/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 20,
    "taskName": "kitchen_ppe",
    "algorithmType": "MASK_HAIRNET_CHEF_HAT",
    "channelId": 1,
    "enable": true,
    "threshold": 70,
    "areaPosition": "[{\"line_id\":\"z1\",\"point\":[{\"x\":50,\"y\":50},{\"x\":600,\"y\":50},{\"x\":600,\"y\":500},{\"x\":50,\"y\":500}],\"direction\":0}]",
    "detailConfig": { "alarmType": ["no_mask", "no_hat"] }
  }'
```

### Example response (200)

```json
{
  "status": "created",
  "task": {
    "taskId": 20,
    "taskName": "kitchen_ppe",
    "algorithmType": "MASK_HAIRNET_CHEF_HAT",
    "channelId": 1,
    "enable": true,
    "threshold": 70,
    "areaPosition": "[{\"line_id\":\"z1\",\"point\":[{\"x\":50,\"y\":50},{\"x\":600,\"y\":50},{\"x\":600,\"y\":500},{\"x\":50,\"y\":500}],\"direction\":0}]",
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

---

## Part I — 5. Tasks — `CASHIER_DRAWER` (cashier monitor)

Set **`YOLO_MODEL`** to cashier weights so FrameBus emits person/drawer/cash on that channel. Use **`/cashier/*`** for zones, status, SSE.

```bash
curl -s -X POST "$BASE/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 30,
    "taskName": "cashier_mon",
    "algorithmType": "CASHIER_DRAWER",
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
    "algorithmType": "CASHIER_DRAWER",
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
    "trackingId": 42,
    "boundingBox": {},
    "attributes": {},
    "confidence": 0.91
  },
  "evidence": { "captureImage": "…", "sceneImage": "…" }
}
```

### SSH — task JSONL (if enabled)

```bash
ssh user@jetson 'tail -f /local/storage/events/task_10.jsonl'
```

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

### SSH — evidence log

```bash
ssh user@jetson 'tail -f /path/to/ml-server/evidence/cashier/logs/events.jsonl'
```

---

## Part I — 9. Error samples

```bash
curl -s -X POST "$BASE/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{"taskId":99,"taskName":"x","algorithmType":"UNKNOWN","channelId":1}'
```

```json
{
  "detail": "Unsupported algorithmType 'UNKNOWN'. Supported: ['CASHIER_DRAWER', 'CROSS_LINE', 'MASK_HAIRNET_CHEF_HAT']"
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
| NORMAL (audit) | N3 | Annotated keyframe + JSONL `triggered` |
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

- **`logs/events.jsonl`** — one line per `triggered` / `resolved` (and related) events; GIF paths after compile.
- **`logs/cashier_drawer_open_totals.json`** — drawer count + duration aggregates (optional).
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
| [`tests/test_cashier_box_open_cases.py`](../tests/test_cashier_box_open_cases.py) | Offline rule / envelope tests |

---

## CI

```bash
python3 -m pytest tests/pipeline_test/ -v --tb=short
```
