# API Usage Guide

Base URL for all examples: `http://localhost:9000`

**See also:** [VISION_PIPELINE_README.md](./VISION_PIPELINE_README.md) — pytest, log/JSONL paths (`EVENTS_DIR`), cURL, SSH, and **CASHIER_BOX_OPEN** `data` / evidence. **ML Image Contract V2** (structured `evidence`, env, disk, SSH): [../service_doc/ml_image_v2.md](../service_doc/ml_image_v2.md). Eyego cURL, mock responses, and full-case JSON: [CASHIER_BOX_OPEN.md](./CASHIER_BOX_OPEN.md).

---

## Table of Contents

1. [Register Cameras](#1-register-cameras)
2. [Register Tasks](#2-register-tasks)
3. [Start Detection](#3-start-detection)
4. [Monitor Status](#4-monitor-status)
5. [Stream Results (SSE)](#5-stream-results-sse)
6. [Stop Detection](#6-stop-detection)
7. [Task Management (CRUD)](#7-task-management-crud)
8. [Common Errors](#8-common-errors)
9. [Full Walkthrough Example](#9-full-walkthrough-example)
10. [Cashier monitor API (`/cashier/*`)](#10-cashier-monitor-api-cashier)
11. [Live stream & annotated frames (`/cameras/{id}/live`)](#11-live-stream-websocket-camerasidlive)
12. [Task-name live stream alias (`/tasks/{task_name}/live`)](#12-task-name-live-stream-alias-taskstask_namelive)

---

## 1. Register Cameras

The body takes a `cameras` array — you can register one or multiple cameras in a single call. The `id` must match the `channelId` used in your tasks.

### Add one camera
```bash
curl -X POST http://localhost:9000/cameras \
  -H "Content-Type: application/json" \
  -d '{
    "cameras": [
      {"id": "1", "url": "rtsp://192.168.1.10/stream"}
    ]
  }'
```

**Response:**
```json
{
  "status": "configured",
  "cameras": {
    "1": "rtsp://192.168.1.10/stream"
  }
}
```

### Add multiple cameras in one request
```bash
curl -X POST http://localhost:9000/cameras \
  -H "Content-Type: application/json" \
  -d '{
    "cameras": [
      {"id": "1", "url": "rtsp://192.168.1.10/stream"},
      {"id": "2", "url": "rtsp://192.168.1.11/stream"}
    ]
  }'
```

**Response:**
```json
{
  "status": "configured",
  "cameras": {
    "1": "rtsp://192.168.1.10/stream",
    "2": "rtsp://192.168.1.11/stream"
  }
}
```

### List all registered cameras
```bash
curl http://localhost:9000/cameras
```

**Response:**
```json
{
  "count": 2,
  "cameras": [
    {"id": "1", "url": "rtsp://192.168.1.10/stream"},
    {"id": "2", "url": "rtsp://192.168.1.11/stream"}
  ]
}
```

### Delete a camera
```bash
curl -X DELETE http://localhost:9000/cameras/1
```

**Response:**
```json
{
  "status": "removed",
  "camera_id": "1"
}
```

---

## 2. Register Tasks

Tasks are registered independently of cameras. The `channelId` links a task to a camera.

### Register a CrossLine task (no age/gender)
```bash
curl -X POST http://localhost:9000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 10,
    "taskName": "entrance_line",
    "algorithmType": "CROSS_LINE",
    "channelId": 1,
    "enable": true,
    "threshold": 60,
    "areaPosition": "[{\"line_id\":\"1\",\"line_name\":\"Entrance\",\"point\":[{\"x\":100,\"y\":400},{\"x\":900,\"y\":400}],\"direction\":1}]",
    "detailConfig": {
      "enableAttrDetect": false
    },
    "validWeekday": ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY"],
    "validStartTime": 28800000,
    "validEndTime": 72000000
  }'
```

**Response:**
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
    "validWeekday": ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY"],
    "validStartTime": 28800000,
    "validEndTime": 72000000
  }
}
```

### Register a CrossLine task (with age/gender on crossing person)
```bash
curl -X POST http://localhost:9000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 11,
    "taskName": "exit_line_with_attrs",
    "algorithmType": "CROSS_LINE",
    "channelId": 1,
    "threshold": 55,
    "areaPosition": "[{\"line_id\":\"2\",\"line_name\":\"Exit\",\"point\":[{\"x\":200,\"y\":600},{\"x\":800,\"y\":600}],\"direction\":2}]",
    "detailConfig": {
      "enableAttrDetect": true
    }
  }'
```

> `validWeekday`, `validStartTime`, and `validEndTime` are optional — they default to all days, all day.

### Register a PPE task (MASK_HAIRNET_CHEF_HAT)
```bash
curl -X POST http://localhost:9000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 20,
    "taskName": "kitchen_ppe_check",
    "algorithmType": "MASK_HAIRNET_CHEF_HAT",
    "channelId": 2,
    "threshold": 70,
    "areaPosition": "[{\"line_id\":\"zone1\",\"point\":[{\"x\":50,\"y\":50},{\"x\":600,\"y\":50},{\"x\":600,\"y\":500},{\"x\":50,\"y\":500}],\"direction\":0}]",
    "detailConfig": {
      "alarmType": ["no_mask", "no_chef_hat", "no_hat"]
    }
  }'
```

**Response:**
```json
{
  "status": "created",
  "task": {
    "taskId": 20,
    "taskName": "kitchen_ppe_check",
    "algorithmType": "MASK_HAIRNET_CHEF_HAT",
    "channelId": 2,
    "enable": true,
    "threshold": 70,
    "areaPosition": "...",
    "detailConfig": {
      "enableAttrDetect": false,
      "enableReid": false,
      "alarmType": ["no_mask", "no_chef_hat", "no_hat"]
    },
    "validWeekday": ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"],
    "validStartTime": 0,
    "validEndTime": 86400000
  }
}
```

### Register a cashier drawer task (`CASHIER_BOX_OPEN`)

Implementation: **`CashierDrawerTask`** and **`CashierService`** in [`services/cashier.py`](../services/cashier.py). Use `/cashier/*` HTTP routes for zones, status, and SSE. Point **`YOLO_MODEL`** at cashier weights on the server so FrameBus emits person/drawer/cash classes on that channel.

```bash
curl -X POST http://localhost:9000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 30,
    "taskName": "cashier_drawer_monitor",
    "algorithmType": "CASHIER_BOX_OPEN",
    "channelId": 1,
    "enable": true,
    "threshold": 50,
    "areaPosition": "[]",
    "detailConfig": {}
  }'
```

**Response:**
```json
{
  "status": "created",
  "task": {
    "taskId": 30,
    "taskName": "cashier_drawer_monitor",
    "algorithmType": "CASHIER_BOX_OPEN",
    "channelId": 1,
    "enable": true,
    "threshold": 50,
    "areaPosition": "[]",
    "detailConfig": {
      "drawerOpenLimit": 20,
      "serviceWaitLimit": 90,
      "enableStaffList": false,
      "staffIds": []
    },
    "validWeekday": ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"],
    "validStartTime": 0,
    "validEndTime": 86400000
  }
}
```

> Cashier-specific knobs are supported in `detailConfig`: `drawerOpenLimit`, `serviceWaitLimit`, `enableStaffList`, `staffIds`.

---

## 3. Start Detection

### Start all cameras (all tasks)
```bash
curl -X POST http://localhost:9000/detection/start
```

**Response:**
```json
{
  "status": "started",
  "cameras": ["1", "2"],
  "tasks": ["10", "11", "20", "30"]
}
```

This spawns:
- 1 FrameBus process for camera `1` (serving tasks 10 and 11)
- 1 FrameBus process for camera `2` (serving task 20)
- 1 task worker process per task (4 total)

### Start one specific camera only
```bash
curl -X POST "http://localhost:9000/detection/start?camera_id=1"
```

**Response:**
```json
{
  "status": "started",
  "cameras": ["1"],
  "tasks": ["10", "11"]
}
```

---

## 4. Monitor Status

```bash
curl http://localhost:9000/detection/status
```

**Response (while running):**
```json
{
  "cameras": {
    "1": {
      "camera_id": "1",
      "rtsp_url": "rtsp://192.168.1.10/stream",
      "running": true,
      "frame_count": 1452,
      "fps": 24.8,
      "last_detections": 3,
      "total_detections": 4210,
      "uptime_seconds": 58.6,
      "error": null
    },
    "2": {
      "camera_id": "2",
      "rtsp_url": "rtsp://192.168.1.11/stream",
      "running": true,
      "frame_count": 1447,
      "fps": 24.7,
      "last_detections": 1,
      "total_detections": 2890,
      "uptime_seconds": 58.6,
      "error": null
    }
  }
}
```

**Response (after stopping or on error):**
```json
{
  "cameras": {
    "1": {
      "camera_id": "1",
      "rtsp_url": "rtsp://192.168.1.10/stream",
      "running": false,
      "frame_count": 1452,
      "fps": 0.0,
      "last_detections": 0,
      "total_detections": 4210,
      "uptime_seconds": 58.6,
      "error": null
    }
  }
}
```

---

## 5. Stream Results (SSE)

Connect once and receive **detection events** (line crossings, PPE violations, cashier frames) in real-time as JSON. For **live annotated video frames**, use the WebSocket endpoint instead — see [section 11](#11-live-stream-websocket-camerasidlive).

Events arrive as they happen — one JSON object per **line crossing**, **PPE violation**, or **cashier frame** (`CASHIER_BOX_OPEN` emits one structured event per processed frame while detection is running).

**Multiple clients** can connect simultaneously; each client receives its own copy of every event. Idle connections receive `: ping` keepalive comments every ~30 seconds.

> **Deployment note:** When Redis is configured (`REDIS_URL`), the SSE bridge subscribes from `live:event:*` on Redis in addition to the in-process queue — multiple uvicorn workers can all serve SSE clients from the same broadcast. Without Redis it falls back to a single-worker in-process broadcast (default [`docker-compose.yml`](../docker-compose.yml) runs one worker).

```bash
# All events from all tasks and cameras
curl -N http://localhost:9000/detection/stream
```

> `-N` disables buffering so you see events immediately.

### Server-side filtering (optional query parameters)

All parameters are optional and combine with **AND** logic:

| Parameter | Type | Description |
|---|---|---|
| `taskId` | int | Only events from this task ID |
| `taskName` | string | Only events whose `taskName` matches (note: not guaranteed unique across tasks) |
| `eventType` | string | Only events of this type (`CROSS_LINE`, `MASK_HAIRNET_CHEF_HAT`, `PHONE_USAGE`, `CASHIER_BOX_OPEN`) |
| `channelId` | int | Only events from this camera channel |

```bash
# Only events from task 10
curl -N "http://localhost:9000/detection/stream?taskId=10"

# Only CROSS_LINE events on camera 1
curl -N "http://localhost:9000/detection/stream?eventType=CROSS_LINE&channelId=1"

# Only events from a named task
curl -N "http://localhost:9000/detection/stream?taskName=entrance_line"

# Combined (AND) — task 10 AND only on channel 1
curl -N "http://localhost:9000/detection/stream?taskId=10&channelId=1"

# Cashier task events only (Eyego-shaped payload under `data`)
curl -N "http://localhost:9000/detection/stream?eventType=CASHIER_BOX_OPEN"
```

### Task events — `evidence` (ML Image Contract V2)

For **`CROSS_LINE`**, **`MASK_HAIRNET_CHEF_HAT`**, and **`PHONE_USAGE`**, `evidence.captureImage` and `evidence.sceneImage` are **structured objects** (`url`, `path`, `type`, `format`, `timestamp`), not bare filesystem strings. On-disk files live under `CAPTURE_DIR` / `SCENE_DIR` with paths like `YYYY-MM-DD/{camera_id}_{event_id}_{uuid8}.jpg` inside each root. Set **`PUBLIC_ML_BASE_URL`** for full `https://…/evidence/…` URLs. See [../service_doc/ml_image_v2.md](../service_doc/ml_image_v2.md).

**Debug on the server (SSH):**

```bash
tail -f /local/storage/events/task_10.jsonl | jq -c .evidence
# or
tail -f /local/storage/events/task_20.jsonl | jq -c .evidence
```

SSE sends **one minified** `data:` line per event. Illustrative **pretty** `evidence` only:

```json
"evidence": {
  "captureImage": {
    "url": "https://ml.example.com/evidence/2026-04-22/cam-1_abc_01a2b3c4.jpg",
    "path": "2026-04-22/cam-1_abc_01a2b3c4.jpg",
    "type": "capture",
    "format": "image/jpeg",
    "timestamp": "2026-04-22T10:00:01.528Z"
  },
  "sceneImage": {
    "url": "https://ml.example.com/evidence/2026-04-22/cam-1_abc_9f8e7d6c.jpg",
    "path": "2026-04-22/cam-1_abc_9f8e7d6c.jpg",
    "type": "scene",
    "format": "image/jpeg",
    "timestamp": "2026-04-22T10:00:01.528Z"
  }
}
```

### CrossLine / PPE / phone — same envelope

The outer fields (`eventType`, `taskId`, `line` or `alert` + `person`, etc.) are unchanged; only `evidence` values moved from string paths to V2 objects. Filter **`eventType=PHONE_USAGE`** the same way as other tasks once a phone task is registered.

### Cashier structured event (`CASHIER_BOX_OPEN`)

One event **per processed frame** while the task worker is running. Top-level fields support SSE filters; the Eyego §4 payload is under **`data`**. **`data.personStructural`** is a **string** containing JSON. By default it is **pretty-printed** (`indent=2`, newlines appear as `\n` inside the SSE/JSONL line). Set **`CASHIER_COMPACT_PERSON_STRUCTURAL=1`** for a single-line minified string. Parse with `jq -r '.data.personStructural | fromjson'` (or the top-level event’s `.data.personStructural` when you hold the full object).

**`data` field notes:** `id` is the **same** UUID as in `captureId`, as 32 hex characters without dashes. `sceneId` uses a **separate** UUID. **`deviceSN`** resolves from task / zone config, then `CASHIER_DEVICE_SN`, `DEVICE_SN`, `HOSTNAME`, else `"UNKNOWN"`. **`captureUrl` / `sceneUrl`:** `CASHIER_CLOUD_IMAGE_BASE` applies to both sides; if unset, use `CASHIER_CAPTURE_URL_BASE` and `CASHIER_SCENE_URL_BASE` independently. If still empty, set **`CASHIER_FORCE_LOCAL_URLS=1`** to use `file:///local/storage/images` per missing side; otherwise URLs are `""` and a **warning** is logged. Details: [VISION_PIPELINE_README.md](./VISION_PIPELINE_README.md) and [CASHIER_BOX_OPEN.md](./CASHIER_BOX_OPEN.md).

SSE sends **one** `data:` line per event (minified outer JSON). Equivalent structure (pretty outer JSON for readability; UUIDs are examples):

```json
{
  "eventId": "a1b2c3d4e5f6789012345678abcdef01",
  "eventType": "CASHIER_BOX_OPEN",
  "timestamp": 1774310589000,
  "timestampUTC": "2026-04-05T10:03:09.000Z",
  "taskId": 30,
  "taskName": "cashier_drawer_monitor",
  "channelId": 1,
  "camera_id": "1",
  "case_id": "N3",
  "severity": "NORMAL",
  "transaction": true,
  "data": {
    "algorithmType": "CASHIER_BOX_OPEN",
    "captureId": "CASHIER_BOX_OPEN_550e8400-e29b-41d4-a716-446655440000.jpg",
    "sceneId": "CASHIER_BOX_OPEN_6ba7b810-9dad-11d1-80b4-00c04fd430c8.jpg",
    "channelId": 1,
    "channelName": "CAM-01-MAIN",
    "deviceSN": "UNKNOWN",
    "id": "550e8400e29b41d4a716446655440000",
    "taskId": 30,
    "taskName": "cashier_drawer_monitor",
    "recordTime": 1774310589000,
    "dateUTC": "2026-04-05T10:03:09.000Z",
    "total_open_count": 9,
    "total_open_duration_ms": 1100000,
    "current_open_duration_ms": 3000,
    "personStructural": "{\n  \"case_matched\": \"N3\",\n  \"case_level\": \"INFO\",\n  \"alert_triggered\": false,\n  \"critical_triggered\": false\n}",
    "captureUrl": "https://storage.example.com/logs-data-images/CASHIER_BOX_OPEN_550e8400-e29b-41d4-a716-446655440000.jpg",
    "sceneUrl": "https://storage.example.com/logs-data-images/CASHIER_BOX_OPEN_6ba7b810-9dad-11d1-80b4-00c04fd430c8.jpg",
    "evidence": {
      "captureImage": {
        "url": "https://ml.example.com/evidence/2026-04-22/CASHIER_BOX_OPEN_550e8400-e29b-41d4-a716-446655440000.jpg",
        "path": "2026-04-22/CASHIER_BOX_OPEN_550e8400-e29b-41d4-a716-446655440000.jpg",
        "type": "capture",
        "format": "image/jpeg",
        "timestamp": "2026-04-05T10:03:09.000Z"
      },
      "sceneImage": {
        "url": "https://ml.example.com/evidence/2026-04-22/CASHIER_BOX_OPEN_6ba7b810-9dad-11d1-80b4-00c04fd430c8.jpg",
        "path": "2026-04-22/CASHIER_BOX_OPEN_6ba7b810-9dad-11d1-80b4-00c04fd430c8.jpg",
        "type": "scene",
        "format": "image/jpeg",
        "timestamp": "2026-04-05T10:03:09.000Z"
      }
    }
  }
}
```

When a frame is saved, **top-level** `evidence` may also be present (same V2 `captureImage`; `sceneImage` may be `{ "url": null, "type": "scene", "status": "not_available" }` if there is no separate scene file). See [../service_doc/ml_image_v2.md](../service_doc/ml_image_v2.md).

Persisted to disk as one JSON line per frame: **`$EVENTS_DIR/task_<taskId>.jsonl`** (default `EVENTS_DIR=/local/storage/events`).

### Event fields reference

**Common to all events:**

| Field | Type | Description |
|---|---|---|
| `eventId` | string | MD5 hash — unique per event |
| `eventType` | string | `"CROSS_LINE"`, `"MASK_HAIRNET_CHEF_HAT"`, `"PHONE_USAGE"`, or `"CASHIER_BOX_OPEN"` |
| `timestamp` | int | Unix timestamp in milliseconds |
| `timestampUTC` | string | ISO 8601 UTC string |
| `taskId` | int | ID of the task that fired the event |
| `taskName` | string | Human-readable name from task config |
| `channelId` | int | Camera that produced the frame |

**`CASHIER_BOX_OPEN` also sets (top-level):** `camera_id` (string), `case_id`, `severity` (`NORMAL` \| `ALERT` \| `CRITICAL`), optional `transaction`, optional **`evidence`** (V2 **`captureImage`** / **`sceneImage`**) when a JPEG was saved this frame.

**Evidence (V2) on all crossing / PPE / phone events:** `evidence.captureImage` and `evidence.sceneImage` are always the structured image objects for those task types. See [../service_doc/ml_image_v2.md](../service_doc/ml_image_v2.md).

**CrossLine specific:**

| Field | Type | Description |
|---|---|---|
| `line.id` | string | Line ID from `areaPosition` config |
| `line.name` | string | Line name from `areaPosition` config |
| `line.direction` | int | `1` = A→B crossing, `2` = B→A crossing |
| `person.trackingId` | string | BoT-SORT track ID |
| `person.boundingBox` | object | `{x, y, width, height}` |
| `person.attributes.gender` | string | `"Male"`, `"Female"`, or `"Unknown"` |
| `person.attributes.age` | string | `"Child"`, `"Adult"`, `"Senior"`, or `"Unknown"` |
| `person.confidence` | int | Detection confidence 0–100 |

**MASK_HAIRNET_CHEF_HAT specific:**

| Field | Type | Description |
|---|---|---|
| `alert.type` | string | `"no_mask"`, `"no_hat"`, or `"no_chef_hat"` |
| `alert.description` | string | Human-readable description of the violation |
| `alert.confidence` | int | PPE model confidence 0–100 |
| `person.areaPoints` | array | Polygon zone points from `areaPosition` config |

**CASHIER_BOX_OPEN specific:**

| Field | Type | Description |
|---|---|---|
| `eventType` | string | `"CASHIER_BOX_OPEN"` |
| `data` | object | Eyego §4 block: `algorithmType`, `captureId`, `sceneId`, `id` (32-hex, same UUID as `captureId`), `recordTime`, `dateUTC`, `personStructural` (string), `total_open_*`, URLs, `taskId` / `channelId` / `deviceSN`, … |
| `data.personStructural` | string | String holding JSON: `case_matched`, `case_level` (`INFO` \| `WARNING` \| `CRITICAL`), zone counts, `detections`, `drawer_open_duration_ms`, … Default formatting is **pretty** (multi-line inside the string); use env `CASHIER_COMPACT_PERSON_STRUCTURAL=1` for one line. |

---

## 6. Stop Detection

### Stop all cameras
```bash
curl -X POST http://localhost:9000/detection/stop
```

**Response:**
```json
{
  "status": "stopped",
  "cameras": ["1", "2"]
}
```

### Stop one specific camera
```bash
curl -X POST "http://localhost:9000/detection/stop?camera_id=2"
```

**Response:**
```json
{
  "status": "stopped",
  "cameras": ["2"]
}
```

---

## 7. Task Management (CRUD)

### List all tasks
```bash
curl http://localhost:9000/api/tasks
```

**Response:**
```json
{
  "count": 4,
  "tasks": [
    { "taskId": 10, "taskName": "entrance_line", "algorithmType": "CROSS_LINE", ... },
    { "taskId": 11, "taskName": "exit_line_with_attrs", "algorithmType": "CROSS_LINE", ... },
    { "taskId": 20, "taskName": "kitchen_ppe_check", "algorithmType": "MASK_HAIRNET_CHEF_HAT", ... },
    { "taskId": 30, "taskName": "cashier_drawer_monitor", "algorithmType": "CASHIER_BOX_OPEN", ... }
  ]
}
```

### Get one task
```bash
curl http://localhost:9000/api/tasks/10
```

**Response:**
```json
{
  "taskId": 10,
  "taskName": "entrance_line",
  "algorithmType": "CROSS_LINE",
  "channelId": 1,
  "enable": true,
  "threshold": 60,
  "areaPosition": "...",
  "detailConfig": { "enableAttrDetect": false, "enableReid": false, "alarmType": [] },
  "validWeekday": ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY"],
  "validStartTime": 28800000,
  "validEndTime": 72000000
}
```

### Update a task (PUT replaces the full config)
```bash
curl -X PUT http://localhost:9000/api/tasks/10 \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 10,
    "taskName": "entrance_line",
    "algorithmType": "CROSS_LINE",
    "channelId": 1,
    "threshold": 75,
    "areaPosition": "[{\"line_id\":\"1\",\"line_name\":\"Entrance\",\"point\":[{\"x\":150,\"y\":400},{\"x\":850,\"y\":400}],\"direction\":1}]",
    "detailConfig": {
      "enableAttrDetect": true
    }
  }'
```

**Response:**
```json
{
  "status": "updated",
  "task": { "taskId": 10, "threshold": 75, "detailConfig": { "enableAttrDetect": true, ... }, ... }
}
```

> **Note:** Updated config takes effect on the **next** `POST /detection/start`. Running workers use the config they were started with.

### Delete a task
```bash
curl -X DELETE http://localhost:9000/api/tasks/20
```

**Response:**
```json
{
  "status": "deleted",
  "taskId": 20
}
```

### Disable a task without deleting it
```bash
curl -X PUT http://localhost:9000/api/tasks/20 \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 20,
    "taskName": "kitchen_ppe_check",
    "algorithmType": "MASK_HAIRNET_CHEF_HAT",
    "channelId": 2,
    "enable": false,
    "detailConfig": { "alarmType": ["no_mask"] }
  }'
```

---

## 8. Common Errors

### 400 — No tasks configured
```bash
curl -X POST http://localhost:9000/detection/start
```
```json
{
  "detail": "No enabled tasks configured. Call POST /api/tasks first."
}
```

### 400 — No cameras configured
```json
{
  "detail": "No cameras configured. Call POST /cameras first."
}
```

### 400 — Unsupported algorithmType
```json
{
  "detail": "Unsupported algorithmType 'UNKNOWN_TASK'. Supported: ['CASHIER_BOX_OPEN', 'CROSS_LINE', 'MASK_HAIRNET_CHEF_HAT', 'PHONE_USAGE']"
}
```

### 404 — Task's channelId has no matching camera
```json
{
  "detail": "No camera registered for channelId '3'. Register it via POST /cameras with id='3'."
}
```

### 404 — Task not found
```json
{
  "detail": "Task 99 not found."
}
```

### 409 — Camera already running
```json
{
  "detail": "Camera '1' is already running."
}
```

### 409 — Nothing is running (stop called when idle)
```json
{
  "detail": "No cameras are currently running."
}
```

---

## 9. Full Walkthrough Example

Complete sequence from scratch to streaming events:

```bash
# 1. Register cameras (both in one call)
curl -X POST http://localhost:9000/cameras \
  -H "Content-Type: application/json" \
  -d '{
    "cameras": [
      {"id": "1", "url": "rtsp://192.168.1.10/stream"},
      {"id": "2", "url": "rtsp://192.168.1.11/stream"}
    ]
  }'

# 2. Register tasks
curl -X POST http://localhost:9000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "taskId":10,"taskName":"entrance","algorithmType":"CROSS_LINE","channelId":1,
    "threshold":60,
    "areaPosition":"[{\"line_id\":\"1\",\"line_name\":\"Entrance\",\"point\":[{\"x\":100,\"y\":400},{\"x\":900,\"y\":400}],\"direction\":1}]",
    "detailConfig":{"enableAttrDetect":true},
    "validWeekday":["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY"],
    "validStartTime":28800000,"validEndTime":72000000
  }'

curl -X POST http://localhost:9000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "taskId":20,"taskName":"kitchen_ppe","algorithmType":"MASK_HAIRNET_CHEF_HAT","channelId":2,
    "threshold":70,
    "areaPosition":"[{\"line_id\":\"z1\",\"point\":[{\"x\":0,\"y\":0},{\"x\":1280,\"y\":0},{\"x\":1280,\"y\":720},{\"x\":0,\"y\":720}],\"direction\":0}]",
    "detailConfig":{"alarmType":["no_mask","no_chef_hat"]}
  }'

curl -X POST http://localhost:9000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "taskId":30,"taskName":"cashier_drawer_monitor","algorithmType":"CASHIER_BOX_OPEN","channelId":1,
    "threshold":50,
    "areaPosition":"[]",
    "detailConfig":{"drawerOpenLimit":20,"serviceWaitLimit":90}
  }'

# 3. Start
curl -X POST http://localhost:9000/detection/start

# 4. Open SSE stream in terminal (keep open) — all tasks, or cashier-only:
curl -N http://localhost:9000/detection/stream
# curl -N "http://localhost:9000/detection/stream?eventType=CASHIER_BOX_OPEN&taskId=30"

# 5. Check status in another terminal
curl http://localhost:9000/detection/status

# 6. Cashier monitor (optional — uses task 30 on channel 1)
curl -s http://localhost:9000/cashier/zones
curl -s http://localhost:9000/cashier/status
curl -s "http://localhost:9000/cashier/events?limit=5"
# curl -N http://localhost:9000/cashier/stream/1

# 7. Open live annotated video stream in a browser (WebSocket — see section 11)
# Paste this HTML into a file and open it in your browser:
# <img id="s"> <script>
#   function c(id){const w=new WebSocket(`ws://localhost:9000/cameras/${id}/live`);
#   w.binaryType="arraybuffer";
#   w.onmessage=e=>{const b=new Blob([e.data],{type:"image/jpeg"});
#   URL.revokeObjectURL(document.getElementById("s").src);
#   document.getElementById("s").src=URL.createObjectURL(b)};
#   w.onclose=()=>setTimeout(()=>c(id),2000);}c("1");
# </script>

# 8. Stop when done
curl -X POST http://localhost:9000/detection/stop
```

---

## 10. Cashier monitor API (`/cashier/*`)

These routes serve the **cashier** monitor (live state, zones on disk, evidence, per-camera SSE). **`GET /detection/stream`** also carries **`CASHIER_BOX_OPEN`** task events (same structured object as below, one line per frame) alongside crossings and PPE — use `?eventType=CASHIER_BOX_OPEN` to filter.

**Prerequisites:** Register a task with `algorithmType: "CASHIER_BOX_OPEN"` and start detection so `GET /cashier/status`, `GET /cashier/events`, and **`GET /detection/stream`** fill from the runtime. **Zone read/write** works whenever the server can access the config file.

**Config file:** `CASHIER_CONFIG` (default `./config/cashier_zones.yaml`). `POST` bodies use **normalized** coordinates in `[0, 1]` as `{"x", "y"}` objects; the file stores `points` as nested lists `[[x, y], …]`. Omitted keys in `POST /cashier/zones` are left unchanged (partial update).

### Get current zone configuration

```bash
curl -s http://localhost:9000/cashier/zones
```

Returns the merged YAML/JSON document: `zones` (`ROI_CASHIER`, `ROI_CUSTOMER`), `thresholds`, and optional keys such as `buffer`, `debounce`, `evidence`, `detail_config`.

### Update zones (partial)

**Rectangle split (cashier left, customer right):**

```bash
curl -X POST http://localhost:9000/cashier/zones \
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
      "drawer_open_max_seconds": 20,
      "customer_wait_max_seconds": 30
    }
  }'
```

**Polygon zones** — use `"shape": "polygon"` and at least three `points`:

```bash
curl -X POST http://localhost:9000/cashier/zones \
  -H "Content-Type: application/json" \
  -d '{
    "ROI_CASHIER": {
      "shape": "polygon",
      "points": [
        {"x": 0.33, "y": 0.55},
        {"x": 0.65, "y": 0.53},
        {"x": 0.69, "y": 0.99},
        {"x": 0.34, "y": 0.99}
      ],
      "active": true
    }
  }'
```

**Optional fields** on the same `POST`: `detail_config` (e.g. `drawerOpenLimit`, `serviceWaitLimit`, `enableStaffList`, `staffIds`), `task` (integration metadata), `detection_threshold` (0–100, stored under `thresholds.detection_threshold`).

**Typical success response:**

```json
{
  "status": "updated",
  "config": { "zones": { "ROI_CASHIER": { "shape": "rectangle", "points": [[0.0, 0.0], [0.45, 1.0]], "active": true } }, "thresholds": {} }
}
```

> Cashier workers reload config on their periodic reload (default `config_reload_interval` in `thresholds`, often 60s).

### Reset zones to built-in defaults

```bash
curl -X POST http://localhost:9000/cashier/zones/reset
```

### Live status (all cameras)

```bash
curl -s http://localhost:9000/cashier/status
```

### Event log (paginated)

While the cashier task worker is running, each processed frame appends a **structured** event (same shape as `GET /detection/stream` for `CASHIER_BOX_OPEN`). Legacy test callers may still log only alert/transaction frames.

```bash
curl -s "http://localhost:9000/cashier/events?limit=50&offset=0"
curl -s "http://localhost:9000/cashier/events?camera_id=1&severity=CRITICAL&case_id=A3"
```

### Clear in-memory event log

```bash
curl -X DELETE http://localhost:9000/cashier/events
```

### Evidence list and download

```bash
curl -s "http://localhost:9000/cashier/evidence?limit=20"
curl -s -o evidence.jpg "http://localhost:9000/cashier/evidence/ALERT/A3/cam_1_2026-04-05_12-00-00.jpg"
```

Use the `path` returned by `GET /cashier/evidence` as the suffix after `/cashier/evidence/`.

### Per-camera SSE (cashier stream)

```bash
curl -N http://localhost:9000/cashier/stream/1
curl -N http://localhost:9000/cashier/stream/1/only
```

`{camera_id}` is the string camera id (same as your registered camera `id`). The `/only` route suppresses routine `frame` events and keeps alerts / `gif_ready`.

### Drawer open-edge count (persisted totals)

Returns cumulative **closed→open** drawer transitions in the cashier ROI for that camera (from `cashier_drawer_open_totals.json`), not a line count from legacy `events.jsonl`.

```bash
curl -s "http://localhost:9000/cashier/media/1/drawer_count"
```

### Latest media (optional)

```bash
curl -s -o latest.jpg "http://localhost:9000/cashier/media/1/latest/jpg"
curl -s -o latest.gif "http://localhost:9000/cashier/media/1/latest/gif"
```

---

## `areaPosition` format reference

`areaPosition` is always a **JSON string** (stringified JSON, not an object).

### For CrossLine (2-point line):
```json
"[{\"line_id\":\"1\",\"line_name\":\"Main Entrance\",\"point\":[{\"x\":100,\"y\":400},{\"x\":900,\"y\":400}],\"direction\":1}]"
```

| `direction` | Meaning |
|---|---|
| `0` | Bidirectional — fires on both crossings |
| `1` | A→B only (left-to-right or top-to-bottom depending on line angle) |
| `2` | B→A only |

### For MASK_HAIRNET_CHEF_HAT (polygon zone):
```json
"[{\"line_id\":\"zone1\",\"point\":[{\"x\":50,\"y\":50},{\"x\":600,\"y\":50},{\"x\":600,\"y\":500},{\"x\":50,\"y\":500}],\"direction\":0}]"
```

Polygon needs at least 3 points. Persons whose centroid falls outside all zones are ignored. If `areaPosition` is `"[]"`, the entire frame is the detection zone.

---

## 11. Live Stream WebSocket (`/cameras/{id}/live`)

**Stream annotation** here means **visually annotated JPEG frames**: FrameBus runs YOLO + BoT-SORT on each RTSP frame, draws bounding boxes and track IDs, encodes JPEG, and publishes to Redis. Clients consume that stream only via **`WS /cameras/{camera_id}/live`** (binary JPEG messages). Task workers may draw extra overlays (lines, cashier zones) on their own copies for evidence; the live WebSocket feed is the FrameBus-annotated frame.

**Related (not the painted live video):** JSON **detection events** (crossings, PPE alerts, phone usage, cashier payloads) are available on **`GET /detection/stream`** (SSE, all cameras, optional filters) and **`WS /cameras/{camera_id}/events`** (one camera, same JSON shape as SSE). Those carry metadata and V2 **evidence** objects (`captureImage` / `sceneImage`) — not a full-motion annotated video stream. Evidence contract: [../service_doc/ml_image_v2.md](../service_doc/ml_image_v2.md).

**Transport design (annotated frame WebSocket):**
- Messages are **binary** (raw JPEG bytes) — no Base64 encoding, no JSON wrapper
- Frames are **dropped** (never queued) when a client cannot receive within 50 ms (`WS_SEND_TIMEOUT_MS`) — prevents memory growth on slow or hidden browser tabs
- Requires Redis (configure `REDIS_URL`; the Docker Compose stack starts Redis automatically)
- FrameBus publishes at `REDIS_LIVE_FPS` (default 13 fps); cadence auto-adjusts to measured camera FPS
- WebSocket does **not** auto-reconnect — implement a 2-second retry loop (see examples below)

### All endpoints: annotated stream vs detection JSON

| What you get | HTTP | Path | When to use |
|---|---|---|---|
| **Annotated JPEG frames** (live video) | `WebSocket` | `/cameras/{camera_id}/live` | Dashboard tile, MJPEG-style preview, one connection **per camera** |
| **Annotated JPEG frames** (live video, by name) | `WebSocket` | `/tasks/{task_name}/live` | Same frames as above; use when you know the task name but not the camera id — see [section 12](#12-task-name-live-stream-alias-taskstask_namelive) |
| **Detection events** (JSON, one camera) | `WebSocket` | `/cameras/{camera_id}/events` | Same event objects as SSE; filter by opening one socket per `camera_id` |
| **Detection events** (JSON, all cameras) | `GET` (SSE) | `/detection/stream` | Single connection; optional query filters (`taskId`, `channelId`, …) — see [section 5](#5-stream-results-sse) |

**Redis channels (for operators / debugging):** FrameBus publishes JPEG bytes to `live:frame:{camera_id}`. Task workers publish JSON to `live:event:{camera_id}`. The FastAPI app bridges Redis → WebSockets; you do not subscribe to Redis directly from a browser.

### How to get the annotated frame stream

1. **`POST /cameras`** — Register each stream; note each camera **`id`** (string, e.g. `"1"`, `"2"`). This `id` must match the WebSocket path and the Redis channel suffix.
2. **`POST /api/tasks`** — For each camera, create tasks with **`channelId`** equal to the numeric channel for that camera (same value as your camera id when it is numeric — e.g. camera `"1"` → `channelId: 1`).
3. **`POST /detection/start`** — Starts FrameBus and workers. Without this, no frames are published.
4. **Redis** — `REDIS_URL` must point at a reachable Redis instance; WebSocket `/live` closes with an error if Redis is unavailable.
5. **`GET /detection/status`** — Confirm each camera shows `"running": true` before expecting frames.
6. **`WebSocket` `ws://<host>:<port>/cameras/<camera_id>/live`** — Use the **same** `camera_id` string you registered (e.g. `1` or `cam-a` if you used that id).

There is **no** single WebSocket that multiplexes all cameras; open **one** `/live` connection per camera you want to display.

### Endpoints (quick reference)

| Endpoint | Protocol | Content | Description |
|---|---|---|---|
| `/cameras/{camera_id}/live` | WebSocket binary | JPEG bytes | Annotated frame stream for one camera |
| `/cameras/{camera_id}/events` | WebSocket text | JSON string | Detection event stream for one camera |
| `/tasks/{task_name}/live` | WebSocket binary | JPEG bytes | Same frames as the task's camera — alias resolved by `taskName` |

### Live annotated frame stream

```bash
# Verify the WebSocket upgrade handshake (expects HTTP 101)
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  -H "Sec-WebSocket-Version: 13" \
  http://localhost:9000/cameras/1/live

# Same check for other registered camera ids (must match POST /cameras "id" values)
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  -H "Sec-WebSocket-Version: 13" \
  http://localhost:9000/cameras/2/live

curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  -H "Sec-WebSocket-Version: 13" \
  http://localhost:9000/cameras/3/live
```

**Browser (minimal HTML — save as `stream.html` and open in browser):**

```html
<!DOCTYPE html>
<html>
<head>
  <style>
    body { background:#111; color:#0f0; font-family:monospace; text-align:center; }
    img  { max-width:100%; border:2px solid #0f0; margin-top:16px; }
  </style>
</head>
<body>
  <h2>Camera 1 — Live Annotated Stream</h2>
  <div id="status">Connecting…</div>
  <img id="stream" alt="stream" />
  <script>
    function connect(cameraId) {
      const ws = new WebSocket(`ws://localhost:9000/cameras/${cameraId}/live`);
      ws.binaryType = "arraybuffer";
      ws.onopen  = () => document.getElementById("status").textContent = "Connected";
      ws.onclose = () => {
        document.getElementById("status").textContent = "Reconnecting…";
        setTimeout(() => connect(cameraId), 2000);   // manual reconnect required
      };
      ws.onerror = () => ws.close();
      ws.onmessage = e => {
        const blob = new Blob([e.data], { type: "image/jpeg" });
        const img  = document.getElementById("stream");
        URL.revokeObjectURL(img.src);                // free previous frame memory
        img.src = URL.createObjectURL(blob);
      };
    }
    connect("1");
  </script>
</body>
</html>
```

### Detection event stream (WebSocket)

```bash
# Connect and print received JSON events (requires websocat or similar CLI tool)
websocat ws://localhost:9000/cameras/1/events

# One process per camera — same host, different path (after POST /cameras for 1, 2, 3)
websocat ws://localhost:9000/cameras/1/events &
websocat ws://localhost:9000/cameras/2/events &
websocat ws://localhost:9000/cameras/3/events &
wait
```

**Browser:**

```javascript
function connectEvents(cameraId) {
    const ws = new WebSocket(`ws://localhost:9000/cameras/${cameraId}/events`);
    ws.onmessage = e => {
        const event = JSON.parse(e.data);
        console.log(event.eventType, event.timestamp, event.taskId);
    };
    ws.onclose = () => setTimeout(() => connectEvents(cameraId), 2000);
}
connectEvents("1");
```

### Multiple cameras — annotated `/live` at once

Each camera uses its **own** WebSocket URL: `…/cameras/{id}/live`. There is no server-side fan-in: your dashboard opens **N** sockets for **N** cameras.

#### Python — save the first JPEG from three streams in parallel

Requires `pip install websockets` (or your project venv). Adjust ids to match your registered cameras.

```python
import asyncio
from pathlib import Path

import websockets

HOST = "localhost:9000"
CAMERA_IDS = ("1", "2", "north_gate")  # string ids from POST /cameras

async def first_jpeg(camera_id: str) -> None:
    uri = f"ws://{HOST}/cameras/{camera_id}/live"
    async with websockets.connect(uri, max_size=None) as ws:
        jpeg = await ws.recv()
        if isinstance(jpeg, str):
            raise SystemExit(f"{camera_id}: expected binary JPEG")
        out = Path(f"first_frame_{camera_id}.jpg")
        out.write_bytes(jpeg)
        print(f"{camera_id}: wrote {out} ({len(jpeg)} bytes)")

async def main() -> None:
    await asyncio.gather(*(first_jpeg(cid) for cid in CAMERA_IDS))

asyncio.run(main())
```

#### Python — continuous multi-camera viewer loop (logging FPS)

```python
import asyncio
import time

import websockets

HOST = "localhost:9000"
CAMERA_IDS = ("1", "2", "3")

async def watch(camera_id: str) -> None:
    uri = f"ws://{HOST}/cameras/{camera_id}/live"
    n = 0
    t0 = time.monotonic()
    while True:
        try:
            async with websockets.connect(uri, max_size=None) as ws:
                while True:
                    await ws.recv()
                    n += 1
                    if n % 30 == 0:
                        dt = time.monotonic() - t0
                        print(f"cam {camera_id}: {n} frames, {n/dt:.1f} fps avg")
        except Exception as exc:
            print(f"cam {camera_id}: {exc!r}, reconnect in 2s")
            await asyncio.sleep(2)

async def main() -> None:
    await asyncio.gather(*(watch(cid) for cid in CAMERA_IDS))

asyncio.run(main())
```

#### Browser — labeled grid (three annotated tiles)

Save as `multi_stream.html` and open while detection is running. Change `HOST` if not on `localhost:9000`.

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Multi-camera annotated streams</title>
  <style>
    body { margin:0; background:#111; color:#ccc; font-family: system-ui, sans-serif; }
    h1 { text-align:center; font-size:1.1rem; margin:12px 0; }
    #grid {
      display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap:12px; padding:12px; max-width:1400px; margin:0 auto;
    }
    figure { margin:0; background:#1a1a1a; border-radius:8px; overflow:hidden;
             border:1px solid #333; }
    figcaption { padding:8px 10px; font-size:0.85rem; color:#9cf; }
    img { display:block; width:100%; height:auto; min-height:120px; background:#000; }
    .err { color:#f66; font-size:0.8rem; padding:8px 10px; }
  </style>
</head>
<body>
  <h1>Live annotated streams (one WebSocket per camera)</h1>
  <div id="grid"></div>
  <script>
    const HOST = "localhost:9000";
    const CAMERAS = [
      { id: "1", label: "Camera 1 — entrance" },
      { id: "2", label: "Camera 2 — kitchen" },
      { id: "3", label: "Camera 3 — parking" },
    ];

    function tile(cam) {
      const fig = document.createElement("figure");
      const cap = document.createElement("figcaption");
      cap.textContent = cam.label + " (`" + cam.id + "`)";
      const st = document.createElement("div");
      st.className = "err";
      st.textContent = "Connecting…";
      const img = document.createElement("img");
      img.alt = cam.label;
      fig.append(cap, st, img);
      document.getElementById("grid").appendChild(fig);

      function connect() {
        const ws = new WebSocket(`ws://${HOST}/cameras/${cam.id}/live`);
        ws.binaryType = "arraybuffer";
        ws.onopen = () => { st.textContent = "Live"; };
        ws.onerror = () => { st.textContent = "WebSocket error"; ws.close(); };
        ws.onclose = () => {
          st.textContent = "Reconnecting in 2s…";
          setTimeout(connect, 2000);
        };
        ws.onmessage = (e) => {
          URL.revokeObjectURL(img.src);
          img.src = URL.createObjectURL(new Blob([e.data], { type: "image/jpeg" }));
        };
      }
      connect();
    }
    CAMERAS.forEach(tile);
  </script>
</body>
</html>
```

#### JavaScript — minimal dynamic tiles (append one `<img>` per id)

```javascript
["1", "2", "3"].forEach((id) => {
  const wrap = document.createElement("div");
  wrap.innerHTML = `<p>Camera ${id}</p>`;
  const img = document.createElement("img");
  wrap.appendChild(img);
  document.body.appendChild(wrap);

  function connect() {
    const ws = new WebSocket(`ws://localhost:9000/cameras/${id}/live`);
    ws.binaryType = "arraybuffer";
    ws.onmessage = (e) => {
      URL.revokeObjectURL(img.src);
      img.src = URL.createObjectURL(new Blob([e.data], { type: "image/jpeg" }));
    };
    ws.onclose = () => setTimeout(connect, 2000);
  }
  connect();
});
```

### Configuration env vars

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection URL (required for live stream) |
| `REDIS_LIVE_FPS` | `13` | Target publish rate in frames per second. FrameBus auto-adjusts cadence to match measured camera FPS. |
| `WS_SEND_TIMEOUT_MS` | `50` | Max milliseconds to wait for a WebSocket send before dropping the frame. Prevents TCP buffer bloat on slow clients. |

### Verify Redis is running

```bash
# From Docker Compose
docker compose ps redis
# Expect: redis   redis:7-alpine   Up

# Check FrameBus connected to Redis
docker compose logs yolo-detect | grep "FrameBus: Redis"
# Expect: [1] FrameBus: Redis connected (redis://redis:6379/0)

# Verify frames are being published (redis-cli inside the redis container)
docker exec -it redis redis-cli SUBSCRIBE live:frame:1
# Messages arrive every ~80ms (13fps)

# Other cameras use the same pattern (id must match POST /cameras)
docker exec -it redis redis-cli SUBSCRIBE live:frame:2
docker exec -it redis redis-cli SUBSCRIBE live:frame:3
```

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| WebSocket closes immediately | Redis not reachable | Check `REDIS_URL`; verify `docker compose ps redis` is Up |
| Stream connects but shows nothing | Detection not started | Run `POST /detection/start` first |
| Very laggy / old frames | `WS_SEND_TIMEOUT_MS` too high | Lower it (e.g. `WS_SEND_TIMEOUT_MS=30`) or check client CPU |
| WebSocket closes after ~30s idle | Normal — client should reconnect | Implement `ws.onclose = () => setTimeout(connect, 2000)` |

---

## 12. Task-Name Live Stream Alias (`/tasks/{task_name}/live`)

`WS /tasks/{task_name}/live` is a **convenience alias** for the camera live stream. The server looks up the task by its exact `taskName`, resolves the `channelId`, and serves the same annotated JPEG frames that `WS /cameras/{channelId}/live` would deliver. No new stream is created — this is a routing shortcut.

### When to use this vs `/cameras/{camera_id}/live`

| Situation | Recommended endpoint |
|---|---|
| You know the camera `id` string | `/cameras/{camera_id}/live` |
| You know the `taskName` but not the camera `id` | `/tasks/{task_name}/live` |
| Multiple tasks share the same camera | Use `/cameras/{camera_id}/live`; the task alias will fail if names collide |

### Prerequisites

Same as section 11: detection must be started (`POST /detection/start`) and `REDIS_URL` must be configured.

### Error close codes

| Code | Meaning |
|---|---|
| `4004` | No task found with that exact `taskName` |
| `4009` | More than one task shares the same `taskName`; use `/cameras/{camera_id}/live` instead |
| `1011` | Redis is unavailable |

### Quick start

```bash
# 1. Register a camera
curl -X POST http://localhost:9000/cameras \
  -H "Content-Type: application/json" \
  -d '{"cameras": [{"id": "cam1", "url": "rtsp://192.168.1.10/stream"}]}'

# 2. Register a task with a descriptive taskName
curl -X POST http://localhost:9000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 10,
    "taskName": "mainentrance1",
    "algorithmType": "CROSS_LINE",
    "channelId": "cam1",
    "enable": true,
    "threshold": 60,
    "areaPosition": "[]"
  }'

# 3. Start detection
curl -X POST http://localhost:9000/detection/start

# 4. Verify the WebSocket upgrade (expects HTTP 101)
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  -H "Sec-WebSocket-Version: 13" \
  http://localhost:9000/tasks/mainentrance1/live
```

### Browser example

```html
<!DOCTYPE html>
<html>
<head>
  <style>
    body { background:#111; color:#0f0; font-family:monospace; text-align:center; }
    img  { max-width:100%; border:2px solid #0f0; margin-top:16px; }
  </style>
</head>
<body>
  <h2>mainentrance1 — Live Annotated Stream</h2>
  <div id="status">Connecting…</div>
  <img id="stream" alt="stream" />
  <script>
    function connect(taskName) {
      const ws = new WebSocket(`ws://localhost:9000/tasks/${taskName}/live`);
      ws.binaryType = "arraybuffer";
      ws.onopen  = () => document.getElementById("status").textContent = "Connected";
      ws.onclose = e => {
        const reason = e.code === 4004 ? "Task not found"
                     : e.code === 4009 ? "Ambiguous task name"
                     : "Reconnecting…";
        document.getElementById("status").textContent = reason;
        if (e.code !== 4004 && e.code !== 4009) {
          setTimeout(() => connect(taskName), 2000);
        }
      };
      ws.onerror = () => ws.close();
      ws.onmessage = e => {
        const blob = new Blob([e.data], { type: "image/jpeg" });
        const img  = document.getElementById("stream");
        URL.revokeObjectURL(img.src);
        img.src = URL.createObjectURL(blob);
      };
    }
    connect("mainentrance1");
  </script>
</body>
</html>
```

### Multiple named tasks on different cameras

```javascript
// Each task name maps to exactly one camera.
// Open one connection per task name you want to display.
["mainentrance1", "kitchenppe", "cashierzone2"].forEach(name => {
  const ws = new WebSocket(`ws://localhost:9000/tasks/${name}/live`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = e => displayFrame(name, e.data);
  ws.onclose   = () => setTimeout(() => connect(name), 2000);
});
```

### Python example — save first frame by task name

```python
import asyncio
from pathlib import Path
import websockets

HOST = "localhost:9000"

async def first_frame_by_task(task_name: str) -> None:
    uri = f"ws://{HOST}/tasks/{task_name}/live"
    async with websockets.connect(uri, max_size=None) as ws:
        jpeg = await ws.recv()
        if isinstance(jpeg, str):
            raise SystemExit(f"{task_name}: expected binary JPEG, got text")
        out = Path(f"first_frame_{task_name}.jpg")
        out.write_bytes(jpeg)
        print(f"{task_name}: saved {out} ({len(jpeg)} bytes)")

asyncio.run(first_frame_by_task("mainentrance1"))
```

### Notes

- The route resolves `taskName` with an **exact string match** (case-sensitive). `mainentrance1` and `MainEntrance1` are different names.
- If two tasks are registered with the same `taskName`, the endpoint closes immediately with code `4009`. Use unique `taskName` values or connect via `/cameras/{camera_id}/live`.
- Reconnect logic is the same as for `/cameras/{camera_id}/live`: the WebSocket does **not** auto-reconnect; implement a 2-second retry in `ws.onclose`.

---

### `validStartTime` / `validEndTime` — milliseconds from midnight

| Time | ms value |
|---|---|
| 00:00 (midnight) | `0` |
| 08:00 | `28800000` |
| 12:00 | `43200000` |
| 18:00 | `64800000` |
| 20:00 | `72000000` |
| 23:59 | `86340000` |
| End of day | `86400000` |
