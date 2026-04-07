# API Usage Guide

Base URL for all examples: `http://localhost:9000`

**See also:** [VISION_PIPELINE_README.md](./VISION_PIPELINE_README.md) — combined pytest, cURL, SSH, and **CASHIER_BOX_OPEN** `data` / evidence (replaces `SERVICE_TEST.md` + `cases-and-repo.md`). Eyego cURL, mock responses, and full-case JSON: [CASHIER_BOX_OPEN.md](./CASHIER_BOX_OPEN.md). [SERVICE_TEST.md](./SERVICE_TEST.md) redirects there.

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

---

## 1. Register Cameras

Register one camera at a time. The `id` must match the `channelId` used in your tasks.

### Add a camera
```bash
curl -X POST http://localhost:9000/cameras \
  -H "Content-Type: application/json" \
  -d '{
    "id": "1",
    "url": "rtsp://192.168.1.10/stream"
  }'
```

**Response:**
```json
{
  "status": "registered",
  "camera_id": "1",
  "url": "rtsp://192.168.1.10/stream"
}
```

### Add a second camera
```bash
curl -X POST http://localhost:9000/cameras \
  -H "Content-Type: application/json" \
  -d '{
    "id": "2",
    "url": "rtsp://192.168.1.11/stream"
  }'
```

### List all registered cameras
```bash
curl http://localhost:9000/cameras
```

**Response:**
```json
{
  "count": 2,
  "cameras": {
    "1": "rtsp://192.168.1.10/stream",
    "2": "rtsp://192.168.1.11/stream"
  }
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

Connect once and receive events in real-time. Events arrive as they happen — one JSON object per line crossing, PPE violation, or cashier state transition.

```bash
curl -N http://localhost:9000/detection/stream
```

> `-N` disables buffering so you see events immediately.

### CrossLine event example
```
data: {"eventId":"a3f92c1d8e4b56f7","eventType":"CROSS_LINE","timestamp":1774310401528,"timestampUTC":"2026-04-05T10:00:01.528Z","taskId":10,"taskName":"entrance_line","channelId":1,"line":{"id":"1","name":"Entrance","direction":1},"person":{"trackingId":"42","reidFeature":[],"boundingBox":{"x":120,"y":200,"width":65,"height":180},"attributes":{"gender":"Unknown","age":"Unknown"},"confidence":87},"evidence":{"captureImage":"/local/storage/captures/2026/04/05/a3f92c1d_crop.jpg","sceneImage":"/local/storage/scenes/2026/04/05/a3f92c1d_scene.jpg"}}

```

### CrossLine event with age/gender (enableAttrDetect: true)
```
data: {"eventId":"b7d21a4c9f3e80ab","eventType":"CROSS_LINE","timestamp":1774310465000,"timestampUTC":"2026-04-05T10:01:05.000Z","taskId":11,"taskName":"exit_line_with_attrs","channelId":1,"line":{"id":"2","name":"Exit","direction":2},"person":{"trackingId":"38","reidFeature":[],"boundingBox":{"x":300,"y":180,"width":58,"height":172},"attributes":{"gender":"Male","age":"Adult"},"confidence":91},"evidence":{"captureImage":"/local/storage/captures/2026/04/05/b7d21a4c_crop.jpg","sceneImage":"/local/storage/scenes/2026/04/05/b7d21a4c_scene.jpg"}}

```

### PPE violation event
```
data: {"eventId":"c9e04f2b1a7d35cc","eventType":"MASK_HAIRNET_CHEF_HAT","timestamp":1774310512000,"timestampUTC":"2026-04-05T10:01:52.000Z","taskId":20,"taskName":"kitchen_ppe_check","channelId":2,"alert":{"type":"no_mask","description":"Face mask not detected","confidence":72},"person":{"trackingId":"15","boundingBox":{"x":88,"y":95,"width":70,"height":195},"areaPoints":[{"x":50,"y":50},{"x":600,"y":50},{"x":600,"y":500},{"x":50,"y":500}]},"evidence":{"captureImage":"/local/storage/captures/2026/04/05/c9e04f2b_crop.jpg","sceneImage":"/local/storage/scenes/2026/04/05/c9e04f2b_scene.jpg"}}

```

### Cashier drawer event
```
data: {"eventId":"d1f98a7c3b2e44aa","eventType":"CASHIER_BOX_OPEN","timestamp":1774310589000,"timestampUTC":"2026-04-05T10:03:09.000Z","taskId":30,"taskName":"cashier_drawer_monitor","channelId":1,"data":{"drawerOpen":true,"drawerOpenSeconds":24,"serviceWaitSeconds":11,"staffPresent":false,"personInZone":true,"labels":["person","cashier_open"],"evidence":{"captureImage":"/local/storage/captures/2026/04/05/d1f98a7c_crop.jpg","sceneImage":"/local/storage/scenes/2026/04/05/d1f98a7c_scene.jpg"}}}

```

### Event fields reference

**Common to all events:**

| Field | Type | Description |
|---|---|---|
| `eventId` | string | MD5 hash — unique per event |
| `eventType` | string | `"CROSS_LINE"`, `"MASK_HAIRNET_CHEF_HAT"`, or `"CASHIER_BOX_OPEN"` |
| `timestamp` | int | Unix timestamp in milliseconds |
| `timestampUTC` | string | ISO 8601 UTC string |
| `taskId` | int | ID of the task that fired the event |
| `taskName` | string | Human-readable name from task config |
| `channelId` | int | Camera that produced the frame |

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
| `data` | object | Cashier payload (status/timers/evidence; shape depends on runtime state) |

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
  "detail": "Unsupported algorithmType 'UNKNOWN_TASK'. Supported: ['CROSS_LINE', 'MASK_HAIRNET_CHEF_HAT', 'CASHIER_BOX_OPEN']"
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
# 1. Register cameras
curl -X POST http://localhost:9000/cameras \
  -H "Content-Type: application/json" \
  -d '{"id":"1","url":"rtsp://192.168.1.10/stream"}'

curl -X POST http://localhost:9000/cameras \
  -H "Content-Type: application/json" \
  -d '{"id":"2","url":"rtsp://192.168.1.11/stream"}'

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

# 4. Open SSE stream in terminal (keep open)
curl -N http://localhost:9000/detection/stream

# 5. Check status in another terminal
curl http://localhost:9000/detection/status

# 6. Cashier monitor (optional — uses task 30 on channel 1)
curl -s http://localhost:9000/cashier/zones
curl -s http://localhost:9000/cashier/status
# curl -N http://localhost:9000/cashier/stream/1

# 7. Stop when done
curl -X POST http://localhost:9000/detection/stop
```

---

## 10. Cashier monitor API (`/cashier/*`)

These routes serve the **cashier** monitor (live state, zones on disk, evidence, per-camera SSE). They are separate from `GET /detection/stream`, which merges **task** events for all algorithms.

**Prerequisites:** Register a task with `algorithmType: "CASHIER_BOX_OPEN"` and start detection so `GET /cashier/status` and event logs fill from the runtime. **Zone read/write** works whenever the server can access the config file.

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
