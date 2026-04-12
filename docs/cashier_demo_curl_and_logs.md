# Cashier Demo Curl And Logs

This document shows a full cashier demo flow with `curl`, from API startup to camera setup, task creation, zone update, start, status, SSE, and cleanup.

It includes:

- a real local API run verified on this machine
- the exact `curl` commands used
- the exact responses/logs captured
- sample cashier action logs for real `N3` / `A5` style events

## 1. Start the API

Use the local virtualenv and force the API to use the repo's existing YOLO file:

```bash
cd /home/a7med/ml-server
YOLO_MODEL="/home/a7med/ml-server/yolov8n.pt" \
  "/home/a7med/ml-server/.venv/bin/python" -m uvicorn app:app --host 127.0.0.1 --port 9000
```

Set a base URL:

```bash
export BASE=http://127.0.0.1:9000
```

## 2. Health Check

```bash
curl -s "$BASE/"
```

Response:

```json
{"service":"Vision Pipeline API","version":"2.0.0"}
```

## 3. Check Initial Cashier State

```bash
curl -s "$BASE/cashier/status"
curl -s "$BASE/cashier/events"
```

Responses:

```json
{}
```

```json
{"total":0,"offset":0,"limit":100,"events":[]}
```

## 4. Add a Demo Camera

This API accepts a local file path as the camera `url`, not only RTSP.

```bash
curl -s -X POST "$BASE/cameras" \
  -H "Content-Type: application/json" \
  -d '{
    "cameras": [
      {
        "id": "1",
        "url": "/home/a7med/ml-server/videos/cashier_demo.mp4"
      }
    ]
  }'
```

Response:

```json
{"status":"configured","cameras":{"1":"/home/a7med/ml-server/videos/cashier_demo.mp4"}}
```

List cameras:

```bash
curl -s "$BASE/cameras"
```

```json
{"count":1,"cameras":[{"id":"1","url":"/home/a7med/ml-server/videos/cashier_demo.mp4"}]}
```

## 5. Add the Cashier Task

```bash
curl -s -X POST "$BASE/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 101,
    "taskName": "cashier_demo",
    "algorithmType": "CASHIER_BOX_OPEN",
    "channelId": 1,
    "enable": true,
    "threshold": 50,
    "areaPosition": "[]",
    "detailConfig": {},
    "validWeekday": [
      "MONDAY","TUESDAY","WEDNESDAY","THURSDAY",
      "FRIDAY","SATURDAY","SUNDAY"
    ],
    "validStartTime": 0,
    "validEndTime": 86400000
  }'
```

Response:

```json
{"status":"created","task":{"taskId":101,"taskName":"cashier_demo","algorithmType":"CASHIER_BOX_OPEN","channelId":1,"enable":true,"threshold":50,"areaPosition":"[]","detailConfig":{"enableAttrDetect":false,"enableReid":false,"alarmType":[],"drawerOpenLimit":30,"serviceWaitLimit":30,"enableStaffList":false,"staffIds":[]},"validWeekday":["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"],"validStartTime":0,"validEndTime":86400000}}
```

List tasks:

```bash
curl -s "$BASE/api/tasks"
```

```json
{"count":1,"tasks":[{"taskId":101,"taskName":"cashier_demo","algorithmType":"CASHIER_BOX_OPEN","channelId":1,"enable":true,"threshold":50,"areaPosition":"[]","detailConfig":{"enableAttrDetect":false,"enableReid":false,"alarmType":[],"drawerOpenLimit":30,"serviceWaitLimit":30,"enableStaffList":false,"staffIds":[]},"validWeekday":["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"],"validStartTime":0,"validEndTime":86400000}]}
```

## 6. Update Cashier Zones

```bash
curl -s -X POST "$BASE/cashier/zones" \
  -H "Content-Type: application/json" \
  -d '{
    "ROI_CASHIER": {
      "shape": "rectangle",
      "points": [{"x": 0.0, "y": 0.0}, {"x": 0.5, "y": 1.0}],
      "active": true
    },
    "ROI_CUSTOMER": {
      "shape": "rectangle",
      "points": [{"x": 0.5, "y": 0.0}, {"x": 1.0, "y": 1.0}],
      "active": true
    },
    "thresholds": {
      "drawer_open_max_seconds": 30,
      "customer_wait_max_seconds": 30,
      "config_reload_interval": 60
    },
    "detail_config": {
      "drawerOpenLimit": 30,
      "serviceWaitLimit": 30
    },
    "task": {
      "task_id": 101,
      "task_name": "cashier_demo",
      "channel_id": "1"
    },
    "detection_threshold": 50
  }'
```

Observed response:

```json
{"status":"updated","config":{"buffer":{"jpeg_quality":75,"size":100},"debounce":{"A3":1,"A4":1,"default":3},"detail_config":{"drawerOpenLimit":30,"enableStaffList":true,"serviceWaitLimit":30,"staffIds":[1001,1002,1003]},"evidence":{"log_rotate_mb":100,"save_gif":true,"save_thumbnail":true},"gif":{"fps":10,"quality":85},"meta":{"version":"1.2.0"},"task":{"algorithmType":"CASHIER_BOX_OPEN","channelId":2,"channelName":"CAM-02-MAIN","deviceSN":"HQDZW1SBCABAH0205","taskId":101,"taskName":"cashier_drawer_monitor","task_id":101,"task_name":"cashier_demo","channel_id":"1"},"thresholds":{"config_reload_interval":60,"customer_wait_max_seconds":30,"detection_threshold":50,"drawer_open_max_seconds":30,"proximity_iou":0.05},"zones":{"ROI_CASHIER":{"shape":"rectangle","points":[[0.0,0.0],[0.5,1.0]],"active":true},"ROI_CUSTOMER":{"shape":"rectangle","points":[[0.5,0.0],[1.0,1.0]],"active":true}}}}
```

Read back the merged config:

```bash
curl -s "$BASE/cashier/zones"
```

## 7. Start Detection

```bash
curl -s -X POST "$BASE/detection/start?camera_id=1"
```

Response:

```json
{"status":"started","cameras":["1"],"tasks":["101"]}
```

## 8. Check Runtime Status

```bash
curl -s "$BASE/detection/status"
```

Observed response in this local test:

```json
{"cameras":{"1":{"camera_id":"1","rtsp_url":"/home/a7med/ml-server/videos/cashier_demo.mp4","running":false,"frame_count":0,"fps":0.0,"last_detections":0,"total_detections":0,"uptime_seconds":0.0,"error":"[VIDEO] Cannot open: /home/a7med/ml-server/videos/cashier_demo.mp4"}}}
```

This means:

- the API flow is correct
- the task and camera were accepted
- the worker started
- the source file did not exist, so no frames were processed

## 9. Watch Cashier SSE

```bash
curl -sN --max-time 2 "$BASE/cashier/stream/1"
```

Observed output:

```text
event: connected
data: {"camera_id": "1", "alert_only": false}
```

Alerts-only stream:

```bash
curl -N "$BASE/cashier/stream/1/only"
```

## 10. Check Cashier Events

```bash
curl -s "$BASE/cashier/events"
```

Observed response:

```json
{"total":0,"offset":0,"limit":100,"events":[]}
```

Because the demo source did not open, no cashier frame or alert events were produced.

## 11. Server Logs From This Demo Run

Observed server log lines:

```text
INFO:     Uvicorn running on http://127.0.0.1:9000 (Press CTRL+C to quit)
INFO:     127.0.0.1 - "POST /cameras HTTP/1.1" 200 OK
INFO:     127.0.0.1 - "POST /api/tasks HTTP/1.1" 200 OK
INFO:     127.0.0.1 - "POST /detection/start?camera_id=1 HTTP/1.1" 200 OK
2026-04-12 ... [CASHIER] Loading model: ./models/best_cashier.onnx
2026-04-12 ... [VIDEO] Opening: /home/a7med/ml-server/videos/cashier_demo.mp4
[1] FrameBus started — tasks: ['101']
[1] FrameBus error: [VIDEO] Cannot open: /home/a7med/ml-server/videos/cashier_demo.mp4
[1] FrameBus stopped. Frames: 0
2026-04-12 ... [CASHIER] Ready — model ./models/best_cashier.onnx | img_size (256, 256) | classes ['Person', 'Drawer_Open', 'Cash'] | GIF enabled
```

## 12. What You Need For A Real Cashier Action Demo

To get real cashier cases like `N3`, `A2`, `A5`, or `A6`, you need:

- a valid RTSP stream or local video file
- cashier-capable detections for `Person`, `Drawer_Open`, and `Cash`
- the cashier model available at `./models/best_cashier.onnx`
- correct `ROI_CASHIER` and `ROI_CUSTOMER` zones

If those are in place, these endpoints become useful:

```bash
curl -s "$BASE/cashier/status"
curl -s "$BASE/cashier/events?limit=20"
curl -s "$BASE/cashier/evidence?limit=20"
curl -N "$BASE/cashier/stream/1"
curl -N "$BASE/cashier/stream/1/only"
```

## 13. Real Image Smoke Test With `best_cashier.onnx`

In addition to the API demo above, the cashier model was tested directly on real images from:

`/mnt/01DA3A868F4FC7D0/frames_with_people/multiple_persons`

Test setup:

- `YOLO_MODEL=/home/a7med/ml-server/models/best_cashier.onnx`
- `CASHIER_MODEL=/home/a7med/ml-server/models/best_cashier.onnx`
- `DEVICE=cpu`
- project services used: `DetectorService` then `CashierService`

What this verifies:

- `best_cashier.onnx` loads correctly
- the detector returns `Person`, `Drawer_Open`, and `Cash`
- cashier business logic produces real `case_id` / `severity` output

### 13.1 Five-image smoke test

Images tested:

- `1000.jpg`
- `1001.jpg`
- `1002.jpg`
- `1003.jpg`
- `1004.jpg`

Observed behavior:

- `1000.jpg` -> `A3`
- `1001.jpg` -> `A1`
- `1002.jpg` -> `A3`
- `1003.jpg` -> `A3`
- `1004.jpg` -> `A3`

Example from `1000.jpg`:

```json
{
  "image": "1000.jpg",
  "detection_count_before_cashier_filter": 7,
  "cashier_summary": {
    "cashier_zone": {"persons": 0, "drawers": 1, "cash": 4},
    "customer_zone": {"persons": 1, "drawers": 0, "cash": 0},
    "case_id": "A3",
    "severity": "CRITICAL",
    "alerts": ["A3 CRITICAL: Cash + open drawer — register unguarded"],
    "transaction": false,
    "frame_saved": true
  }
}
```

The detector also returned the expected cashier classes on that image:

```json
[
  {"class_name": "Person", "confidence": 0.8988},
  {"class_name": "Cash", "confidence": 0.8882},
  {"class_name": "Drawer_Open", "confidence": 0.87},
  {"class_name": "Cash", "confidence": 0.8474}
]
```

### 13.2 Twenty-image summary

Images tested:

- `1000.jpg` through `1019.jpg`

Observed case distribution:

```json
{
  "A3": 13,
  "A1": 2,
  "N5": 5
}
```

Observed class totals before cashier filtering:

```json
{
  "Person": 39,
  "Cash": 54,
  "Drawer_Open": 15
}
```

A few sample rows:

```json
[
  {"image": "1000.jpg", "case_id": "A3", "severity": "CRITICAL", "drawer_count": 1, "cash_count": 4},
  {"image": "1001.jpg", "case_id": "A1", "severity": "CRITICAL", "drawer_count": 1, "cash_count": 0},
  {"image": "1015.jpg", "case_id": "N5", "severity": "NORMAL", "drawer_count": 0, "cash_count": 0},
  {"image": "1019.jpg", "case_id": "N5", "severity": "NORMAL", "drawer_count": 0, "cash_count": 0}
]
```

Conclusion from the real image test:

- `best_cashier.onnx` is working on real frames
- detection is working
- cashier rule evaluation is working
- evidence/GIF generation also triggered during the smoke test

## 14. Sample Logs / Payloads Received By The Backend

This section shows the shape of the data that the backend actually receives and processes during the cashier flow.

### 14.1 Sample payload entering the cashier task worker

This is the shape passed into `CashierDrawerTask.__call__()` from the shared FrameBus pipeline. The real payload also contains a numpy frame in `frame`, but it is omitted here for readability.

```json
{
  "camera_id": "1",
  "frame_id": 1,
  "timestamp": "2026-04-12T11:45:01.980240+00:00",
  "detection": {
    "count": 7,
    "items": [
      {
        "bbox": [958, 0, 1776, 457],
        "class_id": 0,
        "class_name": "Person",
        "confidence": 0.8988,
        "center": [1367, 228],
        "width": 818,
        "height": 457,
        "track_id": -1
      },
      {
        "bbox": [1410, 1150, 1978, 1546],
        "class_id": 1,
        "class_name": "Drawer_Open",
        "confidence": 0.87,
        "center": [1694, 1348],
        "width": 568,
        "height": 396,
        "track_id": -1
      },
      {
        "bbox": [1428, 1182, 1551, 1376],
        "class_id": 2,
        "class_name": "Cash",
        "confidence": 0.8882,
        "center": [1489, 1279],
        "width": 123,
        "height": 194,
        "track_id": -1
      }
    ]
  }
}
```

This means the backend receives:

- camera identity
- frame identity/timestamp
- raw detections for `Person`, `Drawer_Open`, and `Cash`
- the frame itself in memory for zone logic, annotation, and evidence

### 14.2 Sample cashier result produced by the backend

After `CashierService` evaluates the detections and zones, it produces a cashier result like this. This is the shape pushed into backend state and exposed via `/cashier/status`.

```json
{
  "persons": [
    {
      "person_bbox": [958, 0, 1776, 457],
      "confidence": 0.8988,
      "zone": "ROI_CUSTOMER",
      "transaction": false,
      "items": {
        "drawers": [],
        "cash": []
      }
    }
  ],
  "summary": {
    "cashier_zone": {"persons": 0, "drawers": 1, "cash": 4},
    "customer_zone": {"persons": 1, "drawers": 0, "cash": 0},
    "case_id": "A3",
    "severity": "CRITICAL",
    "alerts": ["A3 CRITICAL: Cash + open drawer — register unguarded"],
    "transaction": false,
    "frame_saved": true,
    "evidence_path": "evidence/cashier/abnormal/A3_critical/img-test_20260412T114501_980467.jpg",
    "frame_id": 1,
    "timestamp": "2026-04-12T11:45:01.980240+00:00",
    "cashier_persons": [],
    "total_open_count": 1,
    "total_open_duration_ms": 0,
    "current_open_duration_ms": 0
  },
  "case_id": "A3",
  "severity": "CRITICAL"
}
```

### 14.3 Sample backend event log line

When an alert or transaction is logged, the backend keeps an event entry that can later appear in `GET /cashier/events`.

```json
{
  "camera_id": "img-test",
  "case_id": "A3",
  "severity": "CRITICAL",
  "summary": {
    "alerts": ["A3 CRITICAL: Cash + open drawer — register unguarded"],
    "transaction": false,
    "frame_saved": true,
    "evidence_path": "evidence/cashier/abnormal/A3_critical/img-test_20260412T114501_980467.jpg"
  },
  "logged_at": "2026-04-12T11:45:02.000000+00:00"
}
```

### 14.4 Sample SSE message the backend serves

For the per-camera cashier stream, the backend serves SSE in this form:

```text
event: connected
data: {"camera_id": "1", "alert_only": false}

event: alert
data: {"case_id":"A3","severity":"CRITICAL","camera_id":"1"}
```

The exact alert body can vary by event type, but the backend always emits it as SSE with `event:` and `data:` lines.

## 15. Example Real Cashier Action Logs

These are representative cashier summaries already documented in this repo.

### Example: `N3` active transaction

```json
{
  "summary": {
    "cashier_zone": {"persons": 1, "drawers": 1, "cash": 4},
    "customer_zone": {"persons": 1, "drawers": 0, "cash": 0},
    "case_id": "N3",
    "severity": "NORMAL",
    "alerts": ["N3 EVENT: Transaction in progress"],
    "transaction": true,
    "frame_saved": true
  }
}
```

### Example: `A5` customer waited too long

```json
{
  "summary": {
    "cashier_zone": {"persons": 0, "drawers": 0, "cash": 0},
    "customer_zone": {"persons": 1, "drawers": 0, "cash": 0},
    "case_id": "A5",
    "severity": "ALERT",
    "alerts": ["A5 WARNING: Customer waiting too long"],
    "transaction": false,
    "frame_saved": true
  }
}
```

### Example: cashier SSE connection

```text
event: connected
data: {"camera_id": "1", "alert_only": false}
```

## 16. Stop And Cleanup

```bash
curl -s -X DELETE "$BASE/api/tasks/101"
curl -s -X DELETE "$BASE/cameras/1"
```

If a camera is still running:

```bash
curl -s -X POST "$BASE/detection/stop?camera_id=1"
```

## 17. Short Summary

Start-to-end cashier demo order:

1. start API
2. add camera
3. add `CASHIER_BOX_OPEN` task
4. configure zones
5. start detection
6. read `/detection/status`
7. watch `/cashier/stream/{camera_id}`
8. read `/cashier/events`
9. fetch evidence/media if alerts occur
10. stop and delete task/camera

## 18. Related Docs

- `README.md`
- `docs/CASHIER_BOX_OPEN.md`
- `docs/API_USAGE.md`
- `docs/logs.md` — test command, `EVENTS_DIR` JSONL, curl cheat sheet
- `sse_cashier.md`

## 19. Structured events (`GET /detection/stream` + JSONL)

When `CASHIER_BOX_OPEN` runs under FrameBus, each frame produces **one** task event with the same **outer** envelope as other tasks (`eventId`, `eventType`, `timestamp`, `taskId`, `channelId`, …) plus cashier fields: `camera_id`, `case_id`, `severity`, optional `transaction`, nested Eyego **`data`**, and optional **`evidence`**. Inside **`data`**: `algorithmType`, `captureId`, `sceneId`, **`id`** (32-hex, same UUID as `captureId`), `personStructural` (JSON as a string — **pretty by default**, `\n` escaped on the JSONL line), `captureUrl` / `sceneUrl`, counters, `deviceSN`, etc. CrossLine differs: top-level `line`, `person`, `evidence` paths instead of nested `data`. See [logs.md](./logs.md) for env vars (`CASHIER_CLOUD_IMAGE_BASE`, `CASHIER_COMPACT_PERSON_STRUCTURAL`, …) and side-by-side samples.

Consume globally or filter:

```bash
export BASE=http://127.0.0.1:9000
curl -sN "$BASE/detection/stream?eventType=CASHIER_BOX_OPEN"
```

On disk (default `EVENTS_DIR=/local/storage/events`):

```bash
tail -f /local/storage/events/task_101.jsonl
```

Parse `personStructural` from one JSONL line:

```bash
tail -1 /local/storage/events/task_101.jsonl | jq -r '.data.personStructural | fromjson'
```

See [logs.md](./logs.md) for paths, env, and full samples; [API_USAGE.md](./API_USAGE.md) §5 for the SSE field reference.
