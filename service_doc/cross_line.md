# Cross-line analytics (`CROSS_LINE`)

## What this “service” is

`CROSS_LINE` is **not** a separate microservice or URL prefix. It is an **`algorithmType`** handled inside the vision pipeline: you register a task with `POST /api/tasks`, then run `POST /detection/start`. Events are emitted on the same channels as every other task:

- **`GET /detection/stream`** (SSE) — filter with `eventType=CROSS_LINE`
- **`WS /cameras/{camera_id}/events`** — JSON messages include `eventType: "CROSS_LINE"` when applicable

Implementation: `services/cross_line.py` (`CrossLineTask`).

## HTTP you use (shared pipeline)

| Step | Endpoint | Role |
|------|----------|------|
| Register camera | `POST /cameras` | RTSP for this channel |
| Register task | `POST /api/tasks` | `algorithmType`: `"CROSS_LINE"` |
| Start | `POST /detection/start` | Spawns frame bus + worker |
| Monitor | `GET /detection/status`, `GET /detection/stream?...` | Status + crossing events |
| Hot-update line | `PATCH /api/tasks/{task_id}/lines/{line_id}` | Change one line without full task PUT |
| Stop | `POST /detection/stop` | Tear down |

**Runtime tuning:** `CROSS_LINE_DEBOUNCE_SEC` (default `0`) adds a per `(track_id, line_id)` cooldown between crossing events. See [`docs/OPTIMIZATION_REFERENCE.md`](../docs/OPTIMIZATION_REFERENCE.md).

## Task configuration

| Field | Meaning |
|-------|---------|
| `threshold` | 0–100, minimum **person/detection** confidence (stored as fraction in worker) |
| `areaPosition` | JSON **string** encoding an array of **line** objects (see below) |
| `detailConfig.enableAttrDetect` | If `true`, run age/gender on the crossing person |
| `detailConfig.enableReid` | Reserved |
| `validWeekday`, `validStartTime`, `validEndTime` | Schedule window (ms from midnight for start/end) |

**Common pitfall:** omitting `detailConfig` or sending `{}` leaves `enableAttrDetect` at its schema default **`false`**, so **`AgeGenderService` is never constructed** and every crossing event will show `gender` / `age` as `"Unknown"`. Set `"detailConfig": {"enableAttrDetect": true}` when you want ONNX age/gender on crossings (requires a valid `models/best_aged_gender_6.onnx` and a working ONNX Runtime; for CPU-only hosts set `ONNX_EXECUTION_PROVIDERS_ORDER=cpu_only`).

### `areaPosition` line element

Coordinates are **pixel** values in the **camera frame** (same space as model detections).

```json
{
  "line_id": "1",
  "line_name": "Entrance",
  "point": [{"x": 100, "y": 400}, {"x": 900, "y": 400}],
  "direction": 1
}
```

| `direction` | Meaning |
|-------------|---------|
| `0` | Bidirectional |
| `1` | A → B (first point toward second defines orientation) |
| `2` | B → A |

## Event shape (high level)

Crossing events use `eventType`: **`CROSS_LINE`**. They include line metadata, person/track info, optional attributes, and **`evidence`** with **`captureImage`** and **`sceneImage`** as **ML Image Contract V2** objects (`url`, `path`, `type`, `format`, `timestamp`) — not raw filesystem strings. On-disk files still live under `CAPTURE_DIR` / `SCENE_DIR` with dated paths. See [ml_image_v2.md](./ml_image_v2.md) for the full contract and SSH/tail examples.

Local JSONL (server disk): under `EVENTS_DIR` (default `/local/storage/events`), file `task_{taskId}.jsonl`.

## curl — register cross-line task only

```bash
export BASE="http://localhost:9000"

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

## curl — SSE only cross-line events for one task and camera

```bash
curl -sS -N "${BASE}/detection/stream?eventType=CROSS_LINE&taskId=10&channelId=1"
```

## Edge device checklist (cross-line)

1. `POST /cameras` with `id` equal to the task `channelId`.
2. `POST /api/tasks` with `algorithmType`: `"CROSS_LINE"` and a correct `areaPosition` for your resolution.
3. `POST /detection/start` (or `?camera_id=` for that channel only).
4. Consume `GET /detection/stream` with `eventType=CROSS_LINE` (and optional `taskId` / `channelId`).
5. `POST /detection/stop` when done.

If you change line geometry often, update the task with `PUT /api/tasks/{taskId}` and restart the pipeline for that camera so workers reload config.
