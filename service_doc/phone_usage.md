# Phone usage analytics (`PHONE_USAGE`)

## What this "service" is

`PHONE_USAGE` is **not** a separate HTTP service. It is an **`algorithmType`** handled by the same vision pipeline used by cross-line and PPE tasks: register with `POST /api/tasks`, then run `POST /detection/start`.

Phone events are emitted on the shared channels:

- **`GET /detection/stream`** (SSE) - filter with `eventType=PHONE_USAGE`
- **`WS /cameras/{camera_id}/events`** - JSON messages include `eventType: "PHONE_USAGE"` when a phone is detected

Implementation: `services/phone_usage.py` (`PhoneUsageTask`), using `PhoneService` (`services/phone.py`).

## HTTP you use (shared pipeline)

| Step | Endpoint | Role |
|------|----------|------|
| Register camera | `POST /cameras` | RTSP for this channel |
| Register task | `POST /api/tasks` | `algorithmType`: `"PHONE_USAGE"` |
| Start | `POST /detection/start` | Spawns frame bus + worker |
| Monitor | `GET /detection/status`, `GET /detection/stream?...` | Status + phone usage events |
| Stop | `POST /detection/stop` | Tear down |

## Task configuration

| Field | Meaning |
|-------|---------|
| `threshold` | 0-100, minimum **person detection** confidence used before running phone inference |
| `areaPosition` | JSON **string**: array of **polygon zones**; each zone uses `point` as a list of `{x, y}` vertices (pixels) |
| `detailConfig` | Present for schema compatibility; no required phone-specific keys today |
| `validWeekday`, `validStartTime`, `validEndTime` | Schedule window (ms from midnight for start/end) |

### `areaPosition` zone element

Coordinates are **pixel** values in the camera frame.

```json
{
  "line_id": "zone1",
  "line_name": "Counter area",
  "point": [{"x": 80, "y": 80}, {"x": 920, "y": 80}, {"x": 920, "y": 680}, {"x": 80, "y": 680}],
  "direction": 0
}
```

Zone filtering behavior:

- If at least one zone is configured, a person must be inside one zone to be evaluated.
- If no zones are configured, all people in frame can be evaluated.
- A zone needs at least 3 points to be treated as a polygon.

## Event shape (high level)

Events use `eventType`: **`PHONE_USAGE`**. Payload includes:

- `alert` (`type`, `description`, `confidence`)
- `person` (`trackingId`, person `boundingBox`, `areaPoints` from the selected zone)
- `phone` (`boundingBox`, `confidence`)
- `evidence` with **`captureImage`** and **`sceneImage`** as **ML Image Contract V2** objects (`url`, `path`, `type`, `format`, `timestamp`)

Local JSONL (server disk): under `EVENTS_DIR` (default `/local/storage/events`), file `task_{taskId}.jsonl`.

See [ml_image_v2.md](./ml_image_v2.md) for full evidence contract and SSH/tail examples.

## curl - register phone usage task only

```bash
export BASE="http://localhost:9000"

curl -sS -X POST "${BASE}/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 30,
    "taskName": "phone_zone_check",
    "algorithmType": "PHONE_USAGE",
    "channelId": "3",
    "enable": true,
    "threshold": 60,
    "areaPosition": "[{\"line_id\":\"zone1\",\"line_name\":\"Counter area\",\"point\":[{\"x\":80,\"y\":80},{\"x\":920,\"y\":80},{\"x\":920,\"y\":680},{\"x\":80,\"y\":680}],\"direction\":0}]",
    "detailConfig": {"enableAttrDetect": false, "enableReid": false, "alarmType": []},
    "validWeekday": ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"],
    "validStartTime": 0,
    "validEndTime": 86400000
  }'
```

## curl - SSE only phone events for one task and camera

```bash
curl -sS -N "${BASE}/detection/stream?eventType=PHONE_USAGE&taskId=30&channelId=3"
```

## Edge device checklist (phone usage)

1. `POST /cameras` with `id` equal to the task `channelId`.
2. `POST /api/tasks` with `algorithmType`: `"PHONE_USAGE"` and correct polygon `areaPosition` for your resolution.
3. `POST /detection/start` (or `?camera_id=` for that channel only).
4. Consume `GET /detection/stream` with `eventType=PHONE_USAGE` (and optional `taskId` / `channelId`).
5. `POST /detection/stop` when done.

If you update zone geometry, call `PUT /api/tasks/{taskId}` and restart that camera pipeline so workers reload config.
