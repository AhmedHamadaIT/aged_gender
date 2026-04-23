# PPE / mask & headwear analytics (`MASK_HAIRNET_CHEF_HAT`)

## What this “service” is

`MASK_HAIRNET_CHEF_HAT` is **not** a separate HTTP service. It is an **`algorithmType`** for the same pipeline as cross-line: register with `POST /api/tasks`, start with `POST /detection/start`. Violation events use:

- **`GET /detection/stream`** — filter with `eventType=MASK_HAIRNET_CHEF_HAT`
- **`WS /cameras/{camera_id}/events`**

Implementation: `services/mask_hairnet_chef_hat.py` (`MaskHairnetChefHatTask`), using `PPEService` (`services/ppe.py`).

## HTTP you use (shared pipeline)

| Step | Endpoint | Role |
|------|----------|------|
| Register camera | `POST /cameras` | RTSP for kitchen / zone camera |
| Register task | `POST /api/tasks` | `algorithmType`: `"MASK_HAIRNET_CHEF_HAT"` |
| Start | `POST /detection/start` | Spawns frame bus + worker |
| Monitor | `GET /detection/status`, `GET /detection/stream?...` | Status + PPE violation events |
| Stop | `POST /detection/stop` | Tear down |

## Task configuration

| Field | Meaning |
|-------|---------|
| `threshold` | 0–100, minimum **PPE model** confidence |
| `areaPosition` | JSON **string**: array of **polygon zones**; each item uses `point` as a list of `{x, y}` vertices (pixels). `line_id` labels the zone; `direction` is accepted in examples but zones are polygon-based (see API_USAGE polygon example). |
| `detailConfig.alarmType` | List of violations to emit: `no_mask`, `no_chef_hat`, `no_hat` |

### Alarm types → model behavior

From `services/mask_hairnet_chef_hat.py`:

| `alarmType` | Meaning |
|-------------|---------|
| `no_mask` | Requires **mask** class on the person crop |
| `no_hat` | Requires **hairnet** class (described as hairnet not detected) |
| `no_chef_hat` | Also mapped to **hairnet** in the PPE model (chef hat is not a separate class) |

Only **violation** (`no_*`) types generate alerts; positive detections do not.

If `alarmType` is empty after filtering, the task emits **no** PPE events (worker still loads but schedule/alarms may yield nothing useful — prefer an explicit list).

## Event shape (high level)

Events use `eventType`: **`MASK_HAIRNET_CHEF_HAT`**. Payload includes `alert` (`type`, `description`, `confidence`), `person` (bbox, `trackingId`, etc.), and **`evidence`**: **`captureImage`** and **`sceneImage`** are **V2 image objects** (see [ml_image_v2.md](./ml_image_v2.md)). Implementation: `services/mask_hairnet_chef_hat.py`.

Local JSONL: `EVENTS_DIR/task_{taskId}.jsonl` — tail over SSH: [ml_image_v2.md](./ml_image_v2.md#ssh--watch-task-jsonl-all-cases).

## curl — register PPE task (kitchen example)

```bash
export BASE="http://localhost:9000"

curl -sS -X POST "${BASE}/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 20,
    "taskName": "kitchen_ppe_check",
    "algorithmType": "MASK_HAIRNET_CHEF_HAT",
    "channelId": "2",
    "enable": true,
    "threshold": 70,
    "areaPosition": "[{\"line_id\":\"zone1\",\"point\":[{\"x\":50,\"y\":50},{\"x\":600,\"y\":50},{\"x\":600,\"y\":500},{\"x\":50,\"y\":500}],\"direction\":0}]",
    "detailConfig": {
      "enableAttrDetect": false,
      "enableReid": false,
      "alarmType": ["no_mask", "no_chef_hat", "no_hat"]
    },
    "validWeekday": ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"],
    "validStartTime": 0,
    "validEndTime": 86400000
  }'
```

## curl — SSE only PPE events

```bash
curl -sS -N "${BASE}/detection/stream?eventType=MASK_HAIRNET_CHEF_HAT&channelId=2"
```

Add `taskId=20` if you want a single task:

```bash
curl -sS -N "${BASE}/detection/stream?eventType=MASK_HAIRNET_CHEF_HAT&taskId=20"
```

## Edge device checklist (PPE)

1. `POST /cameras` — ensure `channelId` in the task matches camera `id`.
2. `POST /api/tasks` — draw `areaPosition` polygons in **the same pixel space** as the encoded stream resolution used for inference (adjust if stream resolution differs from UI calibration).
3. Set `detailConfig.alarmType` to the violations you care about.
4. `POST /detection/start`.
5. Consume SSE with `eventType=MASK_HAIRNET_CHEF_HAT`; handle bursty alerts (multiple people, multiple violations).
6. `POST /detection/stop` when done.

## Same camera, multiple algorithms

You can register **multiple tasks** on one `channelId` (e.g. one `CROSS_LINE` and one `MASK_HAIRNET_CHEF_HAT`). Starting detection spawns **one FrameBus** per camera and **one worker per task**. Filter `GET /detection/stream` by `eventType` and/or `taskId` on the client.
