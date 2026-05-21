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
| `areaPosition` | JSON **string**: polygon zone in **pixel coordinates** on the video frame (see below) |
| `detailConfig.alarmType` | List of violations to emit: `no_mask`, `no_chef_hat`, `no_hat` |
| `detailConfig.channelName` | Optional — Eyego `data.channelName` (defaults to camera id string) |
| `detailConfig.deviceSN` | Optional — Eyego `data.deviceSN` (falls back to `PPE_DEVICE_SN`, `DEVICE_SN`, `HOSTNAME`) |

### `areaPosition` — polygon zone (no line)

PPE uses a **closed polygon**, not a line. Each `{x, y}` is a **vertex in pixel space** on the encoded stream used for inference (e.g. 1920×1080). The server tests each detected person's **centroid** against the polygon; persons outside are ignored.

**Preferred — bare polygon** (array of points):

```json
"[{\"x\":1004,\"y\":56},{\"x\":1831,\"y\":89},{\"x\":2013,\"y\":1831},{\"x\":308,\"y\":1876},{\"x\":304,\"y\":1872}]"
```

**Also accepted — wrapped zone** (optional label):

```json
"[{\"point\":[{\"x\":50,\"y\":50},{\"x\":600,\"y\":50},{\"x\":600,\"y\":500},{\"x\":50,\"y\":500}]}]"
```

| Rule | Detail |
|------|--------|
| Minimum points | 3 |
| Empty `"[]"` | Entire frame is the detection zone |
| `line_id` / `direction` | **Not used** for PPE (those are for `CROSS_LINE` only) |
| Resolution | Points must match the **same pixel space** as the RTSP stream / UI calibration |

```
(0,0) ───────────────────────► x
  │
  │    (1004,56)────(1831,89)
  │         │            │
  │         │  ZONE      │
  │    (304,1872)─(308,1876)
  │              (2013,1831)
  ▼
  y
```

### Alarm types → model behavior

From `services/mask_hairnet_chef_hat.py`:

| `alarmType` | Meaning |
|-------------|---------|
| `no_mask` | Requires **mask** class on the person crop |
| `no_hat` | Requires **hairnet** class (described as hairnet not detected) |
| `no_chef_hat` | Also mapped to **hairnet** in the PPE model (chef hat is not a separate class) |

Only **violation** (`no_*`) types generate alerts; positive detections do not.

If `alarmType` is empty after filtering, the task emits **no** PPE events (worker still loads but schedule/alarms may yield nothing useful — prefer an explicit list).

## Event shape — Eyego `data` envelope

Each violation emits one SSE/JSONL event. Top-level fields support **`GET /detection/stream`** filters (`eventType`, `taskId`, `channelId`, `taskName`). The integration payload lives under **`data`** (same pattern as `CASHIER_BOX_OPEN`).

Parse detection details from **`data.personStructural`** (JSON string):

| `personStructural` field | Meaning |
|--------------------------|---------|
| `alarmType` | `"no_mask"`, `"no_hat"`, or `"no_chef_hat"` |
| `areaPoints` | Zone polygon as a **stringified JSON array** of `{x, y}` |
| `objectX`, `objectY`, `objectWidth`, `objectHeight` | Person bounding box (pixels) |
| `score` | PPE confidence 0–100 |
| `smokingX/Y/Width/Height` | Reserved; always `0` for PPE |

**`data` block fields:** `algorithmType`, `captureId`, `sceneId`, `channelId` (int when numeric), `channelName`, `deviceSN`, `id` (32-hex correlation id), `taskId`, `taskName`, `recordTime`, `dateUTC`, `personStructural`, `captureUrl`, `sceneUrl`, `evidence` (V2 `captureImage` / `sceneImage`).

Top-level **`evidence`** mirrors `data.evidence` for consumers that read the outer envelope. See [ml_image_v2.md](./ml_image_v2.md).

### URL env vars

| Env | Purpose |
|-----|---------|
| `PPE_CLOUD_IMAGE_BASE` | Primary base for both `captureUrl` and `sceneUrl` |
| `PPE_CAPTURE_URL_BASE` / `PPE_SCENE_URL_BASE` | Per-side fallback when cloud base unset |
| `PPE_FORCE_LOCAL_URLS=1` | Use `file:///local/storage/images` for missing sides |
| `PPE_DEVICE_SN` / `DEVICE_SN` | Default `deviceSN` |

URL pattern: `{base}/{captureId}{id}.jpg` (correlation `id` appended before final `.jpg`).

Local JSONL: `EVENTS_DIR/task_{taskId}.jsonl` — tail over SSH: [ml_image_v2.md](./ml_image_v2.md#ssh--watch-task-jsonl-all-cases).

## curl — register PPE task

```bash
export BASE="http://localhost:9000"

curl -sS -X POST "${BASE}/api/tasks" \
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
      "alarmType": ["no_mask", "no_chef_hat", "no_hat"],
      "channelName": "7",
      "deviceSN": "HQDZW1SBCABAH0235"
    },
    "validWeekday": ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"],
    "validStartTime": 0,
    "validEndTime": 86400000
  }'
```

**Response:**
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
    "areaPosition": "[{\"x\":1004,\"y\":56},...]",
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

## curl — SSE only PPE events

```bash
curl -sS -N "${BASE}/detection/stream?eventType=MASK_HAIRNET_CHEF_HAT&channelId=7"
```

Add `taskId=8` for a single task:

```bash
curl -sS -N "${BASE}/detection/stream?eventType=MASK_HAIRNET_CHEF_HAT&taskId=8"
```

**Example SSE event** (pretty-printed):

```json
{
  "eventId": "a1b2c3d4e5f678901234567890123456",
  "eventType": "MASK_HAIRNET_CHEF_HAT",
  "timestamp": 1774312985135,
  "timestampUTC": "2026-03-24T00:43:05.135Z",
  "taskId": 8,
  "taskName": "staff_safety_bar_area",
  "channelId": "7",
  "camera_id": "7",
  "data": {
    "algorithmType": "MASK_HAIRNET_CHEF_HAT",
    "captureId": "MASK_HAIRNET_CHEF_HAT_82ae3908-3130-487b-9bf6-13b52c5d788f.jpg",
    "sceneId": "MASK_HAIRNET_CHEF_HAT_88132b76-f9d1-4df2-936f-5a69afabe123.jpg",
    "channelId": 7,
    "channelName": "7",
    "deviceSN": "HQDZW1SBCABAH0235",
    "id": "642a88a75023084fea16bc8828e8d351",
    "taskId": 8,
    "taskName": "staff_safety_bar_area",
    "recordTime": 1774312985135,
    "dateUTC": "2026-03-24T00:43:05.135Z",
    "personStructural": "{\"alarmType\":\"no_chef_hat\",\"areaPoints\":\"[{\\\"x\\\":1004,\\\"y\\\":56},{\\\"x\\\":1831,\\\"y\\\":89},{\\\"x\\\":2013,\\\"y\\\":1831},{\\\"x\\\":308,\\\"y\\\":1876},{\\\"x\\\":304,\\\"y\\\":1872}]\",\"objectHeight\":144,\"objectWidth\":117,\"objectX\":1417,\"objectY\":115,\"score\":79,\"smokingHeight\":0,\"smokingWidth\":0,\"smokingX\":0,\"smokingY\":0}",
    "captureUrl": "https://storage.googleapis.com/logs-data-images/MASK_HAIRNET_CHEF_HAT_82ae3908-3130-487b-9bf6-13b52c5d788f.jpg642a88a75023084fea16bc8828e8d351.jpg",
    "sceneUrl": "https://storage.googleapis.com/logs-data-images/MASK_HAIRNET_CHEF_HAT_88132b76-f9d1-4df2-936f-5a69afabe123.jpg642a88a75023084fea16bc8828e8d351.jpg",
    "evidence": {
      "captureImage": { "url": "…", "path": "2026-03-24/MASK_HAIRNET_CHEF_HAT_….jpg", "type": "capture", "format": "image/jpeg", "timestamp": "…" },
      "sceneImage":   { "url": "…", "path": "2026-03-24/MASK_HAIRNET_CHEF_HAT_….jpg", "type": "scene",   "format": "image/jpeg", "timestamp": "…" }
    }
  },
  "evidence": { "captureImage": { "…": "same as data.evidence" }, "sceneImage": { "…": "same" } }
}
```

Parse `personStructural` on the client:

```bash
# From JSONL or saved SSE payload
jq -r '.data.personStructural | fromjson'
```

## Edge device checklist (PPE)

1. `POST /cameras` — ensure `channelId` in the task matches camera `id`.
2. `POST /api/tasks` — set `areaPosition` polygon in **the same pixel space** as the encoded stream resolution.
3. Set `detailConfig.alarmType` to the violations you care about; optional `channelName` / `deviceSN`.
4. Set `PPE_CLOUD_IMAGE_BASE` (or per-side bases) if downstream needs `captureUrl` / `sceneUrl`.
5. `POST /detection/start`.
6. Consume SSE with `eventType=MASK_HAIRNET_CHEF_HAT`; handle bursty alerts (multiple people, multiple violations).
7. `POST /detection/stop` when done.

## Same camera, multiple algorithms

You can register **multiple tasks** on one `channelId` (e.g. one `CROSS_LINE` and one `MASK_HAIRNET_CHEF_HAT`). Starting detection spawns **one FrameBus** per camera and **one worker per task**. Filter `GET /detection/stream` by `eventType` and/or `taskId` on the client.
