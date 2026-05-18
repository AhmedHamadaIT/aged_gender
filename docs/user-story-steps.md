# User story steps — Cross-line counting & live annotated preview

This document is for the **frontend team** building dashboards that register cameras, draw counting lines, start detection, show **live video with boxes + cross-line overlay**, and display **per-line crossing counts** from events.

**Related docs**

| Topic | Document |
|--------|----------|
| Full API reference | [`API_USAGE.md`](./API_USAGE.md) |
| Live WebSocket rendering | [`frontend-live-stream-guide.md`](./frontend-live-stream-guide.md) |
| Cross-line task contract | [`../service_doc/cross_line.md`](../service_doc/cross_line.md) |
| Longer backend walkthrough | [`sanrio-camera-line-counting-walkthrough.md`](./sanrio-camera-line-counting-walkthrough.md) |

**ML server base URL** in examples: `http://localhost:9000` (replace with your deploy host, e.g. `http://34.47.247.221:9000`).

---

## User stories (what we are building)

### Story A — Operator configures a camera and counting line

> **As an** operator  
> **I want to** register an RTSP camera, draw one or more virtual lines on a still frame, and save them as a `CROSS_LINE` task  
> **So that** the ML server counts people crossing those lines.

### Story B — Operator watches live annotated video

> **As an** operator  
> **I want to** open a live preview that shows **person boxes, track IDs, and my counting lines** on the same image  
> **So that** I can confirm lines align with the floor/door before trusting the counts.

### Story C — Dashboard shows live counts

> **As a** dashboard user  
> **I want to** see **in / out / total** counters per line that update when someone crosses  
> **So that** I can monitor entrance traffic in real time.

**Important:** The ML server emits **one JSON event per crossing**. It does **not** store running totals. The **frontend (or your backend)** increments counters from the event stream.

---

## End-to-end flow (screens)

```mermaid
flowchart LR
  A[1. Register camera] --> B[2. Snapshot / line editor]
  B --> C[3. POST CROSS_LINE task]
  C --> D[4. POST detection/start]
  D --> E[5. WS live preview]
  D --> F[6. SSE / WS events]
  F --> G[7. Aggregate counts in UI]
```

| Step | User action | API / transport |
|------|-------------|-----------------|
| 1 | Add camera (id + RTSP URL) | `POST /cameras` |
| 2 | Draw line(s) on canvas | Use `GET /cameras` → `snapshot`; save geometry in **inference pixels** |
| 3 | Save task | `POST /api/tasks` (`algorithmType`: `CROSS_LINE`, `areaPosition` as JSON **string**) |
| 4 | Start pipeline | `POST /detection/start?camera_id={id}` |
| 5 | Live preview tile | `WS /cameras/{camera_id}/live` (binary JPEG) |
| 6 | Crossing feed | `GET /detection/stream?eventType=CROSS_LINE&channelId={id}` or `WS /cameras/{id}/events` |
| 7 | Stop | `POST /detection/stop?camera_id={id}` |

---

## Coordinate system (do not skip)

Crossing logic and the **live line overlay** use the same pixel space as **YOLO inference** after resize in FrameBus.

| Deploy | Typical `WIDTH` × `HEIGHT` | Mid horizontal line example |
|--------|---------------------------|-----------------------------|
| Local `.env` | **640 × 360** | `(0, 180)` → `(640, 180)` |
| Docker Compose `yolo-detect` | **480 × 360** | `(0, 180)` → `(480, 180)` |
| Custom | Ask ops for `WIDTH` / `HEIGHT` | `y = HEIGHT / 2`, `x` from `0` to `WIDTH` |

`GET /cameras` **snapshots** are often **full-resolution** from the RTSP source. They are **not** always the same size as inference.

**Frontend rule:** either

1. Resize the snapshot in the line editor to **`WIDTH × HEIGHT`** (match server config), draw lines, send those coordinates; or  
2. Draw on the full snapshot, then scale when saving:  
   `x' = round(x * WIDTH / snapWidth)`, `y' = round(y * HEIGHT / snapHeight)`.

If lines look wrong on live video or counts never fire, this scaling step is almost always the cause.

---

## Step 1 — Register the camera

**Example camera** (channel `401`):

```http
POST /cameras
Content-Type: application/json

{
  "cameras": [
    {
      "id": "401",
      "url": "rtsp://admin:Aa112233@10.0.3.71:554/Streaming/channels/401"
    }
  ]
}
```

Flat body is also accepted: `{ "camera_id": "401", "rtsp_url": "rtsp://..." }`.

**Verify:** `GET /cameras` → each entry has `id`, `url`, and optional `snapshot` (path to a recent JPEG for the line editor).

**Frontend:** store `cameraId` (string) — it must equal `channelId` on every task for that camera.

---

## Step 2 — Line editor UI (frontend-owned)

### Canvas requirements

1. Load still image from `GET /cameras` → use `snapshot` URL (or proxy it through your API if the browser cannot reach the ML host filesystem).
2. Set canvas logical size to **`WIDTH × HEIGHT`** (from env/config your ops team documents).
3. Let the user place **exactly two points** per line (segment). Support multiple lines (array).
4. Optional: show **direction** picker (`0` both ways, `1` A→B, `2` B→A). Direction follows the vector from `point[0]` → `point[1]`.

### Export shape for `areaPosition`

`areaPosition` must be a **JSON string** (stringified array), not a nested JSON object in the HTTP body.

Each line object:

```json
{
  "line_id": "1",
  "line_name": "Mid entrance",
  "point": [
    { "x": 0, "y": 180 },
    { "x": 640, "y": 180 }
  ],
  "direction": 0
}
```

| `direction` | Meaning |
|-------------|---------|
| `0` | Count crossings both ways |
| `1` | Count only when crossing from first point toward second |
| `2` | Count only the opposite direction |

**UI export helper (TypeScript):**

```typescript
type Point = { x: number; y: number };
type LineDraft = {
  line_id: string;
  line_name: string;
  start: Point;
  end: Point;
  direction: 0 | 1 | 2;
};

export function toAreaPositionString(lines: LineDraft[]): string {
  const payload = lines.map((ln) => ({
    line_id: ln.line_id,
    line_name: ln.line_name,
    point: [ln.start, ln.end],
    direction: ln.direction,
  }));
  return JSON.stringify(payload);
}
```

### Mid-frame line (common “count everyone crossing the center”)

For **640×360** inference — horizontal line through vertical center:

```json
[
  {
    "line_id": "1",
    "line_name": "Mid horizontal",
    "point": [{ "x": 0, "y": 180 }, { "x": 640, "y": 180 }],
    "direction": 0
  }
]
```

For **480×360** (Docker): use `x: 480` instead of `640` at the second point.

---

## Step 3 — Register the `CROSS_LINE` task

```http
POST /api/tasks
Content-Type: application/json

{
  "taskId": 40101,
  "taskName": "cam401_mid_line",
  "algorithmType": "CROSS_LINE",
  "channelId": "401",
  "enable": true,
  "threshold": 30,
  "areaPosition": "[{\"line_id\":\"1\",\"line_name\":\"Mid horizontal\",\"point\":[{\"x\":0,\"y\":180},{\"x\":640,\"y\":180}],\"direction\":0}]",
  "detailConfig": {
    "enableAttrDetect": false,
    "enableReid": false
  },
  "validWeekday": ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"],
  "validStartTime": 0,
  "validEndTime": 86400000
}
```

**Notes for frontend forms**

- `taskId` — stable integer primary key (your system owns it).
- `channelId` — **must match** camera `id` (`"401"`).
- `threshold` — 0–100; minimum person detection confidence.
- `detailConfig.enableAttrDetect: true` — enables age/gender on each crossing (heavier CPU/GPU).
- Re-posting the same `taskId` **updates** the task (`status: "updated"`).

Optional link after the fact: `POST /cameras/401/tasks` with `{ "taskId": 40101 }`.

---

## Step 4 — Start detection

```http
POST /detection/start?camera_id=401
```

No body required for a single-camera start.

**Poll until running:**

```http
GET /detection/status
```

Expect the camera entry with `"running": true` before showing live preview or expecting events.

### Start failures (show in UI)

`POST /detection/start` may return **422** with a structured body (`error_code`, `message`, `stage`). Common codes:

| `error_code` | User-facing hint |
|--------------|------------------|
| `STREAM_UNREACHABLE` | RTSP URL wrong or ML host cannot reach camera network |
| `NO_TASK_ASSIGNED` | No enabled task for this `channelId` |
| `TASK_CONFIG_INVALID` | Bad `areaPosition` (empty line, missing points) |
| `BUS_INIT_TIMEOUT` / `WORKER_INIT_TIMEOUT` | Models slow to load — retry or check GPU |

Full list: `utils/error_codes.py`.

---

## Step 5 — Live preview with annotation **and cross-line overlay**

### What the server sends

After `POST /detection/start`, FrameBus publishes **annotated JPEG** frames:

- **Person detections:** bounding boxes + track IDs (per `LIVE_ANNOTATION_MODE`, usually `opencv` or `ultralytics`).
- **Cross-line overlay:** segments from every **enabled** `CROSS_LINE` task on that `channelId`, drawn in **inference pixel space** (built at start from `areaPosition` via `utils/live_stream_overlay.py`).

So operators **should see their counting lines on the live feed** without drawing them again in the browser — as long as detection is running and Redis is configured.

### WebSocket URL

```text
ws://<host>:9000/cameras/401/live
```

Alias by task name:

```text
ws://<host>:9000/tasks/cam401_mid_line/live
```

**Requirements**

- `REDIS_URL` must be set on the ML server (Docker Compose includes Redis).
- One WebSocket **per camera** tile (no multiplexed multi-cam socket).
- Messages are **raw JPEG bytes** (not Base64 JSON).

### Minimal frontend player

See [`frontend-live-stream-guide.md`](./frontend-live-stream-guide.md) for memory-safe `createObjectURL` + `requestAnimationFrame` patterns.

```javascript
const wsUrl = `ws://${location.host}/cameras/401/live`;
const img = document.getElementById("live-preview");

function connectLive() {
  const ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";
  let objectUrl = null;
  let pending = null;
  let raf = false;

  ws.onmessage = (ev) => {
    pending = new Uint8Array(ev.data);
    if (!raf) {
      raf = true;
      requestAnimationFrame(() => {
        raf = false;
        if (!pending || pending[0] !== 0xff || pending[1] !== 0xd8) return;
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        objectUrl = URL.createObjectURL(
          new Blob([pending], { type: "image/jpeg" })
        );
        img.src = objectUrl;
        pending = null;
      });
    }
  };

  ws.onclose = () => setTimeout(connectLive, 2000);
  return ws;
}
```

### Optional client-side line overlay

Server JPEG already includes lines. Use a **client overlay** only when:

- You are editing lines **before** save (draft geometry on the snapshot canvas), or  
- Detection is **stopped** and you still want to show saved geometry on a static snapshot.

For draft editing, draw on a `<canvas>` stacked above the image; keep coordinates in **inference space**, scale to display with `scaleX = displayWidth / WIDTH`.

---

## Step 6 — Crossing events (build the counters)

### SSE (simplest for web)

```http
GET /detection/stream?eventType=CROSS_LINE&channelId=401
```

Each `data:` line is JSON. Example fields:

```json
{
  "eventType": "CROSS_LINE",
  "taskId": 40101,
  "taskName": "cam401_mid_line",
  "channelId": "401",
  "line": {
    "id": "1",
    "name": "Mid horizontal",
    "direction": 1
  },
  "person": {
    "trackingId": "42",
    "boundingBox": { "x": 120, "y": 80, "width": 60, "height": 140 },
    "attributes": { "gender": "Unknown", "age": "Unknown" },
    "confidence": 85
  },
  "evidence": {
    "captureImage": { "url": "...", "path": "...", "type": "capture", "format": "jpg" },
    "sceneImage": { "url": "...", "path": "...", "type": "scene", "format": "jpg" }
  },
  "timestamp": 1715900000000
}
```

`line.direction` on the event is the **crossing direction that fired** (`1` or `2`), not the task config value `0`.

### WebSocket (per camera)

```text
ws://<host>:9000/cameras/401/events
```

Same JSON objects as SSE, one message per event.

### Counter logic (frontend state)

```typescript
type LineCounts = Record<string, { in: number; out: number; total: number }>;

function onCrossLineEvent(event: {
  line?: { id?: string; direction?: number };
}) {
  const lineId = event.line?.id ?? "unknown";
  const dir = event.line?.direction;
  if (!counts[lineId]) counts[lineId] = { in: 0, out: 0, total: 0 };
  if (dir === 1) counts[lineId].in += 1;
  else if (dir === 2) counts[lineId].out += 1;
  counts[lineId].total += 1;
  renderCounters(counts);
}
```

Use `taskId` / `line.id` as keys when multiple line tasks exist on one camera.

---

## Step 7 — Stop and edit

```http
POST /detection/stop?camera_id=401
```

**Change lines:** `PUT /api/tasks/{taskId}` with new `areaPosition`, then **stop and start** detection so FrameBus rebuilds the live overlay snapshot.

**Disable without delete:** `PUT /api/tasks/{taskId}` with `"enable": false`.

---

## Suggested dashboard layout

```text
┌─────────────────────────────────────────────────────────────┐
│  Camera: [401 ▼]   [Start] [Stop]   Status: ● Running       │
├──────────────────────────┬──────────────────────────────────┤
│  Live preview (WS /live) │  Line editor (snapshot canvas)    │
│  • boxes + track IDs     │  • drag two points per line       │
│  • server-drawn lines    │  • direction 0/1/2              │
│                          │  [Save task]                     │
├──────────────────────────┴──────────────────────────────────┤
│  Counts:  Line "Mid horizontal"  IN: 12  OUT: 9  TOTAL: 21  │
│  Event log (last 20 crossings)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Frontend checklist

| Item | Done when |
|------|-----------|
| Camera id === task `channelId` | Same string everywhere (`"401"`) |
| Line editor uses inference `WIDTH×HEIGHT` | Lines align on live WS preview |
| `areaPosition` sent as **string** | `JSON.stringify(lines)` in POST body |
| Live tile uses `WS …/cameras/{id}/live` | JPEG renders; revoke blob URLs |
| Counters from SSE/WS events | Not expecting totals from ML API |
| Handle 422 on start | Show `error_code` + message |
| Reconnect WS on close | ~2 s backoff |
| Redis required for live | Ops set `REDIS_URL`; document for deploy |

---

## Quick curl script (QA / backend)

```bash
export BASE=http://localhost:9000
export CAM_ID=401
export RTSP='rtsp://admin:Aa112233@10.0.3.71:554/Streaming/channels/401'

curl -sS -X POST "$BASE/cameras" -H "Content-Type: application/json" \
  -d "{\"cameras\":[{\"id\":\"$CAM_ID\",\"url\":\"$RTSP\"}]}"

curl -sS -X POST "$BASE/api/tasks" -H "Content-Type: application/json" -d '{
  "taskId": 40101,
  "taskName": "cam401_mid_line",
  "algorithmType": "CROSS_LINE",
  "channelId": "401",
  "enable": true,
  "threshold": 30,
  "areaPosition": "[{\"line_id\":\"1\",\"line_name\":\"Mid horizontal\",\"point\":[{\"x\":0,\"y\":180},{\"x\":640,\"y\":180}],\"direction\":0}]",
  "detailConfig": {"enableAttrDetect": false},
  "validWeekday": ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"],
  "validStartTime": 0,
  "validEndTime": 86400000
}'

curl -sS -X POST "$BASE/detection/start?camera_id=$CAM_ID"
curl -sS "$BASE/detection/status"
# Live: ws://localhost:9000/cameras/401/live
# Events: curl -sS -N "$BASE/detection/stream?eventType=CROSS_LINE&channelId=$CAM_ID"
```

---

## Reference (implementation in this repo)

| Concern | Location |
|---------|----------|
| Live overlay (lines + zones) | `utils/live_stream_overlay.py`, `frame_bus.py` |
| Crossing logic | `services/cross_line.py` |
| Start validation gate | `apis/detection.py`, `services/stream_validator.py` |
| WebSocket live | `apis/ws_live.py` |
| OpenAPI | `GET /docs` on running server |
