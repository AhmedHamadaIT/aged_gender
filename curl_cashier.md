# Vision Pipeline API — cURL reference (backend handoff)

Production FastAPI service. Replace `<jetson-ip>` with the device hostname or IP.

**Base URL:** `http://<jetson-ip>:9000`

Run locally (default port from app): `uvicorn app:app --host 0.0.0.0 --port 9000`

Interactive schemas: `http://<jetson-ip>:9000/docs`

**Batch logs:** Local cashier-YOLO run [`outputs/cashier_test/20260329T135320_10106/stream.jsonl`](outputs/cashier_test/20260329T135320_10106/stream.jsonl) (73 frames; see [`sse_cashier.md`](sse_cashier.md)). Full-dataset aggregates: [`outputs/cashier_test/20260329T060741_7464/summary.json`](outputs/cashier_test/20260329T060741_7464/summary.json) (see **Cashier full-dataset validation report** in [`README.md`](README.md)).

---

## SSE event reference (`GET /detection/stream`)

Each SSE line is `data: ` followed by one JSON object (then blank line). Example shape for cashier + age/gender + mood. Each `detection.items[]` entry matches [`Detection.to_dict()`](services/detector.py): `bbox`, `class_id`, `class_name`, `confidence`, `center`, `width`, `height`.

```json
{
  "camera_id": "main_room",
  "frame_count": 97,
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": {
      "count": 2,
      "items": [
        {
          "bbox": [684, 0, 1169, 275],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.92,
          "center": [926, 137],
          "width": 485,
          "height": 275
        },
        {
          "bbox": [811, 874, 1264, 1080],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.88,
          "center": [1037, 977],
          "width": 453,
          "height": 206
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {"bbox": [684, 0, 1169, 275], "gender": "Female", "age_group": "Senior", "confidence": 0.50},
        {"bbox": [811, 874, 1264, 1080], "gender": "Male", "age_group": "Senior", "confidence": 0.74}
      ],
      "mood": [
        {"bbox": [684, 0, 1169, 275], "mood": "Neutral", "confidence": 0.88},
        {"bbox": [811, 874, 1264, 1080], "mood": "Happy", "confidence": 0.91}
      ],
      "cashier": {
        "persons": [
          {
            "person_bbox": [811, 874, 1264, 1080],
            "confidence": 0.87,
            "zone": "ROI_CASHIER",
            "transaction": true,
            "items": {
              "drawers": [{"bbox": [845, 692, 1186, 922], "confidence": 0.87}],
              "cash": [{"bbox": [858, 727, 934, 832], "confidence": 0.85}]
            }
          },
          {
            "person_bbox": [684, 0, 1169, 275],
            "confidence": 0.89,
            "zone": "ROI_CUSTOMER",
            "transaction": false,
            "items": {"drawers": [], "cash": []}
          }
        ],
        "summary": {
          "cashier_zone": {"persons": 1, "drawers": 1, "cash": 4},
          "customer_zone": {"persons": 1, "drawers": 0, "cash": 0},
          "case_id": "N3",
          "severity": "NORMAL",
          "alerts": ["N3 EVENT: Transaction in progress"],
          "transaction": true,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/normal/N3/cam_xxx.jpg",
          "frame_id": 97,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": [{"bbox": [811, 874, 1264, 1080], "confidence": 0.87, "zone": "ROI_CASHIER"}]
        }
      }
    }
  }
}
```

**Implementation note:** Live events from `/detection/stream` also include a top-level **`frame`** field: a base64-encoded JPEG string. It is omitted above because it is large; strip or ignore it when piping to `jq` for debugging.

---

## Recommended setup order

1. `POST /cameras` — register RTSP sources  
2. `POST /detection/setup` — choose pipeline services (`cashier` must be **last** if you use it)  
3. `POST /cashier/zones` — optional; defaults load from `CASHIER_CONFIG` (often `./config/cashier_zones.yaml`)  
4. `GET /cashier/zones` — optional; inspect merged on-disk config (`zones` as `[[x,y],…]`)  
5. `POST /detection/start` — spawn pipeline process(es)  
6. `GET /detection/stream` or `GET /detection/status` — consume results  
7. After traffic flows: `GET /cashier/status`, `GET /cashier/events`, optional `GET /cashier/stream/{camera_id}` — cashier-specific state and SSE  

---

## Health

### `GET /`

Returns API identity.

```bash
curl -s "http://<jetson-ip>:9000/"
```

**Sample JSON response**

```json
{"service": "Vision Pipeline API", "version": "1.0.0"}
```

**Key fields**

- `service` — Product name string.  
- `version` — API version string.  

---

## Cameras

### `POST /cameras`

Register or update one or more cameras in memory (id → RTSP URL).

```bash
curl -s -X POST "http://<jetson-ip>:9000/cameras" \
  -H "Content-Type: application/json" \
  -d '{
    "cameras": [
      {"id": "cam1", "url": "rtsp://user:pass@192.168.1.10/stream"}
    ]
  }'
```

**Sample JSON request body**

```json
{
  "cameras": [
    {"id": "cam1", "url": "rtsp://user:pass@192.168.1.10/stream"}
  ]
}
```

**Sample JSON response**

```json
{
  "status": "configured",
  "cameras": {
    "cam1": "rtsp://user:pass@192.168.1.10/stream"
  }
}
```

**Key fields**

- `status` — Always `"configured"` on success.  
- `cameras` — Map of camera id to RTSP URL (all cameras currently registered).  

---

### `GET /cameras`

List configured cameras as structured rows.

```bash
curl -s "http://<jetson-ip>:9000/cameras"
```

**Sample JSON response**

```json
{
  "count": 1,
  "cameras": [
    {"id": "cam1", "url": "rtsp://user:pass@192.168.1.10/stream"}
  ]
}
```

**Key fields**

- `count` — Number of configured cameras.  
- `cameras` — Array of `{id, url}` (not the same shape as `POST` response).  

---

### `DELETE /cameras/{cam_id}`

Remove a camera by id.

```bash
curl -s -X DELETE "http://<jetson-ip>:9000/cameras/cam1"
```

**Sample JSON response**

```json
{
  "status": "removed",
  "camera_id": "cam1",
  "remaining": []
}
```

**Key fields**

- `status` — `"removed"` on success.  
- `camera_id` — Id that was deleted.  
- `remaining` — Ids still registered.  

---

## Pipeline

### `POST /detection/setup`

Declare which services run per camera process. Order matters for the pipeline.

```bash
curl -s -X POST "http://<jetson-ip>:9000/detection/setup" \
  -H "Content-Type: application/json" \
  -d '{"pipeline": ["detector", "age_gender", "mood", "cashier"]}'
```

**Sample JSON request body (cashier + age/gender + mood)**

```json
{"pipeline": ["detector", "age_gender", "mood", "cashier"]}
```

**Available service names:** `detector`, `age_gender`, `ppe`, `mood`, `cashier` (see `services/__init__.py`).

**Sample JSON response**

```json
{
  "status": "configured",
  "pipeline": ["detector", "age_gender", "mood", "cashier"]
}
```

**Key fields**

- `status` — `"configured"` on success.  
- `pipeline` — Echo of the active pipeline list.  

---

### `POST /detection/start`

Start the pipeline for one camera or for all configured cameras.

```bash
curl -s -X POST "http://<jetson-ip>:9000/detection/start?camera_id=cam1"
```

Omit `camera_id` to start every configured camera.

**Sample JSON response**

```json
{
  "status": "started",
  "cameras": ["cam1"]
}
```

**Key fields**

- `status` — `"started"` on success.  
- `cameras` — List of camera ids that were started in this call.  

**Typical errors:** `400` if no cameras or setup missing; `404` if `camera_id` unknown; `409` if already running.

---

### `POST /detection/stop`

Stop one camera or all running cameras.

```bash
curl -s -X POST "http://<jetson-ip>:9000/detection/stop?camera_id=cam1"
```

Omit `camera_id` to stop **all** running cameras.

**Sample JSON response**

```json
{
  "status": "stopped",
  "cameras": ["cam1"]
}
```

**Key fields**

- `status` — `"stopped"` on success.  
- `cameras` — List of camera ids stopped in this call.  

---

### `GET /detection/status`

Per-camera runtime stats (from shared process state).

```bash
curl -s "http://<jetson-ip>:9000/detection/status"
```

**Sample JSON response (core fields)**

```json
{
  "cameras": {
    "cam1": {
      "running": true,
      "fps": 28.5,
      "frame_count": 420,
      "uptime_seconds": 14.7
    }
  }
}
```

**Additional fields** each camera may include (see `schemas.CameraStatus`): `camera_id`, `rtsp_url`, `last_detections`, `total_detections`, `error`.

**Key fields**

- `cameras` — Map keyed by camera id.  
- `running` — Whether the worker process is active.  
- `fps` — Smoothed frames per second.  
- `frame_count` — Frames processed.  
- `uptime_seconds` — Seconds since start for that worker.  

---

### `GET /detection/stream`

Server-Sent Events: one JSON payload per line prefixed with `data: `. Use **`curl -N`** (no buffer) so events appear in real time.

```bash
curl -N "http://<jetson-ip>:9000/detection/stream"
```

**Wire format:** repeated blocks of `data: <json>\n\n`. See [SSE event reference](#sse-event-reference-get-detectionstream) above; remember the extra **`frame`** (base64 JPEG) on real events.

**Key fields (top level)**

- `camera_id` — Source camera.  
- `frame_count` — Monotonic frame index for that worker.  
- `timestamp` — ISO timestamp when the result was emitted.  
- `frame` — Base64 JPEG (large); optional to decode or omit in logs.  
- `data.detection` — Person/object counts and `items`.  
- `data.use_case` — Service outputs (`age_gender`, `mood`, `cashier`, …).  

---

## Cashier zones

### `GET /cashier/zones`

Read the current zone geometry and **thresholds** from `CASHIER_CONFIG` (default `./config/cashier_zones.yaml`). Response is the raw file: top-level `zones`, `thresholds`, plus any other keys present in the YAML/JSON.

```bash
curl -s "http://<jetson-ip>:9000/cashier/zones"
```

**Inspect only timing / IOU thresholds**

```bash
curl -s "http://<jetson-ip>:9000/cashier/zones" | jq '.thresholds'
```

**Sample JSON fragment** (abridged from [`config/cashier_zones.yaml`](config/cashier_zones.yaml); a live `GET` may also return `buffer`, `debounce`, `evidence`, `gif`, `meta`, etc.)

```json
{
  "thresholds": {
    "config_reload_interval": 60,
    "customer_wait_max_seconds": 30,
    "drawer_open_max_seconds": 30,
    "proximity_iou": 0.05
  },
  "zones": {
    "ROI_CASHIER": {
      "active": true,
      "shape": "polygon",
      "points": [[0.313, 0.561], [0.693, 0.538], [0.733, 1.001], [0.26, 1.001]]
    },
    "ROI_CUSTOMER": {
      "active": true,
      "shape": "polygon",
      "points": [[0.634, 0.282], [0.63, 0.003], [0.295, 0.0], [0.295, 0.326]]
    }
  }
}
```

**Key fields**

- `zones` — `ROI_CASHIER` / `ROI_CUSTOMER` with normalised `points` as `[[x,y],…]` on disk.  
- `thresholds` — Tuning used by cashier logic (wait/drawer timers, IoU, config reload interval).  

#### Thresholds reference (what to change and why)

| Key | Default | Role |
|-----|---------|------|
| `drawer_open_max_seconds` | `30` | Timer for **A6** (drawer open longer than this while the case logic applies). |
| `customer_wait_max_seconds` | `30` | Timer for **A5** (customer in zone, no cashier, wait longer than this). |
| `proximity_iou` | `0.05` | Minimum **IoU** between boxes to associate a **person** with a **drawer** or **cash** (and drawer→cash proxy). Higher = stricter overlap required. |
| `config_reload_interval` | `60` | Seconds between **disk reloads** of `CASHIER_CONFIG`. A successful `POST /cashier/zones` writes the file immediately; the running process picks it up on the next reload cycle. |

#### A5 / A6: thresholds → logic → output

Both are **time-based** escalations in [`CashierService._evaluate`](services/cashier.py). They tie **config** to **rules** to **stream JSON**.

| | **A5** | **A6** |
|---|--------|--------|
| **Threshold** | `customer_wait_max_seconds` | `drawer_open_max_seconds` |
| **Logic (plain language)** | Customer present in **customer zone**, no cashier in **cashier zone**, and cumulative wait exceeds the limit. | Open drawer in **cashier zone** (with the usual case preconditions) stays open longer than the limit. |
| **Output** | `data.use_case.cashier.summary.case_id` = `"A5"`, `severity` = `ALERT`, `alerts[]` text about customer waiting. | `case_id` = `"A6"`, `severity` = `ALERT`, `alerts[]` text about drawer open too long. |

**Do not swap:** **A5** is **customer-wait** (customer zone timer). **A6** is **drawer-open duration** (cashier zone timer).

**How to change thresholds**

1. **HTTP (no file edit):** `POST /cashier/zones` with a JSON body containing only `"thresholds": { … }` (see [thresholds-only update](#post-cashierzones-thresholds-only-update) below). Values merge into the existing file.  
2. **File edit:** Edit `config/cashier_zones.yaml` (or your `CASHIER_CONFIG` path) and save; wait up to `config_reload_interval` seconds, or restart the worker if you need instant pickup.  
3. **Verify:** `curl -s "http://<jetson-ip>:9000/cashier/zones" | jq '.thresholds'`.

---

#### Drawer metrics: per-frame count vs `drawer_open_count` API

Mnemonic (the route name is easy to misread):

- **`drawer_count` ≠** total frames where the drawer is open (or total drawer detections).
- **`drawer_count` =** number of logged **`triggered`** events in `events.jsonl` for that camera (alert/session starts and N3 transaction logs — see implementation note below).

| What you need | Where to get it |
|----------------|-----------------|
| **Open drawers in this frame** (zone tally) | SSE / stream JSON: `data.use_case.cashier.summary.cashier_zone.drawers` (integer count for the current frame). |
| **Lifetime “triggered” events** (log lines) | `GET /cashier/media/{camera_id}/drawer_count` — see below. |

**Important:** The route name says `drawer_count`, but the implementation counts **lines** in `evidence/.../logs/events.jsonl` where `status == "triggered"` for that `camera_id` (see [`apis/cashier.py`](apis/cashier.py)). A `"triggered"` row is written when an **alert/critical evidence session starts** or when a **transaction (N3)** is logged — it is **not** a raw count of “how many times the drawer was detected open” across all frames. For “how many frames had an open drawer”, aggregate `cashier_zone.drawers > 0` from your stream or batch `stream.jsonl`. For “how many alert/critical sessions were opened”, use this endpoint or parse `events.jsonl`.

```bash
curl -s "http://<jetson-ip>:9000/cashier/media/cashier_cam_01/drawer_count"
```

**Sample JSON**

```json
{"camera_id": "cashier_cam_01", "drawer_open_count": 42}
```

---

### `POST /cashier/zones` (thresholds-only update)

You may send **only** `thresholds` to merge into the existing config (zones unchanged). Omitted keys in other sections are left as-is.

```bash
curl -s -X POST "http://<jetson-ip>:9000/cashier/zones" \
  -H "Content-Type: application/json" \
  -d '{
    "thresholds": {
      "drawer_open_max_seconds": 45,
      "customer_wait_max_seconds": 45,
      "proximity_iou": 0.06,
      "config_reload_interval": 60
    }
  }'
```

**Key fields**

- `thresholds.drawer_open_max_seconds` — Before **A6** escalation.  
- `thresholds.customer_wait_max_seconds` — Before **A5** escalation.  
- `thresholds.proximity_iou` — Person–drawer / person–cash association threshold.  
- `thresholds.config_reload_interval` — Seconds between cashier config reloads from disk.  

---

### `POST /cashier/zones`

Update ROI definitions and optional thresholds. Writes through to `CASHIER_CONFIG` (YAML or JSON). Cashier service reloads config on its reload interval (default 60s via `thresholds.config_reload_interval`).

**Important:** The JSON body must use **`points` as `[{"x":…,"y":…}, …]`**. The `[[x,y],[x,y]]` form is what is **stored** on disk and returned by `GET /cashier/zones`, not what `POST` accepts.

```bash
curl -s -X POST "http://<jetson-ip>:9000/cashier/zones" \
  -H "Content-Type: application/json" \
  -d '{
    "ROI_CASHIER": {
      "shape": "rectangle",
      "points": [{"x": 0.0, "y": 0.0}, {"x": 0.5, "y": 1.0}],
      "active": true
    },
    "ROI_CUSTOMER": {
      "shape": "polygon",
      "points": [
        {"x": 0.5, "y": 0.0},
        {"x": 1.0, "y": 0.0},
        {"x": 1.0, "y": 1.0},
        {"x": 0.5, "y": 1.0}
      ],
      "active": true
    },
    "thresholds": {
      "drawer_open_max_seconds": 30,
      "customer_wait_max_seconds": 30
    }
  }'
```

**Sample JSON request body (API shape — use for curl)**

```json
{
  "ROI_CASHIER": {
    "shape": "rectangle",
    "points": [{"x": 0.0, "y": 0.0}, {"x": 0.5, "y": 1.0}],
    "active": true
  },
  "ROI_CUSTOMER": {
    "shape": "polygon",
    "points": [
      {"x": 0.5, "y": 0.0},
      {"x": 1.0, "y": 0.0},
      {"x": 1.0, "y": 1.0},
      {"x": 0.5, "y": 1.0}
    ],
    "active": true
  }
}
```

**On-disk / `GET /cashier/zones` shape (reference only — do not send this to `POST` without wrapping as x/y objects)**

Rectangle:

```json
"ROI_CASHIER": {
  "shape": "rectangle",
  "points": [[0.0, 0.0], [0.5, 1.0]],
  "active": true
}
```

Polygon:

```json
"ROI_CUSTOMER": {
  "shape": "polygon",
  "points": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
  "active": true
}
```

**Sample JSON response (actual API)**

```json
{
  "status": "updated",
  "config": {
    "zones": {
      "ROI_CASHIER": {
        "shape": "rectangle",
        "points": [[0.0, 0.0], [0.5, 1.0]],
        "active": true
      },
      "ROI_CUSTOMER": {
        "shape": "polygon",
        "points": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
        "active": true
      }
    },
    "thresholds": {
      "drawer_open_max_seconds": 30,
      "customer_wait_max_seconds": 30
    }
  }
}
```

**Note:** Some docs use `{"status":"configured","zones":{…}}`; the running server returns **`status: "updated"`** and a full merged **`config`** (including `zones` and `thresholds`).

**Key fields**

- `status` — `"updated"` after a successful write.  
- `config` — Full merged configuration as persisted.  
- `config.zones` — Normalised `ROI_CASHIER` / `ROI_CUSTOMER` entries.  
- `config.thresholds` — Numeric tuning (wait/drawer timers, IOU, reload interval, etc.).  

---

## jq examples

**Current cashier thresholds from HTTP config**

```bash
curl -s "http://<jetson-ip>:9000/cashier/zones" | jq '.thresholds'
```

### `/detection/stream`

**Case id, severity, alerts from live stream**

```bash
curl -sN "http://<jetson-ip>:9000/detection/stream" \
  | grep "^data:" | sed 's/^data: //' \
  | jq '.data.use_case.cashier.summary | {case_id, severity, alerts}'
```

**Persons with zone and transaction (mood lives in a parallel array, not inside each person)**

```bash
curl -sN "http://<jetson-ip>:9000/detection/stream" \
  | grep "^data:" | sed 's/^data: //' \
  | jq '.data.use_case.cashier.persons[] | {zone, transaction}'
```

**Join cashier persons with mood and age_gender by matching `bbox` / `person_bbox`**

```bash
curl -sN "http://<jetson-ip>:9000/detection/stream" \
  | grep "^data:" | sed 's/^data: //' \
  | jq '
    .data.use_case as $u
    | ($u.cashier.persons // [])[]
    | . as $p
    | ($u.mood       // [] | map(select(.bbox == $p.person_bbox)) | .[0]) as $m
    | ($u.age_gender // [] | map(select(.bbox == $p.person_bbox)) | .[0]) as $a
    | {zone: $p.zone, transaction: $p.transaction, mood: $m, age_gender: $a}
  '
```

---

## Cashier ROI shapes, thresholds, and end of pipeline (curl)

Use these after the API is up (`uvicorn app:app --host 0.0.0.0 --port 9000`). Replace `<jetson-ip>` and `cam1` / `cashier_cam_01` with your camera id.

### Read zones and thresholds (on-disk / merged)

Full file-shaped JSON (includes `zones`, `thresholds`, `buffer`, `debounce`, …):

```bash
curl -s "http://<jetson-ip>:9000/cashier/zones"
```

Zones only (`ROI_CASHIER` / `ROI_CUSTOMER` as stored: `shape`, `points` as `[[x,y],…]`):

```bash
curl -s "http://<jetson-ip>:9000/cashier/zones" | jq '.zones'
```

Timing and IoU used for **A5** / **A6** / person–drawer linking:

```bash
curl -s "http://<jetson-ip>:9000/cashier/zones" | jq '.thresholds'
```

### POST polygons (matches default [`config/cashier_zones.yaml`](config/cashier_zones.yaml))

`POST` requires `points` as `[{"x":…,"y":…}, …]` (not the `[[x,y],…]` form returned by `GET`).

```bash
curl -s -X POST "http://<jetson-ip>:9000/cashier/zones" \
  -H "Content-Type: application/json" \
  -d '{
    "ROI_CASHIER": {
      "shape": "polygon",
      "active": true,
      "points": [
        {"x": 0.313, "y": 0.561},
        {"x": 0.693, "y": 0.538},
        {"x": 0.733, "y": 1.001},
        {"x": 0.260, "y": 1.001}
      ]
    },
    "ROI_CUSTOMER": {
      "shape": "polygon",
      "active": true,
      "points": [
        {"x": 0.634, "y": 0.282},
        {"x": 0.630, "y": 0.003},
        {"x": 0.295, "y": 0.000},
        {"x": 0.295, "y": 0.326}
      ]
    },
    "thresholds": {
      "config_reload_interval": 60,
      "customer_wait_max_seconds": 30,
      "drawer_open_max_seconds": 30,
      "proximity_iou": 0.05
    }
  }'
```

### POST rectangle cashier ROI + polygon customer (alternative)

Two corners define the rectangle (normalised): lower-left and upper-right style pair in any order consistent with your layout.

```bash
curl -s -X POST "http://<jetson-ip>:9000/cashier/zones" \
  -H "Content-Type: application/json" \
  -d '{
    "ROI_CASHIER": {
      "shape": "rectangle",
      "active": true,
      "points": [{"x": 0.26, "y": 0.50}, {"x": 0.74, "y": 1.0}]
    },
    "ROI_CUSTOMER": {
      "shape": "polygon",
      "active": true,
      "points": [
        {"x": 0.634, "y": 0.282},
        {"x": 0.630, "y": 0.003},
        {"x": 0.295, "y": 0.000},
        {"x": 0.295, "y": 0.326}
      ]
    }
  }'
```

### POST thresholds only (zones unchanged)

```bash
curl -s -X POST "http://<jetson-ip>:9000/cashier/zones" \
  -H "Content-Type: application/json" \
  -d '{
    "thresholds": {
      "drawer_open_max_seconds": 45,
      "customer_wait_max_seconds": 45,
      "proximity_iou": 0.06,
      "config_reload_interval": 60
    }
  }'
```

### Reset zones file to code defaults

```bash
curl -s -X POST "http://<jetson-ip>:9000/cashier/zones/reset"
```

### End of pipeline (after `POST /detection/start`)

`cashier` last in the pipeline list:

```bash
curl -s -X POST "http://<jetson-ip>:9000/detection/setup" \
  -H "Content-Type: application/json" \
  -d '{"pipeline": ["detector", "age_gender", "mood", "cashier"]}'
```

Runtime health and frame counters:

```bash
curl -s "http://<jetson-ip>:9000/detection/status"
```

Latest cashier summary per camera (from in-memory state updated by `CashierService`):

```bash
curl -s "http://<jetson-ip>:9000/cashier/status"
```

Recent alerts / transactions (newest first):

```bash
curl -s "http://<jetson-ip>:9000/cashier/events?limit=20"
```

Triggered-event tally from `events.jsonl` (not per-frame drawer detections — see **Drawer metrics** in this document):

```bash
curl -s "http://<jetson-ip>:9000/cashier/media/cashier_cam_01/drawer_count"
```

Multiplexed vision SSE (all cameras):

```bash
curl -N "http://<jetson-ip>:9000/detection/stream"
```

Per-camera cashier SSE (`event:` lines: `connected`, `frame`, `alert`, …):

```bash
curl -N "http://<jetson-ip>:9000/cashier/stream/cashier_cam_01"
```

---

## Cashier case reference (N1–N6, A1–A7)

| case_id | severity | condition |
|--------|----------|-----------|
| N1 | NORMAL | Baseline; no other scenario matched (no alerts). |
| N2 | NORMAL | Cashier zone occupied; drawer closed; no customer in customer zone. |
| N3 | NORMAL | Transaction in progress (cashier + drawer + customer + cash/nearby rules satisfied). |
| N4 | NORMAL | Staff handover / supervisor at register (multiple staff + drawer context). |
| N5 | NORMAL | Customer in customer zone; no open drawer in cashier zone. |
| N6 | NORMAL | Drawer open; no cash visible (card / float check). |
| A1 | ALERT or CRITICAL | Unattended open drawer; **CRITICAL** if customer is present. |
| A2 | ALERT | Unexpected person in cashier zone (fallback after other rules). |
| A3 | CRITICAL | Cash + open drawer; cashier zone unoccupied. |
| A4 | CRITICAL | Open register with cash; person not near drawer (unauthorised pattern). |
| A5 | ALERT | Customer waiting longer than threshold; no cashier present. |
| A6 | ALERT | Drawer open longer than threshold. |
| A7 | ALERT | Cash in customer zone; no cashier; drawer closed. |

---

## Further reading

Additional cashier HTTP routes (status, events, evidence, per-camera SSE under `/cashier/stream/...`) are documented in OpenAPI at `/docs`.

**JPEG vs GIF (which case saves what, and cURL for `/cashier/media/...`):** see **Event-level media (GIF vs JPEG)** in [`sse_cashier.md`](sse_cashier.md).
