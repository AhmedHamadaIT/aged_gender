# Vision Pipeline API

The Vision Pipeline API is a FastAPI-based server for multi-camera computer vision. It registers RTSP sources, composes pipelines from **detector**, **age/gender**, **mood**, **PPE**, and **cashier** services, and exposes results over **Server-Sent Events** (`/detection/stream`) plus dedicated **cashier** REST and SSE routes under `/cashier`.

---

##  Setup & Run

### Prerequisites
- Python 3.9+
- CUDA/cuDNN enabled environment (optional but recommended for GPU acceleration)

### Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the FastAPI server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 9000
   ```

*(Alternatively, you can run it via Docker Compose if standard deployment is configured).*

**Default base URL:** `http://localhost:9000` — use `http://<host>:9000` on a Jetson or remote machine. Interactive OpenAPI: `/docs`, `/redoc`.

---

## Features

- **Multi-camera RTSP** — Register streams with `POST /cameras`; list with `GET /cameras`; remove with `DELETE /cameras/{cam_id}`.
- **Composable pipeline** — `POST /detection/setup` chooses services from: `detector`, `age_gender`, `mood`, `ppe`, `cashier` (see `services/__init__.py`).
- **Runtime control** — `POST /detection/start` and `POST /detection/stop` (optional `camera_id` query); `GET /detection/status` for FPS, frame counts, and errors.
- **Global SSE** — `GET /detection/stream` streams one JSON payload per processed frame from all cameras; each event may include a base64 JPEG in `frame`.
- **Cashier monitor** — Zone geometry and timers via `GET` / `POST /cashier/zones` and `POST /cashier/zones/reset`; scenarios **N1–N6** (normal) and **A1–A7** (alert/critical). `GET /cashier/status` for latest per-camera summary; `GET` / `DELETE /cashier/events` for the in-memory alert/transaction log; `GET /cashier/evidence` and `GET /cashier/evidence/{path}` for saved JPEGs. Per-camera SSE: `GET /cashier/stream/{camera_id}` and `GET /cashier/stream/{camera_id}/only` (alerts-focused). Media helpers under `/cashier/media/...` (latest JPG/GIF, per-event assets, `drawer_count`). Case evaluation order in code: critical **A3/A4** → unattended drawer **A1** → timed **A5** (customer wait) / **A6** (drawer duration) → **A7** → normal **N3/N4/N6** → **N5/N2** → fallback **A2** ([`services/cashier.py`](services/cashier.py) `_evaluate`).
- **Configuration on disk** — Cashier YAML/JSON path from env `CASHIER_CONFIG` (default `./config/cashier_zones.yaml`); evidence directory from `CASHIER_EVIDENCE_DIR` (default `./evidence/cashier`).

---

## Sample configuration file (`config/cashier_zones.yaml`)

The cashier service loads **`CASHIER_CONFIG`** (default `./config/cashier_zones.yaml`). On disk, each zone uses **`points` as `[[x, y], …]`** with normalized coordinates in `[0, 1]`. **`POST /cashier/zones`** must send **`points` as `[{"x":…,"y":…}, …]`**; the API writes the nested-list form back to the file. `GET /cashier/zones` returns the merged file (zones, thresholds, and any extra keys such as `buffer`, `debounce`, `evidence`, `gif`, `meta`).

Abbreviated example (see repository file for full values):

```yaml
thresholds:
  config_reload_interval: 60
  customer_wait_max_seconds: 30
  drawer_open_max_seconds: 30
  proximity_iou: 0.05
zones:
  ROI_CASHIER:
    active: true
    shape: polygon
    points:
      - [0.313, 0.561]
      - [0.693, 0.538]
      - [0.733, 1.001]
      - [0.260, 1.001]
  ROI_CUSTOMER:
    active: true
    shape: polygon
    points:
      - [0.634, 0.282]
      - [0.630, 0.003]
      - [0.295, 0.000]
      - [0.295, 0.326]
buffer:
  jpeg_quality: 75
  size: 100
debounce:
  default: 3
  A3: 1
  A4: 1
evidence:
  save_gif: true
  save_thumbnail: true
  log_rotate_mb: 100
gif:
  fps: 10
  quality: 85
meta:
  version: 1.0.0
```

---

## Complete cURL reference (all HTTP routes)

Set `BASE` to your server root, for example:

```bash
export BASE=http://localhost:9000
# or: export BASE=http://<jetson-ip>:9000
```

### `GET /`

```bash
curl -s "$BASE/"
```

### Cameras — `POST` / `GET` / `DELETE /cameras/{cam_id}`

```bash
curl -s -X POST "$BASE/cameras" \
  -H "Content-Type: application/json" \
  -d '{"cameras":[{"id":"cam1","url":"rtsp://user:pass@192.168.1.10/stream"}]}'

curl -s "$BASE/cameras"

curl -s -X DELETE "$BASE/cameras/cam1"
```

### Detection — `setup`, `start`, `stop`, `status`, `stream`

```bash
curl -s -X POST "$BASE/detection/setup" \
  -H "Content-Type: application/json" \
  -d '{"pipeline":["detector","age_gender","mood","cashier"]}'

curl -s -X POST "$BASE/detection/start?camera_id=cam1"
curl -s -X POST "$BASE/detection/start"
curl -s -X POST "$BASE/detection/stop?camera_id=cam1"
curl -s -X POST "$BASE/detection/stop"

curl -s "$BASE/detection/status"

curl -N "$BASE/detection/stream"
```

### Cashier — zones (`GET` / `POST` / `POST …/reset`)

```bash
curl -s "$BASE/cashier/zones"

curl -s "$BASE/cashier/zones" | jq '.thresholds'

curl -s -X POST "$BASE/cashier/zones" \
  -H "Content-Type: application/json" \
  -d '{
    "thresholds": {
      "drawer_open_max_seconds": 45,
      "customer_wait_max_seconds": 45,
      "proximity_iou": 0.06,
      "config_reload_interval": 60
    }
  }'

curl -s -X POST "$BASE/cashier/zones" \
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

curl -s -X POST "$BASE/cashier/zones/reset"
```

Successful `POST /cashier/zones` returns `{"status":"updated","config":{...}}` with the full merged config (including `zones` as `[[x,y],…]`).

### Cashier — `status`, `events`, `evidence`

```bash
curl -s "$BASE/cashier/status"

curl -s "$BASE/cashier/events?severity=ALERT&case_id=A5&camera_id=cam1&limit=50&offset=0"

curl -s -X DELETE "$BASE/cashier/events"

curl -s "$BASE/cashier/evidence?severity=alert&case_id=A5&limit=20"

curl -s -o evidence.jpg "$BASE/cashier/evidence/<relative/path/from/list>"
```

### Cashier — per-camera SSE and media

```bash
curl -N "$BASE/cashier/stream/cam1"
curl -N "$BASE/cashier/stream/cam1/only"

curl -s -o latest.jpg "$BASE/cashier/media/cam1/latest/jpg"
curl -s -o latest.gif "$BASE/cashier/media/cam1/latest/gif"
curl -s -o event.jpg "$BASE/cashier/media/cam1/event/<event_id>/jpg"
curl -s -o event.gif "$BASE/cashier/media/cam1/event/<event_id>/gif"

# Logged "triggered" event count — NOT total frames with drawer open (see curl_cashier.md)
curl -s "$BASE/cashier/media/cam1/drawer_count"
```

### Cashier — thresholds & timers (A5 / A6)

- **`customer_wait_max_seconds`** → **A5** (customer in customer zone, no cashier, wait exceeds limit).  
- **`drawer_open_max_seconds`** → **A6** (drawer open duration exceeds limit).  

Full **thresholds → logic → output** table: [`curl_cashier.md`](curl_cashier.md) (*A5 / A6: thresholds → logic → output*).

```bash
curl -s "$BASE/cashier/zones" | jq '.thresholds'
curl -s "$BASE/cashier/zones" | jq '.zones'
```

### After `POST /detection/start` (runtime checks)

```bash
curl -s "$BASE/detection/status"
curl -s "$BASE/cashier/status"
curl -s "$BASE/cashier/events?limit=20"
```

### jq examples (detection SSE)

```bash
curl -sN "$BASE/detection/stream" \
  | grep "^data:" | sed 's/^data: //' \
  | jq '.data.use_case.cashier.summary | {case_id, severity, alerts}'

curl -sN "$BASE/detection/stream" \
  | grep "^data:" | sed 's/^data: //' \
  | jq '.data.use_case.cashier.persons[] | {zone, transaction}'

curl -s "$BASE/cashier/zones" | jq '.thresholds'
```

### SSH — tail batch JSONL, evidence log, or app log

Replace `<user>`, `<jetson-ip>`, and the repo path on the device.

**Batch JSONL** (same logical shape as live SSE bodies, without the `data: ` prefix):

```bash
# Full-dataset run (includes summary.json in repo when present)
ssh <user>@<jetson-ip> "tail -f /path/to/ml-server/outputs/cashier_test/20260329T060741_7464/stream.jsonl"

# Local cashier-YOLO batch (multiple_persons frames)
ssh <user>@<jetson-ip> "tail -f /path/to/ml-server/outputs/cashier_test/20260329T135320_10106/stream.jsonl"
```

**Cashier evidence log** (`triggered` / `resolved` lines, GIF paths after compile):

```bash
ssh <user>@<jetson-ip> "tail -f /path/to/ml-server/evidence/cashier/logs/events.jsonl"
```

**Application log** (if your deployment writes here):

```bash
ssh <user>@<jetson-ip> "tail -f /path/to/ml-server/logger/app.log"
```

More detail: [`curl_cashier.md`](curl_cashier.md), SSE examples: [`sse_cashier.md`](sse_cashier.md).

---

##  API Endpoints Reference

The application lifecycle works as follows:
1. Register Cameras.
2. Setup the Pipeline Services.
3. Start the Pipeline processing.
4. Consume the Detection Stream (SSE).

---

### **Health Check**

#### `GET /`
Returns basic service information.

**Sample Response**
```json
{
  "service": "Vision Pipeline API",
  "version": "1.0.0"
}
```

---

### **Camera Management Routes**

#### `POST /cameras`
Configure one or multiple cameras.

**Sample Request**
```json
{
  "cameras": [
    {
      "id": "cam1",
      "url": "rtsp://username:password@10.0.0.5:554/stream1"
    }
  ]
}
```

**Sample Response**
```json
{
  "status": "configured",
  "cameras": {
    "cam1": "rtsp://username:password@10.0.0.5:554/stream1"
  }
}
```

#### `GET /cameras`
List all currently configured cameras.

**Sample Response**
```json
{
  "count": 1,
  "cameras": [
    {
      "id": "cam1",
      "url": "rtsp://username:password@10.0.0.5:554/stream1"
    }
  ]
}
```

#### `DELETE /cameras/{cam_id}`
Delete a configured camera.

**Sample Response**
```json
{
  "status": "removed",
  "camera_id": "cam1",
  "remaining": []
}
```

---

### **Detection Pipeline Routes**

#### `POST /detection/setup`
Configure which models/services will run in the pipeline.

**Sample Request**
```json
{
  "pipeline": ["detector", "age_gender", "mood","ppe"]
}
```

**Sample Response**
```json
{
  "status": "configured",
  "pipeline": ["detector", "age_gender", "mood","ppe"]
}
```

#### `POST /detection/start`
Start processing cameras using the configured pipeline. You can optionally specify a `camera_id` as a query parameter string. If not specified, it starts all configured cameras.

**Sample Request**
`POST /detection/start?camera_id=cam1`

**Sample Response**
```json
{
  "status": "started",
  "cameras": ["cam1"]
}
```

#### `POST /detection/stop`
Stop camera streams. You can optionally specify a `camera_id` as a query parameter string. If not specified, it stops all running cameras.

**Sample Request**
`POST /detection/stop?camera_id=cam1`

**Sample Response**
```json
{
  "status": "stopped",
  "cameras": ["cam1"]
}
```

#### `GET /detection/status`
Returns the operational status, FPS, and detection counts for all cameras.

**Sample Response**
```json
{
  "cameras": {
    "cam1": {
      "camera_id": "cam1",
      "rtsp_url": "rtsp://10.0.0.5:554/stream1",
      "running": true,
      "frame_count": 420,
      "fps": 28.5,
      "uptime_seconds": 14.7,
      "error": null
    }
  }
}
```

#### `GET /detection/stream`
An SSE (Server-Sent Events) endpoint that yields one JSON payload per frame, combining inferences from all running cameras.

**Sample Response stream**
```json
data: {
  "camera_id": "main_room",
  "frame_count": 5,
  "timestamp": "2026-03-17T12:50:23+00:00",
  "data": {
    "detection": {
      "count": 1,
      "items": [{
        "bbox": [120, 80, 340, 420],
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.91,
        "center": [230, 250],
        "width": 220,
        "height": 340
      }]
    },
    "use_case": {
      "age_gender": [{
        "bbox": [120, 80, 340, 420],
        "gender": "Female",
        "age_group": "MiddleAged",
        "confidence": 0.87
      }],
      "mood": [{
        "bbox": [120, 80, 340, 420],
        "mood": "Happy",
        "confidence": 0.94
      }],
      "ppe": [
      {
        "person_bbox": [
          677,
          305,
          850,
          523
        ],
        "count": 2,
        "items": [
          {
            "class_id": 2,
            "class_name": "gloves",
            "confidence": 0.9647,
            "x1": 742,
            "y1": 480,
            "x2": 805,
            "y2": 527
          },
          {
            "class_id": 1,
            "class_name": "hairnet",
            "confidence": 0.8999,
            "x1": 769,
            "y1": 312,
            "x2": 863,
            "y2": 374
          }
        ]
    }]
      }
    }
  }
```

---

## ✅ Testing & Real Result Examples

### Test Results Overview

#### Test 1: Sample Image (zidane.jpg)
- **Detections**: 2 Persons
- **Inference Time**: ~2 seconds
- **Confidence Scores**: 81.9% - 83.6%

#### Test 2: F1 Image (f1.webp) - Complete Test Results

**Input Image**: `f1.webp` (WebP format, 33 KB)
- **Local annotated version (provided)**: `/home/a7med/Downloads/f1a.png`

**Annotated Output Image with All Detections:**

![F1 Annotated Image with Detections - 500x281 pixels](outputs/f1_test/annotated.jpg)

**Visual Elements in Image:**
- Green bounding boxes around detected persons
- Blue bounding box around detected chair
- Labels showing: Class, Confidence %, Gender | Age Group, and Mood/Emotion
- All coordinates and detection metadata embedded

**Detection Summary:**
```
Total Objects Detected: 4

Person 1 (Top-Left Area)
├─ Bounding Box: [289, 71, 458, 257]
├─ Confidence: 89.01%
├─ Gender: Male
├─ Age Group: MiddleAged
├─ Mood: Angry
└─ Mood Confidence: 51.92%

Person 2 (Left Side)
├─ Bounding Box: [125, 70, 268, 220]
├─ Confidence: 84.15%
├─ Gender: Female
├─ Age Group: Senior
├─ Mood: Angry
└─ Mood Confidence: 45.12%

Person 3 (Center)
├─ Bounding Box: [209, 73, 301, 217]
├─ Confidence: 83.13%
├─ Gender: Male
├─ Age Group: Senior
├─ Mood: Happy
└─ Mood Confidence: 47.54%

Chair (Right Side)
├─ Bounding Box: [353, 136, 458, 259]
├─ Confidence: 52.47%
└─ Class ID: 56
```

**Performance Metrics:**

| Metric | Value |
|--------|-------|
| Total Processing Time | ~3 seconds |
| Image Resolution | 500×281 pixels |
| Number of Detections | 4 |
| Average Confidence | 82.9% |
| Age/Gender Confidence | 56-75% |
| Mood Detection Confidence | 45-52% |
| Annotated Image Size | 85 KB |
| Output JSON Size | 64 KB |

---

### Complete JSON Response Example (F1 Test)

**Raw API Response Structure:**
```json
{
  "camera_id": "f1_image",
  "frame_count": 1,
  "timestamp": "2026-03-24T09:29:31.395013Z",
  "frame": "<base64_encoded_jpeg_string_containing_annotated_image>",
  "data": {
    "detection": {
      "count": 4,
      "items": [
        {
          "bbox": [289, 71, 458, 257],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.8901,
          "center": [373, 164],
          "width": 169,
          "height": 186
        },
        {
          "bbox": [125, 70, 268, 220],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.8415,
          "center": [196, 145],
          "width": 143,
          "height": 150
        },
        {
          "bbox": [209, 73, 301, 217],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.8313,
          "center": [255, 145],
          "width": 92,
          "height": 144
        },
        {
          "bbox": [353, 136, 458, 259],
          "class_id": 56,
          "class_name": "chair",
          "confidence": 0.5247,
          "center": [405, 197],
          "width": 105,
          "height": 123
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {
          "bbox": [289, 71, 458, 257],
          "gender": "Male",
          "age_group": "MiddleAged",
          "confidence": 0.5693
        },
        {
          "bbox": [125, 70, 268, 220],
          "gender": "Female",
          "age_group": "Senior",
          "confidence": 0.5635
        },
        {
          "bbox": [209, 73, 301, 217],
          "gender": "Male",
          "age_group": "Senior",
          "confidence": 0.7525
        },
        {
          "bbox": [353, 136, 458, 259],
          "gender": "Male",
          "age_group": "Senior",
          "confidence": 0.5758
        }
      ],
      "mood": [
        {
          "bbox": [289, 71, 458, 257],
          "mood": "Angry",
          "confidence": 0.5192
        },
        {
          "bbox": [125, 70, 268, 220],
          "mood": "Angry",
          "confidence": 0.4512
        },
        {
          "bbox": [209, 73, 301, 217],
          "mood": "Happy",
          "confidence": 0.4754
        },
        {
          "bbox": [353, 136, 458, 259],
          "mood": "Angry",
          "confidence": 0.4556
        }
      ]
    }
  }
}
```

**JSON Field Reference:**
- **camera_id**: Identifier of the source camera or image
- **frame_count**: Frame number in stream (1 for single image)
- **timestamp**: ISO 8601 timestamp of processing
- **frame**: Base64-encoded JPEG containing annotated image (with all boxes, labels, confidence scores)
- **detection.count**: Total number of detected objects
- **bbox**: `[x1, y1, x2, y2]` - top-left (x1,y1) and bottom-right (x2,y2) pixel coordinates
- **center**: `[cx, cy]` - center point of bounding box
- **class_id**: COCO dataset class ID (0 = person, 56 = chair, etc.)
- **confidence**: Detection confidence score (0.0-1.0)
- **gender**: Predicted gender ("Male" or "Female")
- **age_group**: Predicted age category ("Young", "MiddleAged", "Senior", "Elderly")
- **mood**: Detected emotion ("Angry", "Happy", "Neutral")

---

HTTP examples for every route are in **[Complete cURL reference (all HTTP routes)](#complete-curl-reference-all-http-routes)** above. This repository’s running app exposes **no** `/process`, `/stream` (POST), `/mood`, `/age-gender`, or `/health` routes — use RTSP cameras plus `/detection/*` and `/cashier/*` as documented.

---

## Cashier full-dataset validation report

**Run ID:** `20260329T060741_7464`  
**Authoritative metrics:** [`outputs/cashier_test/20260329T060741_7464/summary.json`](outputs/cashier_test/20260329T060741_7464/summary.json) (aggregated from [`stream.jsonl`](outputs/cashier_test/20260329T060741_7464/stream.jsonl)).

### Executive summary

| Metric | Value |
|--------|------:|
| Total frames processed | 22,537 |
| Wall-clock span (first → last record timestamp) | 4,798.03 s (**1h 19m 58s**) |
| Average throughput | **4.6971 FPS** (frames ÷ duration) |
| **Total times drawer open (frame count)** | **0** |

The **drawer-open frame count** is the number of frames whose `case_id` is one of **N3, N6, A1, A3, A4, A6** (cases where the cashier-zone drawer is treated as open in the business logic). For this dataset run, no frames fell into those classes, so the aggregate is zero.

### Severity breakdown (observed)

| Severity | Frames | Share of total |
|----------|-------:|---------------:|
| NORMAL | 20,626 | 91.52% |
| ALERT | 1,911 | 8.48% |
| CRITICAL | 0 | 0.00% |

### Case distribution (N1–N6, A1–A7)

Counts are **per-frame classifications** over the full run. Cases not observed are listed as **0**.

| case_id | Severity (typical) | Frames |
|---------|-------------------|-------:|
| N1 | NORMAL | 18,058 |
| N2 | NORMAL | 2,568 |
| N3 | NORMAL | 0 |
| N4 | NORMAL | 0 |
| N5 | NORMAL | 0 |
| N6 | NORMAL | 0 |
| A1 | ALERT / CRITICAL | 0 |
| A2 | ALERT | 68 |
| A3 | CRITICAL | 0 |
| A4 | CRITICAL | 0 |
| A5 | ALERT | 1,843 |
| A6 | ALERT | 0 |
| A7 | ALERT | 0 |

### Visual evidence

**Animated summary (GIF)** — when generated by your batch pipeline, place it next to the run artifacts:

![Cashier test run animation](outputs/cashier_test/20260329T060741_7464/cashier_test.gif)

*If the file is not present in the repository yet, add `cashier_test.gif` under the run directory after export from annotated frames or your tooling.*

**Representative stills from this run** (annotated / evidence paths under `outputs/cashier_test/20260329T060741_7464/`):

| Case | Role | Example path |
|------|------|----------------|
| N1 | Idle | [`evidence/normal/N1/frame_000000_time_00m00s_1of3.jpg`](outputs/cashier_test/20260329T060741_7464/evidence/normal/N1/frame_000000_time_00m00s_1of3.jpg) |
| N2 | Cashier only | [`evidence/normal/N2/frame_000465_time_02m35s_1of3.jpg`](outputs/cashier_test/20260329T060741_7464/evidence/normal/N2/frame_000465_time_02m35s_1of3.jpg) |
| A5 | Customer waiting | [`evidence/alert/A5/frame_000386_time_02m08s_3of3.jpg`](outputs/cashier_test/20260329T060741_7464/evidence/alert/A5/frame_000386_time_02m08s_3of3.jpg) |
| A2 | Unexpected persons in cashier zone | [`evidence/alert/A2/frame_000404_time_02m14s_3of3.jpg`](outputs/cashier_test/20260329T060741_7464/evidence/alert/A2/frame_000404_time_02m14s_3of3.jpg) |

**Critical cases (A3, A4, …)** — no CRITICAL frames in this run; when they occur, evidence follows the same layout, e.g. `evidence/critical/A3/…`, `evidence/critical/A4/…`.

![N1 idle](outputs/cashier_test/20260329T060741_7464/evidence/normal/N1/frame_000000_time_00m00s_1of3.jpg)

![N2 cashier only](outputs/cashier_test/20260329T060741_7464/evidence/normal/N2/frame_000465_time_02m35s_1of3.jpg)

![A5 customer waiting](outputs/cashier_test/20260329T060741_7464/evidence/alert/A5/frame_000386_time_02m08s_3of3.jpg)

![A2 unexpected in cashier zone](outputs/cashier_test/20260329T060741_7464/evidence/alert/A2/frame_000404_time_02m14s_3of3.jpg)

### Artifacts for this run

- [`stream.jsonl`](outputs/cashier_test/20260329T060741_7464/stream.jsonl) — one JSON record per frame  
- [`events/`](outputs/cashier_test/20260329T060741_7464/events/) — per-frame JSON mirrors  
- [`annotated/`](outputs/cashier_test/20260329T060741_7464/annotated/) — annotated JPEGs  
- [`evidence/`](outputs/cashier_test/20260329T060741_7464/evidence/) — normal / alert / critical evidence by case  
- [`summary.json`](outputs/cashier_test/20260329T060741_7464/summary.json) — aggregates documented above  

---

## Complete Cashier Pipeline Lifecycle

Recommended order:

1. **`POST /cameras`** — register RTSP sources.  
2. **`POST /detection/setup`** — e.g. `["detector","age_gender","mood","cashier"]`.  
3. **`POST /cashier/zones`** (optional) — zones/thresholds; defaults come from `CASHIER_CONFIG`.  
4. **`GET /cashier/zones`** (optional) — verify merged YAML on disk.  
5. **`POST /detection/start`** — start workers (`?camera_id=…` or all cameras).  
6. **`GET /detection/stream`** or **`GET /cashier/status`** — consume results.  

Copy-paste **`curl`** for each step: [Complete cURL reference (all HTTP routes)](#complete-curl-reference-all-http-routes). SSH tail examples are in that section.

---

## SSE / SSH Stream Logs (Cashier + Mood + Age/Gender)

### Stream event sources

**Full dataset batch (2026-03-29)** — Line-delimited JSON is archived as [`outputs/cashier_test/20260329T060741_7464/stream.jsonl`](outputs/cashier_test/20260329T060741_7464/stream.jsonl). **Additional batch:** [`outputs/cashier_test/20260329T135320_10106/stream.jsonl`](outputs/cashier_test/20260329T135320_10106/stream.jsonl) (local cashier-YOLO run). Inspect on the Jetson:

```bash
ssh <user>@<jetson-ip> "tail -f /path/to/ml-server/outputs/cashier_test/20260329T060741_7464/stream.jsonl"
ssh <user>@<jetson-ip> "tail -f /path/to/ml-server/outputs/cashier_test/20260329T135320_10106/stream.jsonl"
```

**Earlier simulation (reference still)** — Annotated frame from the production simulation run:

- Image: `outputs/cashier_production_sim/20260326T050929Z/annotated/frame_096720.jpg`

![Annotated Frame Used For Stream Example](outputs/cashier_production_sim/20260326T050929Z/annotated/frame_096720.jpg)

### SSE event shape with `cashier` in `use_case`

The `cashier` block inside `use_case` follows this structure:

```json
{
  "input": {
    "path": "/home/a7med/Documents/all_original_frames/Cashier Drawer/others/frame_097440.jpg",
    "name": "frame_097440.jpg"
  },
  "outputs": {
    "annotated_image": "outputs/cashier_aged_gender_single/20260326T064528Z/annotated/frame_097440.jpg"
  },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": {
      "count": 2,
      "items": [
        {
          "bbox": [
            684,
            0,
            1169,
            275
          ],
          "class_id": 0,
          "class_name": "Person",
          "confidence": 0.8949,
          "center": [
            926,
            137
          ],
          "width": 485,
          "height": 275
        },
        {
          "bbox": [
            811,
            874,
            1264,
            1080
          ],
          "class_id": 0,
          "class_name": "Person",
          "confidence": 0.8725,
          "center": [
            1037,
            977
          ],
          "width": 453,
          "height": 206
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {
          "bbox": [
            684,
            0,
            1169,
            275
          ],
          "gender": "Female",
          "age_group": "Senior",
          "confidence": 0.5035
        },
        {
          "bbox": [
            811,
            874,
            1264,
            1080
          ],
          "gender": "Male",
          "age_group": "Senior",
          "confidence": 0.742
        }
      ],
      "cashier": {
        "persons": [
          {
            "person_bbox": [
              811,
              874,
              1264,
              1080
            ],
            "confidence": 0.8725,
            "zone": "ROI_CASHIER",
            "transaction": true,
            "items": {
              "drawers": [
                {
                  "bbox": [
                    845,
                    692,
                    1186,
                    922
                  ],
                  "confidence": 0.87
                }
              ],
              "cash": [
                {
                  "bbox": [
                    858,
                    727,
                    934,
                    832
                  ],
                  "confidence": 0.85
                },
                {
                  "bbox": [
                    1038,
                    697,
                    1107,
                    804
                  ],
                  "confidence": 0.85
                },
                {
                  "bbox": [
                    973,
                    705,
                    1045,
                    811
                  ],
                  "confidence": 0.85
                },
                {
                  "bbox": [
                    1100,
                    690,
                    1168,
                    800
                  ],
                  "confidence": 0.85
                }
              ]
            }
          },
          {
            "person_bbox": [
              684,
              0,
              1169,
              275
            ],
            "confidence": 0.8949,
            "zone": "ROI_CUSTOMER",
            "transaction": false,
            "items": {
              "drawers": [],
              "cash": []
            }
          }
        ],
        "summary": {
          "cashier_zone": {
            "persons": 1,
            "drawers": 1,
            "cash": 4
          },
          "customer_zone": {
            "persons": 1,
            "drawers": 0,
            "cash": 0
          },
          "case_id": "N3",
          "severity": "NORMAL",
          "alerts": [
            "N3 EVENT: Transaction in progress"
          ],
          "transaction": true,
          "frame_saved": true,
          "evidence_path": "/home/a7med/ml-server/outputs/cashier_aged_gender_single/20260326T064528Z/evidence/normal/N3/cam_20260326T064542_598744.jpg",
          "frame_id": 1,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": [
            {
              "bbox": [
                811,
                874,
                1264,
                1080
              ],
              "confidence": 0.8725,
              "zone": "ROI_CASHIER"
            }
          ]
        }
      }
    }
  }
}
```

**Cashier Field Reference:**
- **persons**: Array of all detected persons with their zone assignment and associated items
  - **person_bbox**: `[x1, y1, x2, y2]` bounding box of the person
  - **zone**: Either `ROI_CASHIER` (behind the register) or `ROI_CUSTOMER` (customer-facing side)
  - **transaction**: `true` if this person is involved in an active transaction
  - **items.drawers**: List of open drawer detections associated with this person
  - **items.cash**: List of cash detections associated with this person
- **summary**: Aggregated zone counts and case classification
  - **cashier_zone / customer_zone**: Count of `persons`, `drawers`, and `cash` per zone
  - **case_id**: Classified scenario (see case table below)
  - **severity**: `NORMAL`, `ALERT`, or `CRITICAL`
  - **alerts**: List of human-readable alert messages for this frame
  - **transaction**: `true` if an active transaction is in progress
  - **frame_saved**: `true` if an evidence frame was written to disk
  - **evidence_path**: Absolute path to the saved evidence image (null if not saved)
  - **cashier_persons**: Subset list of persons confirmed to be in the cashier zone

---

### Cashier Case IDs (All Cases)

#### Normal Cases
| Case | Severity | Condition |
|------|----------|-----------|
| **N1** | NORMAL | Idle register — nothing happening |
| **N2** | NORMAL | Cashier on duty, no customer |
| **N3** | NORMAL | Active transaction (cashier + drawer + customer + cash, nearby) |
| **N4** | NORMAL | Staff handover / supervisor at register (2 persons near drawer) |
| **N5** | NORMAL | Customer waiting (no transaction started) |
| **N6** | NORMAL | Drawer open, no cash (card transaction / float check) |

#### Alert Cases
| Case | Severity | Condition |
|------|----------|-----------|
| **A1** | ALERT / CRITICAL | Unattended open drawer (CRITICAL if customer present) |
| **A2** | ALERT | Unexpected person in cashier zone |
| **A3** | CRITICAL | Cash + open drawer, no cashier (theft signature) |
| **A4** | CRITICAL | Unauthorised person at open register with cash |
| **A5** | ALERT | Customer waiting too long (>30s) without cashier |
| **A6** | ALERT | Drawer open too long (>30s) |
| **A7** | ALERT | Cash in customer zone, no cashier present |

---

### Target Format Reference: Cashier Test Cases (Normal & Alert)

All cashier test cases (both normal operations and anomalies) yield an annotated image mapped to a detailed `result.json` log. The structure below demonstrates exactly how the response block changes for every single case:

#### N1 (Idle register — nothing happening)
```json
{
  "input": { "path": "...", "name": "frame_N1.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_N1.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 0, "items": [] },
    "use_case": {
      "cashier": {
        "persons": [],
        "summary": {
          "cashier_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "customer_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "case_id": "N1",
          "severity": "NORMAL",
          "alerts": [],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/normal/N1/cam_N1.jpg",
          "frame_id": 1,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": []
        }
      }
    }
  }
}
```

#### N2 (Cashier on duty, no customer)
```json
{
  "input": { "path": "...", "name": "frame_N2.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_N2.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 1, "items": [{ "class_name": "Person" }] },
    "use_case": {
      "cashier": {
        "persons": [{ "zone": "ROI_CASHIER", "transaction": false, "items": { "drawers": [], "cash": [] } }],
        "summary": {
          "cashier_zone": { "persons": 1, "drawers": 0, "cash": 0 },
          "customer_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "case_id": "N2",
          "severity": "NORMAL",
          "alerts": [],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/normal/N2/cam_N2.jpg",
          "frame_id": 2,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": [{ "zone": "ROI_CASHIER" }]
        }
      }
    }
  }
}
```

#### N3 (Active transaction — e.g. frame_097440.jpg)
```json
{
  "input": { "path": "...", "name": "frame_097440.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_097440.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 6, "items": [{ "class_name": "Person" }, { "class_name": "Person" }, "..."] },
    "use_case": {
      "cashier": {
        "persons": [
          { "zone": "ROI_CASHIER", "transaction": true, "items": { "drawers": [{...}], "cash": [{...}] } },
          { "zone": "ROI_CUSTOMER", "transaction": false, "items": { "drawers": [], "cash": [] } }
        ],
        "summary": {
          "cashier_zone": { "persons": 1, "drawers": 1, "cash": 4 },
          "customer_zone": { "persons": 1, "drawers": 0, "cash": 0 },
          "case_id": "N3",
          "severity": "NORMAL",
          "alerts": ["N3 EVENT: Transaction in progress"],
          "transaction": true,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/normal/N3/cam_N3.jpg",
          "frame_id": 3,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": [{ "zone": "ROI_CASHIER" }]
        }
      }
    }
  }
}
```

#### N4 (Staff handover / supervisor at register)
```json
{
  "input": { "path": "...", "name": "frame_N4.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_N4.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 2, "items": [{ "class_name": "Person" }, { "class_name": "Person" }] },
    "use_case": {
      "cashier": {
        "persons": [
          { "zone": "ROI_CASHIER", "transaction": false, "items": { "drawers": [], "cash": [] } },
          { "zone": "ROI_CASHIER", "transaction": false, "items": { "drawers": [], "cash": [] } }
        ],
        "summary": {
          "cashier_zone": { "persons": 2, "drawers": 0, "cash": 0 },
          "customer_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "case_id": "N4",
          "severity": "NORMAL",
          "alerts": ["N4 EVENT: Multiple staff at register"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/normal/N4/cam_N4.jpg",
          "frame_id": 4,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": [{ "zone": "ROI_CASHIER" }, { "zone": "ROI_CASHIER" }]
        }
      }
    }
  }
}
```

#### N5 (Customer waiting without transaction)
```json
{
  "input": { "path": "...", "name": "frame_N5.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_N5.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 1, "items": [{ "class_name": "Person" }] },
    "use_case": {
      "cashier": {
        "persons": [{ "zone": "ROI_CUSTOMER", "transaction": false, "items": { "drawers": [], "cash": [] } }],
        "summary": {
          "cashier_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "customer_zone": { "persons": 1, "drawers": 0, "cash": 0 },
          "case_id": "N5",
          "severity": "NORMAL",
          "alerts": ["N5 EVENT: Customer waiting"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/normal/N5/cam_N5.jpg",
          "frame_id": 5,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": []
        }
      }
    }
  }
}
```

#### N6 (Drawer open, no cash — card transaction)
```json
{
  "input": { "path": "...", "name": "frame_N6.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_N6.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 2, "items": [{ "class_name": "Person" }, { "class_name": "drawer_open" }] },
    "use_case": {
      "cashier": {
        "persons": [{ "zone": "ROI_CASHIER", "transaction": true, "items": { "drawers": [{...}], "cash": [] } }],
        "summary": {
          "cashier_zone": { "persons": 1, "drawers": 1, "cash": 0 },
          "customer_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "case_id": "N6",
          "severity": "NORMAL",
          "alerts": [],
          "transaction": true,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/normal/N6/cam_N6.jpg",
          "frame_id": 6,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": [{ "zone": "ROI_CASHIER" }]
        }
      }
    }
  }
}
```

#### A1 (Unattended open drawer)
```json
{
  "input": { "path": "...", "name": "frame_A1.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_A1.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 1, "items": [{ "class_name": "drawer_open" }] },
    "use_case": {
      "cashier": {
        "persons": [],
        "summary": {
          "cashier_zone": { "persons": 0, "drawers": 1, "cash": 0 },
          "customer_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "case_id": "A1",
          "severity": "ALERT",
          "alerts": ["A1 WARNING: Unattended open drawer"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/alert/A1/cam_A1.jpg",
          "frame_id": 7,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": []
        }
      }
    }
  }
}
```

#### A2 (Unexpected person in cashier zone)
```json
{
  "input": { "path": "...", "name": "frame_A2.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_A2.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 2, "items": [{ "class_name": "Person" }, { "class_name": "Person" }] },
    "use_case": {
      "cashier": {
        "persons": [
          { "zone": "ROI_CASHIER", "transaction": false, "items": { "drawers": [], "cash": [] } },
          { "zone": "ROI_CASHIER", "transaction": false, "items": { "drawers": [], "cash": [] } }
        ],
        "summary": {
          "cashier_zone": { "persons": 2, "drawers": 0, "cash": 0 },
          "customer_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "case_id": "A2",
          "severity": "ALERT",
          "alerts": ["A2 WARNING: Unexpected person in cashier zone"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/alert/A2/cam_A2.jpg",
          "frame_id": 8,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": [{ "zone": "ROI_CASHIER" }, { "zone": "ROI_CASHIER" }]
        }
      }
    }
  }
}
```

#### A3 (Cash + open drawer, no cashier - theft signature)
```json
{
  "input": { "path": "...", "name": "frame_A3.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_A3.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 2, "items": [{ "class_name": "drawer_open" }, { "class_name": "cash" }] },
    "use_case": {
      "cashier": {
        "persons": [],
        "summary": {
          "cashier_zone": { "persons": 0, "drawers": 1, "cash": 1 },
          "customer_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "case_id": "A3",
          "severity": "CRITICAL",
          "alerts": ["A3 CRITICAL: Cash + open drawer, no cashier"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/alert/A3/cam_A3.jpg",
          "frame_id": 9,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": []
        }
      }
    }
  }
}
```

#### A4 (Unauthorised person at open register with cash)
```json
{
  "input": { "path": "...", "name": "frame_A4.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_A4.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 3, "items": [{ "class_name": "Person" }, { "class_name": "drawer_open" }, { "class_name": "cash" }] },
    "use_case": {
      "cashier": {
        "persons": [{ "zone": "ROI_CASHIER", "transaction": true, "items": { "drawers": [{...}], "cash": [{...}] } }],
        "summary": {
          "cashier_zone": { "persons": 1, "drawers": 1, "cash": 1 },
          "customer_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "case_id": "A4",
          "severity": "CRITICAL",
          "alerts": ["A4 CRITICAL: Unauthorised person at open register with cash"],
          "transaction": true,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/alert/A4/cam_A4.jpg",
          "frame_id": 10,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": [{ "zone": "ROI_CASHIER" }]
        }
      }
    }
  }
}
```

#### A5 (Customer waiting too long >30s)
```json
{
  "input": { "path": "...", "name": "frame_A5.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_A5.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 1, "items": [{ "class_name": "Person" }] },
    "use_case": {
      "cashier": {
        "persons": [{ "zone": "ROI_CUSTOMER", "transaction": false, "items": { "drawers": [], "cash": [] } }],
        "summary": {
          "cashier_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "customer_zone": { "persons": 1, "drawers": 0, "cash": 0 },
          "case_id": "A5",
          "severity": "ALERT",
          "alerts": ["A5 WARNING: Customer waiting too long"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/alert/A5/cam_A5.jpg",
          "frame_id": 11,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": []
        }
      }
    }
  }
}
```

#### A6 (Drawer open too long >30s)
```json
{
  "input": { "path": "...", "name": "frame_A6.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_A6.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 2, "items": [{ "class_name": "Person" }, { "class_name": "drawer_open" }] },
    "use_case": {
      "cashier": {
        "persons": [{ "zone": "ROI_CASHIER", "transaction": false, "items": { "drawers": [{...}], "cash": [] } }],
        "summary": {
          "cashier_zone": { "persons": 1, "drawers": 1, "cash": 0 },
          "customer_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "case_id": "A6",
          "severity": "ALERT",
          "alerts": ["A6 WARNING: Drawer open too long"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/alert/A6/cam_A6.jpg",
          "frame_id": 12,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": [{ "zone": "ROI_CASHIER" }]
        }
      }
    }
  }
}
```

#### A7 (Cash in customer zone, no cashier present)
```json
{
  "input": { "path": "...", "name": "frame_A7.jpg" },
  "outputs": { "annotated_image": "outputs/cashier_aged_gender_single/.../annotated/frame_A7.jpg" },
  "timestamp": "2026-03-26T06:45:42.941111+00:00",
  "data": {
    "detection": { "count": 2, "items": [{ "class_name": "Person" }, { "class_name": "cash" }] },
    "use_case": {
      "cashier": {
        "persons": [{ "zone": "ROI_CUSTOMER", "transaction": false, "items": { "drawers": [], "cash": [{...}] } }],
        "summary": {
          "cashier_zone": { "persons": 0, "drawers": 0, "cash": 0 },
          "customer_zone": { "persons": 1, "drawers": 0, "cash": 1 },
          "case_id": "A7",
          "severity": "ALERT",
          "alerts": ["A7 WARNING: Cash in customer zone, no cashier present"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/.../evidence/alert/A7/cam_A7.jpg",
          "frame_id": 13,
          "timestamp": "2026-03-26T06:45:42.588678+00:00",
          "cashier_persons": []
        }
      }
    }
  }
}
```


### Full-dataset cashier test (2026-03-29)
- `outputs/cashier_test/20260329T060741_7464/summary.json` — metrics and case table (source of truth for the validation report above)
- `outputs/cashier_test/20260329T060741_7464/stream.jsonl` — per-frame pipeline JSON
- `outputs/cashier_test/20260329T060741_7464/annotated/`, `evidence/`, `events/`
- `outputs/cashier_test/20260329T060741_7464/cashier_test.gif` — optional animated export

### Production simulation artifacts (earlier reference run)
- `outputs/cashier_production_sim/20260326T050929Z/annotated_sequence.gif`
- `outputs/cashier_production_sim/20260326T050929Z/stream.jsonl`
- `outputs/cashier_production_sim/20260326T050929Z/summary.json`
- `outputs/cashier_production_sim/20260326T050929Z/annotated/`

### Single-frame test output (cashier + aged-gender)
For the one-off test on `frame_097440`, outputs are saved to:
- Annotated frame: `outputs/cashier_aged_gender_single/20260326T064528Z/annotated/frame_097440.jpg`
- JSON results: `outputs/cashier_aged_gender_single/20260326T064528Z/result.json`

### Response processing examples (SSE + jq)

Use `export BASE=http://localhost:9000`. Strip the SSE `data: ` prefix before `jq` (see [Complete cURL reference](#complete-curl-reference-all-http-routes)).

```bash
# First streamed frame: detection class + confidence
curl -sN "$BASE/detection/stream" \
  | grep "^data:" | head -1 | sed 's/^data: //' \
  | jq '.data.detection.items[]? | {class_name, confidence}'

# Same frame: high-confidence detections only
curl -sN "$BASE/detection/stream" \
  | grep "^data:" | head -1 | sed 's/^data: //' \
  | jq '.data.detection.items[]? | select(.confidence > 0.85)'

# Mood histogram for one frame (if mood is in the pipeline)
curl -sN "$BASE/detection/stream" \
  | grep "^data:" | head -1 | sed 's/^data: //' \
  | jq '[.data.use_case.mood[]? | .mood] | group_by(.) | map({mood: .[0], count: length})'

# Cashier summary (continuous; interrupt with Ctrl+C)
curl -sN "$BASE/detection/stream" \
  | grep "^data:" | sed 's/^data: //' \
  | jq '.data.use_case.cashier.summary | {case_id, severity, alerts}'

# Cashier persons with drawer/cash counts
curl -sN "$BASE/detection/stream" \
  | grep "^data:" | sed 's/^data: //' \
  | jq '.data.use_case.cashier.persons[]? | {zone, transaction, drawers: (.items.drawers | length), cash: (.items.cash | length)}'
```

---

## 🎯 Models & Services

### 1. YOLO v8 Nano (Object Detection)
- **Model**: `models/yolov8n.pt` (or `models/yolov8n.engine`)
- **Framework**: PyTorch/TensorRT
- **Purpose**: Person & object detection with bounding boxes
- **Input**: RGB images (any size)
- **Output**: Bounding boxes, class IDs, confidence scores
- **Supported Classes**: 80 COCO classes (persons, chairs, tables, etc.)

### 2. Age/Gender Classification (ONNX)
- **Model**: `models/best_aged_gender_6.onnx`
- **Framework**: ONNX Runtime
- **Gender Classes**: `[Female, Male]`
- **Age Groups**: `[Young, MiddleAged, Senior, Elderly]`
- **Input**: Face crops (224×224 pixels)
- **Output**: Class predictions + confidence scores

### 3. Mood/Emotion Detection (ONNX)
- **Model**: `models/best_mood.onnx`
- **Framework**: ONNX Runtime
- **Classes**: `[Angry, Happy, Neutral]`
- **Input**: Face crops (128×128 pixels)
- **Output**: Mood class + confidence score

### 4. PPE Detection (ONNX)
- **Model**: `models/best_ppe.onnx`
- **Framework**: ONNX Runtime
- **Classes**: `[mask, hairnet, gloves]`
- **Input**: person crops (224×224 pixels)
- **Output**: PPE class + confidence score

### 5. Cashier monitor (YOLO + zone logic)
- **Purpose**: Assign persons to `ROI_CASHIER` / `ROI_CUSTOMER`, detect drawers and cash, classify **N1–N6** / **A1–A7**, optional evidence GIFs and logs under `CASHIER_EVIDENCE_DIR`.
- **Config**: `CASHIER_CONFIG` (default `./config/cashier_zones.yaml`); see [Sample configuration file](#sample-configuration-file-configcashier_zonesyaml).

---

## 📁 Project Structure

```
.
├── app.py                      # FastAPI main application
├── pipeline.py                 # Per-camera worker pipeline
├── apis/                       # Route handlers
│   ├── cameras.py
│   ├── detection.py
│   └── cashier.py              # /cashier/* (zones, SSE, evidence, media)
├── config/                     # Default cashier zones YAML
│   └── cashier_zones.yaml
├── requirements.txt            # Python dependencies
├── README.md                   # This file (complete documentation)
├── .gitignore                  # Git ignore rules (models/ excluded)
├── models/                     # ML models (not tracked in git)
│   ├── yolov8n.pt             # YOLO v8 Nano (~25 MB)
│   ├── best_ppe.onnx           # PPE ONNX (~38 MB)
│   ├── best_aged_gender_6.onnx # Age/Gender ONNX (~85 MB)
│   └── best_mood.onnx          # Mood/Emotion ONNX (~15 MB)
├── services/                   # Service modules
│   ├── detector.py             # YOLO detection service
│   ├── age_gender.py           # Age/Gender classification
|   ├── ppe.py                  # PPE detection
│   ├── mood.py                 # Mood/Emotion detection
│   └── cashier.py              # Cashier zone + case logic
├── scripts/                    # Testing and utility scripts
│   └── test_image_pipeline.py  # Image inference testing script
├── tests/                      # pytest suite (`pytest.ini`)
├── logger/                     # Logging configuration
│   └── logger_config.py        # Logger setup
└── outputs/                    # Test results directory
    ├── test_image/             # Sample image test results
    │   ├── annotated.jpg       # Annotated image with boxes
    │   ├── stream_data.json    # Full inference results
    │   └── run.log             # Execution logs
    └── f1_test/                # F1.webp test results
        ├── annotated.jpg       # 500×281px annotated JPEG
        ├── stream_data.json    # Complete detection/classification data
        └── run.log             # Pipeline execution log
```

---

## 🔄 Git Workflow

```bash
git log --oneline -10
git status
git add .
git commit -m "Description of changes"
git push origin <your-branch>
```

### Model Files
Models are excluded from git tracking to reduce repository size:
- `best_aged_gender_6.onnx` (~85 MB)
- `best_mood.onnx` (~15 MB)
- `best_ppe.onnx` (~38 MB)
- `yolov8n.pt` (~25 MB)

Download separately or configure via environment variables.

---

## ⚙️ Configuration

### Model Paths
```bash
# Environment variables
export YOLO_MODEL="./models/yolov8n.pt"
export AGE_GENDER_MODEL="./models/best_aged_gender_6.onnx"
export MOOD_MODEL="./models/best_mood.onnx"
export PPE_MODEL="./models/best_PPE.onnx"
```

### Detection Thresholds
- **YOLO Confidence**: 0.35 (configurable)
- **Face Detection Minimum Size**: 10×10 pixels
- **Mood Classification**: All 3 classes enabled

---

## 📋 API Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Service info |
| `/cameras` | POST | Configure cameras |
| `/cameras` | GET | List configured cameras |
| `/cameras/{cam_id}` | DELETE | Remove camera |
| `/detection/setup` | POST | Configure pipeline services |
| `/detection/start` | POST | Start processing (optional `camera_id`) |
| `/detection/stop` | POST | Stop processing (optional `camera_id`) |
| `/detection/status` | GET | Operational status per camera |
| `/detection/stream` | GET | SSE: all cameras, one JSON per frame |
| `/cashier/status` | GET | Latest cashier summary per camera |
| `/cashier/events` | GET | Paginated alert/transaction log (`severity`, `case_id`, `camera_id`, `limit`, `offset`) |
| `/cashier/events` | DELETE | Clear in-memory event log |
| `/cashier/evidence` | GET | List evidence JPEGs (`severity`, `case_id`, `limit`) |
| `/cashier/evidence/{path}` | GET | Download one evidence JPEG |
| `/cashier/zones` | GET | Read `CASHIER_CONFIG` |
| `/cashier/zones` | POST | Update zones and/or thresholds |
| `/cashier/zones/reset` | POST | Restore default zones to config file |
| `/cashier/stream/{camera_id}` | GET | Per-camera SSE (all events) |
| `/cashier/stream/{camera_id}/only` | GET | Per-camera SSE (alerts-oriented) |
| `/cashier/media/{camera_id}/latest/jpg` | GET | Latest evidence JPG |
| `/cashier/media/{camera_id}/latest/gif` | GET | Latest evidence GIF |
| `/cashier/media/{camera_id}/event/{event_id}/jpg` | GET | JPG for event |
| `/cashier/media/{camera_id}/event/{event_id}/gif` | GET | GIF for event |
| `/cashier/media/{camera_id}/drawer_count` | GET | Drawer-open count from JSONL log |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc |

---

## 📞 Support & Documentation

After starting the server (`uvicorn` on port **9000** by default):

- Swagger UI: `http://localhost:9000/docs`
- ReDoc: `http://localhost:9000/redoc`

Supplementary handoff docs: [`curl_cashier.md`](curl_cashier.md), [`sse_cashier.md`](sse_cashier.md).

## Models file
https://drive.google.com/drive/folders/1oAROlqkBo8C3rzTe4hAcS7abaIKC_Ugq?usp=drive_link
