# Full pipeline test

`test_pipeline_api.py` checks **`POST /detection/setup`** with **all** services in `services.REGISTRY` (`detector`, `age_gender`, `ppe`, `mood`, `cashier`), then hits the main REST routes. It does **not** call `POST /detection/start` (that would spawn RTSP workers).

Fixtures live in `conftest.py` (including an `ultralytics` stub so CI does not need GPU/YOLO weights).

---

## Run the test

From the **repository root** (where `app.py` lives):

```bash
python3 -m pytest tests/pipeline_test/test_pipeline_api.py -v
```

Run the whole folder:

```bash
python3 -m pytest tests/pipeline_test/ -v
```

---

## cURL (against a running server)

Start the API (example):

```bash
uvicorn app:app --host 0.0.0.0 --port 9000
```

Set a base URL:

```bash
export BASE=http://127.0.0.1:9000
```

Below, **example responses** match the usual JSON shape; field order may differ.

### `GET /`

```bash
curl -s "$BASE/"
```

**Example response**

```json
{"service":"Vision Pipeline API","version":"1.0.0"}
```

### Full pipeline setup

```bash
curl -s -X POST "$BASE/detection/setup" \
  -H "Content-Type: application/json" \
  -d '{"pipeline":["detector","age_gender","ppe","mood","cashier"]}'
```

**Example response**

```json
{
  "status": "configured",
  "pipeline": ["detector", "age_gender", "ppe", "mood", "cashier"]
}
```

### Cameras

```bash
curl -s -X POST "$BASE/cameras" \
  -H "Content-Type: application/json" \
  -d '{"cameras":[{"id":"cam1","url":"rtsp://user:pass@192.168.1.10/stream"}]}'
```

**Example response**

```json
{
  "status": "configured",
  "cameras": {"cam1": "rtsp://user:pass@192.168.1.10/stream"}
}
```

```bash
curl -s "$BASE/cameras"
```

**Example response**

```json
{"count": 1, "cameras": [{"id": "cam1", "url": "rtsp://..."}]}
```

*(Exact `cameras` list shape follows your API implementation.)*

### Detection status / stop (no workers running)

```bash
curl -s "$BASE/detection/status"
```

**Example response**

```json
{"cameras": {}}
```

```bash
curl -s -X POST "$BASE/detection/stop"
```

**Example response** (HTTP 409)

```json
{"detail": "No cameras are currently running."}
```

### SSE streams (raw wire; use `-N` so data is not buffered)

**Detection** — one JSON object per `data:` line (long base64 `frame` when present):

```bash
curl -sN --max-time 8 "$BASE/detection/stream" | head -n 5
```

**Example lines** (truncated)

```text
data: {"camera_id":"cam1","frame_count":1,"timestamp":"2026-03-30T12:00:00.123456","frame":"/9j/4AAQ...","data":{"detection":{"count":2,"items":[]}}}
```

**Cashier** — named events:

```bash
curl -sN --max-time 8 "$BASE/cashier/stream/cam1" | head -n 20
```

**Example lines**

```text
event: connected
data: {"camera_id": "cam1", "alert_only": false}

event: frame
data: {"case_id": "N3", "severity": "NORMAL", "summary": {}}
```

Alerts-only stream:

```bash
curl -sN --max-time 8 "$BASE/cashier/stream/cam1/only" | head -n 15
```

### Cashier REST (samples)

```bash
curl -s "$BASE/cashier/status"
```

**Example response** (empty until frames are processed)

```json
{}
```

```bash
curl -s "$BASE/cashier/events?limit=5"
```

**Example response**

```json
{"total": 0, "offset": 0, "limit": 5, "events": []}
```

```bash
curl -s "$BASE/cashier/zones"
```

**Example response** (truncated; includes `zones`, `thresholds`, and often `buffer`, `debounce`, …)

```json
{
  "zones": {
    "ROI_CASHIER": {"shape": "rectangle", "points": [[0.0, 0.0], [0.5, 1.0]], "active": true},
    "ROI_CUSTOMER": {"shape": "rectangle", "points": [[0.5, 0.0], [1.0, 1.0]], "active": true}
  },
  "thresholds": {"drawer_open_max_seconds": 30}
}
```

```bash
curl -s "$BASE/cashier/media/cam1/drawer_count"
```

**Example response**

```json
{"camera_id": "cam1", "drawer_open_count": 0}
```

---

## CI

GitHub Actions runs:

```bash
python3 -m pytest tests/pipeline_test/ -v --tb=short
```
