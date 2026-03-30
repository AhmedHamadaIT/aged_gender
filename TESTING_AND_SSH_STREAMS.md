# Automated tests, endpoints, and remote log / SSE streams

This document lists HTTP routes, pipeline **services** (inference modules), how to run **pytest**, and **SSH** one-liners to follow live streams or batch logs on a Jetson/host where the repo lives at `REMOTE_ROOT` (adjust paths and user).

Set locally:

```bash
export REMOTE_USER=ubuntu
export REMOTE_HOST=192.168.1.50
export REMOTE_ROOT=/home/ubuntu/ml-server
export BASE=http://${REMOTE_HOST}:9000
```

---

## Pipeline services (`services.REGISTRY`)

These are **not** separate OS services; they are Python classes registered for `POST /detection/setup` (`pipeline` array):

| Key          | Module              | Role |
|-------------|---------------------|------|
| `detector`  | `services.detector` | YOLO detection |
| `age_gender`| `services.age_gender` | Age/gender use-case |
| `ppe`       | `services.ppe`      | PPE use-case |
| `mood`      | `services.mood`     | Mood use-case |
| `cashier`   | `services.cashier`  | Cashier zones / scenarios |

**pytest:** `tests/services_registry_suite.py` asserts registry keys, types, default cashier config, and `app.build_pipeline` wiring.

---

## HTTP endpoints (FastAPI)

| Method | Path | Notes |
|--------|------|--------|
| GET | `/` | Service name + version |
| GET | `/openapi.json` | Machine-readable schema |
| POST | `/cameras` | Register cameras (JSON body) |
| GET | `/cameras` | List configured cameras |
| DELETE | `/cameras/{cam_id}` | Remove camera |
| POST | `/detection/setup` | Set `pipeline: ["detector", …]` |
| POST | `/detection/start` | Optional `?camera_id=` |
| POST | `/detection/stop` | Optional `?camera_id=` |
| GET | `/detection/status` | Per-camera runtime status |
| GET | `/detection/stream` | **SSE** — one JSON frame per line (`data: …`) |
| GET | `/cashier/status` | Latest cashier summary per camera |
| GET | `/cashier/events` | Query: `severity`, `case_id`, `camera_id`, `limit`, `offset` |
| DELETE | `/cashier/events` | Clear in-memory log |
| GET | `/cashier/evidence` | List JPEGs under `CASHIER_EVIDENCE_DIR` |
| GET | `/cashier/evidence/{path}` | Download one JPEG |
| GET | `/cashier/zones` | YAML/JSON config |
| POST | `/cashier/zones` | Partial zone/threshold update |
| POST | `/cashier/zones/reset` | Defaults |
| GET | `/cashier/stream/{camera_id}` | **SSE** — cashier events |
| GET | `/cashier/stream/{camera_id}/only` | **SSE** — alerts-oriented |
| GET | `/cashier/media/{camera_id}/latest/jpg` | Latest `{camera_id}_*.jpg` |
| GET | `/cashier/media/{camera_id}/latest/gif` | Latest `{camera_id}_*.gif` |
| GET | `/cashier/media/{camera_id}/event/{event_id}/jpg` | Per-event JPEG |
| GET | `/cashier/media/{camera_id}/event/{event_id}/gif` | Per-event GIF |
| GET | `/cashier/media/{camera_id}/drawer_count` | Count from `logs/events.jsonl` |

**Environment:**

- `CASHIER_CONFIG` — default `./config/cashier_zones.yaml`
- `CASHIER_EVIDENCE_DIR` — default `./evidence/cashier`
- `CASHIER_LOG_MAX` — max in-memory events (default `5000`)

---

## cURL examples with sample responses

Set `BASE` (local or remote). Examples use **`curl -s`** (silent body). Shapes are **illustrative**; field names match the live API.

### `GET /`

```bash
curl -s "$BASE/"
```

**Sample response:**

```json
{"service":"Vision Pipeline API","version":"1.0.0"}
```

### Cameras

```bash
curl -s "$BASE/cameras"
```

**Sample (no cameras):**

```json
{"count":0,"cameras":[]}
```

```bash
curl -s -X POST "$BASE/cameras" \
  -H "Content-Type: application/json" \
  -d '{"cameras":[{"id":"cam1","url":"rtsp://192.168.1.10/stream"}]}'
```

**Sample response:**

```json
{"status":"configured","cameras":{"cam1":"rtsp://192.168.1.10/stream"}}
```

```bash
curl -s -X DELETE "$BASE/cameras/cam1"
```

**Sample response:**

```json
{"status":"removed","camera_id":"cam1","remaining":[]}
```

### Detection

```bash
curl -s -X POST "$BASE/detection/setup" \
  -H "Content-Type: application/json" \
  -d '{"pipeline":["detector","cashier"]}'
```

**Sample response:**

```json
{"status":"configured","pipeline":["detector","cashier"]}
```

```bash
curl -s "$BASE/detection/status"
```

**Sample (idle / no workers):**

```json
{"cameras":{}}
```

**Sample (one camera running — fields vary by implementation):**

```json
{
  "cameras": {
    "cam1": {
      "camera_id": "cam1",
      "rtsp_url": "rtsp://192.168.1.10/stream",
      "running": true,
      "frame_count": 1240,
      "fps": 12.4,
      "last_detections": 2,
      "total_detections": 5800,
      "uptime_seconds": 100.5,
      "error": null
    }
  }
}
```

**Error examples (HTTP 4xx, JSON body):**

```json
{"detail":"Unknown services: ['bad']. Available: ['detector', 'age_gender', 'ppe', 'mood', 'cashier']"}
```

```json
{"detail":"No cameras configured. Call POST /cameras first."}
```

```json
{"detail":"Camera 'cam1' already running."}
```

```json
{"detail":"No cameras are currently running."}
```

### Cashier — zones, status, events

```bash
curl -s "$BASE/cashier/zones"
```

**Sample (truncated):** top-level keys often include `zones`, `thresholds`, `buffer`, `debounce`, `evidence`, `gif`, `meta`.

```json
{
  "meta": {"version": "1.2.0"},
  "zones": {
    "ROI_CASHIER": {"shape": "rectangle", "points": [[0.0, 0.0], [0.5, 1.0]], "active": true},
    "ROI_CUSTOMER": {"shape": "rectangle", "points": [[0.5, 0.0], [1.0, 1.0]], "active": true}
  },
  "thresholds": {
    "drawer_open_max_seconds": 45,
    "customer_wait_max_seconds": 45
  }
}
```

```bash
curl -s -X POST "$BASE/cashier/zones" \
  -H "Content-Type: application/json" \
  -d '{"thresholds":{"drawer_open_max_seconds": 30}}'
```

**Sample response:**

```json
{"status":"updated","config": { "zones": {}, "thresholds": { "drawer_open_max_seconds": 30 } } }
```

*(Real `config` is the full merged document.)*

```bash
curl -s -X POST "$BASE/cashier/zones/reset"
```

**Sample response:**

```json
{"status":"reset_to_default","config": { "meta": {"version": "1.2.0"}, "zones": {} } }
```

```bash
curl -s "$BASE/cashier/status"
```

**Sample (no frames yet):**

```json
{}
```

**Sample (one camera):**

```json
{
  "cam1": {
    "camera_id": "cam1",
    "case_id": "N3",
    "severity": "NORMAL",
    "alerts": ["N3 EVENT: Transaction in progress"],
    "transaction": true,
    "cashier_zone": {"persons": 1, "drawers": 1, "cash": 0},
    "customer_zone": {"persons": 1, "drawers": 0, "cash": 1}
  }
}
```

```bash
curl -s "$BASE/cashier/events?limit=5"
```

**Sample (empty log):**

```json
{"total":0,"offset":0,"limit":5,"events":[]}
```

```bash
curl -s -X DELETE "$BASE/cashier/events"
```

**Sample response:**

```json
{"cleared":0}
```

### Cashier — evidence list

```bash
curl -s "$BASE/cashier/evidence?limit=3"
```

**Sample (no files on disk):**

```json
{"total":0,"files":[]}
```

**Sample (with JPEGs under `CASHIER_EVIDENCE_DIR`):**

```json
{
  "total": 12,
  "files": [
    {
      "path": "alert/A5/cam1_20260329T120000_001.jpg",
      "size_kb": 84.2,
      "modified": "2026-03-29T12:00:05+00:00"
    }
  ]
}
```

### Cashier — media / drawer count

```bash
curl -s "$BASE/cashier/media/cam1/drawer_count"
```

**Sample response:**

```json
{"camera_id":"cam1","drawer_open_count":0}
```

`GET /cashier/media/cam1/latest/jpg` returns **raw JPEG** (`Content-Type: image/jpeg`), not JSON — use `-o file.jpg` to save.

### SSE (detection / cashier) — wire format

**Detection** (`curl -N "$BASE/detection/stream"`): each frame is one SSE message, JSON after `data: `:

```text
data: {"camera_id":"cam1","frame_count":42,"timestamp":"2026-03-30T10:00:00.123456","frame":"<base64>","data":{"detection":{"count":1,"items":[]},"use_case":{}}}

```

**Cashier** (`curl -N "$BASE/cashier/stream/cam1"`): first line is usually `connected`, then `frame` / `alert` / `gif_ready`:

```text
event: connected
data: {"camera_id": "cam1", "alert_only": false}

event: frame
data: {"case_id": "N1", "severity": "NORMAL", "summary": {}}

```

---

## Run pytest (local, uses `.venv`)

```bash
cd /path/to/ml-server
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

| File | What it covers |
|------|----------------|
| `tests/conftest.py` | `TestClient` with lifespan (startup / `detection.set_pipeline`), isolated cashier config fixture |
| `tests/api_full_suite.py` | All routes above except long-lived SSE bodies (see note below) |
| `tests/cashier_evidence_fixtures.py` | Evidence list/download/media when `outputs/...` JPEG exists |
| `tests/services_registry_suite.py` | `REGISTRY`, `_default_config`, `build_pipeline` |

**Note on SSE in pytest:** `TestClient` can block on long-lived streams. SSE is validated via **`/openapi.json`** path presence; use **curl** over SSH (below) for live checks.

**Evidence fixtures:** `tests/cashier_evidence_fixtures.py` **skips** if this file is missing (batch output is gitignored):

`outputs/cashier_aged_gender_single/20260326T064528Z/evidence/normal/N3/cam_20260326T064542_598744.jpg`

---

## SSH: discover paths with `ls`, then stream

Run these **on the remote host** (after `ssh user@host`) or wrap with `ssh user@host '…'`.

### List batch runs under `outputs/` (pick a timestamp folder)

```bash
ls -la "${REMOTE_ROOT}/outputs/"
```

**Sample `ls` output (illustrative):**

```text
drwxrwxr-x  3 ubuntu ubuntu 4096 Mar 29 15:53 .
drwxrwxr-x 18 ubuntu ubuntu 4096 Mar 29 16:21 ..
drwxrwxr-x  3 ubuntu ubuntu 4096 Mar 26 08:45 cashier_aged_gender_single
drwxrwxr-x  3 ubuntu ubuntu 4096 Mar 29 15:53 cashier_test
drwxrwxr-x  4 ubuntu ubuntu 4096 Mar 29 11:05 cashier_test_first100
```

```bash
ls -la "${REMOTE_ROOT}/outputs/cashier_test/"
```

**Sample:**

```text
drwxrwxr-x 3 ubuntu ubuntu 4096 Mar 29 15:53 .
drwxrwxr-x 5 ubuntu ubuntu 4096 Mar 29 15:46 ..
drwxrwxr-x 6 ubuntu ubuntu 4096 Mar 29 06:07 20260329T060741_7464
```

Then list files inside the run directory:

```bash
ls -la "${REMOTE_ROOT}/outputs/cashier_test/20260329T060741_7464/"
```

**Sample:**

```text
-rw-rw-r-- 1 ubuntu ubuntu  1234567 Mar 29 06:15 stream.jsonl
-rw-rw-r-- 1 ubuntu ubuntu     8901 Mar 29 06:15 summary.json
drwxrwxr-x 4 ubuntu ubuntu     4096 Mar 29 06:08 evidence
drwxrwxr-x 2 ubuntu ubuntu     4096 Mar 29 06:08 annotated
drwxrwxr-x 2 ubuntu ubuntu     4096 Mar 29 06:08 events
```

### List cashier evidence tree and log file

```bash
ls -la "${REMOTE_ROOT}/evidence/cashier/"
ls -la "${REMOTE_ROOT}/evidence/cashier/logs/" 2>/dev/null || true
```

**Sample:**

```text
drwxrwxr-x 5 ubuntu ubuntu 4096 Mar 29 15:15 .
drwxrwxr-x 3 ubuntu ubuntu 4096 Mar 26 09:14 ..
drwxrwxr-x 4 ubuntu ubuntu 4096 Mar 29 12:00 alert
drwxrwxr-x 4 ubuntu ubuntu 4096 Mar 29 12:00 normal
drwxrwxr-x 2 ubuntu ubuntu 4096 Mar 29 12:00 logs

logs/events.jsonl
```

---

## SSH: follow HTTP SSE streams (remote API)

Replace `CAM` with a real `camera_id` (e.g. `cam1`, `cashier_cam_01`).

### Detection — combined frame stream

```bash
ssh "${REMOTE_USER}@${REMOTE_HOST}" "curl -sN 'http://127.0.0.1:9000/detection/stream' | head -n 20"
```

**Sample first lines (long `frame` base64 truncated):**

```text
data: {"camera_id":"cam1","frame_count":1,"timestamp":"2026-03-30T12:00:00","frame":"/9j/4AAQ...","data":{"detection":{"count":2,"items":[]}}}

```

Leave unbounded (Ctrl+C locally breaks SSH):

```bash
ssh -t "${REMOTE_USER}@${REMOTE_HOST}" "curl -N 'http://127.0.0.1:9000/detection/stream'"
```

### Cashier — all events for one camera

```bash
ssh -t "${REMOTE_USER}@${REMOTE_HOST}" "curl -N 'http://127.0.0.1:9000/cashier/stream/CAM'"
```

**Sample first lines:**

```text
event: connected
data: {"camera_id": "cam1", "alert_only": false}

: ping

```

*(After ~30s idle the server may emit an SSE **comment** line `: ping` to keep intermediaries happy.)*

### Cashier — alerts-oriented stream

```bash
ssh -t "${REMOTE_USER}@${REMOTE_HOST}" "curl -N 'http://127.0.0.1:9000/cashier/stream/CAM/only'"
```

**Sample:**

```text
event: connected
data: {"camera_id": "cam1", "alert_only": true}

```

Use `-sN` for less progress noise; always use **`-N`** (no buffer) on SSE.

---

## SSH: follow on-disk batch / evidence logs

Use the **`ls`** section above to copy the exact run directory name, then:

### Batch run — line-delimited frame JSON (`stream.jsonl`)

```bash
ssh "${REMOTE_USER}@${REMOTE_HOST}" "tail -f '${REMOTE_ROOT}/outputs/cashier_test/20260329T060741_7464/stream.jsonl'"
```

**Sample line (one JSON object per line, wrapped for readability):**

```json
{"camera_id":"cam1","frame_count":150,"timestamp":"2026-03-29T06:08:12.456789","data":{"detection":{"count":1,"items":[]},"use_case":{"cashier":{"summary":{"case_id":"N2","severity":"NORMAL"}}}}}
```

### Cashier evidence log (`triggered` / `resolved`, GIF paths after compile)

```bash
ssh "${REMOTE_USER}@${REMOTE_HOST}" "tail -f '${REMOTE_ROOT}/evidence/cashier/logs/events.jsonl'"
```

**Sample lines:**

```json
{"camera_id":"cam1","case_id":"A5","status":"triggered","severity":"ALERT","logged_at":"2026-03-29T12:00:01+00:00"}
{"camera_id":"cam1","case_id":"A5","status":"resolved","severity":"ALERT","gif_path":".../cam1_20260329_....gif"}
```

### Application log (if you log to file)

```bash
ssh "${REMOTE_USER}@${REMOTE_HOST}" "tail -f '${REMOTE_ROOT}/logger/app.log'"
```

**Sample:**

```text
2026-03-30 10:00:00 [INFO] [APP] Ready.
2026-03-30 10:00:05 [INFO] Camera cam1 frame_count=100
```

### Optional: filter SSE lines to JSON only (on the SSH host)

```bash
ssh "${REMOTE_USER}@${REMOTE_HOST}" "curl -sN 'http://127.0.0.1:9000/detection/stream' | grep --line-buffered '^data:' | sed -u 's/^data: //' | head -n 5"
```

**Sample stdout (one JSON per line):**

```text
{"camera_id":"cam1","frame_count":1,...}
```

---

## Quick health check over SSH

```bash
ssh "${REMOTE_USER}@${REMOTE_HOST}" "curl -s 'http://127.0.0.1:9000/' && echo && curl -s 'http://127.0.0.1:9000/detection/status'"
```

**Sample combined output:**

```text
{"service":"Vision Pipeline API","version":"1.0.0"}
{"cameras":{}}
```

---

## Related docs

- Full curl reference: [`README.md`](README.md)
- Cashier thresholds / scenarios detail: [`curl_cashier.md`](curl_cashier.md), [`sse_cashier.md`](sse_cashier.md)
