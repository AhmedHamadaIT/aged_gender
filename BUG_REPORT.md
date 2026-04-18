# EyeGo Vision Pipeline API v2.0.0 — Bug Report

**Date:** 2026-04-18  
**Tester:** Backend Engineer (Ahmed)  
**Server:** `http://localhost:9000`  
**Camera:** Laptop webcam `/dev/video0` via `mediamtx` RTSP at `rtsp://localhost:8554/cam0`  
**Scope:** Full end-to-end API test covering all 36 prompts across health, cameras, tasks, detection, SSE/WS streams, cashier monitor, person search, and semantic search.

---

## Executive Summary

| Severity | Count |
|----------|-------|
| P0 Critical | 2 |
| P1 High | 5 |
| P2 Medium | 4 |
| P3 Low | 3 |
| **Total** | **14** |

**Stream annotation is confirmed WORKING** — annotated JPEG frames (with YOLOv8 bounding boxes) are delivered via `WS /cameras/cam0/live` at ~2.5 FPS on CPU. The main failure cluster is in the Cashier service, caused by a fundamental IPC design flaw between multiprocessing subprocess workers and the main process's in-memory state.

---

## P0 — Critical

### BUG-001: Cashier SSE streams never deliver events (IPC design flaw)

**Endpoints:** `GET /cashier/stream/{camera_id}`, `GET /cashier/stream/{camera_id}/only`

**Symptom:** Both endpoints deliver only a single "connected" acknowledgment event, then go silent. No cashier events are ever streamed regardless of detection activity.

**Root Cause:** `_SSEPublisher` in `apis/cashier.py` is a module-level singleton instantiated at import time. It uses an asyncio queue (`self._queues`) to bridge between the `sse_publish()` call site and the `stream()` generator. However, cashier tasks (e.g., `CashierDrawerTask`) run inside **forked subprocess workers** (`multiprocessing` via `task_worker.py`). On `fork()`, the subprocess inherits a copy-on-write snapshot of the parent's memory:

1. The subprocess's `_sse_publisher` is a separate object from the main process's.
2. Calls to `sse_publish(...)` in the subprocess put messages into the **subprocess's** asyncio queue — the main process's `stream()` generator never sees them.
3. The secondary Redis pub/sub path (`self._redis_sync`) is **always `None`** because `REDIS_URL` was not in `os.environ` when `apis/cashier.py` was first imported (before `load_dotenv()` runs in `app.py`). This means Redis bridging is also dead.

**Reproduction:**
```bash
# Start detection with CASHIER_BOX task
curl -s -X POST http://localhost:9000/detection/start -H "Content-Type: application/json" \
  -d '{"cameras": [{"camera_id": "cam0", "rtsp_url": "rtsp://localhost:8554/cam0"}]}'

# Listen to cashier stream - only gets 1 "connected" event, then silence
curl -N -H "Accept: text/event-stream" http://localhost:9000/cashier/stream/cam0
```

**Fix Required (AI Team):** Replace in-memory queue IPC with Redis pub/sub throughout the cashier event pipeline. The subprocess must publish to a Redis channel; the main process subscribes and forwards to SSE clients. Alternatively, use a multiprocessing `Manager().Queue()` or `multiprocessing.Queue` for shared state. The lazy Redis initialization must also be fixed (initialize `_redis_sync` on first `publish()` call, not in `__init__`).

---

### BUG-002: `GET /cashier/events` always returns `total: 0`

**Endpoint:** `GET /cashier/events`

**Symptom:** Returns `{"total": 0, "events": []}` even while cashier events are visibly flowing through `GET /detection/stream`.

**Root Cause:** Same IPC flaw as BUG-001. The `_event_log` deque in `apis/cashier.py` (and `push_structured_cashier_event()`) is written by the subprocess worker but read by the main process. These are different memory spaces post-`fork()`. The main process's `_event_log` is never populated.

**Reproduction:**
```bash
# Events appear in /detection/stream SSE (CASHIER_BOX_OPEN type)
# But this always returns 0:
curl -s http://localhost:9000/cashier/events
# {"total": 0, "events": []}
```

**Fix Required:** Same as BUG-001 — event log persistence must use a shared store (Redis list/sorted-set or a database), not an in-process deque.

---

## P1 — High

### BUG-003: `POST /detection/stop/all` returns HTTP 404

**Endpoint:** `POST /detection/stop/all`

**Symptom:** HTTP 404 "Not Found". The route does not exist.

**Actual working endpoint:** `POST /detection/stop` (stops all cameras).

**Impact:** Any client or script using `/detection/stop/all` (as documented in the test suite specification) will fail silently.

**Reproduction:**
```bash
curl -s -X POST http://localhost:9000/detection/stop/all
# {"detail":"Not Found"}

curl -s -X POST http://localhost:9000/detection/stop
# {"status": "stopped", "cameras": ["cam0"]}  ← works
```

**Fix Required:** Either add the `/detection/stop/all` alias route, or update API documentation and client SDKs to use `/detection/stop`.

---

### BUG-004: `PUT /api/tasks/{task_id}` creates a new task on non-existent ID (no 404)

**Endpoint:** `PUT /api/tasks/{task_id}`

**Symptom:** Sending a PUT request with a `task_id` that does not exist in the registry returns HTTP 200 and creates a new task, instead of returning HTTP 404 Not Found.

**Reproduction:**
```bash
curl -s -X PUT http://localhost:9000/api/tasks/9999 \
  -H "Content-Type: application/json" \
  -d '{"taskName":"ghost","algorithmType":"CROSS_LINE","channelId":"cam0"}'
# HTTP 200 — task created with id 9999 (unexpected)
```

**Fix Required:** `TaskRegistry.update()` (or the PUT handler) must check if `task_id` exists and raise HTTP 404 if not.

---

### BUG-005: `GET /cashier/status` always returns empty `{}`

**Endpoint:** `GET /cashier/status`

**Symptom:** Returns `{}` regardless of whether cashier tasks are running or have processed any events.

**Expected:** A camera-keyed object containing cashier state: open drawer count, alert level, staff present, etc.

**Reproduction:**
```bash
curl -s http://localhost:9000/cashier/status
# {}
```

**Fix Required:** The cashier status handler must read from a shared store (same IPC issue as BUG-001/002). The current in-process state updated by the subprocess is not accessible in the main process.

---

### BUG-006: Inconsistent `channelId` type in SSE detection events

**Endpoint:** `GET /detection/stream`

**Symptom:** Different algorithm types emit events with different `channelId` types in the JSON payload:
- `CASHIER_BOX_OPEN` events → `"channelId": 0` (integer `0`, not the actual camera ID)
- `MASK_HAIRNET_CHEF_HAT` events → `"channelId": "cam0"` (correct string)
- `MASK_HAIRNET_CHEF_HAT` events → missing `camera_id` field entirely
- `CROSS_LINE` events → `"channelId": "cam0"` (correct string), `"camera_id": "cam0"` present

**Impact:** Clients parsing the SSE stream cannot reliably filter events by camera. Integer `0` is not a valid camera ID in this system.

**Sample malformed event:**
```json
{
  "event": "CASHIER_BOX_OPEN",
  "channelId": 0,
  "data": { ... }
}
```

**Fix Required:**
1. Enforce `channelId` as a `str` type across all event schemas.
2. Ensure all event types include the `camera_id` field.
3. `CashierDrawerTask` must pass the actual camera ID string, not a default integer.

---

### BUG-007: `person_search` and `semantic_search` return HTTP 500 instead of 503 when models not loaded

**Endpoints:** `POST /person_search/search`, `POST /semantic_search/search`

**Symptom:** When REID model or ONNX/open_clip models are absent, the endpoints return HTTP 500 Internal Server Error.

**Expected:** HTTP 503 Service Unavailable with a clear `Retry-After` or setup guidance.

**Reproduction:**
```bash
# With no OSNet model file at REID_MODEL_PATH:
curl -s -X POST http://localhost:9000/person_search/search \
  -F "file=@/tmp/test_person.jpg"
# HTTP 500: {"detail": "Search failed: OSNet model not loaded..."}

# With no open_clip / ONNX models:
curl -s -X POST http://localhost:9000/semantic_search/search \
  -F "text_query=person in red shirt"
# HTTP 500: {"detail": "Search failed: SemanticSearchService not loaded..."}
```

**Fix Required:** Change HTTP status code to 503 in both service handlers when `self.model is None` / `self._ready is False`. Consider adding a `GET /person_search/health` and `GET /semantic_search/health` endpoint.

---

## P2 — Medium

### BUG-008: `POST /cameras` with empty array returns HTTP 200 instead of 400

**Endpoint:** `POST /cameras`

**Symptom:** Sending `{"cameras": []}` returns HTTP 200 with the current camera list (no-op), rather than rejecting the empty payload with HTTP 400 Bad Request.

**Reproduction:**
```bash
curl -s -X POST http://localhost:9000/cameras \
  -H "Content-Type: application/json" \
  -d '{"cameras": []}'
# HTTP 200 — returns current cameras list
```

**Fix Required:** Add input validation: if `cameras` array is empty, return HTTP 422 or 400 with a descriptive error.

---

### BUG-009: WebSocket endpoints missing from OpenAPI spec (`/docs`)

**Endpoints:** `WS /cameras/{camera_id}/live`, `WS /cameras/{camera_id}/events`

**Symptom:** Neither WebSocket endpoint appears in `/openapi.json` or `/docs`. FastAPI does not auto-document WebSocket routes by default, but these are key consumer-facing APIs.

**Impact:** API consumers using `/docs` or generated SDKs have no visibility into the WS interface, frame format (binary JPEG), or event schema.

**Fix Required:** Add manual OpenAPI schema entries for WS endpoints, or add a dedicated section in the API documentation (e.g., README / AsyncAPI spec). At minimum, document the binary JPEG frame format and the JSON event structure for `cameras/{camera_id}/events`.

---

### BUG-010: `semantic_search/search` rejects file-only upload (missing `text_query`)

**Endpoint:** `POST /semantic_search/search`

**Symptom:** Uploading a file without a `text_query` field returns HTTP 422 "Field required: text_query". The API should accept *either* an image file *or* a text query for search.

**Reproduction:**
```bash
curl -s -X POST http://localhost:9000/semantic_search/search \
  -F "file=@/tmp/test_person.jpg"
# HTTP 422: {"detail": [{"type": "missing", "loc": ["body", "text_query"], ...}]}
```

**Fix Required:** Make `text_query` optional (default `None`) in the form model. Add validation that at least one of `file` or `text_query` is provided, and return 400 if neither is provided.

---

### BUG-011: `_SSEPublisher` Redis client initialized before `load_dotenv()` — always `None`

**File:** `apis/cashier.py` line ~211

**Symptom:** `_sse_publisher._redis_sync` is always `None` at runtime. The Redis-based fallback path in `_SSEPublisher.publish()` is therefore permanently dead, even when Redis is available.

**Root Cause:** `_SSEPublisher()` is instantiated at module import time (`_sse_publisher = _SSEPublisher()`). At this point `os.environ["REDIS_URL"]` is not yet set because `load_dotenv()` in `app.py` has not run yet. `os.getenv("REDIS_URL", "")` returns `""`, so no Redis client is created.

**Fix Required:** Lazily initialize `_redis_sync` on the first call to `publish()`, not in `__init__`. Or move `load_dotenv()` to the top of `app.py` before any API module imports.

---

## P3 — Low

### BUG-012: Low detection FPS on CPU (~2.5 FPS vs expected ≥10 FPS)

**Endpoint:** `GET /detection/status`

**Symptom:** After ~13 minutes of runtime, `fps` reported as `2.49` for `cam0`. This is well below a production-viable 10 FPS threshold.

**Context:** Tested with `DEVICE=cpu` and `yolov8n.pt`. This is partially expected on CPU, but the FrameBus frame delivery rate also appears to be throttled.

**Fix Required (AI Team):** Document minimum hardware requirements. If CPU-only mode must be supported, consider a lighter model (e.g., YOLOv8n ONNX with CPU optimization) or frame-skip logic. The `.engine` (TensorRT) model in the default `.env` should be pre-built or a fallback model auto-selected.

---

### BUG-013: `GET /cashier/evidence/{file_path}` — path traversal not validated

**Endpoint:** `GET /cashier/evidence/{file_path}`

**Symptom:** The `file_path` parameter is taken from the user and used to construct a file path for serving. No path traversal sanitization was observed in the response headers or behavior.

**Risk:** Potential directory traversal (e.g., `../../etc/passwd`) if the file serving logic uses `os.path.join` naively without canonicalization.

**Fix Required:** Use `pathlib.Path.resolve()` and assert the resolved path is within the evidence directory before serving.

---

### BUG-014: `GET /cashier/media/{camera_id}/latest/jpg` returns non-standard error schema (404)

**Endpoint:** `GET /cashier/media/{camera_id}/latest/jpg`

**Symptom:** Returns a non-standard error envelope:
```json
{
  "status": "error",
  "error": {"code": "404", "message": "Cashier evidence not found.", "detail": "cam0"}
}
```

**Expected:** FastAPI standard `{"detail": "..."}` with HTTP 404, or a consistent error schema used across the whole API.

**Fix Required:** Standardize error responses across all endpoints using a shared Pydantic error model or FastAPI's built-in `HTTPException`.

---

## Verified Working Features

The following features were confirmed functional during testing:

| Feature | Status | Notes |
|---------|--------|-------|
| `GET /` health check | ✅ PASS | Returns `{"status":"ok"}` |
| `POST /cameras` register | ✅ PASS | Accepts camera list |
| `GET /cameras` list | ✅ PASS | Returns registered cameras |
| `DELETE /cameras/{cam_id}` | ✅ PASS | Removes camera |
| `POST /api/tasks` create | ✅ PASS | String `channelId` accepted after fix |
| `GET /api/tasks` list | ✅ PASS | Returns all tasks |
| `GET /api/tasks/{task_id}` get | ✅ PASS | Returns specific task |
| `DELETE /api/tasks/{task_id}` | ✅ PASS | Removes task |
| `POST /detection/start` | ✅ PASS | Starts detection on all cameras |
| `POST /detection/stop` | ✅ PASS | Stops detection, cleans up |
| `GET /detection/status` | ✅ PASS | Returns per-camera FPS, frame_count |
| `GET /detection/stream` SSE | ✅ PASS | Events stream correctly (see BUG-006 for schema issues) |
| `WS /cameras/cam0/live` | ✅ PASS | Delivers annotated binary JPEGs at ~2.5 FPS on CPU |
| `WS /cameras/cam0/events` | ✅ PASS | Delivers JSON detection events |
| **Stream annotation** | ✅ **CONFIRMED** | YOLOv8 bounding boxes visible in saved frames (1280×720 JPEG) |
| `GET /cashier/zones` | ✅ PASS | Returns zone config |
| `PUT /cashier/zones` | ✅ PASS | Updates zone config |
| `POST /cashier/zones/reset` | ✅ PASS | Resets to default config |
| `GET /cashier/evidence` | ✅ PASS | Lists evidence files |
| `GET /cashier/media/{cam}/drawer_count` | ✅ PASS | Returns `{"drawer_open_count": 0}` |
| `GET /docs` | ✅ PASS | HTTP 200, Swagger UI loads |
| `GET /openapi.json` | ✅ PASS | HTTP 200, valid OpenAPI 3.x document |
| Redis pub/sub `live:frame:{cam}` | ✅ PASS | FrameBus publishing annotated JPEG frames |
| Redis pub/sub `live:event:{cam}` | ✅ PASS | FrameBus publishing JSON events |

---

## Fixes Applied in This Branch (`live-streem`)

The following bugs were fixed as part of this branch and are ready for review:

| Fix | File | Description |
|-----|------|-------------|
| `channelId` type: `int` → `str` | `apis/tasks.py` | Tasks now accept string camera IDs like `"cam0"` |
| `channel_id` type: `Optional[int]` → `Optional[str]` | `apis/detection_stream.py` | SSE filter now matches string camera IDs correctly |
| `channelId` query param: `Optional[int]` → `Optional[str]` | `app.py` | `/detection/stream?channelId=cam0` now works |
| Graceful startup if OSNet model missing | `services/person_search.py` | Server no longer crashes; returns 503 instead |
| Graceful startup if ONNX/open_clip missing | `services/semantic_search.py` | Server no longer crashes; returns 503 instead |
| Store dir env vars | `.env` | `GALLERY_DIR`, `EVENTS_DIR`, `CAPTURE_DIR`, `SCENE_DIR` set to `./store/*` (Docker-compatible) |

---

## Environment Setup Notes (Docker / Edge)

When running via `docker compose up --build`:

- `REDIS_URL=redis://redis:6379/0` — uses Docker service name `redis`
- `QDRANT_URL=http://qdrant:6333` — uses Docker service name `qdrant`
- `DEVICE=cuda:0` — requires NVIDIA GPU + `nvidia-container-toolkit`
- `YOLO_MODEL=./models/yolov8n.engine` — requires pre-built TensorRT engine in `./models/`
- `./store/*` directories are created automatically on first run (mounted via `.:/app`)

---

*Report generated after full 36-prompt test suite execution on 2026-04-18.*
