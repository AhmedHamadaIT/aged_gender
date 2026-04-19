# Testing on edge devices — **curl-only** runbook

End-to-end checks for the Vision Pipeline API using **`curl` only** for HTTP (same style as production scripts and load balancers). **RTSP URLs are sent verbatim in JSON** bodies—no `jq`, no Python helpers.

SSH is only used for the tunnel to reach the edge. WebSocket streams are noted separately (not HTTP `curl`).

---

## 1. Prerequisites

| Item | Purpose |
|------|---------|
| `curl` | All API calls |
| `ssh` | Port forward to the edge API |
| On the **edge**: Docker Compose | `yolo-detect`, Redis, Qdrant |

Compose exposes the API on **9000** by default. Your tunnel may map another local port to that host—set `BASE` to match.

---

## 2. SSH tunnel (keep this terminal open)

Compose in this repo binds **uvicorn on `0.0.0.0:9000`** (`docker-compose.yml`). Forward **9000** on the edge host unless you know an ingress listens elsewhere (e.g. nginx on 8080).

```bash
ssh -L 9000:10.0.2.177:9000 -L 8081:127.0.0.1:80 -p 26060 eyego@34.47.247.221
```

```bash
export BASE="http://127.0.0.1:9000"
```

**Where to run `curl`:** `-L 9000:10.0.2.177:9000` listens on **9000 on the machine that runs the `ssh` client** (your laptop). Use `BASE=http://127.0.0.1:9000` there. If you run `curl` **on the edge** (e.g. `eyego@eyego-desktop`): use **`http://127.0.0.1:9000`** when Docker runs on that same box; use **`http://10.0.2.177:9000`** only if the API container runs on another host at that IP. **`Connection refused` on `:8080`** usually means nothing listens on 8080—this stack defaults to **9000**.

If something in your deployment exposes the API on **8080** instead, point the tunnel and `BASE` at that port, for example:

```bash
ssh -L 8080:10.0.2.177:8080 -L 8081:127.0.0.1:80 -p 26060 eyego@34.47.247.221
export BASE="http://127.0.0.1:8080"
```

---

## 3. Register cameras — production-style JSON (curl only)

Use a **here-document** so the JSON stays readable and you do not fight shell escaping inside RTSP URLs.

In **the same shell** as `curl`, set `BASE` (see section 2). If `BASE` is unset, `"${BASE}/cameras"` becomes `/cameras` without `http://`, and `curl` often returns **`curl: (3) URL using bad/illegal format or missing URL`**.

Put the RTSP URLs **inside the JSON** and use a **quoted** heredoc (`<<'EOF'`) so the shell does not expand `$` inside passwords. Edit `YOUR_PASSWORD` once in the body (do not commit real passwords to git). These paths match your NVR layout: **401**, **1101**, **201**.

```bash
curl -sS -X POST "${BASE}/cameras" \
  -H "Content-Type: application/json" \
  --data-binary @- <<'EOF'
{
  "cameras": [
    {
      "id": "cam401",
      "url": "rtsp://admin:YOUR_PASSWORD@10.0.3.71:554/Streaming/channels/401"
    },
    {
      "id": "cam1101",
      "url": "rtsp://admin:YOUR_PASSWORD@10.0.3.71:554/Streaming/channels/1101"
    },
    {
      "id": "cam201",
      "url": "rtsp://admin:YOUR_PASSWORD@10.0.3.71:554/Streaming/channels/201"
    }
  ]
}
EOF
```

**Heredoc:** the closing line must be exactly `EOF` alone (same word as in `<<'EOF'`), with no `}` or JSON on that line. Paste **one** `curl` command only—the first line ends with `\` alone, not `\"`. Duplicate `curl` lines or merged JSON (`"cameras": [m401"`, `},,`, `}OF`) break the request.

**Single-line `curl` (edge-friendly paste)** — avoids broken line continuations. On the edge, match the port Compose publishes (**9000** here). Replace `YOUR_PASSWORD`; if it contains `'`, use the heredoc block instead.

```bash
export BASE="http://127.0.0.1:9000"
curl -sS -X POST "${BASE}/cameras" -H "Content-Type: application/json" -d '{"cameras":[{"id":"cam401","url":"rtsp://admin:YOUR_PASSWORD@10.0.3.71:554/Streaming/channels/401"},{"id":"cam1101","url":"rtsp://admin:YOUR_PASSWORD@10.0.3.71:554/Streaming/channels/1101"},{"id":"cam201","url":"rtsp://admin:YOUR_PASSWORD@10.0.3.71:554/Streaming/channels/201"}]}'
```

If the API runs on another LAN host, use `export BASE="http://10.0.2.177:9000"` instead.

**Optional — same payload from a file** (good for CI or secrets injected into `cameras.json` at deploy time):

```bash
curl -sS -X POST "${BASE}/cameras" \
  -H "Content-Type: application/json" \
  --data-binary @cameras.json
```

**Optional — password from env** (only if you want zero password bytes in the command line history; still needs `BASE`). Use an **unquoted** delimiter so variables expand. If the password contains `$`, `` ` ``, or `\`, prefer the `<<'EOF'` literal block above instead.

```bash
export NVR_USER="admin"
export NVR_PASS="your_password_here"

curl -sS -X POST "${BASE}/cameras" \
  -H "Content-Type: application/json" \
  --data-binary @- <<EOF
{
  "cameras": [
    {"id": "cam401",  "url": "rtsp://${NVR_USER}:${NVR_PASS}@10.0.3.71:554/Streaming/channels/401"},
    {"id": "cam1101", "url": "rtsp://${NVR_USER}:${NVR_PASS}@10.0.3.71:554/Streaming/channels/1101"},
    {"id": "cam201",  "url": "rtsp://${NVR_USER}:${NVR_PASS}@10.0.3.71:554/Streaming/channels/201"}
  ]
}
EOF
```

The `yolo-detect` container must reach `10.0.3.71:554` on the network (same as production).

---

## 4. Docker on the edge

```bash
cd /path/to/ml-server
docker compose up -d --build
docker compose logs -f yolo-detect
```

**Compose v1 on older hosts:** if `docker compose` fails with `unknown command`, use **`docker-compose`** from the directory that contains **`docker-compose.yml`** (the `cd` path above). `docker-compose ...` without `cd` reports *no configuration file provided*.

Local health on the box:

```bash
ss -tlnp | grep 9000
curl -sS "http://127.0.0.1:9000/"
```

If `ss` shows **`ssh`** bound to `127.0.0.1:9000`, that port is an **SSH local forward** (`ssh -L 9000:...`), **not** the API container. `curl` to localhost then hits the tunnel (and may **reset** if the far end is wrong). Stop that SSH session, move the tunnel to another local port, or set **`BASE`** to the host that runs Docker (e.g. `http://10.0.2.177:9000`) if the network allows it.

Through the tunnel on your laptop, use `"${BASE}/"` instead.

---

## 5. Full **curl** scenario list (`$BASE`)

Use this section as a **checklist**: run **5.0** once for orientation, then **5.1**–**5.15** in order for smoke, or jump to a group for regression.

### 5.0 Endpoint map (all HTTP routes)

| Method | Path | Notes |
|--------|------|--------|
| GET | `/` | Health JSON |
| POST | `/cameras` | Body: `{ "cameras": [ { "id", "url" } ] }`, min 1 camera |
| GET | `/cameras` | List registered cameras |
| DELETE | `/cameras/{cam_id}` | 404 if unknown |
| POST | `/api/tasks` | `algorithmType`: `CROSS_LINE`, `MASK_HAIRNET_CHEF_HAT`, `CASHIER_BOX_OPEN` |
| GET | `/api/tasks` | List tasks |
| GET | `/api/tasks/{task_id}` | 404 if missing |
| PUT | `/api/tasks/{task_id}` | Body `taskId` must match URL; 404 if missing |
| DELETE | `/api/tasks/{task_id}` | 404 if missing |
| POST | `/detection/start` | Optional `?camera_id=` |
| POST | `/detection/stop` | Optional `?camera_id=`; no id = stop all running |
| POST | `/detection/stop/all` | Same as stop-all |
| GET | `/detection/status` | Camera / process status |
| GET | `/detection/stream` | SSE; query: `taskId`, `taskName`, `eventType`, `channelId` (AND) |
| GET | `/cashier/status` | Latest structured event per camera |
| GET | `/cashier/events` | Query: `severity`, `case_id`, `camera_id`, `limit`, `offset` |
| DELETE | `/cashier/events` | Clear log (+ Redis backing when configured) |
| GET | `/cashier/evidence` | Query: `severity`, `case_id`, `limit` |
| GET | `/cashier/evidence/{file_path}` | Relative path under evidence dir; 403 traversal, 404 missing |
| GET | `/cashier/zones` | Zone config from `CASHIER_CONFIG` |
| POST | `/cashier/zones` | Partial update JSON body |
| POST | `/cashier/zones/reset` | Defaults |
| GET | `/cashier/stream/{camera_id}` | SSE all events |
| GET | `/cashier/stream/{camera_id}/only` | SSE alerts only (no `frame`) |
| GET | `/cashier/media/{camera_id}/latest/jpg` | 404 if no files |
| GET | `/cashier/media/{camera_id}/latest/gif` | 404 if no GIF |
| GET | `/cashier/media/{camera_id}/event/{event_id}/jpg` | Per-event JPEG |
| GET | `/cashier/media/{camera_id}/event/{event_id}/gif` | Per-event GIF |
| GET | `/cashier/media/{camera_id}/drawer_count` | JSON counter |
| GET | `/person_search/health` | Model loaded flag |
| POST | `/person_search/search` | `multipart`: `file` (required), `top_k` |
| GET | `/semantic_search/health` | Model ready flag |
| POST | `/semantic_search/search` | `multipart`: `text_query` and/or `file`, `top_k` |
| GET | `/openapi.json`, `/docs` | OpenAPI / Swagger UI |

WebSockets (not curl): `WS /cameras/{camera_id}/live`, `WS /cameras/{camera_id}/events` — see section 6.

---

### 5.1 Health

```bash
curl -sS "${BASE}/"
```

---

### 5.2 OpenAPI / docs

```bash
curl -sS -o /dev/null -w "%{http_code}\n" "${BASE}/openapi.json"
curl -sS -o /dev/null -w "%{http_code}\n" "${BASE}/docs"
```

---

### 5.3 Cameras — list, validation, delete errors

**List (after registration):**

```bash
curl -sS "${BASE}/cameras"
```

**Validation — empty `cameras` array** (expect **422**):

```bash
curl -sS -w "\nHTTP %{http_code}\n" -X POST "${BASE}/cameras" \
  -H "Content-Type: application/json" \
  -d '{"cameras":[]}'
```

**Validation — invalid JSON** (expect **422**):

```bash
curl -sS -w "\nHTTP %{http_code}\n" -X POST "${BASE}/cameras" \
  -H "Content-Type: application/json" \
  -d '{"cameras":[}'
```

**Delete unknown camera** (expect **404**):

```bash
curl -sS -w "\nHTTP %{http_code}\n" -X DELETE "${BASE}/cameras/no_such_camera"
```

**Upsert behavior:** posting again with the same `id` **overwrites** the URL for that id (no error).

---

### 5.4 Tasks — create variants, read, validation errors

Supported `algorithmType` values: **`CROSS_LINE`**, **`MASK_HAIRNET_CHEF_HAT`**, **`CASHIER_BOX_OPEN`**.

**CROSS_LINE** (edit `areaPosition` and lines for your site):

```bash
curl -sS -X POST "${BASE}/api/tasks" \
  -H "Content-Type: application/json" \
  --data-binary @- <<'EOF'
{
  "taskId": 1,
  "taskName": "line_cam401",
  "algorithmType": "CROSS_LINE",
  "channelId": "cam401",
  "enable": true,
  "threshold": 50,
  "areaPosition": "[]",
  "detailConfig": {
    "enableAttrDetect": false,
    "enableReid": false
  },
  "validWeekday": ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"],
  "validStartTime": 0,
  "validEndTime": 86400000
}
EOF
```

**MASK_HAIRNET_CHEF_HAT** (example `detailConfig.alarmType`):

```bash
curl -sS -X POST "${BASE}/api/tasks" \
  -H "Content-Type: application/json" \
  --data-binary @- <<'EOF'
{
  "taskId": 2,
  "taskName": "ppe_cam1101",
  "algorithmType": "MASK_HAIRNET_CHEF_HAT",
  "channelId": "cam1101",
  "enable": true,
  "threshold": 50,
  "areaPosition": "[]",
  "detailConfig": {
    "alarmType": ["NO_MASK", "NO_HAIRNET"]
  },
  "validWeekday": ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"],
  "validStartTime": 0,
  "validEndTime": 86400000
}
EOF
```

**CASHIER_BOX_OPEN** (cashier drawer monitor):

```bash
curl -sS -X POST "${BASE}/api/tasks" \
  -H "Content-Type: application/json" \
  --data-binary @- <<'EOF'
{
  "taskId": 3,
  "taskName": "cashier_cam201",
  "algorithmType": "CASHIER_BOX_OPEN",
  "channelId": "cam201",
  "enable": true,
  "threshold": 50,
  "areaPosition": "[]",
  "detailConfig": {
    "drawerOpenLimit": 30,
    "serviceWaitLimit": 30,
    "enableStaffList": false,
    "staffIds": []
  },
  "validWeekday": ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"],
  "validStartTime": 0,
  "validEndTime": 86400000
}
EOF
```

**Unsupported `algorithmType`** (expect **400**):

```bash
curl -sS -w "\nHTTP %{http_code}\n" -X POST "${BASE}/api/tasks" \
  -H "Content-Type: application/json" \
  --data-binary @- <<'EOF'
{
  "taskId": 99,
  "taskName": "bad_algo",
  "algorithmType": "UNKNOWN_ALGO",
  "channelId": "cam401",
  "enable": true,
  "threshold": 50,
  "areaPosition": "[]",
  "detailConfig": {},
  "validWeekday": ["MONDAY"],
  "validStartTime": 0,
  "validEndTime": 86400000
}
EOF
```

**List / get one:**

```bash
curl -sS "${BASE}/api/tasks"
curl -sS "${BASE}/api/tasks/1"
```

**GET missing task** (expect **404**):

```bash
curl -sS -w "\nHTTP %{http_code}\n" "${BASE}/api/tasks/99998"
```

**PUT — `taskId` in body must match URL** (expect **400**):

```bash
curl -sS -w "\nHTTP %{http_code}\n" -X PUT "${BASE}/api/tasks/1" \
  -H "Content-Type: application/json" \
  --data-binary @- <<'EOF'
{
  "taskId": 2,
  "taskName": "mismatch",
  "algorithmType": "CROSS_LINE",
  "channelId": "cam401",
  "enable": true,
  "threshold": 50,
  "areaPosition": "[]",
  "detailConfig": {"enableAttrDetect": false, "enableReid": false},
  "validWeekday": ["MONDAY"],
  "validStartTime": 0,
  "validEndTime": 86400000
}
EOF
```

**PUT non-existent task** (expect **404**):

```bash
curl -sS -w "\nHTTP %{http_code}\n" -X PUT "${BASE}/api/tasks/99999" \
  -H "Content-Type: application/json" \
  --data-binary @- <<'EOF'
{
  "taskId": 99999,
  "taskName": "ghost",
  "algorithmType": "CROSS_LINE",
  "channelId": "cam401",
  "enable": true,
  "threshold": 50,
  "areaPosition": "[]",
  "detailConfig": {"enableAttrDetect": false, "enableReid": false},
  "validWeekday": ["MONDAY"],
  "validStartTime": 0,
  "validEndTime": 86400000
}
EOF
```

**DELETE missing task** (expect **404**):

```bash
curl -sS -w "\nHTTP %{http_code}\n" -X DELETE "${BASE}/api/tasks/99997"
```

**Disabled task:** set `"enable": false` on a task; `POST /detection/start` only starts **enabled** tasks. Use a separate task id to verify workers are not spawned for disabled rows.

---

### 5.5 Detection — start / status / stop (happy + error cases)

Prerequisites for a successful start: at least one **enabled** task and matching **POST /cameras** entries for every `channelId` in those tasks.

**Start all cameras that have enabled tasks:**

```bash
curl -sS -X POST "${BASE}/detection/start"
```

**Start one camera** (only tasks whose `channelId` matches):

```bash
curl -sS -X POST "${BASE}/detection/start?camera_id=cam401"
```

**Status:**

```bash
curl -sS "${BASE}/detection/status"
```

**Stop one / stop all:**

```bash
curl -sS -X POST "${BASE}/detection/stop?camera_id=cam401"
curl -sS -X POST "${BASE}/detection/stop"
curl -sS -X POST "${BASE}/detection/stop/all"
```

**Errors (run when appropriate — do not assume ordering):**

| Case | Expect |
|------|--------|
| `POST /detection/start` with **no tasks** | **400** — configure tasks first |
| `POST /detection/start` with **no cameras** | **400** — configure cameras first |
| `POST /detection/start?camera_id=unknown` | **404** — no enabled task for that channel |
| `POST /detection/start` while that camera **already running** | **409** |
| `POST /detection/stop` when **nothing** is running | **409** |
| `POST /detection/stop?camera_id=not_running` | **409** |

Examples:

```bash
# After stopping everything and deleting all tasks (adjust to your state):
curl -sS -w "\nHTTP %{http_code}\n" -X POST "${BASE}/detection/start"

# Unknown camera id for enabled tasks:
curl -sS -w "\nHTTP %{http_code}\n" -X POST "${BASE}/detection/start?camera_id=no_such_cam"
```

---

### 5.6 Detection SSE — filters and keepalive

**Unfiltered broadcast:**

```bash
curl -sSN --max-time 15 "${BASE}/detection/stream"
```

**Single-filter examples** (server uses **AND** when multiple params are set):

```bash
curl -sSN --max-time 15 "${BASE}/detection/stream?eventType=CROSS_LINE&channelId=cam401"
curl -sSN --max-time 15 "${BASE}/detection/stream?taskId=1"
curl -sSN --max-time 15 "${BASE}/detection/stream?taskName=line_cam401"
```

**Combined filters:**

```bash
curl -sSN --max-time 15 \
  "${BASE}/detection/stream?taskId=1&eventType=CROSS_LINE&channelId=cam401"
```

Idle streams emit SSE comments `: ping` about every 30 seconds.

---

### 5.7 Cashier — HTTP (status, events, evidence, zones)

```bash
curl -sS "${BASE}/cashier/status"
```

**Events — pagination and filters** (`severity`: NORMAL | ALERT | CRITICAL; `case_id`: N1–N6 / A1–A7 style per your data):

```bash
curl -sS "${BASE}/cashier/events?limit=20&offset=0"
curl -sS "${BASE}/cashier/events?severity=ALERT&limit=50"
curl -sS "${BASE}/cashier/events?case_id=N3&camera_id=cam201&limit=10&offset=0"
```

**Clear event log:**

```bash
curl -sS -X DELETE "${BASE}/cashier/events"
```

**Evidence listing** (optional `severity`, `case_id`, `limit`):

```bash
curl -sS "${BASE}/cashier/evidence?limit=50"
curl -sS "${BASE}/cashier/evidence?severity=alert&case_id=N3&limit=20"
```

**Download one evidence file** — use a path returned by `GET /cashier/evidence` (replace `RELATIVE_PATH.jpg`):

```bash
curl -sS -o /tmp/evidence.jpg -w "\nHTTP %{http_code}\n" \
  "${BASE}/cashier/evidence/RELATIVE_PATH.jpg"
```

**Path traversal guard** (expect **403**):

```bash
curl -sS -w "\nHTTP %{http_code}\n" "${BASE}/cashier/evidence/../../../etc/passwd"
```

**Zones — read:**

```bash
curl -sS "${BASE}/cashier/zones"
```

**Zones — partial update** (omit fields you do not want to change):

```bash
curl -sS -X POST "${BASE}/cashier/zones" \
  -H "Content-Type: application/json" \
  --data-binary @- <<'EOF'
{
  "ROI_CASHIER": {
    "shape": "rectangle",
    "points": [{"x": 0.0, "y": 0.0}, {"x": 0.45, "y": 1.0}],
    "active": true
  },
  "ROI_CUSTOMER": {
    "shape": "rectangle",
    "points": [{"x": 0.45, "y": 0.0}, {"x": 1.0, "y": 1.0}],
    "active": true
  },
  "thresholds": {"drawer_open_max_seconds": 20},
  "detail_config": {"drawerOpenLimit": 30, "serviceWaitLimit": 30},
  "detection_threshold": 50
}
EOF
```

**Zones — reset to defaults:**

```bash
curl -sS -X POST "${BASE}/cashier/zones/reset"
```

---

### 5.8 Cashier — SSE streams (all vs alerts-only)

```bash
curl -sS -N --max-time 15 "${BASE}/cashier/stream/cam401"
curl -sS -N --max-time 15 "${BASE}/cashier/stream/cam401/only"
```

`/only` suppresses `frame` events; `alert` and `gif_ready` still flow.

---

### 5.9 Cashier — media and drawer count

```bash
curl -sS -w "\nHTTP %{http_code}\n" "${BASE}/cashier/media/cam401/latest/jpg"
curl -sS -w "\nHTTP %{http_code}\n" "${BASE}/cashier/media/cam401/latest/gif"
curl -sS "${BASE}/cashier/media/cam401/drawer_count"
```

**Per-event media** (replace `EVENT_ID` with an id from cashier events / logs; **404** if no matching file):

```bash
curl -sS -o /tmp/event.jpg -w "\nHTTP %{http_code}\n" \
  "${BASE}/cashier/media/cam401/event/EVENT_ID/jpg"
curl -sS -o /tmp/event.gif -w "\nHTTP %{http_code}\n" \
  "${BASE}/cashier/media/cam401/event/EVENT_ID/gif"
```

---

### 5.10 ReID / person search

**Health:**

```bash
curl -sS -w "\nHTTP %{http_code}\n" "${BASE}/person_search/health"
```

**Search** (requires a real image path; **503** if model not loaded; **400** if not an image):

```bash
curl -sS -w "\nHTTP %{http_code}\n" -X POST "${BASE}/person_search/search" \
  -F "file=@/path/to/query.jpg" \
  -F "top_k=5"
```

**Non-image upload** (expect **400**):

```bash
curl -sS -w "\nHTTP %{http_code}\n" -X POST "${BASE}/person_search/search" \
  -F "file=@/etc/hosts;type=text/plain" \
  -F "top_k=3"
```

---

### 5.11 Semantic search

**Health:**

```bash
curl -sS -w "\nHTTP %{http_code}\n" "${BASE}/semantic_search/health"
```

**Text only:**

```bash
curl -sS -w "\nHTTP %{http_code}\n" -X POST "${BASE}/semantic_search/search" \
  -F "text_query=person in red shirt" \
  -F "top_k=5"
```

**Image only:**

```bash
curl -sS -w "\nHTTP %{http_code}\n" -X POST "${BASE}/semantic_search/search" \
  -F "file=@/path/to/query.jpg" \
  -F "top_k=5"
```

**Neither text nor file** (expect **400**):

```bash
curl -sS -w "\nHTTP %{http_code}\n" -X POST "${BASE}/semantic_search/search" \
  -F "top_k=5"
```

**Empty text** (expect **400**):

```bash
curl -sS -w "\nHTTP %{http_code}\n" -X POST "${BASE}/semantic_search/search" \
  -F "text_query=   " \
  -F "top_k=5"
```

**Non-image file with image-only intent** (expect **400**):

```bash
curl -sS -w "\nHTTP %{http_code}\n" -X POST "${BASE}/semantic_search/search" \
  -F "file=@/etc/hosts;type=text/plain" \
  -F "top_k=3"
```

When `text_query` is non-empty, the handler uses **text search** (image part ignored for that request).

---

### 5.12 Cleanup (order: stop detection → delete tasks → delete cameras)

```bash
curl -sS -X POST "${BASE}/detection/stop/all"
curl -sS -X DELETE "${BASE}/api/tasks/1"
curl -sS -X DELETE "${BASE}/api/tasks/2"
curl -sS -X DELETE "${BASE}/api/tasks/3"
curl -sS -X DELETE "${BASE}/cameras/cam401"
curl -sS -X DELETE "${BASE}/cameras/cam1101"
curl -sS -X DELETE "${BASE}/cameras/cam201"
```

Adjust task ids to match what you created.

---

## 6. WebSockets (not `curl`)

`GET /detection/stream` and `GET /cashier/stream/...` cover many **SSE** cases with **curl**. Binary **`WS /cameras/{id}/live`** and **`WS /cameras/{id}/events`** need a WebSocket client (e.g. `websocat`), not HTTP curl.

Example (after installing `websocat`):

```bash
websocat -n -t "${BASE/http/ws}/cameras/cam401/live"
websocat "${BASE/http/ws}/cameras/cam401/events"
```

Replace `http` with `ws` in `BASE` (e.g. `ws://127.0.0.1:9000/...`).

---

## 7. Quick checklist (curl-only HTTP)

| Step | Action |
|------|--------|
| Tunnel | `ssh -L ...` then `export BASE=...` |
| Stack | `docker compose up -d --build` on edge |
| Map | Skim **5.0** — know every route |
| Cameras | `POST /cameras`, `GET /cameras`, validation + delete 404 (**5.3**) |
| Tasks | `POST /api/tasks` for each **algorithmType** you use (**5.4**) |
| Detection | `POST /detection/start`, errors (**5.5**), `GET /detection/status` |
| SSE | `GET /detection/stream` with filters (**5.6**) |
| Cashier | status, events, evidence, zones, streams, media (**5.7**–**5.9**) |
| Search | health + multipart cases (**5.10**–**5.11**) |
| Stop | `POST /detection/stop` or `/detection/stop/all` |
| Clean | `DELETE` tasks + cameras (**5.12**) |

---

## 8. RTSP reference (your channels)

| Camera `id` | NVR path |
|-------------|----------|
| `cam401` | `/Streaming/channels/401` |
| `cam1101` | `/Streaming/channels/1101` |
| `cam201` | `/Streaming/channels/201` |

Full form (same as in JSON `url`):

`rtsp://admin:PASSWORD@10.0.3.71:554/Streaming/channels/<channel>`

---

## 9. Troubleshooting

| Symptom | Check |
|---------|--------|
| `curl: (3) URL using bad/illegal format` | **`export BASE=...` in this shell** (see section 2). Also: heredoc closing `EOF` alone on its line; **full** JSON through final `]` `}`. |
| `curl: (7) ... Connection refused` | Wrong host/port: this repo’s API is **9000** by default (`docker-compose.yml`). On the edge, try `http://127.0.0.1:9000` (same box as Docker) or `http://10.0.2.177:9000`. **`127.0.0.1:8080`** is only valid on the **laptop** if your `ssh -L` forwards local 8080. Confirm `docker compose ps` and published ports. |
| `curl: (56) ... Connection reset by peer` | Often **`ss -tlnp`** shows **`ssh`** on **9000**: a **tunnel** is bound there, not uvicorn (see section 4). Or **yolo-detect** crash/restart — check logs from the compose directory. Run **`curl -v "${BASE}/"`** to confirm. |
| Connection refused (other) | Tunnel up? `BASE` port matches `-L`? API listening on that host/port? |
| RTSP errors in container logs | Reachability of `10.0.3.71:554`, credentials, channel id |
| No SSE | Redis + detection started |
| **400** on `/detection/start` | No enabled tasks or no cameras — see **5.5** |
| **409** on start/stop | Already running / nothing running — see **5.5** |
| **503** on search | Models / ONNX / Qdrant / `REID_MODEL_PATH` on edge |
| **403** on `/cashier/evidence/...` | Path outside evidence dir (traversal) |
| **500** on `/cashier/zones` | `CASHIER_CONFIG` read/write failure on disk |

---

*Runbook uses **curl** for HTTP/SSE and multipart. Replace passwords and paths for your environment; do not commit secrets to git.*
