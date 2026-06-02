# Cashier monitor service (`/cashier`)

## Description

HTTP API for **cashier drawer / zone monitoring** when you run a task with `algorithmType` **`CASHIER_BOX_OPEN`**. It exposes live status, paginated structured events, zone configuration, evidence files, and **SSE** streams per camera.

When **`REDIS_URL`** is set, status and events are mirrored to Redis so **forked task workers** and the API process stay consistent.

Evidence directory default: `./evidence/cashier` (override with `CASHIER_EVIDENCE_DIR`). Zone file default: `./config/cashier_zones.yaml` (override with `CASHIER_CONFIG`).

**ML Image Contract V2:** the Eyego **`data`** block may include **`data.evidence`** with **`captureImage`** and **`sceneImage`** as structured image objects (in addition to `captureUrl` / `sceneId` / `sceneUrl`). Per-frame **structured** events on `GET /detection/stream` use top-level `evidence` with V2 objects; when only one JPEG exists, `sceneImage` may be `{ "url": null, "type": "scene", "status": "not_available" }`. Details: [ml_image_v2.md](./ml_image_v2.md).

**Rule engine:** `CashierService._evaluate()` uses a declarative priority table (`_build_transition_table`) preserving **N1–N6** / **A1–A7** behavior. Tests: `tests/unit/test_cashier_parametrized.py`. Evidence listing supports optional `since` and `camera_id` query filters (QW-15).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/cashier/status` | Latest structured cashier snapshot per `camera_id` |
| GET | `/cashier/events` | Paginated event log (`severity`, `case_id`, `camera_id`, `limit`, `offset`) |
| DELETE | `/cashier/events` | Clear in-memory log and Redis-backed cashier lists |
| GET | `/cashier/evidence` | List saved JPEG evidence files |
| GET | `/cashier/evidence/{file_path}` | Download one JPEG by relative path |
| GET | `/cashier/zones` | Current zone YAML/JSON config |
| POST | `/cashier/zones` | Update zones/thresholds (picked up on reload cycle, default ~60s) |
| POST | `/cashier/zones/reset` | Restore default split zones |
| GET | `/cashier/stream/{camera_id}` | SSE: all events (`connected`, `frame`, `alert`, `gif_ready`) |
| GET | `/cashier/stream/{camera_id}/only` | SSE: alerts only (suppresses `frame`) |
| GET | `/cashier/media/{camera_id}/latest/jpg` | Latest evidence JPEG |
| GET | `/cashier/media/{camera_id}/latest/gif` | Latest evidence GIF |
| GET | `/cashier/media/{camera_id}/event/{event_id}/jpg` | JPEG for event id |
| GET | `/cashier/media/{camera_id}/event/{event_id}/gif` | GIF for event id |
| GET | `/cashier/media/{camera_id}/drawer_count` | Cumulative drawer-open edge count from persisted totals |

## curl — live status

```bash
export BASE="http://localhost:9000"
curl -sS "${BASE}/cashier/status"
```

## curl — events (newest-first, paginated)

```bash
curl -sS "${BASE}/cashier/events?limit=50&offset=0"
```

Filter examples:

```bash
curl -sS "${BASE}/cashier/events?camera_id=1&severity=ALERT"
curl -sS "${BASE}/cashier/events?case_id=N3"
```

## curl — clear event log

```bash
curl -sS -X DELETE "${BASE}/cashier/events"
```

## curl — list evidence files

```bash
curl -sS "${BASE}/cashier/evidence?limit=20"
```

## curl — download evidence file

Use a `path` value returned from `GET /cashier/evidence` (URL-encoded):

```bash
curl -sS -o evidence.jpg "${BASE}/cashier/evidence/ALERT%2Fsome_subfolder%2Ffile.jpg"
```

## curl — get zone config

```bash
curl -sS "${BASE}/cashier/zones"
```

## curl — update zones (partial)

```bash
curl -sS -X POST "${BASE}/cashier/zones" \
  -H "Content-Type: application/json" \
  -d '{
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
    "thresholds": {"drawer_open_max_seconds": 20}
  }'
```

## curl — reset zones

```bash
curl -sS -X POST "${BASE}/cashier/zones/reset"
```

## curl — SSE (all events)

```bash
curl -sS -N "${BASE}/cashier/stream/1"
```

## curl — SSE alerts only

```bash
curl -sS -N "${BASE}/cashier/stream/1/only"
```

## curl — latest media

```bash
curl -sS -o latest.jpg "${BASE}/cashier/media/1/latest/jpg"
curl -sS -o latest.gif "${BASE}/cashier/media/1/latest/gif"
```

## curl — drawer count

```bash
curl -sS "${BASE}/cashier/media/1/drawer_count"
```

## Edge device E2E (cashier)

1. Register camera with id e.g. `"1"` — [cameras.md](./cameras.md).
2. `POST /api/tasks` with `"algorithmType": "CASHIER_BOX_OPEN"` and `"channelId": "1"` (plus `detailConfig` limits as needed).
3. `POST /detection/start`.
4. Poll `GET /cashier/status` or open `GET /cashier/stream/1` for live updates.
5. Optional: tune `POST /cashier/zones`; wait for worker reload window.
6. `POST /detection/stop` when finished.

Crossing-line events still appear on `GET /detection/stream` when applicable; cashier-specific UX typically uses `/cashier/*`.

## Further reading

Repository docs: `docs/CASHIER_BOX_OPEN.md`, `docs/API_USAGE.md` (section 10).
