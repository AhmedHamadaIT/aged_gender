# Service documentation (Vision Pipeline API)

This folder describes each HTTP/WebSocket surface of the **Vision Pipeline API** (FastAPI app in `app.py`): purpose, endpoints, and **curl** examples. For a single linear checklist from an edge gateway or NVR, start with **[edge_device_e2e.md](./edge_device_e2e.md)**.

| Document | What it covers |
|----------|----------------|
| [edge_device_e2e.md](./edge_device_e2e.md) | End-to-end steps for edge devices / integrators (order of calls, networking, streaming) |
| [health.md](./health.md) | Root health / service identity |
| [cameras.md](./cameras.md) | Camera registry (`POST/GET/DELETE /cameras`) |
| [tasks.md](./tasks.md) | Task registry (`/api/tasks`) — all `algorithmType` values |
| [cross_line.md](./cross_line.md) | **CROSS_LINE** algorithm — lines, schedule, SSE filters, curl |
| [mask_hairnet_chef_hat.md](./mask_hairnet_chef_hat.md) | **MASK_HAIRNET_CHEF_HAT** algorithm — zones, `alarmType`, SSE filters, curl |
| [phone_usage.md](./phone_usage.md) | **PHONE_USAGE** algorithm — polygon zones, schedule, SSE filters, curl |
| [ml_image_v2.md](./ml_image_v2.md) | **ML Image Contract V2** — `evidence` object shape, env vars, disk paths, `PIPELINE_IMAGE_MODE`, SSH + JSONL for all task types |
| [detection.md](./detection.md) | Start/stop pipeline, status, SSE event stream, stream metrics, worker lifecycle |
| [websockets.md](./websockets.md) | Live JPEG frames and per-camera events over WebSocket; camera-ID validation |
| [cashier.md](./cashier.md) | Cashier monitor (`/cashier/*`) for `CASHIER_BOX_OPEN` |
| [person_search.md](./person_search.md) | Re-ID image search |
| [semantic_search.md](./semantic_search.md) | Text/image semantic search |

For the **internal process model, annotation flow, RTSP transport override API, zombie-process reaping, and E2E test guide** see [`docs/pipeline-architecture.md`](../docs/pipeline-architecture.md).

## Defaults

- **Default bind** (from `app.py`): `uvicorn app:app --host 0.0.0.0 --port 9000`
- **Base URL in examples**: `http://<ML_SERVER_HOST>:9000` — replace with your server IP or hostname.
- **Redis**: Set `REDIS_URL` (default `redis://localhost:6379/0`) on the ML server for WebSocket live fan-out (`/cameras/.../live`, `/cameras/.../events`, `/tasks/.../live`) and for cashier multi-process sync. Without Redis, some live features degrade gracefully.
- **Local dev without Redis**: set `REDIS_CONNECT_RETRIES=1` so FrameBus fails fast on the Redis publish path and continues processing.

## OpenAPI

Interactive docs are served by FastAPI at `/docs` (Swagger UI) and `/redoc` when the server is running.
