# Health / root

## Description

Lightweight identity endpoint for load balancers and integrators to confirm the Vision Pipeline API process is responding.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Returns service name and API version |
| GET | `/health` | Liveness probe |
| GET | `/metrics` | Prometheus text (only when `PROMETHEUS_ENABLED=true`) |

## curl

```bash
export BASE="http://localhost:9000"
curl -sS "${BASE}/"
curl -sS "${BASE}/health"
# Optional metrics (set PROMETHEUS_ENABLED=true first):
curl -sS "${BASE}/metrics"
```

**Example response (`GET /` or `/health`):**

```json
{
  "service": "Vision Pipeline API",
  "version": "2.0.0",
  "model_version": "unknown",
  "model_file": "yolov8n.pt",
  "model_loaded_at": null
}
```

`model_version`, `model_file`, and `model_loaded_at` are additive fields (QW-13); clients that ignore unknown keys remain compatible.
