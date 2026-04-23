# Health / root

## Description

Lightweight identity endpoint for load balancers and integrators to confirm the Vision Pipeline API process is responding.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Returns service name and API version |

## curl

```bash
export BASE="http://localhost:9000"
curl -sS "${BASE}/"
```

**Example response:**

```json
{"service":"Vision Pipeline API","version":"2.0.0"}
```
