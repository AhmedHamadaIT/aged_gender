# Person search (Re-ID image search)

## Description

Upload a **query image**; the service embeds it and searches the vector index (Qdrant) for similar identities (**OSNet** / Re-ID pipeline). Independent of `/detection/start` — uses `PersonSearchService` loaded at API startup.

Returns **503** if the Re-ID model is not loaded (check `REID_MODEL_PATH` and server logs).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/person_search/search` | Multipart: image file + optional `top_k` (form field, default 10) |
| GET | `/person_search/health` | `{ "model_loaded": bool, "status": "ok" \| "unavailable" }` |

## curl — health

```bash
export BASE="http://localhost:9000"
curl -sS "${BASE}/person_search/health"
```

## curl — search by image

```bash
curl -sS -X POST "${BASE}/person_search/search" \
  -F "file=@/path/to/query.jpg" \
  -F "top_k=5"
```

## Edge device note

- Use **multipart/form-data**; field name for the file is **`file`**.
- Large images: consider resizing on the edge to reduce uplink latency.
- Handle **503** with backoff when models are still loading or GPU memory is exhausted.
