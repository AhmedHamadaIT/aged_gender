# Semantic search (text / image → similar images)

## Description

Queries a **semantic (CLIP-style) embedding** index in Qdrant. The route accepts optional `text_query` and optional image **`file`** — at least one must be provided. If **`text_query` is non-empty after trimming**, the server runs **text search** and ignores an uploaded file; if text is empty and a file is present, it runs **image search**.

Returns **503** if ONNX / OpenCLIP models failed to load.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/semantic_search/search` | Form: `text_query` (optional), `file` (optional image), `top_k` (default 10) |
| GET | `/semantic_search/health` | Model readiness |

## curl — health

```bash
export BASE="http://localhost:9000"
curl -sS "${BASE}/semantic_search/health"
```

## curl — search by text only

```bash
curl -sS -X POST "${BASE}/semantic_search/search" \
  -F "text_query=person wearing red jacket" \
  -F "top_k=10"
```

## curl — search by image only

```bash
curl -sS -X POST "${BASE}/semantic_search/search" \
  -F "file=@/path/to/query.jpg" \
  -F "top_k=10"
```

## curl — both fields sent (text wins)

If you send both, **only `text_query` is used** (same as “text only”):

```bash
curl -sS -X POST "${BASE}/semantic_search/search" \
  -F "text_query=shopping cart" \
  -F "file=@/path/to/scene.jpg" \
  -F "top_k=5"
```

## Edge device note

- Prefer **clear, short text queries** for latency and recall.
- For image-only queries, enforce JPEG/PNG and reasonable resolution before upload.
