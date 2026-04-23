# ML Image Contract V2 — evidence, disk layout, env, SSH / logs

This document describes how **image metadata** and **on-disk files** are exposed in API responses, where data is written, and how to inspect **JSONL** and **application logs** over **SSH** for every **task** type.

## Environment variables (images + pipeline)

| Variable | Role |
|----------|------|
| `PUBLIC_ML_BASE_URL` | Public origin for V2 `url` fields (no trailing slash). Example: `https://ml.example.com`. If unset, `url` is path-only: `/evidence/...`. |
| `CAPTURE_DIR` | Root for **person crop** JPEGs (default `/local/storage/captures`). |
| `SCENE_DIR` | Root for **full frame** scene JPEGs (default `/local/storage/scenes`). |
| `EVENTS_DIR` | Root for **task JSONL** append logs (default `/local/storage/events`). |
| `CASHIER_EVIDENCE_DIR` | Cashier tree (default `./evidence/cashier`). |
| `PIPELINE_IMAGE_MODE` | `full` (default) \| `light` \| `none` — see [Batch pipeline `images`](#batch-pipeline-images-pipelinepy). |
| `OUTPUT_DIR` | Per-camera JPEG saves when `SAVE_OUTPUT` is true in batch `CameraPipeline` (default `./outputs`). |

V2 public URL rule: **`{PUBLIC_ML_BASE_URL}/evidence/{relative_path}`** — the relative path has **no** algorithm name in the path segment (uniqueness is in the filename).

## Standard V2 image object

Anywhere an image is described in the **structured** form, the value is an object (not a bare string path):

```json
{
  "url": "https://ml.example.com/evidence/2026-04-22/cam-1_abc123_01a2b3c4.jpg",
  "path": "2026-04-22/cam-1_abc123_01a2b3c4.jpg",
  "type": "capture",
  "format": "image/jpeg",
  "timestamp": "2026-04-22T14:33:21Z"
}
```

- **`type`**: `capture` (crop) or `scene` (full frame), as emitted by line/PPE/phone tasks; cashier uses the same object under nested `evidence` / `data.evidence` keys.
- On-disk file layout for **line / PPE / phone** tasks:  
  `CAPTURE_DIR/YYYY-MM-DD/{camera_id}_{event_id}_{uuid8}.jpg` and the same date folder under `SCENE_DIR` for the scene file (separate `uuid8` per image).

## Task types — `evidence` in SSE / WebSocket / JSONL

| `eventType` | `evidence.captureImage` | `evidence.sceneImage` | JSONL file |
|-------------|-------------------------|------------------------|------------|
| `CROSS_LINE` | V2 object (`capture`) | V2 object (`scene`) | `$EVENTS_DIR/task_<taskId>.jsonl` |
| `MASK_HAIRNET_CHEF_HAT` | V2 object | V2 object | same |
| `PHONE_USAGE` | V2 object | V2 object | same |
| `CASHIER_BOX_OPEN` (multiplexed SSE) | V2 `capture` when a frame is saved, **or** top-level `evidence` on structured event | V2: **either** full `build_image` for `scene` **inside `data`**, or `sceneImage` with `status: "not_available"` when only one file exists (see below) | `$EVENTS_DIR/task_<taskId>.jsonl` |

**Cashier** additionally embeds:

- Under **`data`** (Eyego): `captureUrl` / `sceneUrl` (integration), and **`data.evidence`** with **`captureImage`** and **`sceneImage`** as V2 image objects (dated paths like `2026-04-22/CASHIER_BOX_OPEN_….jpg` in `path`).
- On **structured** per-frame events (same channel as other tasks), optional top-level **`evidence`**: `captureImage` = V2 object from saved path; `sceneImage` = `{ "url": null, "type": "scene", "status": "not_available" }` when there is no separate scene file (single JPEG reused for V1-style consumers is no longer emitted as raw strings; use the V2 fields).

## SSH — watch task JSONL (all cases)

Set `ML_HOST` and `TASK_ID` to your edge device and task. Default log root is `/local/storage/events` unless `EVENTS_DIR` is overridden.

```bash
export ML_USER=ubuntu
export ML_HOST=192.168.1.50
export EVENTS_DIR=/local/storage/events   # or your custom EVENTS_DIR

# Live tail one task (cross-line, PPE, phone, or cashier)
ssh ${ML_USER}@${ML_HOST} "tail -f ${EVENTS_DIR}/task_10.jsonl"
```

**Examples by algorithm** (use the real `taskId` from `GET /api/tasks`):

| Case | Example command |
|------|------------------|
| Cross-line | `tail -f $EVENTS_DIR/task_10.jsonl` |
| Mask / PPE | `tail -f $EVENTS_DIR/task_20.jsonl` |
| Phone usage | `tail -f $EVENTS_DIR/task_40.jsonl` |
| Cashier | `tail -f $EVENTS_DIR/task_30.jsonl` |

One **JSON object per line**; each line matches the same envelope as one SSE `data:` event for that task (pretty-printing is only for human inspection — SSE sends minified JSON).

**Filter last line with `jq` (evidence only):**

```bash
ssh ${ML_USER}@${ML_HOST} "tail -n1 ${EVENTS_DIR}/task_10.jsonl" | jq '.evidence'
```

## SSH — application / uvicorn logs

Log file location depends on your deployment (systemd, docker, or manual `uvicorn`). Common patterns:

```bash
# If logging to a file under the repo
ssh ${ML_USER}@${ML_HOST} "tail -f /path/to/ml-server/logger/app.log"

# Journal on systemd
ssh ${ML_USER}@${ML_HOST} "journalctl -u ml-server -f"
```

## SSH — cURL against localhost on the device

```bash
ssh ${ML_USER}@${ML_HOST} 'curl -sN http://127.0.0.1:9000/detection/stream?eventType=CROSS_LINE\&taskId=10 | head -n 5'
```

## Cashier evidence **files** (HTTP + disk)

- **List / download** via `GET /cashier/evidence` and `GET /cashier/evidence/{path}` (see [cashier.md](./cashier.md)).
- On disk: under `CASHIER_EVIDENCE_DIR` with subfolders by case/severity; filenames are not the same as the V2 `path` in JSON (V2 `path` is a logical CDN key; the server still writes real files to the evidence tree). Use **`data.evidence.*.path`** and **`captureUrl` / `sceneUrl`** for integration, or download via `/cashier/evidence/...` for debugging.

## Batch pipeline images (`pipeline.py`)

Registry/batch runs emit each frame (optional) as:

- **`frame`**: base64 JPEG (raw, pre-service) — **legacy** field, unchanged.
- **`images`**: optional, controlled by `PIPELINE_IMAGE_MODE`:
  - `full` — `images.raw` / `images.annotated` each include `type` and `base64`.
  - `light` — only `type` (`raw` / `annotated`), no `base64`.
  - `none` — `images` key omitted; `frame` may still be present.

This is **not** the same as FrameBus `GET /detection/stream` task events; it is for offline `CameraPipeline` consumers.

## See also

- [detection.md](./detection.md) — SSE and filters
- [tasks.md](./tasks.md) — supported `algorithmType` values
- [API_USAGE.md](../docs/API_USAGE.md) — long-form examples
- [VISION_PIPELINE_README.md](../docs/VISION_PIPELINE_README.md) — operations and cashier
