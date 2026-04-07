# Cashier framing test run — results

**Date (UTC):** 2026-04-06  

## Source material

| Item | Value |
|------|--------|
| Original dataset | `/home/a7med/Downloads/cashier_framing/cashier_framing_720p/` (directory of JPEG frames, **not** a single video file) |
| Resolution of frames | 1280×720 |
| Video used for API test | `/home/a7med/ml-server/videos/cashier_framing_sample_720p.mp4` (built from the **first 180** JPEGs, 3 FPS, ~60 s playback) |

To reproduce the sample MP4 locally:

```bash
# Example: regenerate with Python/OpenCV (same logic as the test run)
python3 -c "
import cv2, glob, os
src = '/home/a7med/Downloads/cashier_framing/cashier_framing_720p'
paths = sorted(glob.glob(os.path.join(src, 'frame_*.jpg')))[:180]
# ... write mp4 (see project history) ...
"
```

## Server configuration (this run)

| Variable / file | Value |
|-----------------|--------|
| `YOLO_MODEL` | `yolov8n.pt` (Ultralytics default; **COCO classes** — not cashier-specific weights) |
| `CASHIER_CONFIG` | `./config/cashier_zones.yaml` |
| `CASHIER_EVIDENCE_DIR` | `./evidence/cashier_test_run` |
| `DEVICE` | `cpu` (via venv run) |

**Note:** For production cashier behaviour (person / drawer / cash), configure `YOLO_MODEL` to your **cashier-trained** weights so FrameBus class indices match `CashierBoxOpenTask` expectations (`0=Person`, `1=Drawer_Open`, `2=Cash` per `services/cashier.py`).

## API sequence

1. `POST /cameras` — camera `id`: `"1"`, `url`: absolute path to `cashier_framing_sample_720p.mp4`
2. `POST /api/tasks` — `taskId`: `30`, `algorithmType`: `CASHIER_DRAWER`, `channelId`: `1`, `taskName`: `cashier_framing_test`
3. `POST /detection/start`
4. Polled `GET /detection/status` during run
5. `POST /detection/stop`

### Register camera (response excerpt)

```json
{
  "status": "configured",
  "cameras": {
    "1": "/home/a7med/ml-server/videos/cashier_framing_sample_720p.mp4"
  },
  "error": null
}
```

### Start detection (response excerpt)

```json
{
  "status": "started",
  "cameras": ["1"],
  "tasks": ["30"]
}
```

### Detection status (samples while running)

Early idle / warmup:

```json
{
  "cameras": {
    "1": {
      "camera_id": "1",
      "rtsp_url": "/home/a7med/ml-server/videos/cashier_framing_sample_720p.mp4",
      "running": true,
      "frame_count": 0,
      "fps": 0.0,
      "last_detections": 0,
      "total_detections": 0,
      "uptime_seconds": 0.0,
      "error": null
    }
  }
}
```

Later sample (processing underway):

```json
{
  "cameras": {
    "1": {
      "camera_id": "1",
      "rtsp_url": "/home/a7med/ml-server/videos/cashier_framing_sample_720p.mp4",
      "running": true,
      "frame_count": 6,
      "fps": 1.9,
      "last_detections": 1,
      "total_detections": 4,
      "uptime_seconds": 22.6,
      "error": null
    }
  }
}
```

### Stop detection (response excerpt)

```json
{
  "status": "stopped",
  "cameras": ["1"]
}
```

## Cashier HTTP endpoints (this run)

| Endpoint | Result |
|----------|--------|
| `GET /cashier/status` | `{}` |
| `GET /cashier/events` | `{"total":0,"offset":0,"limit":100,"events":[]}` |

No cashier events were recorded in this short sample with default COCO weights.

## Evidence directory

Path: `./evidence/cashier_test_run` (relative to repo root).  
If empty or unused for this run, ensure alerts fire (`severity` ALERT/CRITICAL) and `CASHIER_EVIDENCE_DIR` is writable when using cashier-specific detection.

## Conclusion

- **Local file path** as camera `url` works: FrameBus reads the file via `stream.frames()` (non-`rtsp://` → `cv2.VideoCapture` on file).
- The **framing dataset** shipped as **extracted frames**; a **short MP4** was generated for this test.
- Full **two-hour** runs should use a single video file or stream; expect proportionally longer wall time and more disk use for evidence if alerts occur.
