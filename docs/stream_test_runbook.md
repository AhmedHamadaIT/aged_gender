# Stream / detection deployment and manual test

## Build and run

The Compose file documents:

```bash
docker compose up -d --build
```

Not `docker compose up build` (no such service). If you need a rebuild, always use `--build` with `up`.

Typical update on the host:

```bash
docker compose down
git pull
docker compose up -d --build
```

## JSON on minimal hosts

If `jq` is not installed, format JSON with:

```bash
curl -s http://127.0.0.1:9000/detection/status | python3 -m json.tool
```

## Redis: vm.overcommit (host)

If logs show *Memory overcommit must be enabled*, on the **host** (not only inside the container) run once:

```bash
sudo sysctl vm.overcommit_memory=1
```

To persist, add `vm.overcommit_memory = 1` to `/etc/sysctl.conf` and reboot or run `sysctl -p`.

## API order of operations

1. `POST /cameras` — register `id` and RTSP `url` for each stream.
2. `POST /api/tasks` — register tasks. For `CROSS_LINE` with `enable: true`, `areaPosition` must be a non-empty JSON array of line objects (see `services/cross_line.py`).
3. `POST /detection/start?camera_id=<id>` — start **one** channel. Recommended to avoid starting every camera that has a task.
4. To start every enabled task channel: `POST /detection/start?all_channels=true` (use only when you intend to run all channels).
5. `GET /detection/status` and `GET /stream/metrics` — pipeline health; if `task_queue_drop_rate` stays high, raise `TASK_QUEUE_MAXSIZE`, lower `WIDTH`, use an H.264 substream, and check `task_queue_coalesced_by_task` when `TASK_QUEUE_COALESCE=true`. Default `TASK_QUEUE_INCLUDE_FRAME=false` sends JPEG over the queue (lighter IPC than raw BGR).

## RTSP / HEVC

HEVC/RTSP log lines (`PPS id out of range`, `Could not find ref`, etc.) are often from a noisy or 4K stream. Prefer a **stable H.264 substream** (main stream 4K, substream 720p) if decode errors persist. The default `RTSP_PROFILE=balanced` in Compose favors stability. For a clean LAN and H.264, you can try `RTSP_PROFILE=performance` and `RTSP_LOW_DELAY=true` (see `utils/rtsp_ffmpeg.py`).

## Jetson: clocks and GStreamer (NVDEC)

On the **Jetson host** (before or independently of Docker), max performance mode reduces stutter and helps sustained decode + YOLO:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

With `RTSP_BACKEND=auto` (default) on a Jetson with OpenCV+GStreamer, the app prefers a **GStreamer** pipeline using `nvv4l2decoder` before Adaptive FFmpeg. Do not replace the L4T OpenCV in the `dustynv/l4t-pytorch` image with `pip install opencv-python*`, or `cv2.CAP_GSTREAMER` may break. Use `GET /stream/metrics` and `rtsp_backend` in shared state to see which path is active. Tune live overlay CPU with `LIVE_ANNOTATION_MODE` (`ultralytics` / `opencv` / `none`).

## Credentials

Never commit camera passwords or real RTSP URLs into the repo. Use environment-specific `.env` or secret storage.

## Real-time drops runbook (production)

Frame drops are expected under pressure when queue coalescing is enabled (`TASK_QUEUE_COALESCE=true`): old frames are discarded so workers process fresher frames with lower end-to-end latency.

### Quick health checks

```bash
curl -s http://127.0.0.1:9000/stream/metrics | python3 -m json.tool
curl -s http://127.0.0.1:9000/stream/pipeline-health | python3 -m json.tool
curl -s http://127.0.0.1:9000/detection/status | python3 -m json.tool
```

### Alert thresholds (suggested)

- `critical`: `running=false`
- `critical`: `reconnects > 5`
- `critical`: `drop_rate > 0.8`
- `warning`: `drop_rate > 0.6`
- `warning`: `reconnects > 2`
- `warning`: no events while worker is running

### When to ignore vs act

- **Ignore / normal behavior**
  - `drop_rate` around `0.3` to `0.6` and events continue to flow.
  - Queue coalescing counters increase but latest frames are still processed.
- **Investigate soon**
  - `drop_rate` remains above `0.6` for several minutes.
  - `reconnects` keeps increasing (network jitter, RTSP instability).
- **Immediate action**
  - Worker not running (`running=false`).
  - `drop_rate > 0.8` sustained.
  - No detection events at all for active scenes/tasks.

### Fast mitigations if drop rate is high

1. Use camera H.264 substream (avoid high-bitrate/HEVC main stream on edge boxes).
2. Lower input resolution (`WIDTH`) or source FPS.
3. Increase worker capacity or reduce expensive per-frame operations.
4. Increase `TASK_QUEUE_MAXSIZE` moderately and keep coalescing enabled.

### `.env` tuning behavior in Compose

`docker-compose.yml` uses `${VAR:-default}` for stream throttle settings, so values in `.env` are applied at container start. If `.env` changes are not reflected, recreate the service (`docker compose down && docker compose up -d`).

Recommended low-load Jetson profile:

```bash
WIDTH=320
HEIGHT=240
FRAME_SKIP=2
STREAM_TARGET_FPS=3
LIVE_ANNOTATION_MODE=none
```
