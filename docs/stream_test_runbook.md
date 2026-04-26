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
