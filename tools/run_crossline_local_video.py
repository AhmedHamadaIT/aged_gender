#!/usr/bin/env python3
"""
Run CROSS_LINE against a local file path like production: uvicorn + POST /cameras,
POST /api/tasks, POST /detection/start, wait until the file is exhausted.

Typical CPU / weights (override via env):
  DEVICE=cpu YOLO_MODEL=./models/yolov8n.pt WIDTH=640 HEIGHT=360

Writes an MP4 of the same annotated frames as the live stream (see SAVE_ANNOTATED_VIDEO in frame_bus).

Example:
  python tools/run_crossline_local_video.py \\
    --video "/path/to/clip.mp4" \\
    --artifact-dir /tmp/crossline_run

Full-length file (slow on CPU):
  USE_FULL_VIDEO=1 python tools/run_crossline_local_video.py --video "/path/to/full.mp4"
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def _http(base: str, method: str, path: str, body: object | None = None, timeout: float = 120.0) -> object:
    url = base + path
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=pathlib.Path, required=True)
    ap.add_argument("--artifact-dir", type=pathlib.Path, default=pathlib.Path("/tmp/crossline_local_run"))
    ap.add_argument("--port", type=int, default=19103)
    ap.add_argument("--camera-id", default="local_file_cam")
    ap.add_argument("--task-id", type=int, default=99002)
    args = ap.parse_args()

    video = args.video.expanduser().resolve()
    if not video.is_file():
        print("Video not found:", video)
        return 2

    art: pathlib.Path = args.artifact_dir
    art.mkdir(parents=True, exist_ok=True)
    for sub in ("events", "captures", "scenes", "gallery", "outputs"):
        (art / sub).mkdir(parents=True, exist_ok=True)

    root = pathlib.Path(__file__).resolve().parents[1]
    w = int(os.environ.get("WIDTH", "640"))
    h = int(os.environ.get("HEIGHT", "360"))
    mid_y = h // 2
    area = json.dumps(
        [
            {
                "line_id": "L1",
                "line_name": "cross",
                "point": [{"x": 0, "y": mid_y}, {"x": w, "y": mid_y}],
                "direction": 0,
            }
        ]
    )

    env = {
        **os.environ,
        "DEVICE": os.environ.get("DEVICE", "cpu"),
        "YOLO_MODEL": os.environ.get("YOLO_MODEL", str(root / "models" / "yolov8n.pt")),
        "CONF_THRESHOLD": os.environ.get("CONF_THRESHOLD", "0.35"),
        "FILTER_CLASSES": os.environ.get("FILTER_CLASSES", "0"),
        "WIDTH": str(w),
        "HEIGHT": str(h),
        "FRAME_SKIP": os.environ.get("FRAME_SKIP", "2"),
        "LIVE_ANNOTATION_MODE": os.environ.get("LIVE_ANNOTATION_MODE", "opencv"),
        "SAVE_OUTPUT": os.environ.get("SAVE_OUTPUT", "true"),
        "REDIS_CONNECT_RETRIES": os.environ.get("REDIS_CONNECT_RETRIES", "1"),
        "EVENTS_DIR": str(art / "events"),
        "CAPTURE_DIR": str(art / "captures"),
        "SCENE_DIR": str(art / "scenes"),
        "GALLERY_DIR": str(art / "gallery"),
        "OUTPUT_DIR": str(art / "outputs"),
        "LOG_LEVEL": os.environ.get("LOG_LEVEL", "WARNING"),
        # Same BGR annotations as WebSocket/Redis live preview; path under artifact dir.
        "SAVE_ANNOTATED_VIDEO": os.environ.get("SAVE_ANNOTATED_VIDEO", "true"),
        "SAVE_ANNOTATED_VIDEO_PATH": os.environ.get(
            "SAVE_ANNOTATED_VIDEO_PATH", str(art / "annotated_live_stream.mp4")
        ),
        "SAVE_ANNOTATED_VIDEO_MATCH_WS": os.environ.get(
            "SAVE_ANNOTATED_VIDEO_MATCH_WS", "false"
        ),
    }

    port = args.port
    cam = args.camera_id
    task_id = args.task_id
    base = f"http://127.0.0.1:{port}"
    log_path = art / "uvicorn.log"
    log_f = log_path.open("w")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            "1",
        ],
        cwd=str(root),
        env=env,
        stdout=log_f,
        stderr=log_f,
    )

    def row(status_obj: object) -> dict:
        if isinstance(status_obj, dict):
            cams = status_obj.get("cameras", {})
            if cam in cams:
                r = cams[cam]
                return dict(r) if isinstance(r, dict) else {}
        return {}

    try:
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(base + "/health", timeout=2)
                break
            except (urllib.error.URLError, OSError):
                time.sleep(0.5)
        else:
            print(log_path.read_text()[-4000:])
            return 1

        _http(base, "POST", "/cameras", body={"cameras": [{"id": cam, "url": str(video)}]})
        _http(
            base,
            "POST",
            "/api/tasks",
            body={
                "taskId": task_id,
                "taskName": "local_crossline",
                "algorithmType": "CROSS_LINE",
                "channelId": cam,
                "enable": True,
                "threshold": 30,
                "areaPosition": area,
                "detailConfig": {},
                "validWeekday": [
                    "MONDAY",
                    "TUESDAY",
                    "WEDNESDAY",
                    "THURSDAY",
                    "FRIDAY",
                    "SATURDAY",
                    "SUNDAY",
                ],
                "validStartTime": 0,
                "validEndTime": 86400000,
            },
        )
        qs = urllib.parse.urlencode({"camera_id": cam})
        _http(base, "POST", f"/detection/start?{qs}")

        boot_deadline = time.monotonic() + 120.0
        last_status: object = None
        while time.monotonic() < boot_deadline:
            last_status = _http(base, "GET", "/detection/status")
            r = row(last_status)
            if r and r.get("running"):
                break
            time.sleep(0.5)
        else:
            print("Timeout: camera never running. Log tail:\n", log_path.read_text()[-4000:])
            return 1

        use_full = os.environ.get("USE_FULL_VIDEO", "").lower() in ("1", "true", "yes")
        end_deadline = time.monotonic() + (7200.0 if use_full else 900.0)
        while time.monotonic() < end_deadline:
            last_status = _http(base, "GET", "/detection/status")
            r = row(last_status)
            running = bool(r.get("running")) if r else False
            alive = bool(r.get("framebus_process_alive", True))
            if not running and not alive:
                break
            time.sleep(1.0)
        else:
            print("Timeout waiting for finish. Last status:", last_status)
            try:
                _http(base, "POST", f"/detection/stop?{qs}")
            except Exception:
                pass
            return 1

        jpegs = sorted((art / "outputs").rglob("*.jpg")) + sorted((art / "outputs").rglob("*.jpeg"))
        jsonl = list((art / "events").rglob("*.jsonl"))
        r_final = row(last_status)
        mp4_guess = pathlib.Path(env.get("SAVE_ANNOTATED_VIDEO_PATH", str(art / "annotated_live_stream.mp4")))
        mp4_reported = r_final.get("annotated_video_path") if r_final else None
        mp4_path = pathlib.Path(mp4_reported) if mp4_reported else mp4_guess
        print("video:", video)
        print("saved_annotated_frames:", len(jpegs))
        print("jsonl_event_files:", len(jsonl))
        print("annotated_stream_mp4:", mp4_path, "exists:", mp4_path.is_file(), "bytes:", mp4_path.stat().st_size if mp4_path.is_file() else 0)
        print("final_status_row:", json.dumps(r_final, indent=2))
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_f.close()


if __name__ == "__main__":
    raise SystemExit(main())
