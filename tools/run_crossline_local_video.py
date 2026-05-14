#!/usr/bin/env python3
"""
Run CROSS_LINE against a local file path like production: uvicorn + POST /cameras,
POST /api/tasks, POST /detection/start, wait until the file is exhausted.

Typical CPU / weights (override via env):
  DEVICE=cpu YOLO_MODEL=./models/yolov8n.pt WIDTH=640 HEIGHT=360

Writes an MP4 of the same annotated frames as the live stream (see SAVE_ANNOTATED_VIDEO in frame_bus).

Example with annotator export (start/end points):

  python3 tools/run_crossline_local_video.py \\
    --video "/path/to/clip.mp4" \\
    --lines-json /path/to/lines_export.json \\
    --artifact-dir /tmp/crossline_run

Full-length file (slow on CPU):

  USE_FULL_VIDEO=1 python3 tools/run_crossline_local_video.py --video "/path/to/full.mp4"

Production-like logs + evidence layout under ``--artifact-dir``:

  USE_FULL_VIDEO=1 python3 tools/run_crossline_local_video.py --production-artifacts ...
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


def _lines_export_to_area_position(data: dict) -> str:
    """
    Convert UI/export shape { line_count, lines: [{ id, start, end, ... }] }
    into CROSS_LINE areaPosition (JSON string for POST /api/tasks).
    ``direction_deg`` is ignored; use ``direction`` per line if needed (0=both, 1/2=one way).
    """
    rows = []
    for ln in data.get("lines") or []:
        start = ln["start"]
        end = ln["end"]
        lid = ln.get("id", ln.get("line_id", ""))
        rows.append(
            {
                "line_id": str(lid),
                "line_name": str(ln.get("name", ln.get("line_name", f"line_{lid}"))),
                "point": [
                    {"x": int(start["x"]), "y": int(start["y"])},
                    {"x": int(end["x"]), "y": int(end["y"])},
                ],
                "direction": int(ln.get("direction", 0)),
            }
        )
    return json.dumps(rows)


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
    ap.add_argument(
        "--lines-json",
        type=pathlib.Path,
        default=None,
        help="Path to {line_count, lines:[{id,start,end,...}]} export; overrides default mid-line.",
    )
    ap.add_argument(
        "--area-position-json",
        type=pathlib.Path,
        default=None,
        help="Path to raw CROSS_LINE areaPosition array JSON (overrides --lines-json).",
    )
    ap.add_argument(
        "--production-artifacts",
        action="store_true",
        help="DEBUG logging, app.log under <artifact>/logs/ (LOG_APP_FILE), uvicorn.log there too, "
        "SAVE_JPEG_QUALITY=80, queue warn interval like compose.",
    )
    ap.add_argument(
        "--disable-attr-detect",
        action="store_true",
        help="Turn off detailConfig.enableAttrDetect (age/gender on crossings). Default is ON — "
        "the harness used to send empty detailConfig so AgeGender never loaded.",
    )
    args = ap.parse_args()

    video = args.video.expanduser().resolve()
    if not video.is_file():
        print("Video not found:", video)
        return 2

    art: pathlib.Path = args.artifact_dir
    art.mkdir(parents=True, exist_ok=True)
    for sub in ("events", "captures", "scenes", "gallery", "outputs"):
        (art / sub).mkdir(parents=True, exist_ok=True)

    logs_dir = art / "logs"
    if args.production_artifacts:
        logs_dir.mkdir(parents=True, exist_ok=True)

    root = pathlib.Path(__file__).resolve().parents[1]
    w = int(os.environ.get("WIDTH", "640"))
    h = int(os.environ.get("HEIGHT", "360"))
    mid_y = h // 2
    if args.area_position_json:
        area = args.area_position_json.expanduser().resolve().read_text().strip()
    elif args.lines_json:
        spec = json.loads(args.lines_json.expanduser().resolve().read_text())
        area = _lines_export_to_area_position(spec)
    else:
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

    log_level = os.environ.get("LOG_LEVEL", "DEBUG" if args.production_artifacts else "WARNING")

    device_raw = os.environ.get("DEVICE", "cpu")

    env = {
        **os.environ,
        "LOG_LEVEL": log_level,
        "DEVICE": device_raw,
        "YOLO_MODEL": os.environ.get("YOLO_MODEL", str(root / "models" / "yolov8n.pt")),
        "AGE_GENDER_MODEL": os.environ.get(
            "AGE_GENDER_MODEL", str(root / "models" / "best_aged_gender_6.onnx")
        ),
        "CONF_THRESHOLD": os.environ.get("CONF_THRESHOLD", "0.35"),
        "FILTER_CLASSES": os.environ.get("FILTER_CLASSES", "0"),
        "WIDTH": str(w),
        "HEIGHT": str(h),
        "FRAME_SKIP": os.environ.get("FRAME_SKIP", "2"),
        "LIVE_ANNOTATION_MODE": os.environ.get("LIVE_ANNOTATION_MODE", "opencv"),
        "LIVE_PREVIEW_ANNOTATE": os.environ.get("LIVE_PREVIEW_ANNOTATE", "true"),
        "SAVE_OUTPUT": os.environ.get("SAVE_OUTPUT", "true"),
        "SAVE_FORMAT": os.environ.get("SAVE_FORMAT", "jpg"),
        "SAVE_JPEG_QUALITY": os.environ.get("SAVE_JPEG_QUALITY", "80"),
        "REDIS_CONNECT_RETRIES": os.environ.get("REDIS_CONNECT_RETRIES", "1"),
        "EVENTS_DIR": str(art / "events"),
        "CAPTURE_DIR": str(art / "captures"),
        "SCENE_DIR": str(art / "scenes"),
        "GALLERY_DIR": str(art / "gallery"),
        "OUTPUT_DIR": str(art / "outputs"),
        "PERF_LOG_INTERVAL": os.environ.get("PERF_LOG_INTERVAL", "300"),
        "FRAMEBUS_QUEUE_WARN_INTERVAL_SEC": os.environ.get(
            "FRAMEBUS_QUEUE_WARN_INTERVAL_SEC", "15" if args.production_artifacts else "5.0"
        ),
        # Same BGR annotations as WebSocket/Redis live preview; path under artifact dir.
        "SAVE_ANNOTATED_VIDEO": os.environ.get("SAVE_ANNOTATED_VIDEO", "true"),
        "SAVE_ANNOTATED_VIDEO_PATH": os.environ.get(
            "SAVE_ANNOTATED_VIDEO_PATH", str(art / "annotated_live_stream.mp4")
        ),
        "SAVE_ANNOTATED_VIDEO_MATCH_WS": os.environ.get(
            "SAVE_ANNOTATED_VIDEO_MATCH_WS", "false"
        ),
    }
    if args.production_artifacts:
        env["LOG_APP_FILE"] = os.environ.get("LOG_APP_FILE", str(logs_dir / "app.log"))
        env["PYTHONUNBUFFERED"] = os.environ.get("PYTHONUNBUFFERED", "1")

    if "ONNX_EXECUTION_PROVIDERS_ORDER" not in os.environ:
        if str(device_raw).strip().lower() == "cpu":
            env["ONNX_EXECUTION_PROVIDERS_ORDER"] = "cpu_only"

    port = args.port
    cam = args.camera_id
    task_id = args.task_id
    base = f"http://127.0.0.1:{port}"
    log_path = (logs_dir / "uvicorn.log") if args.production_artifacts else (art / "uvicorn.log")
    if args.production_artifacts:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "video": str(video),
        "camera_id": args.camera_id,
        "task_id": args.task_id,
        "width": w,
        "height": h,
        "production_artifacts": args.production_artifacts,
        "enable_attr_detect": not args.disable_attr_detect,
        "areaPosition": json.loads(area),
    }
    (art / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
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
        print("areaPosition:", area)
        detail = (
            {"enableAttrDetect": False, "enableReid": False}
            if args.disable_attr_detect
            else {"enableAttrDetect": True, "enableReid": False}
        )
        print("detailConfig:", detail)
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
                "detailConfig": detail,
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
        for jf in jsonl[:5]:
            nlines = sum(1 for ln in jf.read_text().splitlines() if ln.strip())
            print(f"  events {jf.name}: {nlines} line(s)")
        if jsonl and not args.disable_attr_detect:
            attrs_found = 0
            unknown_only = 0
            for jf in jsonl:
                for ln in jf.read_text().splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        ev = json.loads(ln)
                        att = (ev.get("person") or {}).get("attributes") or {}
                        g, a = att.get("gender"), att.get("age")
                        if g and g != "Unknown" and a and a != "Unknown":
                            attrs_found += 1
                        elif g == "Unknown" and a == "Unknown":
                            unknown_only += 1
                    except json.JSONDecodeError:
                        pass
            print(
                "age_gender_in_events: non-Unknown pairs:",
                attrs_found,
                "Unknown-only events:",
                unknown_only,
            )
        caps = list((art / "captures").rglob("*.jpg")) + list((art / "captures").rglob("*.jpeg"))
        scns = list((art / "scenes").rglob("*.jpg")) + list((art / "scenes").rglob("*.jpeg"))
        crops = list((art / "gallery").rglob("*.jpg")) + list((art / "gallery").rglob("*.jpeg"))
        print("evidence_captures:", len(caps), "evidence_scenes:", len(scns), "gallery_crops:", len(crops))
        if args.production_artifacts:
            alog = logs_dir / "app.log"
            print("logs/uvicorn.log:", log_path, "bytes:", log_path.stat().st_size if log_path.is_file() else 0)
            print("logs/app.log:", alog, "bytes:", alog.stat().st_size if alog.is_file() else 0)
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
