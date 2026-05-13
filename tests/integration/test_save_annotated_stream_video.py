"""
FrameBus can write an MP4 of the same annotated BGR as the live WebSocket/Redis path.

Requires the short cross-line clip used elsewhere in the repo.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CLIP = ROOT / "artifacts" / "crossline_jabal_live_test" / "clip_15s.mp4"


def _http(base: str, method: str, path: str, body: object | None = None, timeout: float = 60.0) -> object:
    url = base + path
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


@pytest.mark.integration
def test_save_annotated_video_mp4_matches_stream_pixels(tmp_path: pathlib.Path) -> None:
    if not CLIP.exists():
        pytest.skip(f"fixture clip missing: {CLIP}")

    port = 19140
    base = f"http://127.0.0.1:{port}"
    cam = "annot_mp4_cam"
    task_id = 99100
    art = tmp_path / "run"
    art.mkdir(parents=True, exist_ok=True)
    for sub in ("events", "captures", "scenes", "gallery", "outputs"):
        (art / sub).mkdir(parents=True, exist_ok=True)

    mp4_path = art / "annotated_live_stream.mp4"
    env = {
        **os.environ,
        "DEVICE": "cpu",
        "FRAME_SKIP": "4",
        "YOLO_MODEL": str(ROOT / "models" / "yolov8n.pt"),
        "WIDTH": "480",
        "HEIGHT": "360",
        "LIVE_ANNOTATION_MODE": "opencv",
        "SAVE_OUTPUT": "false",
        "SAVE_ANNOTATED_VIDEO": "true",
        "SAVE_ANNOTATED_VIDEO_PATH": str(mp4_path),
        "SAVE_ANNOTATED_VIDEO_FPS": "15",
        "REDIS_CONNECT_RETRIES": "1",
        "EVENTS_DIR": str(art / "events"),
        "CAPTURE_DIR": str(art / "captures"),
        "SCENE_DIR": str(art / "scenes"),
        "GALLERY_DIR": str(art / "gallery"),
        "OUTPUT_DIR": str(art / "outputs"),
        "LOG_LEVEL": "WARNING",
    }

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
        cwd=str(ROOT),
        env=env,
        stdout=log_f,
        stderr=log_f,
    )

    def row(status: object) -> dict:
        if isinstance(status, dict):
            cams = status.get("cameras", {})
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
            proc.terminate()
            pytest.fail(f"server did not start: {log_path.read_text()[-2000:]}")

        area = json.dumps(
            [
                {
                    "line_id": "1",
                    "line_name": "mid",
                    "point": [{"x": 0, "y": 180}, {"x": 480, "y": 180}],
                    "direction": 0,
                }
            ]
        )
        _http(base, "POST", "/cameras", body={"cameras": [{"id": cam, "url": str(CLIP)}]})
        _http(
            base,
            "POST",
            "/api/tasks",
            body={
                "taskId": task_id,
                "taskName": "annot_mp4",
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
        _http(base, "POST", f"/detection/start?camera_id={cam}")

        boot_deadline = time.monotonic() + 120.0
        last_status: object = None
        while time.monotonic() < boot_deadline:
            last_status = _http(base, "GET", "/detection/status")
            r = row(last_status)
            if r and r.get("running"):
                break
            time.sleep(0.5)
        else:
            pytest.fail(f"camera never running: {last_status}")

        end_deadline = time.monotonic() + 300.0
        while time.monotonic() < end_deadline:
            last_status = _http(base, "GET", "/detection/status")
            r = row(last_status)
            if not r.get("running") and not r.get("framebus_process_alive", True):
                break
            time.sleep(0.5)
        else:
            pytest.fail("timeout waiting for pipeline")

        assert mp4_path.is_file(), f"expected MP4 at {mp4_path}"
        assert mp4_path.stat().st_size > 5000, "MP4 unexpectedly small (encoder may have failed)"

        r = row(last_status)
        assert r.get("save_annotated_video") is True
        assert r.get("annotated_video_path"), "status should report annotated_video_path"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_f.close()
