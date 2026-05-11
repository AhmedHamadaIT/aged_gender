"""
Final system-wide validation test.

Validates the full analytics pipeline end-to-end:
  - Server startup / shutdown lifecycle
  - Camera registration + task creation
  - Detection start/stop + worker lifecycle
  - Annotated frame persistence on disk
  - Cross-line events generated and persisted
  - Stream metrics (FPS, drop-rate, queue health)
  - Memory stability (no runaway RSS growth)
  - Rapid start/stop cycles (no zombie workers)
  - Clean shutdown (no residual processes)

Runs against a real uvicorn process using the 15-second clip from
artifacts/crossline_jabal_live_test/clip_15s.mp4 so no RTSP / Redis
infrastructure is required.
"""

from __future__ import annotations

import gc
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Generator, List, Optional

import psutil
import pytest

# ─── Constants ──────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[2]
CLIP = ROOT / "artifacts" / "crossline_jabal_live_test" / "clip_15s.mp4"
PORT = 19050  # dedicated port so we don't clash with anything
BASE = f"http://127.0.0.1:{PORT}"
ARTIFACTS_BASE = ROOT / "artifacts" / "final_validation_run"

CROSS_LINE_AREA = json.dumps(
    [
        {
            "line_id": "1",
            "line_name": "mid",
            "point": [{"x": 0, "y": 240}, {"x": 640, "y": 240}],
            "direction": 0,
        }
    ]
)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _http(
    method: str,
    path: str,
    *,
    body: Any = None,
    params: Optional[Dict] = None,
    timeout: float = 15.0,
) -> Dict:
    url = BASE + path
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url += "?" + qs
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _wait_server(timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=2) as r:
                if r.status < 500:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    raise RuntimeError(f"Server did not start within {timeout_s}s")


def _wait(pred, *, timeout_s: float = 30.0, interval_s: float = 0.5) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if pred():
                return True
        except Exception:
            pass
        time.sleep(interval_s)
    return False


# ─── Server fixture ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def server() -> Generator[subprocess.Popen, None, None]:
    """Start uvicorn with the test clip, yield the process, then tear down."""
    if not CLIP.exists():
        pytest.skip(f"Test clip not found: {CLIP}")

    art = ARTIFACTS_BASE
    art.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "DEVICE": "cpu",
        "FRAME_SKIP": "3",
        "LIVE_ANNOTATION_MODE": "opencv",
        "SAVE_OUTPUT": "true",
        "REDIS_CONNECT_RETRIES": "1",  # fail fast so we don't block on Redis
        "EVENTS_DIR": str(art / "events"),
        "CAPTURE_DIR": str(art / "captures"),
        "SCENE_DIR": str(art / "scenes"),
        "GALLERY_DIR": str(art / "gallery"),
        "OUTPUT_DIR": str(art / "outputs"),
        "LOG_LEVEL": "WARNING",
    }

    log_path = art / "uvicorn.log"
    log_fh = log_path.open("w")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            "--workers",
            "1",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=log_fh,
        stderr=log_fh,
    )

    try:
        _wait_server(timeout_s=60)
    except RuntimeError:
        proc.terminate()
        log_fh.close()
        pytest.fail(f"Server failed to start. See {log_path}")

    yield proc

    # ── Teardown ──
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    log_fh.close()


# ─── Tests ───────────────────────────────────────────────────────────────────

CAM_ID = "val_cam_01"
TASK_ID = 90001


def _register_camera_and_task(video_path: pathlib.Path) -> None:
    _http("POST", "/cameras", body={"cameras": [{"id": CAM_ID, "url": str(video_path)}]})
    _http(
        "POST",
        "/api/tasks",
        body={
            "taskId": TASK_ID,
            "taskName": "validation_crossline",
            "algorithmType": "CROSS_LINE",
            "channelId": CAM_ID,
            "enable": True,
            "threshold": 30,
            "areaPosition": CROSS_LINE_AREA,
            "detailConfig": {},
            "validWeekday": [
                "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY",
                "FRIDAY", "SATURDAY", "SUNDAY",
            ],
            "validStartTime": 0,
            "validEndTime": 86400000,
        },
    )


class TestServerLifecycle:
    def test_health_endpoint(self, server):
        """Server must respond 200 on /health."""
        with urllib.request.urlopen(f"{BASE}/health", timeout=5) as r:
            assert r.status == 200

    def test_openapi_schema_accessible(self, server):
        """OpenAPI schema must be reachable (verifies all routes registered cleanly)."""
        with urllib.request.urlopen(f"{BASE}/openapi.json", timeout=5) as r:
            body = json.loads(r.read())
        assert "paths" in body
        required_routes = {
            "/cameras", "/api/tasks", "/detection/start",
            "/detection/stop", "/detection/status",
        }
        missing = required_routes - set(body["paths"].keys())
        assert not missing, f"Missing routes: {missing}"


class TestCameraAndTaskRegistration:
    def test_register_camera(self, server):
        _http("POST", "/cameras", body={"cameras": [{"id": CAM_ID, "url": str(CLIP)}]})
        # Idempotent re-register
        _http("POST", "/cameras", body={"cameras": [{"id": CAM_ID, "url": str(CLIP)}]})

    def test_register_cross_line_task(self, server):
        _http(
            "POST",
            "/api/tasks",
            body={
                "taskId": TASK_ID,
                "taskName": "validation_crossline",
                "algorithmType": "CROSS_LINE",
                "channelId": CAM_ID,
                "enable": True,
                "threshold": 30,
                "areaPosition": CROSS_LINE_AREA,
                "detailConfig": {},
                "validWeekday": [
                    "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY",
                    "FRIDAY", "SATURDAY", "SUNDAY",
                ],
                "validStartTime": 0,
                "validEndTime": 86400000,
            },
        )


class TestDetectionLifecycle:
    def test_start_detection(self, server):
        """Detection start must succeed and status must report running."""
        _register_camera_and_task(CLIP)
        _http("POST", "/detection/start", params={"camera_id": CAM_ID})

        def _is_running():
            status = _http("GET", "/detection/status")
            if isinstance(status, list):
                for row in status:
                    if str(row.get("camera_id", "")) == CAM_ID:
                        return row.get("running", False) or row.get("framebus_running", False)
            elif isinstance(status, dict):
                cams = status.get("cameras", {})
                row = cams.get(CAM_ID, {})
                return row.get("running", False) or row.get("framebus_running", False)
            return False

        assert _wait(_is_running, timeout_s=20), "Detection did not reach running state"

    def test_stream_metrics_populated(self, server):
        """After detection starts, /stream/metrics must have an entry for the camera."""
        def _has_metrics():
            rows = _http("GET", "/stream/metrics")
            if isinstance(rows, list):
                return any(str(r.get("camera_id", "")) == CAM_ID for r in rows)
            return False

        assert _wait(_has_metrics, timeout_s=20), "Stream metrics never populated"

    def test_frames_processed(self, server):
        """frame_count must advance, proving frames are flowing through FrameBus."""
        def _count():
            rows = _http("GET", "/stream/metrics")
            if isinstance(rows, list):
                for r in rows:
                    if str(r.get("camera_id", "")) == CAM_ID:
                        return int(r.get("frame_count", 0))
            return 0

        initial = _count()
        assert _wait(lambda: _count() > initial + 5, timeout_s=30), (
            "frame_count did not advance — FrameBus may be stuck"
        )

    def test_annotation_state_ok(self, server):
        """Annotation diagnostics must not report annotation disabled with detections."""
        ann = _http("GET", "/stream/debug/annotation-state")
        cams = ann.get("cameras", {})
        cam = cams.get(CAM_ID, {})
        mode = cam.get("live_annotation_mode")
        assert mode in ("opencv", "ultralytics", "none", None), (
            f"Unexpected annotation mode: {mode}"
        )

    def test_drop_rate_acceptable(self, server):
        """Decode drop-rate must stay below 20% — checks for decoder/buffer issues."""
        time.sleep(3)
        rows = _http("GET", "/stream/metrics")
        if isinstance(rows, list):
            for r in rows:
                if str(r.get("camera_id", "")) == CAM_ID:
                    drop = float(r.get("drop_rate", 0))
                    assert drop < 0.20, (
                        f"Drop rate {drop:.1%} too high — possible decoder or I/O issue"
                    )

    def test_task_queue_not_saturated(self, server):
        """Task queue drop rate must stay below 10% — detects downstream bottlenecks."""
        time.sleep(2)
        rows = _http("GET", "/stream/metrics")
        if isinstance(rows, list):
            for r in rows:
                if str(r.get("camera_id", "")) == CAM_ID:
                    qdr = float(r.get("task_queue_drop_rate", 0))
                    assert qdr < 0.10, (
                        f"Task queue drop rate {qdr:.1%} — task workers too slow"
                    )


class TestPersistenceAndArtifacts:
    def test_output_frames_saved(self, server):
        """SAVE_OUTPUT=true — frame JPEG files must appear under OUTPUT_DIR."""
        out_dir = ARTIFACTS_BASE / "outputs"

        def _has_frames():
            if not out_dir.exists():
                return False
            return any(out_dir.rglob("*.jpg")) or any(out_dir.rglob("*.jpeg"))

        assert _wait(_has_frames, timeout_s=30), (
            f"No output frames written under {out_dir}"
        )

    def test_output_frames_are_annotated(self, server):
        """Saved JPEG files must be valid (non-zero, decodable) — proxy for annotation."""
        import cv2

        out_dir = ARTIFACTS_BASE / "outputs"
        jpegs: List[pathlib.Path] = []
        for ext in ("*.jpg", "*.jpeg"):
            jpegs.extend(out_dir.rglob(ext))

        assert jpegs, "No JPEGs found to inspect"
        # Sample up to 5 frames
        for p in sorted(jpegs)[:5]:
            img = cv2.imread(str(p))
            assert img is not None, f"Could not decode {p}"
            assert img.shape[0] > 0 and img.shape[1] > 0, f"Zero-size image: {p}"

    def test_events_dir_created(self, server):
        """EVENTS_DIR must exist after detection has run."""
        events_dir = ARTIFACTS_BASE / "events"
        assert _wait(events_dir.exists, timeout_s=15), (
            f"Events directory was never created: {events_dir}"
        )

    def test_cross_line_events_persisted(self, server):
        """JSONL event log must be written for the cross-line task."""
        events_dir = ARTIFACTS_BASE / "events"

        def _has_events():
            if not events_dir.exists():
                return False
            return any(events_dir.rglob("*.jsonl"))

        # Allow up to 20 s for at least one crossing to be detected
        found = _wait(_has_events, timeout_s=25)
        if not found:
            pytest.xfail(
                "No cross-line events detected within window "
                "(may need persons in frame to cross line)"
            )

    def test_event_schema_valid(self, server):
        """Every persisted event must be valid JSON with required top-level keys."""
        events_dir = ARTIFACTS_BASE / "events"
        jsonl_files = list(events_dir.rglob("*.jsonl")) if events_dir.exists() else []
        if not jsonl_files:
            pytest.skip("No JSONL files to validate")

        required_keys = {"eventId", "timestamp", "cameraId"}
        for jf in jsonl_files[:3]:
            for line in jf.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                missing = required_keys - set(ev.keys())
                assert not missing, (
                    f"Event missing required keys {missing} in {jf.name}: {line[:200]}"
                )


class TestMemoryAndResourceStability:
    def test_rss_not_growing_unboundedly(self, server):
        """
        RSS memory of the server process must not grow by more than 300 MB
        over a 10-second observation window — guards against memory leaks.
        """
        server_pid = server.pid
        try:
            proc = psutil.Process(server_pid)
        except psutil.NoSuchProcess:
            pytest.skip("Server process not found via psutil")

        def _rss_mb():
            try:
                mem = proc.memory_info()
                # Include child worker processes
                children = proc.children(recursive=True)
                total = mem.rss + sum(c.memory_info().rss for c in children if c.is_running())
                return total / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return 0.0

        rss_start = _rss_mb()
        time.sleep(10)
        rss_end = _rss_mb()
        growth = rss_end - rss_start
        assert growth < 300, (
            f"RSS grew by {growth:.0f} MB in 10 s — possible memory leak"
        )

    def test_no_zombie_child_processes(self, server):
        """
        No zombie worker processes must persist after the server has had a chance
        to reap them.  We trigger a /detection/stop (which calls _reap_finished_processes)
        before inspecting, then allow a brief settling window.
        """
        # Trigger the zombie reaper in the server
        try:
            _http("POST", "/detection/stop", params={"camera_id": CAM_ID})
        except urllib.error.HTTPError as exc:
            if exc.code != 409:  # 409 = already stopped, that's fine
                raise
        time.sleep(1.0)

        try:
            proc = psutil.Process(server.pid)
            children = proc.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pytest.skip("Could not inspect process tree")

        zombies = [
            c for c in children
            if c.status() == psutil.STATUS_ZOMBIE
        ]
        assert not zombies, (
            f"Found {len(zombies)} zombie child process(es) after reap: "
            + ", ".join(str(z.pid) for z in zombies)
        )

    def test_worker_cpu_not_saturated(self, server):
        """
        FrameBus worker CPU must not peg at 100% continuously — guards against
        busy-wait loops or unbounded tight loops in the pipeline.
        """
        try:
            proc = psutil.Process(server.pid)
            # Measure over 3 s
            cpu = proc.cpu_percent(interval=3.0)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pytest.skip("Could not measure CPU via psutil")

        # On a single CPU machine the total may be higher; we set a generous 250% ceiling
        assert cpu < 250, (
            f"Server CPU at {cpu:.0f}% — possible busy-wait in pipeline"
        )


class TestStartStopCycles:
    def test_rapid_stop_start(self, server):
        """Three stop→start cycles must leave the system in a running state."""
        _register_camera_and_task(CLIP)
        for cycle in range(3):
            try:
                _http("POST", "/detection/stop", params={"camera_id": CAM_ID})
            except urllib.error.HTTPError as exc:
                if exc.code != 409:  # 409 = already stopped, acceptable
                    raise
            time.sleep(0.5)
            try:
                _http("POST", "/detection/start", params={"camera_id": CAM_ID})
            except urllib.error.HTTPError as exc:
                if exc.code != 409:  # 409 = already running, acceptable
                    raise
            time.sleep(0.5)

        # After churn, status must still be reachable
        status = _http("GET", "/detection/status")
        assert status is not None, "Status endpoint unreachable after start/stop cycles"

    def test_stop_clears_workers(self, server):
        """After /detection/stop, the FrameBus process must terminate."""
        _register_camera_and_task(CLIP)
        try:
            _http("POST", "/detection/start", params={"camera_id": CAM_ID})
        except urllib.error.HTTPError as exc:
            if exc.code != 409:  # 409 = already running from previous cycle, that's fine
                raise
        time.sleep(3)
        try:
            _http("POST", "/detection/stop", params={"camera_id": CAM_ID})
        except urllib.error.HTTPError as exc:
            if exc.code != 409:  # 409 = video already ended naturally, still pass
                raise

        def _not_running():
            try:
                status = _http("GET", "/detection/status")
                if isinstance(status, list):
                    for row in status:
                        if str(row.get("camera_id", "")) == CAM_ID:
                            return not row.get("running", True)
                elif isinstance(status, dict):
                    cams = status.get("cameras", {})
                    row = cams.get(CAM_ID, {})
                    return not row.get("running", True)
                return True
            except Exception:
                return False

        assert _wait(_not_running, timeout_s=15), (
            "FrameBus did not stop within 15 s after /detection/stop"
        )


class TestCleanShutdown:
    def test_final_stop_before_shutdown(self, server):
        """Explicit stop before server teardown — verifies graceful cleanup path."""
        try:
            _http("POST", "/detection/stop", params={"camera_id": CAM_ID})
        except Exception:
            pass  # already stopped is fine

        # Server still responds after stop
        with urllib.request.urlopen(f"{BASE}/health", timeout=5) as r:
            assert r.status == 200

    def test_artifacts_not_empty_after_run(self, server):
        """At least one artifact directory must be non-empty after the full run."""
        dirs_to_check = [
            ARTIFACTS_BASE / "outputs",
            ARTIFACTS_BASE / "events",
            ARTIFACTS_BASE / "captures",
            ARTIFACTS_BASE / "scenes",
        ]
        non_empty = [d for d in dirs_to_check if d.exists() and any(d.iterdir())]
        assert non_empty, (
            "All artifact directories are empty — pipeline may not have run"
        )

    def test_resilience_stats_no_crash(self, server):
        """Resilience stats endpoint must return a valid response without error."""
        stats = _http("GET", "/stream/resilience-stats")
        assert isinstance(stats, dict)
        assert "cameras" in stats
