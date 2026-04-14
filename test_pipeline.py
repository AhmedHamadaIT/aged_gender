"""
test_pipeline.py
----------------
Integration test — uses the real pipeline with a video file as camera source.

Since stream.py already supports both RTSP and local video files,
this script simply:
    1. Registers a camera pointing to the video file
    2. Registers tasks (if any)
    3. Starts detection via the real DetectionResource
    4. Monitors progress until the video ends
    5. Stops and prints results

Usage:
    python test_pipeline.py --video test.mp4
    python test_pipeline.py --video test.mp4 --max-seconds 30
"""

import os
import sys
import time
import argparse

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Integration test — run the full pipeline with a video file as camera source."
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--camera-id", default="test_cam", help="Simulated camera ID (default: test_cam)")
    parser.add_argument("--max-seconds", type=int, default=0, help="Max seconds to run (0 = until video ends)")
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # ── Import the real infrastructure ────────────────────────────────────
    from apis.cameras import camera_registry, CameraSetupRequest
    from apis.detection import detection
    from schemas import DetectionRequest

    # ── 1. Register camera pointing to the video file ─────────────────────
    camera_id = args.camera_id
    print(f"\n[TEST] Registering camera '{camera_id}' → {video_path}")
    camera_registry.on_post(CameraSetupRequest(id=camera_id, url=video_path))

    # ── 2. Register a dummy CROSS_LINE task so detection has something ────
    from apis.tasks import task_registry, TaskConfig
    task_config = TaskConfig(
        taskId=999,
        taskName="test_cross_line",
        algorithmType="CROSS_LINE",
        channelId=int(camera_id) if camera_id.isdigit() else 1,
        enable=True,
    )
    # Use the camera_id as channelId if it's numeric, otherwise map it
    task_config_dict = task_config.model_dump()
    task_config_dict["channelId"] = camera_id  # Override to match camera registration
    # Register directly so channelId can be a string
    task_registry._tasks[task_config.taskId] = task_config_dict

    print(f"[TEST] Registered task: {task_config.taskName} (id={task_config.taskId})")

    # ── 3. Start detection (uses the real FrameBus + EmbeddingWorker) ─────
    print(f"[TEST] Starting detection...")
    try:
        result = detection.on_post(DetectionRequest(action="start", camera_id=camera_id))
        print(f"[TEST] Started: {result}")
    except Exception as e:
        print(f"[TEST] Failed to start: {e}")
        sys.exit(1)

    # ── 4. Monitor until video ends or timeout ────────────────────────────
    print(f"\n[TEST] Monitoring pipeline (Ctrl+C to stop)...\n")
    start_time = time.time()

    try:
        while True:
            time.sleep(2)

            status = detection.on_get()
            cam_status = status.cameras.get(camera_id)

            if cam_status:
                elapsed = time.time() - start_time
                print(f"  [{elapsed:6.1f}s] Frames: {cam_status.frame_count:5d} | "
                      f"FPS: {cam_status.fps:5.1f} | "
                      f"Detections: {cam_status.last_detections:3d} | "
                      f"Total: {cam_status.total_detections:6d} | "
                      f"Running: {cam_status.running}")

                # Stop if video finished (FrameBus sets running=False)
                if not cam_status.running:
                    print(f"\n[TEST] Video finished processing.")
                    break

            # Timeout check
            if args.max_seconds > 0 and (time.time() - start_time) > args.max_seconds:
                print(f"\n[TEST] Max time ({args.max_seconds}s) reached.")
                break

    except KeyboardInterrupt:
        print(f"\n[TEST] Interrupted by user.")

    # ── 5. Stop detection ─────────────────────────────────────────────────
    print(f"[TEST] Stopping detection...")
    try:
        result = detection.on_post(DetectionRequest(action="stop", camera_id=camera_id))
        print(f"[TEST] Stopped: {result}")
    except Exception as e:
        print(f"[TEST] Stop error (may already be stopped): {e}")

    # ── 6. Print results ──────────────────────────────────────────────────
    gallery_dir = os.getenv("GALLERY_DIR", "/local/storage/gallery")
    crop_dir = os.path.join(gallery_dir, "crops", camera_id)

    print(f"\n{'='*60}")
    print(f"  TEST COMPLETE")
    print(f"{'='*60}")

    if os.path.exists(crop_dir):
        crops = [f for f in os.listdir(crop_dir) if f.endswith(".jpg")]
        print(f"  Crops on disk : {len(crops)} files in {crop_dir}")
        for c in sorted(crops)[:10]:
            size = os.path.getsize(os.path.join(crop_dir, c))
            print(f"    - {c} ({size} bytes)")
        if len(crops) > 10:
            print(f"    ... and {len(crops) - 10} more")
    else:
        print(f"  No crops found at: {crop_dir}")

    # Check SSE result queue for any events
    events = []
    while True:
        try:
            event = detection.result_queue().get_nowait()
            events.append(event)
        except Exception:
            break

    print(f"  SSE events   : {len(events)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
