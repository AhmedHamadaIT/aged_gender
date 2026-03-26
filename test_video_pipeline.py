"""
test_video_pipeline.py
----------------------
Standalone test script to run the full pipeline on a local video file
instead of an RTSP stream.

Usage:
    python test_video_pipeline.py                          # uses default ./videos/sample.mp4
    python test_video_pipeline.py --video /path/to/video.mp4
    python test_video_pipeline.py --video /path/to/video.mp4 --services detector age_gender
    python test_video_pipeline.py --video /path/to/video.mp4 --max-frames 100
    python test_video_pipeline.py --video /path/to/video.mp4 --show   # display frames in a window

Output:
    - Annotated frames saved to ./outputs/test_video/ (if SAVE_OUTPUT=True)
    - Per-frame JSON results printed to console
    - Summary statistics printed at the end
"""

import argparse
import json
import os
import sys
import time
import base64
from datetime import datetime

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from stream import frames
from services import REGISTRY
from utils import resize


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the ML pipeline using a local video file."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=os.getenv("INPUT_VIDEO", "./videos/sample.mp4"),
        help="Path to the input video file (default: ./videos/sample.mp4)",
    )
    parser.add_argument(
        "--services",
        nargs="+",
        default=None,
        help=(
            "Pipeline services to run, in order. "
            f"Available: {list(REGISTRY.keys())}. "
            "Default: uses PIPELINE from .env"
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames to process (0 = all frames in video)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated frames in an OpenCV window",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default="",
        help="Path to save JSON results (e.g. results.json). Empty = no file output.",
    )
    parser.add_argument(
        "--no-save-frames",
        action="store_true",
        help="Disable saving annotated frames to disk (overrides SAVE_OUTPUT env)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validate video path ──
    if not os.path.isfile(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        sys.exit(1)

    # ── Determine services ──
    if args.services:
        service_names = args.services
    else:
        env_pipeline = os.getenv("PIPELINE", "detector")
        service_names = [s.strip() for s in env_pipeline.split(",") if s.strip()]

    # Validate service names
    unknown = [s for s in service_names if s not in REGISTRY]
    if unknown:
        print(f"[ERROR] Unknown services: {unknown}")
        print(f"[ERROR] Available: {list(REGISTRY.keys())}")
        sys.exit(1)

    # ── Override save if requested ──
    if args.no_save_frames:
        os.environ["SAVE_OUTPUT"] = "False"

    # ── Configuration ──
    width  = int(os.getenv("WIDTH",  "1280"))
    height = int(os.getenv("HEIGHT", "0"))
    save_output = os.getenv("SAVE_OUTPUT", "True").lower() in ("true", "1", "yes")
    out_dir = os.path.join(os.getenv("OUTPUT_DIR", "./outputs"), "test_video")

    print("=" * 60)
    print("  VIDEO PIPELINE TEST")
    print("=" * 60)
    print(f"  Video      : {args.video}")
    print(f"  Services   : {service_names}")
    print(f"  Max frames : {args.max_frames or 'all'}")
    print(f"  Resolution : {width}x{height or 'auto'}")
    print(f"  Save frames: {save_output} → {out_dir}")
    print(f"  Show window: {args.show}")
    print("=" * 60)

    # ── Initialize services ──
    print("\n[INIT] Loading services...")
    services = []
    for name in service_names:
        print(f"  → Loading '{name}'...")
        svc = REGISTRY[name]()
        services.append(svc)
        print(f"  ✓ '{name}' ready")

    if save_output:
        os.makedirs(out_dir, exist_ok=True)

    # ── Run pipeline ──
    print(f"\n[RUN] Processing video: {args.video}\n")

    frame_count    = 0
    total_time     = 0.0
    all_results    = []
    fps_counter    = 0
    fps_timer      = time.time()
    fps            = 0.0

    try:
        for frame in frames(args.video):
            frame_count += 1
            fps_counter += 1

            if args.max_frames and frame_count > args.max_frames:
                print(f"\n[DONE] Reached max frames limit ({args.max_frames})")
                break

            t_start = time.time()

            # Resize
            resized_frame = resize(frame, width, height)

            # Encode raw frame as base64 (before annotation)
            _, buf    = cv2.imencode(".jpg", resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_b64 = base64.b64encode(buf).decode("utf-8")

            # Build context
            context = {
                "data": {
                    "frame"    : resized_frame.copy(),
                    "detection": {},
                    "use_case" : {},
                }
            }

            # Run all services
            for svc in services:
                context = svc(context)

            t_elapsed = time.time() - t_start
            total_time += t_elapsed

            # FPS calculation
            elapsed_fps = time.time() - fps_timer
            if elapsed_fps >= 1.0:
                fps         = round(fps_counter / elapsed_fps, 2)
                fps_counter = 0
                fps_timer   = time.time()

            # Extract results
            detection_data = context["data"].get("detection", {})
            use_case_data  = context["data"].get("use_case",  {})

            det_count = detection_data.get("count", 0)
            det_items = [
                d.to_dict() for d in detection_data.get("items", [])
            ]
            use_case_serialized = {
                k: [r.to_dict() for r in v] if isinstance(v, list) else v
                for k, v in use_case_data.items()
            }

            result = {
                "frame_count"  : frame_count,
                "timestamp"    : datetime.utcnow().isoformat(),
                "inference_ms" : round(t_elapsed * 1000, 1),
                "fps"          : fps,
                "data": {
                    "detection": {
                        "count": det_count,
                        "items": det_items,
                    },
                    "use_case": use_case_serialized,
                },
            }
            all_results.append(result)

            # Print per-frame summary
            uc_summary = ""
            if use_case_serialized:
                parts = []
                for k, v in use_case_serialized.items():
                    if isinstance(v, list):
                        parts.append(f"{k}={len(v)}")
                    else:
                        parts.append(f"{k}={v}")
                uc_summary = " | " + ", ".join(parts)

            print(
                f"  Frame {frame_count:5d} | "
                f"{t_elapsed*1000:6.1f}ms | "
                f"FPS: {fps:5.1f} | "
                f"Detections: {det_count:3d}"
                f"{uc_summary}"
            )

            # Save annotated frame
            if save_output:
                out_path = os.path.join(out_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(out_path, context["data"]["frame"], [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Display frame
            if args.show:
                cv2.imshow("Pipeline Test", context["data"]["frame"])
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n[DONE] User pressed 'q' to quit")
                    break

    except KeyboardInterrupt:
        print("\n[DONE] Interrupted by user")
    finally:
        if args.show:
            cv2.destroyAllWindows()

    # ── Summary ──
    avg_ms = (total_time / frame_count * 1000) if frame_count else 0
    avg_fps = frame_count / total_time if total_time > 0 else 0

    total_detections = sum(r["data"]["detection"]["count"] for r in all_results)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Frames processed : {frame_count}")
    print(f"  Total time       : {total_time:.2f}s")
    print(f"  Avg inference    : {avg_ms:.1f}ms/frame")
    print(f"  Avg FPS          : {avg_fps:.1f}")
    print(f"  Total detections : {total_detections}")
    if save_output:
        print(f"  Output saved to  : {out_dir}")
    print("=" * 60)

    # ── Save results JSON ──
    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[SAVED] Results → {args.save_results}")


if __name__ == "__main__":
    main()
