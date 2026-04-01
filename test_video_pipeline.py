import argparse
import json
import os
import sys
import time
import base64
from datetime import datetime

import cv2
import numpy as np
from services import REGISTRY


def run_pipeline(video_path=None, service_names=None, max_frames=0, save_results_path="results.json", output_video_path="/content/l.mp4"):
    # 1. Defaults & Configuration
    width = int(os.getenv("WIDTH", "1280"))
    height = int(os.getenv("HEIGHT", "0"))
    save_output = os.getenv("SAVE_OUTPUT", "True").lower() in ("true", "1", "yes")
    out_dir = os.path.join(os.getenv("OUTPUT_DIR", "./outputs"), "test_video")

    print("=" * 60)
    print("  VIDEO PIPELINE RUN")
    print("=" * 60)
    print(f"  Video      : {video_path}")
    print(f"  Services   : {service_names}")
    print(f"  Max frames : {max_frames or 'all'}")
    print(f"  Resolution : {width}x{height or 'auto'}")
    print("=" * 60)

    # 2. Initialize Services
    services = []
    for name in service_names:
        if name in REGISTRY:
            svc = REGISTRY[name]()
            services.append(svc)
            print(f"  ✓ '{name}' ready")

    if save_output:
        os.makedirs(out_dir, exist_ok=True)

    # 3. Video Writer Setup
    video_writer = None
    if output_video_path:
        cap = cv2.VideoCapture(video_path)
        fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        out_w = width
        out_h = height if height > 0 else int(orig_h * width / orig_w) if orig_w > 0 else 720
        # Use 'avc1' (H.264) for better browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps_val, (out_w, out_h))

    # 4. Process Frames
    frame_count = 0
    total_time = 0.0
    all_results = []

    try:
        for frame in frames(video_path):
            frame_count += 1
            if max_frames > 0 and frame_count > max_frames:
                break

            t_start = time.time()
            resized_frame = resize(frame, width, height)

            context = {"data": {"frame": resized_frame.copy(), "detection": {}, "use_case": {}}}

            for svc in services:
                context = svc(context)

            t_elapsed = time.time() - t_start
            total_time += t_elapsed

            # Collect results
            res = {
                "frame": frame_count,
                "inference_ms": round(t_elapsed * 1000, 1),
                "detections": context["data"]["detection"].get("count", 0)
            }
            all_results.append(res)

            if video_writer:
                out_frame = context["data"]["frame"]
                
                # 1. Force NumPy array and uint8 type
                import numpy as np
                if not isinstance(out_frame, np.ndarray) or out_frame.dtype != np.uint8:
                    out_frame = np.array(out_frame, dtype=np.uint8)
                
                # 2. Force exactly 3 channels (if it became grayscale or RGBA)
                if len(out_frame.shape) == 2: 
                    out_frame = cv2.cvtColor(out_frame, cv2.COLOR_GRAY2BGR)
                elif out_frame.shape[2] == 4: 
                    out_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGRA2BGR)
                
                # 3. Force exact dimensions (just in case a service cropped/padded it)
                if out_frame.shape[:2] != (out_h, out_w):
                    out_frame = cv2.resize(out_frame, (out_w, out_h))

                # Write the sanitized frame
                video_writer.write(out_frame)

            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames...")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if video_writer:
            video_writer.release()
        if save_results_path:
            print(save_results_path,"oooooooooooooooooooooooooooooooooooooooooooo")
            with open(save_results_path, "w") as f:
                json.dump(all_results, f)

    print(f"\nDone. Processed {frame_count} frames.")
    if output_video_path: print(f"Video saved: {output_video_path}")



if __name__ == "__main__":
    run_pipeline(
    video_path="test.mp4",
    service_names=["detector","reid"],
    max_frames=500,
    )