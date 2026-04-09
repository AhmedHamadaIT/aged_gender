import os
import cv2
import time
import json
from datetime import datetime

# Import your services
from services.detector import DetectorService
from services.semantic_search import SemanticSearchTask

def run_video_pipeline(
    video_path,
    detector,
    tasks=None,
    max_frames=0,
    save_results_path="results.json",
    output_video_path="./output_test.mp4"
):
    # 1. Defaults & Configuration
    width = int(os.getenv("WIDTH", "1280"))
    height = int(os.getenv("HEIGHT", "0"))

    print("=" * 60)
    print("  HYBRID ARCHITECTURE VIDEO PIPELINE RUN")
    print("=" * 60)
    print(f"  Video      : {video_path}")
    print(f"  Tasks      : {[t.task_name for t in tasks] if tasks else 'None'}")
    print(f"  Max frames : {max_frames or 'all'}")
    print("=" * 60)

    if not tasks:
        tasks = []

    # 2. Video Capture & Writer Setup
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w = width
    out_h = height if height > 0 else int(orig_h * width / orig_w) if orig_w > 0 else 720

    video_writer = None
    if output_video_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
        # Handle mp4 extension safely
        if not output_video_path.endswith('.mp4'):
            output_video_path += '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps_val, (out_w, out_h))

    # 3. Process Frames
    frame_count = 0
    total_time = 0.0
    all_events = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if max_frames > 0 and frame_count > max_frames:
                break

            t_start = time.time()

            # Resize frame for processing
            if frame.shape[:2] != (out_h, out_w):
                frame = cv2.resize(frame, (out_w, out_h))

            timestamp = datetime.utcnow().isoformat() + "Z"

            # --- STEP A: Run Detector (Old Context Pattern) ---
            # We pass a copy of the frame so the detector can draw on it without
            # ruining the clean pixels we need for Semantic Search cropping.
            context = {
                "data": {
                    "frame": frame.copy()
                }
            }
            context = detector(context)

            # If SAVE_OUTPUT=True, the detector draws boxes on context["data"]["frame"]
            annotated_frame = context["data"]["frame"]
            clean_frame = context["data"].get("clean_frame", frame)

            # --- STEP B: Build Payload (New Architecture Pattern) ---
            payload = {
                "camera_id": "test_cam_1",
                "frame_id": frame_count,
                "timestamp": timestamp,
                "frame": clean_frame, # Important: Pass the clean frame to the task for high-quality crops
                "detection": context["data"]["detection"]
            }

            # --- STEP C: Run Tasks ---
            frame_events = []
            for task in tasks:
                events = task(payload)
                if events:
                    frame_events.extend(events)
                    all_events.extend(events)

            t_elapsed = time.time() - t_start
            total_time += t_elapsed

            # --- STEP D: Visualize Results for the Video Writer ---
            if video_writer:
                vis_frame = annotated_frame.copy()

                # Draw Event Highlights from SemanticSearch
                for event in frame_events:
                    if "person" in event and "boundingBox" in event["person"]:
                        bbox = event["person"]["boundingBox"]
                        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

                        # Flash a thick green box and label when an extraction/overwrite occurs
                        cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
                        cv2.putText(vis_frame, "INDEXED", (x, y - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                video_writer.write(vis_frame)

            if frame_count % 5 == 0:
                print(f"Processed {frame_count} frames... Events generated: {len(all_events)}")

    except Exception as e:
        print(f"Pipeline Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        if save_results_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_results_path)), exist_ok=True)
            with open(save_results_path, "w") as f:
                json.dump(all_events, f, indent=2)

    print("=" * 60)
    print(f"Done. Processed {frame_count} frames.")
    print(f"Average Inference Time: {round((total_time/frame_count)*1000, 2)} ms/frame")
    print(f"Total Events Generated: {len(all_events)}")
    if output_video_path:
        print(f"Video saved to: {output_video_path}")
    if save_results_path:
        print(f"Events saved to: {save_results_path}")


if __name__ == "__main__":

    # 1. Initialize the Detector Service
    print("Loading Detector...")
    detector_service = DetectorService()

    # 2. Configure the Semantic Search Task
    semantic_task_config = {
        "taskId": 105,
        "taskName": "Semantic Search",
        "algorithmType": "SEMANTIC_SEARCH",
        "channelId": 1,
        "enable": True,
        "detailConfig": {"padding": 10}
    }

    # 3. Instantiate tasks
    print("Loading Semantic Search Task...")
    tasks_to_run = [
        SemanticSearchTask(semantic_task_config)
    ]


     # 4. Run Pipeline
    run_video_pipeline(
        video_path="test.mp4",
        detector=detector_service,
        tasks=tasks_to_run,
        max_frames=300, # Set to 0 to process whole video
        save_results_path="./outputs/semantic_events.json",
        output_video_path="./outputs/semantic_test_output.mp4"
    )
