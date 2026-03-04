#!/usr/bin/env python3
"""
live_inference.py — Combined Age/Gender + Mood Live Inference
=============================================================
One image in → Age + Gender + Mood out (both models run on every frame).

Modes:
  webcam   — real-time webcam feed (default, press Q to quit)
  folder   — run on a folder of images and save an annotated copy
  image    — run on a single image file

Usage:
  # Live webcam
  python3 live_inference.py \
      --yolo       best.pt \
      --mood       "best mood.pt" \
      --device     cuda

  # Folder of images
  python3 live_inference.py \
      --yolo       best.pt \
      --mood       "best mood.pt" \
      --mode       folder \
      --input      ./images \
      --output     ./live_output \
      --device     cpu

  # Single image
  python3 live_inference.py \
      --yolo       best.pt \
      --mood       "best mood.pt" \
      --mode       image \
      --input      ./images/1001.jpg \
      --device     cpu

Notes:
  • torch MUST be imported before cv2/numpy (Jetson Orin/Nano safety rule).
  • Both models are loaded onto --device. On Jetson, load to CPU first then
    move to CUDA (handled automatically below).
  • os._exit(0) is used to bypass Python shutdown on Jetson to avoid glibc
    heap corruption from CUDA atexit handlers.
"""

import os
import sys
import time
import json
import gc
import argparse
import warnings
warnings.filterwarnings("ignore")

# NOTE: torch MUST come before numpy / cv2 on Jetson Nano / Orin.
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────────

MOOD_CLASSES        = ["angry", "happy", "neutral"]
IMAGE_EXTS          = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Display colours per mood class (BGR)
MOOD_COLOURS = {
    "angry":    (0,   0,   220),   # red
    "happy":    (0,   200, 50),    # green
    "neutral":  (220, 150, 30),    # blue-ish
    "positive": (0,   200, 50),    # alias
    "negative": (0,   0,   220),   # alias
}

AGE_GENDER_COLOUR = (220, 200, 50)   # cyan-ish for age/gender label


# ── Model loader (Jetson-safe) ─────────────────────────────────────────────────

def _load_yolo(path: str, device: str) -> tuple:
    """
    Load a YOLO model.
    - Loads to CPU first (Jetson-safe).
    - Moves to CUDA and runs warmup if device=='cuda'.
    - Falls back to CPU on CUDA arch error.
    Returns (model, effective_device).
    """
    print(f"  Loading: {path}")
    model = YOLO(path)
    model.to("cpu")

    if device == "cuda" and torch.cuda.is_available():
        model.to("cuda")
        try:
            dummy = np.zeros((128, 128, 3), dtype=np.uint8)
            model.predict(dummy, verbose=False, device="cuda", half=True)
            del dummy
            print(f"    ✓ CUDA warmup OK — {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"    [!] CUDA warmup failed ({e}) — falling back to CPU")
            device = "cpu"
            model.to("cpu")
    else:
        device = "cpu"

    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"    ✓ Loaded ({size_mb:.2f} MB) on {device}")
    return model, device


def _cuda_cleanup():
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()


# ── Single-frame inference ─────────────────────────────────────────────────────

def infer_frame(frame_bgr: np.ndarray,
                yolo_model, yolo_device: str,
                mood_model, mood_device: str) -> dict:
    """
    Run both models on one BGR frame.
    Returns a dict with all predictions and combined timing.

    Return keys:
      ag_class, ag_conf, ag_time_ms   ← age/gender YOLO
      mood_class, mood_conf, mood_time_ms  ← mood YOLO
      total_time_ms                   ← ag + mood combined
    """
    is_half_ag   = yolo_device == "cuda"
    is_half_mood = mood_device == "cuda"

    # ── Age / Gender ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    ag_preds = yolo_model.predict(frame_bgr, verbose=False, augment=False,
                                   device=yolo_device, half=is_half_ag)
    ag_ms = (time.perf_counter() - t0) * 1000

    ag_class = "unknown"
    ag_conf  = 0.0
    if ag_preds and ag_preds[0].probs is not None:
        p       = ag_preds[0].probs
        top1    = p.top1
        ag_conf = float(p.data[top1])
        ag_class = ag_preds[0].names[top1]

    # ── Mood ──────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    mood_preds = mood_model.predict(frame_bgr, verbose=False, augment=False,
                                     device=mood_device, half=is_half_mood)
    mood_ms = (time.perf_counter() - t0) * 1000

    mood_class = "unknown"
    mood_conf  = 0.0
    if mood_preds and mood_preds[0].probs is not None:
        p          = mood_preds[0].probs
        top1       = p.top1
        mood_conf  = float(p.data[top1])
        mood_class = mood_preds[0].names[top1]

    return {
        "ag_class":     ag_class,
        "ag_conf":      ag_conf,
        "ag_time_ms":   ag_ms,
        "mood_class":   mood_class,
        "mood_conf":    mood_conf,
        "mood_time_ms": mood_ms,
        "total_time_ms": ag_ms + mood_ms,
    }


# ── Overlay drawing ────────────────────────────────────────────────────────────

def draw_overlay(frame_bgr: np.ndarray, result: dict) -> np.ndarray:
    """
    Burn predictions onto the frame.
    Layout (top-left):
      Line 1 (cyan) : Age/Gender — class  (conf%)
      Line 2 (mood colour): Mood — class  (conf%)
      Line 3 (white): Total: X ms  |  FPS: Y
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    ag_text   = (f"Age/Gender: {result['ag_class']}"
                 f"  ({result['ag_conf']*100:.0f}%)")
    mood_col  = MOOD_COLOURS.get(result['mood_class'].lower(), (200, 200, 200))
    mood_text = (f"Mood: {result['mood_class']}"
                 f"  ({result['mood_conf']*100:.0f}%)")
    fps       = 1000 / result['total_time_ms'] if result['total_time_ms'] > 0 else 0
    perf_text = (f"Total: {result['total_time_ms']:.1f} ms"
                 f"  |  FPS: {fps:.1f}")

    font      = cv2.FONT_HERSHEY_DUPLEX
    scale     = max(0.55, w / 1200)
    thick     = 1
    pad       = 8

    lines = [
        (ag_text,   AGE_GENDER_COLOUR),
        (mood_text, mood_col),
        (perf_text, (255, 255, 255)),
    ]

    y = pad
    for text, colour in lines:
        (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
        # semi-transparent black background
        cv2.rectangle(out, (pad - 2, y), (pad + tw + 4, y + th + bl + 2),
                      (0, 0, 0), -1)
        cv2.putText(out, text, (pad, y + th),
                    font, scale, colour, thick, cv2.LINE_AA)
        y += th + bl + pad

    return out


# ── Mode: single image ─────────────────────────────────────────────────────────

def run_image(args, yolo_model, yolo_device, mood_model, mood_device):
    frame = cv2.imread(args.input)
    if frame is None:
        print(f"ERROR: cannot read image: {args.input}")
        return

    result   = infer_frame(frame, yolo_model, yolo_device, mood_model, mood_device)
    annotated = draw_overlay(frame, result)

    print(f"\n{'='*55}")
    print(f"  Image     : {os.path.basename(args.input)}")
    print(f"  Age/Gender: {result['ag_class']}  ({result['ag_conf']*100:.1f}%)  "
          f"[{result['ag_time_ms']:.1f} ms]")
    print(f"  Mood      : {result['mood_class']}  ({result['mood_conf']*100:.1f}%)  "
          f"[{result['mood_time_ms']:.1f} ms]")
    print(f"  Total     : {result['total_time_ms']:.1f} ms  "
          f"(FPS: {1000/result['total_time_ms']:.1f})")
    print(f"{'='*55}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        out_path = os.path.join(args.output,
                                "annotated_" + os.path.basename(args.input))
        cv2.imwrite(out_path, annotated)
        print(f"  Saved → {out_path}")

    # Show preview if display available
    try:
        cv2.imshow("Live Inference", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass


# ── Mode: folder ───────────────────────────────────────────────────────────────

def run_folder(args, yolo_model, yolo_device, mood_model, mood_device):
    paths = sorted(
        str(p) for p in Path(args.input).rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    )
    if not paths:
        print(f"ERROR: No images found in {args.input}")
        return

    print(f"\nFound {len(paths)} images — running on {yolo_device}/{mood_device}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    ag_times, mood_times, total_times = [], [], []
    ag_counts   = defaultdict(int)
    mood_counts = defaultdict(int)
    all_results = []

    for img_path in tqdm(paths, desc="Inferencing"):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        result = infer_frame(frame, yolo_model, yolo_device, mood_model, mood_device)
        ag_times.append(result["ag_time_ms"])
        mood_times.append(result["mood_time_ms"])
        total_times.append(result["total_time_ms"])
        ag_counts[result["ag_class"]] += 1
        mood_counts[result["mood_class"]] += 1

        all_results.append({
            "image":      os.path.basename(img_path),
            "ag_class":   result["ag_class"],
            "ag_conf":    round(result["ag_conf"], 4),
            "mood_class": result["mood_class"],
            "mood_conf":  round(result["mood_conf"], 4),
            "total_ms":   round(result["total_time_ms"], 2),
        })

        if args.output and args.save_vis:
            annotated = draw_overlay(frame, result)
            cv2.imwrite(os.path.join(args.output,
                                     "ann_" + os.path.basename(img_path)),
                        annotated)

    n = len(all_results)
    if n == 0:
        print("No images processed.")
        return

    avg_ag    = np.mean(ag_times)
    avg_mood  = np.mean(mood_times)
    avg_total = np.mean(total_times)
    fps       = 1000 / avg_total if avg_total > 0 else 0

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Images processed   : {n}")
    print(f"  Age/Gender avg ms  : {avg_ag:.2f}")
    print(f"  Mood avg ms        : {avg_mood:.2f}")
    print(f"  Combined avg ms    : {avg_total:.2f}  (FPS: {fps:.1f})")
    print(f"\n  Age/Gender distribution:")
    for cls, cnt in sorted(ag_counts.items(), key=lambda x: -x[1]):
        print(f"    {cls:<30} {cnt:>5}  ({100*cnt/n:.1f}%)")
    print(f"\n  Mood distribution:")
    for cls, cnt in sorted(mood_counts.items(), key=lambda x: -x[1]):
        print(f"    {cls:<30} {cnt:>5}  ({100*cnt/n:.1f}%)")
    print(f"{'='*60}")

    if args.output:
        json_path = os.path.join(args.output, "live_inference_results.json")
        with open(json_path, "w") as f:
            json.dump({
                "summary": {
                    "total_images":       n,
                    "avg_ag_ms":         round(avg_ag, 3),
                    "avg_mood_ms":       round(avg_mood, 3),
                    "avg_combined_ms":   round(avg_total, 3),
                    "fps":               round(fps, 2),
                    "ag_distribution":   dict(ag_counts),
                    "mood_distribution": dict(mood_counts),
                },
                "predictions": all_results,
            }, f, indent=2)
        print(f"\n  JSON saved → {json_path}")


# ── Mode: webcam ───────────────────────────────────────────────────────────────

def run_webcam(args, yolo_model, yolo_device, mood_model, mood_device):
    cam_id = int(args.input) if (args.input and args.input.isdigit()) else 0
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {cam_id}")
        return

    print(f"\nWebcam {cam_id} opened — press Q to quit")
    print(f"Running both models on {yolo_device}")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        result    = infer_frame(frame, yolo_model, yolo_device,
                                mood_model, mood_device)
        annotated = draw_overlay(frame, result)

        cv2.imshow("Live Inference  (Q = quit)", annotated)

        frame_count += 1
        if frame_count % 30 == 0:
            fps = 1000 / result["total_time_ms"] if result["total_time_ms"] > 0 else 0
            print(f"  Frame {frame_count:5d} | "
                  f"AG: {result['ag_class']:<22} "
                  f"Mood: {result['mood_class']:<10} "
                  f"| {result['total_time_ms']:.1f} ms  ({fps:.1f} FPS)")

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combined Age/Gender + Mood live inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--yolo",   required=True,
                        help="Age/Gender YOLO model (.pt)")
    parser.add_argument("--mood",   required=True,
                        help="Mood YOLO model (.pt) — 3-class: angry/happy/neutral")
    parser.add_argument("--mode",   default="webcam",
                        choices=["webcam", "folder", "image"],
                        help=(
                            "webcam=live camera, "
                            "folder=run on images folder, "
                            "image=single image"
                        ))
    parser.add_argument("--input",  default="0",
                        help="Camera ID (webcam), image path, or images folder")
    parser.add_argument("--output", default=None,
                        help="Output directory for annotated images / JSON")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"],
                        help="Device — works on desktop GPU and Jetson Nano")
    parser.add_argument("--save-vis", action="store_true", default=False,
                        help="Save annotated frame for each image (folder mode)")
    args = parser.parse_args()

    # Validate model paths
    for p, label in [(args.yolo, "YOLO age/gender"), (args.mood, "Mood")]:
        if not os.path.exists(p):
            print(f"ERROR: {label} model not found: {p}")
            sys.exit(1)

    # Load both models
    print(f"\nLoading models on [{args.device}] ...")
    yolo_model, yolo_device = _load_yolo(args.yolo, args.device)
    mood_model, mood_device = _load_yolo(args.mood, args.device)

    print(f"\n{'='*55}")
    print(f"  Age/Gender model : {os.path.basename(args.yolo)}")
    print(f"  Mood model       : {os.path.basename(args.mood)}")
    print(f"  Mode             : {args.mode}")
    print(f"  Device           : {yolo_device}")
    print(f"{'='*55}\n")

    # Dispatch
    if args.mode == "image":
        run_image(args, yolo_model, yolo_device, mood_model, mood_device)
    elif args.mode == "folder":
        run_folder(args, yolo_model, yolo_device, mood_model, mood_device)
    else:
        run_webcam(args, yolo_model, yolo_device, mood_model, mood_device)

    # Jetson-safe teardown
    del yolo_model, mood_model
    _cuda_cleanup()


if __name__ == "__main__":
    main()
    os._exit(0)
