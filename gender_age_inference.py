#!/usr/bin/env python3
"""
gender_age_inference.py
-----------------------
Standalone inference script for the MobileNetV3-based GenderAge model.

Usage:
  # Single image
  python3 gender_age_inference.py --checkpoint best_checkpoint.pth --input path/to/image.jpg

  # Folder of images
  python3 gender_age_inference.py --checkpoint best_checkpoint.pth --input ./images --device cuda

  # Save results to JSON
  python3 gender_age_inference.py --checkpoint best_checkpoint.pth --input ./images --output results.json
"""

import os
import sys
import time
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import atexit
import gc

@atexit.register
def _cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# Import the model definition from the local module
try:
    from gender_age_model import (
        load_gender_age_model,
        GENDER_LABELS,
        AGE_LABELS,
        AGE_SHORT_LABELS,
    )
except ImportError as e:
    print(f"ERROR: Cannot import gender_age_model.py — {e}")
    print("Make sure gender_age_model.py is in the same directory.")
    sys.exit(1)

# ── Image pre-processing (matches training pipeline) ──────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 224

def preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 HxWx3  →  float32 torch 1x3x224x224 (ImageNet-normalised)"""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)                        # HWC → CHW
    return torch.from_numpy(img).unsqueeze(0)            # → 1×C×H×W


# ── Main inference class ──────────────────────────────────────────────────────

class GenderAgeInference:
    """Wraps the GenderAgeModel for convenient single-image / folder inference."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = load_gender_age_model(checkpoint_path, device)
        self.model_size_mb = os.path.getsize(checkpoint_path) / (1024 ** 2)

        # ── CUDA arch compatibility check ─────────────────────────────────────
        # Some GPUs (e.g. older Jetson) don't have a CUDA kernel image for the
        # installed PyTorch build.  Run a tiny warmup and fall back to CPU if it
        # fails — mirrors the same check in compare_models.py for YOLO.
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=self.device)
                with torch.no_grad():
                    self.model(dummy)
            except Exception:
                print("  [!] GenderAge: CUDA arch unsupported — falling back to CPU")
                self.device = "cpu"
                self.model = self.model.cpu()

    # ── Single image ──────────────────────────────────────────────────────────
    def predict_image(self, img_input):
        """
        Predict gender + age group for one image.

        Parameters
        ----------
        img_input : str (path) or np.ndarray (BGR uint8)

        Returns
        -------
        dict with keys: gender, gender_conf, age_class, age_label, age_conf,
                         combined_class (YOLO-style string), inference_time_ms
        """
        if isinstance(img_input, str):
            img = cv2.imread(img_input)
            if img is None:
                raise ValueError(f"Cannot read image: {img_input}")
        else:
            img = img_input

        tensor = preprocess(img).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            gender_logits, age_logits = self.model(tensor)
        inference_time_ms = (time.perf_counter() - t0) * 1000

        # Probabilities
        gender_probs = F.softmax(gender_logits, dim=1)[0].cpu().numpy()
        age_probs    = F.softmax(age_logits,    dim=1)[0].cpu().numpy()

        gender_idx = int(np.argmax(gender_probs))
        age_idx    = int(np.argmax(age_probs))

        # YOLO-style combined class name  e.g. "Female_Child"
        combined = f"{GENDER_LABELS[gender_idx]}_{AGE_SHORT_LABELS[age_idx]}"

        return {
            "gender":           GENDER_LABELS[gender_idx],
            "gender_conf":      float(gender_probs[gender_idx]),
            "age_class":        age_idx,
            "age_label":        AGE_LABELS[age_idx],
            "age_conf":         float(age_probs[age_idx]),
            "combined_class":   combined,
            "confidence":       float(gender_probs[gender_idx]),   # primary confidence
            "inference_time_ms": inference_time_ms,
            "gender_probs":     gender_probs.tolist(),
            "age_probs":        age_probs.tolist(),
        }

    # ── Benchmark ─────────────────────────────────────────────────────────────
    def benchmark(self, image_paths: list, num_images: int = 100):
        """
        Returns timing & memory stats over up to num_images images.
        Compatible with the dict format used by compare_models.py.
        """
        subset = image_paths[:min(num_images, len(image_paths))]

        inference_times = []
        confidences     = []
        predictions     = []

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        mem_start = torch.cuda.memory_allocated() if (self.device == "cuda") else 0

        total_start = time.perf_counter()

        for p in tqdm(subset, desc="  GenderAge inference"):
            img = cv2.imread(str(p))
            if img is None:
                continue
            result = self.predict_image(img)
            inference_times.append(result["inference_time_ms"])
            confidences.append(result["confidence"])
            predictions.append({
                "image":      os.path.basename(p),
                "class":      result["combined_class"],
                "confidence": result["confidence"],
            })

        total_time = time.perf_counter() - total_start
        mem_end    = torch.cuda.memory_allocated() if (self.device == "cuda") else 0
        peak_mem   = (torch.cuda.max_memory_allocated() / (1024**2)
                      if (self.device == "cuda") else 0)

        return {
            "model_name":     "GenderAge-MobileNetV3",
            "model_size_mb":  self.model_size_mb,
            "inference_times": inference_times,
            "confidences":    confidences,
            "predictions":    predictions,
            "total_time":     total_time,
            "fps":            len(predictions) / total_time if total_time > 0 else 0,
            "avg_time_ms":    np.mean(inference_times) if inference_times else 0,
            "std_time_ms":    np.std(inference_times)  if inference_times else 0,
            "min_time_ms":    min(inference_times)     if inference_times else 0,
            "max_time_ms":    max(inference_times)     if inference_times else 0,
            "avg_confidence": np.mean(confidences)     if confidences else 0,
            "peak_memory_mb": peak_mem,
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GenderAge MobileNetV3 Inference"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_checkpoint.pth")
    parser.add_argument("--input", required=True,
                        help="Image file or folder of images")
    parser.add_argument("--output", default=None,
                        help="Optional: save JSON results to this path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"])
    parser.add_argument("--num-images", type=int, default=None,
                        help="Limit number of images processed (useful for quick tests)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not os.path.exists(args.input):
        print(f"ERROR: input not found: {args.input}")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    infer = GenderAgeInference(args.checkpoint, args.device)

    # ── Collect images ────────────────────────────────────────────────────────
    if os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_paths = [
            str(p) for p in Path(args.input).rglob("*")
            if p.suffix.lower() in exts
        ]
        image_paths.sort()

    if args.num_images:
        image_paths = image_paths[:args.num_images]

    print(f"\nFound {len(image_paths)} image(s) — running on {args.device}")

    # ── Single-image demo ─────────────────────────────────────────────────────
    if len(image_paths) == 1:
        res = infer.predict_image(image_paths[0])
        print(f"\n{'='*60}")
        print(f"Image   : {os.path.basename(image_paths[0])}")
        print(f"Gender  : {res['gender']}  (conf {res['gender_conf']:.3f})")
        print(f"Age     : {res['age_label']}  (conf {res['age_conf']:.3f})")
        print(f"Combined: {res['combined_class']}")
        print(f"Time    : {res['inference_time_ms']:.2f} ms")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(res, f, indent=2)
            print(f"\n✓ Saved to {args.output}")
        return

    # ── Folder inference ───────────────────────────────────────────────────────
    print(f"\nProcessing {len(image_paths)} images one by one…")
    t0 = time.perf_counter()
    results = []
    for p in tqdm(image_paths, desc="Processing images"):
        res = infer.predict_image(str(p))
        res["image"] = os.path.basename(p)
        res["path"] = str(p)
        results.append(res)
    elapsed = time.perf_counter() - t0

    fps = len(results) / elapsed if elapsed > 0 else 0
    avg_g_conf = np.mean([r["gender_conf"] for r in results]) if results else 0
    avg_a_conf = np.mean([r["age_conf"]    for r in results]) if results else 0

    # Class distribution
    from collections import Counter
    class_dist = Counter(r["combined_class"] for r in results)

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Processed   : {len(results)} images")
    print(f"  Total time  : {elapsed:.2f} s")
    print(f"  FPS         : {fps:.2f}")
    print(f"  Avg gender conf : {avg_g_conf:.4f}")
    print(f"  Avg age conf    : {avg_a_conf:.4f}")
    print(f"\nClass distribution:")
    for cls, cnt in sorted(class_dist.items()):
        print(f"  {cls:<32} {cnt}")

    if args.output:
        payload = {
            "timestamp":     datetime.now().isoformat(),
            "checkpoint":    args.checkpoint,
            "device":        args.device,
            "total_images":  len(results),
            "fps":           fps,
            "avg_gender_conf": float(avg_g_conf),
            "avg_age_conf":    float(avg_a_conf),
            "class_distribution": dict(class_dist),
            "predictions":   results,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
