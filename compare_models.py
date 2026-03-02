#!/usr/bin/env python3
"""
compare_models.py — Unified Model Comparison Tool
==================================================
Compares two types of models on the same dataset:
  • YOLO  (.pt / .onnx / .engine) — Ultralytics YOLO classification
  • GenderAge (.pth)              — MobileNetV3-Small custom model

Usage:
  python3 compare_models.py \\
      --model-yolo      best.pt \\
      --model-gender-age best_checkpoint.pth \\
      --images          ./images \\
      --output          ./comparison_output \\
      --device          cpu \\
      --num-images      100
"""

import os
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import atexit
import gc

@atexit.register
def _cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# Optional pretty tables
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# ── Conditional imports ────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("WARNING: ultralytics not found — YOLO model will be skipped.")

try:
    from gender_age_inference import GenderAgeInference
    HAS_GENDER_AGE = True
except ImportError:
    HAS_GENDER_AGE = False
    print("WARNING: gender_age_inference.py not found — GenderAge model will be skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def collect_images(root: str, recursive: bool = True) -> list:
    p = Path(root)
    if p.is_file():
        return [str(p)]
    fn = p.rglob if recursive else p.glob
    paths = [str(f) for f in fn("*") if f.suffix.lower() in IMAGE_EXTS]
    paths.sort()
    return paths


def save_annotated_image(img_bgr: np.ndarray, label: str, conf: float,
                         vis_dir: str, filename: str):
    """
    Write a copy of img_bgr with prediction text burned in.
    Organises into vis_dir/<class_name>/<filename>.
    """
    class_dir = os.path.join(vis_dir, label)
    os.makedirs(class_dir, exist_ok=True)

    out = img_bgr.copy()
    text = f"{label}  {conf:.2f}"

    # Background rectangle for readability
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(out, (8, 6), (14 + tw, 14 + th + 6), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 10 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(os.path.join(class_dir, filename), out)


# ─────────────────────────────────────────────────────────────────────────────
# YOLO benchmarker
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_yolo(model_path: str, image_paths: list, device: str,
                   num_images: int, vis_dir: str = None):
    if not HAS_YOLO:
        return None

    subset = image_paths[:min(num_images, len(image_paths))]
    model_size_mb = os.path.getsize(model_path) / (1024 ** 2)

    print(f"\nLoading YOLO model: {model_path}")
    model = YOLO(model_path)
    if device == "cuda" and torch.cuda.is_available():
        model.to("cuda")
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            model.predict(dummy, verbose=False, device=device, half=True)
        except Exception:
            print("  [!] CUDA arch unsupported — falling back to CPU")
            device = "cpu"
            model.to("cpu")

    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        print(f"  Saving annotated images → {vis_dir}")

    print(f"  Size: {model_size_mb:.2f} MB | Device: {device}")
    print(f"  Benchmarking on {len(subset)} images…")

    is_half = device == "cuda"
    inference_times = []
    confidences     = []
    predictions     = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    total_start = time.perf_counter()

    for img_path in tqdm(subset, desc="  YOLO"):
        img = cv2.imread(img_path)
        if img is None:
            continue

        t0 = time.perf_counter()
        preds = model.predict(img, verbose=False, augment=False,
                              device=device, half=is_half)
        t1 = time.perf_counter()

        inference_times.append((t1 - t0) * 1000)

        if preds and preds[0].probs is not None:
            probs   = preds[0].probs
            top1_i  = probs.top1
            top1_c  = float(probs.data[top1_i])
            top1_n  = preds[0].names[top1_i]
            confidences.append(top1_c)
            predictions.append({"image": os.path.basename(img_path),
                                 "class": top1_n,
                                 "confidence": top1_c})

            if vis_dir:
                save_annotated_image(img, top1_n, top1_c,
                                     vis_dir, os.path.basename(img_path))

    total_time = time.perf_counter() - total_start
    peak_mem   = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                  if torch.cuda.is_available() else 0)

    return {
        "model_name":     "YOLO",
        "model_file":     os.path.basename(model_path),
        "model_size_mb":  model_size_mb,
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


# ─────────────────────────────────────────────────────────────────────────────
# GenderAge benchmarker  (delegates to GenderAgeInference.benchmark)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_gender_age(checkpoint: str, image_paths: list, device: str,
                         num_images: int, vis_dir: str = None):
    if not HAS_GENDER_AGE:
        return None

    infer = GenderAgeInference(checkpoint, device)

    if vis_dir:
        # Run inference with image saving inline (can't delegate to .benchmark)
        os.makedirs(vis_dir, exist_ok=True)
        print(f"  Saving annotated images → {vis_dir}")

        subset = image_paths[:min(num_images, len(image_paths))]
        inference_times = []
        confidences     = []
        predictions     = []

        if infer.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        total_start = time.perf_counter()

        for img_path in tqdm(subset, desc="  GenderAge inference"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            res = infer.predict_image(img)
            inference_times.append(res["inference_time_ms"])
            confidences.append(res["confidence"])
            predictions.append({"image":      os.path.basename(img_path),
                                 "class":      res["combined_class"],
                                 "confidence": res["confidence"]})
            save_annotated_image(img, res["combined_class"], res["confidence"],
                                 vis_dir, os.path.basename(img_path))

        total_time = time.perf_counter() - total_start
        peak_mem   = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                      if (infer.device == "cuda") else 0)

        result = {
            "model_name":      "GenderAge-MobileNetV3",
            "model_file":      os.path.basename(checkpoint),
            "model_size_mb":   infer.model_size_mb,
            "inference_times": inference_times,
            "confidences":     confidences,
            "predictions":     predictions,
            "total_time":      total_time,
            "fps":             len(predictions) / total_time if total_time > 0 else 0,
            "avg_time_ms":     np.mean(inference_times) if inference_times else 0,
            "std_time_ms":     np.std(inference_times)  if inference_times else 0,
            "min_time_ms":     min(inference_times)     if inference_times else 0,
            "max_time_ms":     max(inference_times)     if inference_times else 0,
            "avg_confidence":  np.mean(confidences)     if confidences else 0,
            "peak_memory_mb":  peak_mem,
        }
    else:
        result = infer.benchmark(image_paths, num_images=num_images)
        result["model_file"] = os.path.basename(checkpoint)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reporting & plots
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(results: dict):
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":            name,
            "File":             r.get("model_file", ""),
            "Size (MB)":        f"{r['model_size_mb']:.2f}",
            "FPS":              f"{r['fps']:.1f}",
            "Avg Time (ms)":    f"{r['avg_time_ms']:.2f} ± {r['std_time_ms']:.2f}",
            "Min Time (ms)":    f"{r['min_time_ms']:.2f}",
            "Max Time (ms)":    f"{r['max_time_ms']:.2f}",
            "Peak Mem (MB)":    f"{r['peak_memory_mb']:.2f}",
            "Avg Confidence":   f"{r['avg_confidence']:.4f}",
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 90)
    if HAS_TABULATE:
        print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
    else:
        print(df.to_string(index=False))
    print("=" * 90)
    return df


def generate_plots(results: dict, output_dir: str, timestamp: str):
    names = list(results.keys())
    if not names:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "Model Comparison Dashboard — YOLO vs GenderAge-MobileNetV3",
        fontsize=15, fontweight="bold", y=1.01
    )

    palette = ["#4A90D9", "#E67E22", "#2ECC71", "#E74C3C"]

    # 1. Avg inference time
    ax = axes[0, 0]
    times  = [results[m]["avg_time_ms"] for m in names]
    errors = [results[m]["std_time_ms"]  for m in names]
    bars   = ax.bar(names, times, yerr=errors, capsize=6,
                    color=palette[:len(names)], alpha=0.85)
    ax.set_ylabel("Inference Time (ms)")
    ax.set_title("Average Inference Time per Image")
    ax.grid(axis="y", alpha=0.3)
    for b, t in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{t:.1f} ms", ha="center", va="bottom", fontsize=9)

    # 2. FPS
    ax = axes[0, 1]
    fps_vals = [results[m]["fps"] for m in names]
    bars = ax.bar(names, fps_vals, color=palette[:len(names)], alpha=0.85)
    ax.set_ylabel("FPS")
    ax.set_title("Frames Per Second")
    ax.grid(axis="y", alpha=0.3)
    for b, f in zip(bars, fps_vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                f"{f:.1f}", ha="center", va="bottom", fontsize=9)

    # 3. Model size
    ax = axes[0, 2]
    sizes = [results[m]["model_size_mb"] for m in names]
    bars  = ax.bar(names, sizes, color=palette[:len(names)], alpha=0.85)
    ax.set_ylabel("Model Size (MB)")
    ax.set_title("Model File Size")
    ax.grid(axis="y", alpha=0.3)
    for b, s in zip(bars, sizes):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2,
                f"{s:.1f} MB", ha="center", va="bottom", fontsize=9)

    # 4. Peak memory
    ax = axes[1, 0]
    mem = [results[m]["peak_memory_mb"] for m in names]
    bars = ax.bar(names, mem, color=palette[:len(names)], alpha=0.85)
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title("Peak GPU Memory Usage")
    ax.grid(axis="y", alpha=0.3)
    for b, m_ in zip(bars, mem):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2,
                f"{m_:.1f}", ha="center", va="bottom", fontsize=9)

    # 5. Avg confidence
    ax = axes[1, 1]
    conf = [results[m]["avg_confidence"] for m in names]
    bars = ax.bar(names, conf, color=palette[:len(names)], alpha=0.85)
    ax.set_ylabel("Average Confidence")
    ax.set_title("Average Prediction Confidence")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    for b, c in zip(bars, conf):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{c:.3f}", ha="center", va="bottom", fontsize=9)

    # 6. Inference time distribution (box-like bar with min/avg/max)
    ax = axes[1, 2]
    x = np.arange(len(names))
    width = 0.25
    mins = [results[m]["min_time_ms"] for m in names]
    avgs = [results[m]["avg_time_ms"]  for m in names]
    maxs = [results[m]["max_time_ms"]  for m in names]

    ax.bar(x - width, mins, width, label="Min",  color="#2ECC71", alpha=0.8)
    ax.bar(x,         avgs, width, label="Avg",  color="#3498DB", alpha=0.8)
    ax.bar(x + width, maxs, width, label="Max",  color="#E74C3C", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Inference Time: Min / Avg / Max")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"comparison_plots_{timestamp}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Plots saved: {plot_path}")
    return plot_path


def save_results(results: dict, df: pd.DataFrame, output_dir: str, timestamp: str):
    # CSV summary
    csv_path = os.path.join(output_dir, f"comparison_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV saved : {csv_path}")

    # JSON detailed (drop raw lists)
    exportable = {}
    for name, r in results.items():
        exportable[name] = {
            k: v for k, v in r.items()
            if k not in ("inference_times", "confidences", "predictions")
        }
    json_path = os.path.join(output_dir, f"comparison_detail_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(exportable, f, indent=2, default=str)
    print(f"✓ JSON saved: {json_path}")

    # Per-model predictions (one JSON each)
    for name, r in results.items():
        preds_path = os.path.join(output_dir,
                                  f"{name.replace(' ', '_')}_predictions_{timestamp}.json")
        with open(preds_path, "w") as f:
            json.dump(r.get("predictions", []), f, indent=2)
    return csv_path, json_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified YOLO + GenderAge model comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-yolo",       default=None,
                        help="Path to YOLO model (.pt / .onnx / .engine)")
    parser.add_argument("--model-gender-age", default=None,
                        help="Path to GenderAge checkpoint (.pth)")
    parser.add_argument("--images", required=True,
                        help="Image file or folder of images to test on")
    parser.add_argument("--output", default="./comparison_output",
                        help="Directory to save plots, CSV & JSON")
    parser.add_argument("--num-images", type=int, default=100,
                        help="Max number of images to benchmark per model")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"])
    parser.add_argument("--recursive", action="store_true", default=True,
                        help="Recurse into image subdirectories")
    parser.add_argument("--save-vis", action="store_true", default=False,
                        help="Save annotated images per model to <output>/YOLO_vis/ and <output>/GenderAge_vis/")
    args = parser.parse_args()

    # At least one model required
    if not args.model_yolo and not args.model_gender_age:
        parser.error("Provide at least one of --model-yolo or --model-gender-age")

    # Validate paths
    for attr, label in [("model_yolo", "YOLO"), ("model_gender_age", "GenderAge")]:
        p = getattr(args, attr)
        if p and not os.path.exists(p):
            print(f"ERROR: {label} model not found: {p}")
            sys.exit(1)

    if not os.path.exists(args.images):
        print(f"ERROR: --images path not found: {args.images}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Collect images
    image_paths = collect_images(args.images, recursive=args.recursive)
    if not image_paths:
        print(f"ERROR: No images found in {args.images}")
        sys.exit(1)
    print(f"\nFound {len(image_paths)} image(s) in {args.images}")
    print(f"Benchmarking up to {args.num_images} per model on [{args.device}]")

    # ── Visualisation dirs ──────────────────────────────────────────────────
    yolo_vis_dir = os.path.join(args.output, "YOLO_vis")       if args.save_vis else None
    ga_vis_dir   = os.path.join(args.output, "GenderAge_vis")  if args.save_vis else None
    if args.save_vis:
        print(f"  [save-vis] YOLO     → {yolo_vis_dir}")
        print(f"  [save-vis] GenderAge → {ga_vis_dir}")

    # ── Run benchmarks ─────────────────────────────────────────────────────
    all_results = {}

    if args.model_yolo:
        r = benchmark_yolo(args.model_yolo, image_paths, args.device,
                           args.num_images, vis_dir=yolo_vis_dir)
        if r:
            all_results["YOLO"] = r
            print(f"  → YOLO : FPS={r['fps']:.1f}  "
                  f"avg={r['avg_time_ms']:.2f}ms  "
                  f"conf={r['avg_confidence']:.4f}")

    if args.model_gender_age:
        r = benchmark_gender_age(args.model_gender_age, image_paths,
                                 args.device, args.num_images, vis_dir=ga_vis_dir)
        if r:
            all_results["GenderAge-MobileNetV3"] = r
            print(f"  → GenderAge : FPS={r['fps']:.1f}  "
                  f"avg={r['avg_time_ms']:.2f}ms  "
                  f"conf={r['avg_confidence']:.4f}")

    if not all_results:
        print("ERROR: No models ran successfully.")
        sys.exit(1)

    # ── Report ─────────────────────────────────────────────────────────────
    df = print_comparison_table(all_results)
    save_results(all_results, df, args.output, timestamp)
    plot_path = generate_plots(all_results, args.output, timestamp)

    print(f"\n✅  Comparison complete! Results saved to: {args.output}")

    # Quick winner summary
    if len(all_results) >= 2:
        names = list(all_results.keys())
        fastest = min(names, key=lambda n: all_results[n]["avg_time_ms"])
        smallest = min(names, key=lambda n: all_results[n]["model_size_mb"])
        print(f"\n  🏆 Fastest model : {fastest} "
              f"({all_results[fastest]['avg_time_ms']:.2f} ms avg)")
        print(f"  🏆 Smallest model: {smallest} "
              f"({all_results[smallest]['model_size_mb']:.2f} MB)")


if __name__ == "__main__":
    main()
