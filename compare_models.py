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
import threading
import subprocess
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import cv2
import psutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import atexit
import gc

# Only True after CUDA is actually used — prevents atexit from touching
# the CUDA context when it was never initialized (e.g. early import error).
_cuda_was_used = False

@atexit.register
def _cleanup():
    try:
        if _cuda_was_used and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

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
# Resource Monitor  — background thread that samples GPU / CPU / RAM
# ─────────────────────────────────────────────────────────────────────────────

def _get_gpu_util_pct() -> float:
    """Query GPU utilization % via nvidia-smi (works on Jetson JetPack 6)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=1
        ).decode().strip().split("\n")[0]
        return float(out)
    except Exception:
        return 0.0


class ResourceMonitor:
    """
    Polls CPU %, RAM %, and GPU util % in a background thread.
    Call start() before inference and stop() after.
    """
    INTERVAL = 0.25   # seconds between samples

    def __init__(self):
        self._samples_cpu  = []
        self._samples_ram  = []
        self._samples_gpu  = []
        self._running      = False
        self._thread       = None

    def start(self):
        self._samples_cpu.clear()
        self._samples_ram.clear()
        self._samples_gpu.clear()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            self._samples_cpu.append(psutil.cpu_percent(interval=None))
            self._samples_ram.append(psutil.virtual_memory().percent)
            self._samples_gpu.append(_get_gpu_util_pct())
            time.sleep(self.INTERVAL)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def avg_cpu(self):
        return float(np.mean(self._samples_cpu)) if self._samples_cpu else 0.0

    @property
    def avg_ram(self):
        return float(np.mean(self._samples_ram)) if self._samples_ram else 0.0

    @property
    def avg_gpu_util(self):
        return float(np.mean(self._samples_gpu)) if self._samples_gpu else 0.0

    @property
    def max_gpu_util(self):
        return float(np.max(self._samples_gpu)) if self._samples_gpu else 0.0


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


def _gpu_mem_snapshot():
    """Return (total_mb, alloc_mb, cached_mb) from torch CUDA."""
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return 0.0, 0.0, 0.0
    total  = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    alloc  = torch.cuda.memory_allocated()  / (1024**2)
    cached = torch.cuda.memory_reserved()   / (1024**2)
    return total, alloc, cached


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

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(out, (8, 6), (14 + tw, 14 + th + 6), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 10 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(os.path.join(class_dir, filename), out)


def _fmt(v, fmt=".2f"):
    return format(v, fmt) if isinstance(v, float) else str(v)


def _print_table(rows, headers):
    df = pd.DataFrame(rows, columns=headers)
    if HAS_TABULATE:
        print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
    else:
        print(df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# YOLO benchmarker
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_yolo(model_path: str, image_paths: list, device: str,
                   num_images: int, vis_dir: str = None):
    if not HAS_YOLO:
        return None

    subset = image_paths[:min(num_images, len(image_paths))]
    model_size_mb = os.path.getsize(model_path) / (1024 ** 2)
    fmt = Path(model_path).suffix.lstrip(".")

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

    monitor = ResourceMonitor()
    monitor.start()
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
    monitor.stop()

    gpu_total, gpu_alloc, gpu_cached = _gpu_mem_snapshot()
    peak_mem = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                if torch.cuda.is_available() else 0)

    # Clean up model to avoid Jetson memory corruption on exit
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "model_name":      "YOLO",
        "model_file":      os.path.basename(model_path),
        "model_format":    fmt.upper(),
        "model_size_mb":   model_size_mb,
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
        "gpu_mem_total_mb":  gpu_total,
        "gpu_mem_alloc_mb":  gpu_alloc,
        "gpu_mem_cached_mb": gpu_cached,
        "avg_cpu_pct":     monitor.avg_cpu,
        "avg_ram_pct":     monitor.avg_ram,
        "avg_gpu_util_pct": monitor.avg_gpu_util,
        "max_gpu_util_pct": monitor.max_gpu_util,
        "total_images":    len(predictions),
        "device":          device,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GenderAge benchmarker
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_gender_age(checkpoint: str, image_paths: list, device: str,
                         num_images: int, vis_dir: str = None):
    if not HAS_GENDER_AGE:
        return None

    infer = GenderAgeInference(checkpoint, device)
    fmt   = Path(checkpoint).suffix.lstrip(".")

    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        print(f"  Saving annotated images → {vis_dir}")

    subset = image_paths[:min(num_images, len(image_paths))]
    inference_times = []
    confidences     = []
    predictions     = []

    if infer.device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    monitor = ResourceMonitor()
    monitor.start()
    total_start = time.perf_counter()

    for img_path in tqdm(subset, desc="  GenderAge"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        res = infer.predict_image(img)
        inference_times.append(res["inference_time_ms"])
        confidences.append(res["confidence"])
        predictions.append({"image":      os.path.basename(img_path),
                             "class":      res["combined_class"],
                             "confidence": res["confidence"]})
        if vis_dir:
            save_annotated_image(img, res["combined_class"], res["confidence"],
                                 vis_dir, os.path.basename(img_path))

    total_time = time.perf_counter() - total_start
    monitor.stop()

    gpu_total, gpu_alloc, gpu_cached = _gpu_mem_snapshot()
    peak_mem = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                if (infer.device == "cuda" and torch.cuda.is_available()) else 0)

    # Clean up model to avoid Jetson memory corruption on exit
    del infer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "model_name":      "GenderAge-MobileNetV3",
        "model_file":      os.path.basename(checkpoint),
        "model_format":    fmt.upper(),
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
        "gpu_mem_total_mb":  gpu_total,
        "gpu_mem_alloc_mb":  gpu_alloc,
        "gpu_mem_cached_mb": gpu_cached,
        "avg_cpu_pct":     monitor.avg_cpu,
        "avg_ram_pct":     monitor.avg_ram,
        "avg_gpu_util_pct": monitor.avg_gpu_util,
        "max_gpu_util_pct": monitor.max_gpu_util,
        "total_images":    len(predictions),
        "device":          infer.device,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rich terminal output
# ─────────────────────────────────────────────────────────────────────────────

def print_all_reports(results: dict):
    vm   = psutil.virtual_memory()
    sep  = "=" * 100

    for name, r in results.items():
        print(f"\n{sep}")
        print(f"  MODEL: {name}  |  file: {r['model_file']}  |  format: {r['model_format']}  |  device: {r['device']}")
        print(sep)

        # ── Table 1: per-model live metrics ──────────────────────────────────
        print("\n[ Inference Metrics ]\n")
        rows1 = [{
            "Model Name":       name,
            "GPU Util %":       f"{r['avg_gpu_util_pct']:.1f}",
            "GPU Mem Total(MB)":f"{r['gpu_mem_total_mb']:.0f}",
            "GPU Mem Alloc(MB)":f"{r['gpu_mem_alloc_mb']:.1f}",
            "GPU Cached(MB)":   f"{r['gpu_mem_cached_mb']:.1f}",
            "CPU %":            f"{r['avg_cpu_pct']:.1f}",
            "RAM %":            f"{r['avg_ram_pct']:.1f}",
            "Format":           r["model_format"],
            "Size (MB)":        f"{r['model_size_mb']:.2f}",
            "FPS":              f"{r['fps']:.1f}",
            "Avg Time (ms)":    f"{r['avg_time_ms']:.2f}",
            "Peak Mem (MB)":    f"{r['peak_memory_mb']:.1f}",
            "Total Time (s)":   f"{r['total_time']:.1f}",
        }]
        _print_table(rows1, list(rows1[0].keys()))

        # ── Table 2: summary stats ────────────────────────────────────────────
        print("\n[ Run Summary ]\n")
        rows2 = [{
            "Total Images":  r["total_images"],
            "Total Time (s)":f"{r['total_time']:.2f}",
            "Avg Time (ms)": f"{r['avg_time_ms']:.2f}",
            "FPS":           f"{r['fps']:.1f}",
            "Avg CPU %":     f"{r['avg_cpu_pct']:.1f}",
            "Avg RAM %":     f"{r['avg_ram_pct']:.1f}",
            "Avg GPU Util %":f"{r['avg_gpu_util_pct']:.1f}",
        }]
        _print_table(rows2, list(rows2[0].keys()))

        # ── Table 3: system memory snapshot ─────────────────────────────────
        print("\n[ System Memory ]\n")
        rows3 = [{
            "RAM Total (GB)":      f"{vm.total / (1024**3):.1f}",
            "RAM Available (GB)":  f"{vm.available / (1024**3):.1f}",
            "RAM Used %":          f"{vm.percent:.1f}",
            "GPU Mem Total (MB)":  f"{r['gpu_mem_total_mb']:.0f}",
            "GPU Mem Alloc (MB)":  f"{r['gpu_mem_alloc_mb']:.1f}",
            "GPU Mem Cached (MB)": f"{r['gpu_mem_cached_mb']:.1f}",
            "Peak Infer Mem (MB)": f"{r['peak_memory_mb']:.1f}",
        }]
        _print_table(rows3, list(rows3[0].keys()))

        # ── Table 4: class distribution ──────────────────────────────────────
        print("\n[ Class Results ]\n")
        class_counts = Counter(p["class"] for p in r["predictions"])
        total = sum(class_counts.values())
        rows4 = [{"Class": cls,
                  "Count": cnt,
                  "% of Total": f"{100 * cnt / total:.1f}%" if total else "0%"}
                 for cls, cnt in sorted(class_counts.items(),
                                        key=lambda x: -x[1])]
        _print_table(rows4, ["Class", "Count", "% of Total"])

    # ── Cross-model comparison (if more than one model ran) ──────────────────
    if len(results) >= 2:
        print(f"\n{sep}")
        print("  CROSS-MODEL COMPARISON")
        print(sep)
        rows_cmp = []
        for name, r in results.items():
            rows_cmp.append({
                "Model":          name,
                "FPS":            f"{r['fps']:.1f}",
                "Avg ms":         f"{r['avg_time_ms']:.2f}",
                "Peak Mem (MB)":  f"{r['peak_memory_mb']:.1f}",
                "GPU Util %":     f"{r['avg_gpu_util_pct']:.1f}",
                "CPU %":          f"{r['avg_cpu_pct']:.1f}",
                "RAM %":          f"{r['avg_ram_pct']:.1f}",
                "Size (MB)":      f"{r['model_size_mb']:.2f}",
                "Avg Conf":       f"{r['avg_confidence']:.4f}",
            })
        _print_table(rows_cmp, list(rows_cmp[0].keys()))

        names    = list(results.keys())
        fastest  = min(names, key=lambda n: results[n]["avg_time_ms"])
        smallest = min(names, key=lambda n: results[n]["model_size_mb"])
        print(f"\n  🏆 Fastest : {fastest} ({results[fastest]['avg_time_ms']:.2f} ms avg)")
        print(f"  🏆 Smallest: {smallest} ({results[smallest]['model_size_mb']:.2f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def generate_plots(results: dict, output_dir: str, timestamp: str):
    names = list(results.keys())
    if not names:
        return

    palette = ["#4A90D9", "#E67E22", "#2ECC71", "#E74C3C"]

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle("Model Comparison Dashboard", fontsize=16, fontweight="bold", y=1.01)

    def bar(ax, vals, title, ylabel, labels=None):
        labels = labels or names
        b = ax.bar(labels, vals, color=palette[:len(labels)], alpha=0.85)
        ax.set_title(title); ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        for bar_, v in zip(b, vals):
            ax.text(bar_.get_x() + bar_.get_width() / 2,
                    bar_.get_height() * 1.02,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    # Row 0
    bar(axes[0,0], [results[m]["avg_time_ms"] for m in names], "Avg Inference Time", "ms")
    bar(axes[0,1], [results[m]["fps"] for m in names], "FPS", "frames/s")
    bar(axes[0,2], [results[m]["model_size_mb"] for m in names], "Model Size", "MB")

    # Row 1
    bar(axes[1,0], [results[m]["peak_memory_mb"] for m in names], "Peak GPU Memory", "MB")
    bar(axes[1,1], [results[m]["avg_gpu_util_pct"] for m in names], "Avg GPU Util", "%")
    bar(axes[1,2], [results[m]["avg_cpu_pct"] for m in names], "Avg CPU Usage", "%")

    # Row 2
    bar(axes[2,0], [results[m]["avg_ram_pct"] for m in names], "Avg RAM Usage", "%")
    bar(axes[2,1], [results[m]["avg_confidence"] for m in names], "Avg Confidence", "score")

    # Class distribution for last model
    last_name = names[-1]
    class_counts = Counter(p["class"] for p in results[last_name]["predictions"])
    top_n = 10
    top_classes = class_counts.most_common(top_n)
    if top_classes:
        cls_labels, cls_vals = zip(*top_classes)
        axes[2,2].barh(cls_labels[::-1], cls_vals[::-1], color=palette[1], alpha=0.85)
        axes[2,2].set_title(f"Top Class Distribution ({last_name})")
        axes[2,2].set_xlabel("Count")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"comparison_plots_{timestamp}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Plots saved: {plot_path}")
    return plot_path


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: dict, output_dir: str, timestamp: str):
    # CSV summary (one row per model)
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":             name,
            "File":              r["model_file"],
            "Format":            r["model_format"],
            "Size_MB":           r["model_size_mb"],
            "FPS":               r["fps"],
            "Avg_ms":            r["avg_time_ms"],
            "Std_ms":            r["std_time_ms"],
            "Min_ms":            r["min_time_ms"],
            "Max_ms":            r["max_time_ms"],
            "Total_Time_s":      r["total_time"],
            "Total_Images":      r["total_images"],
            "Avg_Confidence":    r["avg_confidence"],
            "Peak_Mem_MB":       r["peak_memory_mb"],
            "GPU_Mem_Total_MB":  r["gpu_mem_total_mb"],
            "GPU_Mem_Alloc_MB":  r["gpu_mem_alloc_mb"],
            "GPU_Mem_Cached_MB": r["gpu_mem_cached_mb"],
            "Avg_CPU_Pct":       r["avg_cpu_pct"],
            "Avg_RAM_Pct":       r["avg_ram_pct"],
            "Avg_GPU_Util_Pct":  r["avg_gpu_util_pct"],
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"comparison_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV saved : {csv_path}")

    # JSON — structured with all requested sections per model
    vm = psutil.virtual_memory()
    exportable = {}
    for name, r in results.items():
        class_counts = Counter(p["class"] for p in r.get("predictions", []))
        total_preds  = sum(class_counts.values()) or 1
        exportable[name] = {
            "inference_metrics": {
                "model_name":         name,
                "model_file":         r["model_file"],
                "format":             r["model_format"],
                "size_mb":            round(r["model_size_mb"], 2),
                "fps":                round(r["fps"], 2),
                "avg_time_ms":        round(r["avg_time_ms"], 3),
                "peak_mem_mb":        round(r["peak_memory_mb"], 2),
                "total_time_s":       round(r["total_time"], 2),
                "gpu_util_pct":       round(r["avg_gpu_util_pct"], 1),
                "gpu_mem_total_mb":   round(r["gpu_mem_total_mb"], 1),
                "gpu_mem_alloc_mb":   round(r["gpu_mem_alloc_mb"], 2),
                "gpu_mem_cached_mb":  round(r["gpu_mem_cached_mb"], 2),
                "cpu_usage_pct":      round(r["avg_cpu_pct"], 1),
                "ram_usage_pct":      round(r["avg_ram_pct"], 1),
            },
            "run_summary": {
                "total_images":    r["total_images"],
                "total_time_s":    round(r["total_time"], 2),
                "avg_time_ms":     round(r["avg_time_ms"], 3),
                "fps":             round(r["fps"], 2),
                "avg_cpu_pct":     round(r["avg_cpu_pct"], 1),
                "avg_ram_pct":     round(r["avg_ram_pct"], 1),
                "avg_gpu_util_pct":round(r["avg_gpu_util_pct"], 1),
            },
            "system_memory": {
                "ram_total_gb":       round(vm.total / (1024**3), 2),
                "ram_available_gb":   round(vm.available / (1024**3), 2),
                "ram_used_pct":       round(vm.percent, 1),
                "gpu_mem_total_mb":   round(r["gpu_mem_total_mb"], 1),
                "gpu_mem_alloc_mb":   round(r["gpu_mem_alloc_mb"], 2),
                "gpu_mem_cached_mb":  round(r["gpu_mem_cached_mb"], 2),
                "peak_infer_mem_mb":  round(r["peak_memory_mb"], 2),
            },
            "class_results": [
                {
                    "class":        cls,
                    "count":        cnt,
                    "pct_of_total": round(100 * cnt / total_preds, 2),
                }
                for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])
            ],
        }
    json_path = os.path.join(output_dir, f"comparison_detail_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(exportable, f, indent=2, default=str)
    print(f"✓ JSON saved: {json_path}")

    # Per-model predictions
    for name, r in results.items():
        preds_path = os.path.join(
            output_dir,
            f"{name.replace(' ', '_')}_predictions_{timestamp}.json"
        )
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
                        help="Save annotated images per model")
    args = parser.parse_args()

    if not args.model_yolo and not args.model_gender_age:
        parser.error("Provide at least one of --model-yolo or --model-gender-age")

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

    image_paths = collect_images(args.images, recursive=args.recursive)
    if not image_paths:
        print(f"ERROR: No images found in {args.images}")
        sys.exit(1)
    print(f"\nFound {len(image_paths)} image(s) in {args.images}")
    print(f"Benchmarking up to {args.num_images} per model on [{args.device}]")

    yolo_vis_dir = os.path.join(args.output, "YOLO_vis")      if args.save_vis else None
    ga_vis_dir   = os.path.join(args.output, "GenderAge_vis") if args.save_vis else None
    if args.save_vis:
        print(f"  [save-vis] YOLO      → {yolo_vis_dir}")
        print(f"  [save-vis] GenderAge → {ga_vis_dir}")

    # ── Run benchmarks ──────────────────────────────────────────────────────
    all_results = {}

    if args.model_yolo:
        r = benchmark_yolo(args.model_yolo, image_paths, args.device,
                           args.num_images, vis_dir=yolo_vis_dir)
        if r:
            all_results["YOLO"] = r
            global _cuda_was_used
            if r["device"] == "cuda":
                _cuda_was_used = True

    if args.model_gender_age:
        r = benchmark_gender_age(args.model_gender_age, image_paths,
                                 args.device, args.num_images, vis_dir=ga_vis_dir)
        if r:
            all_results["GenderAge-MobileNetV3"] = r
            if r["device"] == "cuda":
                _cuda_was_used = True

    if not all_results:
        print("ERROR: No models ran successfully.")
        sys.exit(1)

    # ── Reports ─────────────────────────────────────────────────────────────
    print_all_reports(all_results)
    save_results(all_results, args.output, timestamp)
    generate_plots(all_results, args.output, timestamp)

    print(f"\n✅  Comparison complete! Results saved to: {args.output}")

    # Explicit global cleanup to prevent Jetson memory corruption
    del all_results
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
