#!/usr/bin/env python3
"""
Standalone ONNX age+gender inference + model-to-model comparison.

This script is intentionally independent from the existing YOLO-based tools
in this repository. It compares two ONNX models without modifying either.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: numpy. Install with `pip install numpy`."
    ) from e

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: opencv-python. Install with `pip install opencv-python`."
    ) from e

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pandas. Install with `pip install pandas`."
    ) from e

try:
    import onnxruntime as ort  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: onnxruntime. Install with `pip install onnxruntime` "
        "(or `pip install onnxruntime-gpu` if you have CUDA)."
    ) from e


AGE_BIN_LABELS: List[str] = [
    "Child (0-16)",
    "Young Adults (17-30)",
    "Middle-aged Adults (31-45)",
    "Old-aged Adults (45+)",
]

ALL_8_CLASSES: List[str] = [
    "Female_Child",
    "Female_YoungAdult",
    "Female_MiddleAged",
    "Female_OldAged",
    "Male_Child",
    "Male_YoungAdult",
    "Male_MiddleAged",
    "Male_OldAged",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def _available_providers() -> List[str]:
    try:
        return list(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]


def _select_providers(prefer_cuda: bool) -> List[str]:
    providers = _available_providers()
    if prefer_cuda and "CUDAExecutionProvider" in providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _infer_input_hw(session: "ort.InferenceSession", fallback: int = 224) -> Tuple[int, int]:
    inp = session.get_inputs()[0]
    shape = list(inp.shape) if inp.shape is not None else []

    # Expect NCHW: [N, 3, H, W]. Some exports may have dynamic dims as None or strings.
    h = None
    w = None
    if len(shape) >= 4:
        h_raw, w_raw = shape[-2], shape[-1]
        if isinstance(h_raw, int):
            h = h_raw
        if isinstance(w_raw, int):
            w = w_raw

    if not h or not w:
        return (fallback, fallback)
    return (int(h), int(w))


def _preprocess_from_bgr(
    img_bgr: np.ndarray,
    hw: Tuple[int, int],
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
) -> np.ndarray:
    h, w = hw
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, 0)  # NCHW
    return img.astype(np.float32, copy=False)


@dataclass
class ModelPrediction:
    gender: Optional[str] = None  # "female" | "male"
    age_group: Optional[str] = None  # label from AGE_BIN_LABELS
    gender_conf: Optional[float] = None
    age_conf: Optional[float] = None
    combined_class: Optional[str] = None  # e.g. Female_YoungAdult
    combined_conf: Optional[float] = None
    inference_ms: Optional[float] = None
    output_names: Optional[List[str]] = None


def _decode_prediction(output_map: Dict[str, np.ndarray]) -> ModelPrediction:
    pred = ModelPrediction(output_names=list(output_map.keys()))

    # Flatten batch dimension when present (N=1)
    def squeeze(arr: np.ndarray) -> np.ndarray:
        if arr is None:
            return arr
        if arr.ndim >= 2 and arr.shape[0] == 1:
            return arr[0]
        return arr

    squeezed: Dict[str, np.ndarray] = {k: squeeze(v) for k, v in output_map.items()}
    outputs = list(squeezed.items())

    # 1) Combined 8-class head
    for name, arr in outputs:
        arr = np.asarray(arr)
        if arr.ndim == 1 and arr.shape[0] == 8:
            probs = _softmax(arr)
            idx = int(np.argmax(probs))
            cls = ALL_8_CLASSES[idx] if idx < len(ALL_8_CLASSES) else f"class_{idx}"
            pred.combined_class = cls
            pred.combined_conf = float(probs[idx])
            if "_" in cls:
                g, a = cls.split("_", 1)
                pred.gender = g.lower()
                pred.age_group = {
                    "Child": AGE_BIN_LABELS[0],
                    "YoungAdult": AGE_BIN_LABELS[1],
                    "MiddleAged": AGE_BIN_LABELS[2],
                    "OldAged": AGE_BIN_LABELS[3],
                }.get(a, a)
            return pred

    # 2) Separate heads: find a 2-logit gender head and a 4-logit age head
    gender_logits: Optional[np.ndarray] = None
    age_logits: Optional[np.ndarray] = None

    # Prefer well-known names if present
    for key in ("gender_logits", "gender", "sex", "gender_output"):
        if key in squeezed:
            arr = np.asarray(squeezed[key])
            if arr.ndim == 1 and arr.shape[0] == 2:
                gender_logits = arr
                break

    for key in ("age_output", "age_logits", "age", "age_group"):
        if key in squeezed:
            arr = np.asarray(squeezed[key])
            if arr.ndim == 1 and arr.shape[0] == len(AGE_BIN_LABELS):
                age_logits = arr
                break

    # Otherwise, infer by shape
    if gender_logits is None:
        for _, arr in outputs:
            arr = np.asarray(arr)
            if arr.ndim == 1 and arr.shape[0] == 2:
                gender_logits = arr
                break

    if age_logits is None:
        for _, arr in outputs:
            arr = np.asarray(arr)
            if arr.ndim == 1 and arr.shape[0] == len(AGE_BIN_LABELS):
                age_logits = arr
                break

    if gender_logits is not None:
        probs = _softmax(gender_logits)
        idx = int(np.argmax(probs))
        pred.gender = "female" if idx == 0 else "male"
        pred.gender_conf = float(probs[idx])

    if age_logits is not None:
        probs = _softmax(age_logits)
        idx = int(np.argmax(probs))
        pred.age_group = AGE_BIN_LABELS[idx] if idx < len(AGE_BIN_LABELS) else f"class_{idx}"
        pred.age_conf = float(probs[idx])

    return pred


def run_onnx(
    session: "ort.InferenceSession",
    input_tensor: np.ndarray,
) -> Tuple[ModelPrediction, Dict[str, np.ndarray], float]:
    input_name = session.get_inputs()[0].name
    start = time.perf_counter()
    out_list = session.run(None, {input_name: input_tensor})
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    out_names = [o.name for o in session.get_outputs()]
    output_map = {name: np.asarray(val) for name, val in zip(out_names, out_list)}

    pred = _decode_prediction(output_map)
    pred.inference_ms = float(elapsed_ms)
    return pred, output_map, elapsed_ms


def iter_image_paths(input_path: str, recursive: bool) -> List[str]:
    p = Path(input_path)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    if p.is_file():
        return [str(p)]

    if not p.is_dir():
        return []

    if recursive:
        files = [x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in exts]
    else:
        files = [x for x in p.glob("*") if x.is_file() and x.suffix.lower() in exts]

    return [str(x) for x in sorted(files)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare two ONNX models on images (gender + age-group decoding when possible)."
    )
    ap.add_argument("--new-model", required=True, help="Path to the new age+gender ONNX model")
    ap.add_argument(
        "--old-model",
        required=True,
        help="Path to the old model ONNX to compare against (won't be modified)",
    )
    ap.add_argument("--input", required=True, help="Image path or directory of images")
    ap.add_argument("--recursive", action="store_true", help="Recursively search for images in directory input")
    ap.add_argument("--output-dir", default="./onnx_comparison", help="Where to write CSV/JSON outputs")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device preference")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of images processed (0 = no limit)")

    args = ap.parse_args()

    new_model_path = args.new_model
    old_model_path = args.old_model

    if not os.path.exists(new_model_path):
        raise SystemExit(f"New model not found: {new_model_path}")
    if not os.path.exists(old_model_path):
        raise SystemExit(f"Old model not found: {old_model_path}")

    prefer_cuda = args.device in ("auto", "cuda")
    if args.device == "cpu":
        prefer_cuda = False

    providers = _select_providers(prefer_cuda=prefer_cuda)
    print(f"Providers: {providers}")

    print(f"Loading NEW model: {new_model_path}")
    new_sess = ort.InferenceSession(new_model_path, providers=providers)
    print(f"Loading OLD model: {old_model_path}")
    old_sess = ort.InferenceSession(old_model_path, providers=providers)

    new_hw = _infer_input_hw(new_sess, fallback=224)
    old_hw = _infer_input_hw(old_sess, fallback=224)
    print(f"NEW input HxW: {new_hw[0]}x{new_hw[1]}")
    print(f"OLD input HxW: {old_hw[0]}x{old_hw[1]}")

    image_paths = iter_image_paths(args.input, recursive=args.recursive)
    if not image_paths:
        raise SystemExit(f"No images found at: {args.input}")

    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]

    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    rows: List[Dict[str, Any]] = []

    for img_path in image_paths:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        new_inp = _preprocess_from_bgr(img_bgr, new_hw)
        old_inp = _preprocess_from_bgr(img_bgr, old_hw)

        new_pred, _, _ = run_onnx(new_sess, new_inp)
        old_pred, _, _ = run_onnx(old_sess, old_inp)

        agree_gender = (
            (new_pred.gender is not None)
            and (old_pred.gender is not None)
            and (new_pred.gender == old_pred.gender)
        )
        agree_age = (
            (new_pred.age_group is not None)
            and (old_pred.age_group is not None)
            and (new_pred.age_group == old_pred.age_group)
        )

        rows.append(
            {
                "path": img_path,
                "filename": os.path.basename(img_path),
                "new_gender": new_pred.gender,
                "new_gender_conf": new_pred.gender_conf,
                "new_age_group": new_pred.age_group,
                "new_age_conf": new_pred.age_conf,
                "new_combined_class": new_pred.combined_class,
                "new_combined_conf": new_pred.combined_conf,
                "new_inference_ms": new_pred.inference_ms,
                "new_output_names": ",".join(new_pred.output_names or []),
                "old_gender": old_pred.gender,
                "old_gender_conf": old_pred.gender_conf,
                "old_age_group": old_pred.age_group,
                "old_age_conf": old_pred.age_conf,
                "old_combined_class": old_pred.combined_class,
                "old_combined_conf": old_pred.combined_conf,
                "old_inference_ms": old_pred.inference_ms,
                "old_output_names": ",".join(old_pred.output_names or []),
                "agree_gender": bool(agree_gender),
                "agree_age_group": bool(agree_age),
            }
        )

    if not rows:
        raise SystemExit("No valid images were processed (cv2.imread failures?).")

    df = pd.DataFrame(rows)

    # Aggregate metrics
    gender_compared = df[(df["new_gender"].notna()) & (df["old_gender"].notna())]
    age_compared = df[(df["new_age_group"].notna()) & (df["old_age_group"].notna())]

    summary = {
        "timestamp": ts,
        "new_model": os.path.abspath(new_model_path),
        "old_model": os.path.abspath(old_model_path),
        "num_images": int(len(df)),
        "providers": providers,
        "new_input_hw": {"h": int(new_hw[0]), "w": int(new_hw[1])},
        "old_input_hw": {"h": int(old_hw[0]), "w": int(old_hw[1])},
        "gender_agreement_rate": float(gender_compared["agree_gender"].mean()) if len(gender_compared) else None,
        "age_agreement_rate": float(age_compared["agree_age_group"].mean()) if len(age_compared) else None,
        "avg_new_inference_ms": float(df["new_inference_ms"].mean()) if "new_inference_ms" in df else None,
        "avg_old_inference_ms": float(df["old_inference_ms"].mean()) if "old_inference_ms" in df else None,
        "notes": "Agreement rates are computed only on images where both models could be decoded for that field.",
    }

    csv_path = os.path.join(args.output_dir, f"comparison_{ts}.csv")
    json_path = os.path.join(args.output_dir, f"summary_{ts}.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print a concise console summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Images processed: {len(df)}")
    if summary["gender_agreement_rate"] is not None:
        print(f"Gender agreement: {summary['gender_agreement_rate'] * 100:.2f}% "
              f"(n={len(gender_compared)})")
    else:
        print("Gender agreement: N/A (could not decode gender for both models)")

    if summary["age_agreement_rate"] is not None:
        print(f"Age-group agreement: {summary['age_agreement_rate'] * 100:.2f}% "
              f"(n={len(age_compared)})")
    else:
        print("Age-group agreement: N/A (could not decode age-group for both models)")

    print(f"Avg NEW inference: {summary['avg_new_inference_ms']:.2f} ms")
    print(f"Avg OLD inference: {summary['avg_old_inference_ms']:.2f} ms")
    print(f"\nWrote:\n  - {csv_path}\n  - {json_path}")

    # If single image, also print per-image predictions
    if Path(args.input).is_file():
        r = df.iloc[0].to_dict()
        print("\n" + "-" * 80)
        print("SINGLE IMAGE DETAILS")
        print("-" * 80)
        print(f"File: {r['path']}")
        print(f"NEW: gender={r['new_gender']} ({r['new_gender_conf']}) | "
              f"age={r['new_age_group']} ({r['new_age_conf']}) | "
              f"combined={r['new_combined_class']} ({r['new_combined_conf']})")
        print(f"OLD: gender={r['old_gender']} ({r['old_gender_conf']}) | "
              f"age={r['old_age_group']} ({r['old_age_conf']}) | "
              f"combined={r['old_combined_class']} ({r['old_combined_conf']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

