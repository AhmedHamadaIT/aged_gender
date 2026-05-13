# Model weights (not in Git)

This directory is **empty of real weights in the repository**. The zero-byte files here are **placeholders** so paths exist on edge devices and in containers after clone; **replace each file** with the actual weight from your internal mirror or Google Drive (see repo root `README.md` “Models file”, or run `python tools/download_models.py` where network allows).

| Placeholder | Used by (default env) |
|-------------|------------------------|
| `yolov8n.pt` | `YOLO_MODEL` (FrameBus / detector) |
| `yolov8n.engine` | `YOLO_MODEL` (TensorRT export on Jetson — see `.env.example` Jetson section) |
| `best_aged_gender_6.onnx` | `AGE_GENDER_MODEL` |
| `best_mood.onnx` | `MOOD_MODEL` |
| `best_ppe.onnx` | `PPE_MODEL_PATH` |
| `best_cashier.onnx` | `CASHIER_MODEL` |
| `osnet_x1_0.pt` | `REID_MODEL_PATH` |
| OSNet `*.onnx` export | `REID_MODEL_ONNX` (preferred on Jetson with GPU ONNX Runtime) |
| `image_encoder.onnx` | `IMAGE_ENCODER_ONNX` |
| `text_encoder.onnx` | `TEXT_ENCODER_ONNX` |

After copying real files, verify sizes are non-zero (e.g. `ls -lh models/`).

## Jetson Nano / Orin Nano (YOLO engine + ONNX)

- **YOLO TensorRT:** `yolo export model=models/yolov8n.pt format=engine device=0 half=True imgsz=640` then set **`YOLO_MODEL=models/yolov8n.engine`** (Ultralytics loads `.engine` on GPU).
- **ONNX services:** use **`ONNX_EXECUTION_PROVIDERS_ORDER=cuda_first`** on tight VRAM; **`ONNX_TENSORRT_CACHE_PATH`** should be writable (Compose mounts `trt_engine_cache` at `/app/trt_cache`).
- **ReID:** set **`REID_MODEL_ONNX`** to an OSNet-compatible ONNX file when possible so inference uses ORT GPU EPs (`utils/onnx_runtime.py`).
