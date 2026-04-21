# Model weights (not in Git)

This directory is **empty of real weights in the repository**. The zero-byte files here are **placeholders** so paths exist on edge devices and in containers after clone; **replace each file** with the actual weight from your internal mirror or Google Drive (see repo root `README.md` “Models file”, or run `python tools/download_models.py` where network allows).

| Placeholder | Used by (default env) |
|-------------|------------------------|
| `yolov8n.pt` | `YOLO_MODEL` (FrameBus / detector) |
| `best_aged_gender_6.onnx` | `AGE_GENDER_MODEL` |
| `best_mood.onnx` | `MOOD_MODEL` |
| `best_ppe.onnx` | `PPE_MODEL_PATH` |
| `best_cashier.onnx` | `CASHIER_MODEL` |
| `osnet_x1_0.pt` | `REID_MODEL_PATH` |
| `image_encoder.onnx` | `IMAGE_ENCODER_ONNX` |
| `text_encoder.onnx` | `TEXT_ENCODER_ONNX` |

After copying real files, verify sizes are non-zero (e.g. `ls -lh models/`).
