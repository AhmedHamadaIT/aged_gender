import argparse
from ultralytics import YOLO

def export_model(model_path="best.pt", format="onnx", imgsz=640, dynamic=False, simplify=True, half=False):
    print(f"Loading YOLO model from: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Exporting model to {format.upper()} format...")
    # Using dynamic=False and simplify=True is highly recommended for
    # subsequent TensorRT engine generation on NVIDIA Orin.
    # half=True is recommended for TensorRT engine for FP16 precision.
    try:
        exported_path = model.export(
            format=format, 
            imgsz=imgsz, 
            dynamic=dynamic, 
            simplify=simplify if format == 'onnx' else False,
            half=half
        )
        print(f"Export successful! {format.upper()} model saved to: {exported_path}")
    except Exception as e:
        print(f"Error exporting model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX or TensorRT Engine for deployment")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to the YOLO PyTorch model (.pt)")
    parser.add_argument("--format", type=str, choices=['onnx', 'engine'], default="onnx", help="Export format: onnx or engine (default: onnx)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for export (default: 640)")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic axes for export")
    parser.add_argument("--half", action="store_true", help="FP16 half-precision export (recommended for TensorRT)")
    args = parser.parse_args()
    
    export_model(
        model_path=args.model, 
        format=args.format, 
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        half=args.half
    )
