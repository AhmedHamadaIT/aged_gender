import argparse
from ultralytics import YOLO

def export_model(model_path="best.pt", imgsz=640):
    print(f"Loading YOLO model from: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Exporting model to ONNX format...")
    # Using dynamic=False and simplify=True is highly recommended for
    # subsequent TensorRT engine generation on NVIDIA Orin.
    try:
        exported_path = model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True)
        print(f"Export successful! ONNX model saved to: {exported_path}")
    except Exception as e:
        print(f"Error exporting model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX for NVIDIA Orin deployment")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to the YOLO PyTorch model (.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for export (default: 640)")
    args = parser.parse_args()
    
    export_model(model_path=args.model, imgsz=args.imgsz)
