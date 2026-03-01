#!/bin/bash

echo "==========================================="
echo "   Running Nvidia L4T ML Container         "
echo "==========================================="

# The L4T ML container contains PyTorch, OpenCV (CUDA), ONNX, and TensorRT.
# You do NOT need a virtual environment inside this container.

# You can change the image to a custom built one if needed (e.g. aged_gender_jetson)
# by running: docker build -t aged_gender_jetson -f Dockerfile.jetson .
# and changing the image name below.

CONTAINER_IMAGE="nvcr.io/nvidia/l4t-ml:r36.2.0-py3"

echo "Using image: $CONTAINER_IMAGE"
echo "(If you don't have it, downloading may take some time)."
echo ""
echo "Once inside, you can run commands like:"
echo "  python export_model.py --model best.pt --format onnx"
echo "  python realtime_monitor.py --model best.engine --source 0"

docker run -it --rm \
  --runtime nvidia \
  --network host \
  --privileged \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "$CONTAINER_IMAGE" \
  /bin/bash
