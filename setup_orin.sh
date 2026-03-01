#!/bin/bash
set -e

echo "==========================================="
echo "   NVIDIA Orin Environment Setup Script    "
echo "==========================================="
echo "Note: It's recommended to run this inside an Nvidia L4T container,"
echo "or ensure PyTorch/Torchvision are installed via Jetson wheels first."

# Create virtual environment allowing access to system-site-packages.
# This is necessary on Jetson to access system-installed TensorRT and PyTorch.
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment '.venv'..."
    python3 -m venv .venv --system-site-packages
else
    echo "Virtual environment '.venv' already exists."
fi

# Activate environment
source .venv/bin/activate

echo "Upgrading pip..."
python3 -m pip install --upgrade pip

echo "Installing regular dependencies..."
# We use --no-deps for ultralytics if user wants to avoid overriding system PyTorch,
# but using requirements.txt directly is standard.
pip install -r requirements.txt

echo "==========================================="
echo "Setup complete! To activate, run:"
echo "source .venv/bin/activate"
echo "==========================================="
