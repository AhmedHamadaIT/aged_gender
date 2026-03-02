#!/usr/bin/env bash
# run_comparison.sh
# -----------------
# Compare YOLO (best.pt) and MobileNetV3 GenderAge (best_checkpoint.pth)
# on the images/ folder and save results to ./comparison_output/
#
# Usage:
#   ./run_comparison.sh                        # edge GPU (cuda, default)
#   ./run_comparison.sh --device cpu           # local CPU
#   ./run_comparison.sh --num-images 50        # quick test with 50 images
#   ./run_comparison.sh --device cpu --num-images 200

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

YOLO_MODEL="$SCRIPT_DIR/best.pt"
GENDER_AGE_MODEL="$SCRIPT_DIR/best_checkpoint.pth"
IMAGES_DIR="$SCRIPT_DIR/images"
OUTPUT_DIR="$SCRIPT_DIR/comparison_output"
DEVICE="cuda"       # default: edge GPU (Jetson Orin / CUDA device)
NUM_IMAGES=100      # default: 100 images per model

# ── Parse optional overrides ──────────────────────────────────────────────────
SAVE_VIS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)     DEVICE="$2";     shift 2 ;;
        --num-images) NUM_IMAGES="$2"; shift 2 ;;
        --images)     IMAGES_DIR="$2"; shift 2 ;;
        --output)     OUTPUT_DIR="$2"; shift 2 ;;
        --save-vis)   SAVE_VIS="--save-vis"; shift 1 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "  Aged Gender — Model Comparison Runner"
echo "========================================"
echo "  YOLO model      : $YOLO_MODEL"
echo "  GenderAge model : $GENDER_AGE_MODEL"
echo "  Images dir      : $IMAGES_DIR"
echo "  Output dir      : $OUTPUT_DIR"
echo "  Device          : $DEVICE"
echo "  Max images      : $NUM_IMAGES"
echo "========================================"

# Check that at least one model exists
if [ ! -f "$YOLO_MODEL" ] && [ ! -f "$GENDER_AGE_MODEL" ]; then
    echo "ERROR: Neither best.pt nor best_checkpoint.pth found in $SCRIPT_DIR"
    exit 1
fi

# Build argument list
ARGS="--images $IMAGES_DIR --output $OUTPUT_DIR --num-images $NUM_IMAGES --device $DEVICE $SAVE_VIS"
[ -f "$YOLO_MODEL" ]        && ARGS="$ARGS --model-yolo $YOLO_MODEL"
[ -f "$GENDER_AGE_MODEL" ]  && ARGS="$ARGS --model-gender-age $GENDER_AGE_MODEL"

echo ""
echo "Running: python3 compare_models.py $ARGS"
echo ""

python3 compare_models.py $ARGS

echo ""
echo "✅ Done! Check the results in: $OUTPUT_DIR"

