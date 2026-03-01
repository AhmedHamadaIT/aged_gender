# Age & Gender YOLO Inference Scripts

Model: `best.pt` — 8-class YOLO classifier  
Classes: `Female_Child`, `Female_YoungAdult`, `Female_MiddleAged`, `Female_OldAged`, `Male_Child`, `Male_YoungAdult`, `Male_MiddleAged`, `Male_OldAged`

## Install Dependencies

```bash
pip install ultralytics opencv-python psutil pandas numpy matplotlib seaborn tqdm humanize tabulate
```

---

## Docker Build & Run

You can run the inference scripts inside an isolated Docker container with GPU support.

1. **Build and start the container:**
   ```bash
   sudo docker-compose -f docker-compose.jetson.yml up -d --build
   ```

2. **Access the container shell:**
   ```bash
   sudo docker exec -it yolov8_inference bash
   ```
   *Alternatively, you can use the provided script:*
   ```bash
   bash run_in_docker.sh
   ```

---

## Model Export (ONNX & TensorRT Engine)

For maximum inference speed, especially on NVIDIA Jetson/Orin devices, export the `.pt` model to ONNX or TensorRT Engine format.

```bash
# Export to ONNX
python export_model.py --model best.pt --format onnx

# Export to TensorRT Engine (Recommended for Jetson/Orin - requires GPU)
python export_model.py --model best.pt --format engine --half
```

**Note:** Once exported, you can pass the `.onnx` or `.engine` file directly to the `--model` argument in any of the scripts below instead of `.pt`. This usually results in significantly higher FPS.

---

## Scripts

### 1. `model_inference.py` — Full Inference + Reports

Runs inference on a folder or single image and generates JSON / Markdown / HTML reports with visualizations.

```bash
# Full folder analysis
python model_inference.py \
    --model /home/a7med/Documents/aged_gender/best.pt \
    --input /home/a7med/Documents/aged_gender/images \
    --output ./reports \
    --recursive \
    --save-vis

# Single image
python model_inference.py \
    --model best.pt \
    --input photo.jpg \
    --output ./reports
```

**Output (`./reports/`):**
- `report_<timestamp>.json` — machine-readable report
- `report_<timestamp>.md` — Markdown report
- `report_<timestamp>.html` — HTML report
- `visualizations_<timestamp>.png` — charts

---

### 2. `realtime_monitor.py` — Live Webcam / Video Monitor

Runs inference frame-by-frame with on-screen FPS, CPU/RAM/GPU overlays and top-5 predictions.

```bash
# Webcam
python realtime_monitor.py --model best.pt --source 0

# Video file
python realtime_monitor.py --model best.pt --source video.mp4
```

**Controls:** `q` quit · `s` screenshot · `p` pause/resume

---

### 3. `batch_processor.py` — Batch Processing

Processes every image in a folder, saves per-image predictions to CSV + JSON, and generates 4 analysis plots.

```bash
python batch_processor.py \
    --model best.pt \
    --input /home/a7med/Documents/aged_gender/images \
    --output ./batch_output \
    --recursive \
    --save-vis        # optional: save annotated images
```

**Output (`./batch_output/`):**
- `results_<timestamp>.csv` — per-image predictions
- `results_<timestamp>.json` — JSON version
- `summary_<timestamp>.json` — aggregated stats
- `plots_<timestamp>.png` — distribution charts
- `visualizations/` — annotated images (if `--save-vis`)

---

### 4. `compare_models.py` — Multi-Model Comparison

Benchmarks two or more `.pt` models on the same test set and produces a side-by-side comparison.

```bash
python compare_models.py \
    --models best.pt other_model.pt \
    --names "best" "other" \
    --test-dir /home/a7med/Documents/aged_gender/images \
    --output ./model_comparison \
    --num-images 50
```

**Output (`./model_comparison/`):**
- `comparison_<timestamp>.csv` — summary table
- `detailed_results_<timestamp>.json` — per-model stats
- `comparison_plots_<timestamp>.png` — bar charts

---

## Quick Start (your paths)

```bash
# Run full analysis right now:
python /home/a7med/Documents/aged_gender/model_inference.py \
    --model /home/a7med/Documents/aged_gender/best.pt \
    --input /home/a7med/Documents/aged_gender/images \
    --output /home/a7med/Documents/aged_gender/reports \
    --recursive
```
