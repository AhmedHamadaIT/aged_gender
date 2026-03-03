# Aged-Gender Prediction — Project README

A dual-model pipeline for **gender + age-group classification** on images and video streams.  
Two models are supported and can be benchmarked side-by-side:

| Model | File | Architecture |
|-------|------|--------------|
| YOLO classifier | `best.pt` | Ultralytics YOLO (classification head) |
| GenderAge MobileNetV3 | `best_checkpoint.pth` | MobileNetV3-Small multi-task CNN |

---

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Age & Gender Classes](#age--gender-classes)
- [File Declarations](#file-declarations)
  - [gender_age_model.py](#gender_age_modelpy)
  - [gender_age_inference.py](#gender_age_inferencepy)
  - [compare_models.py](#compare_modelspy)
  - [model_inference.py](#model_inferencepy)
  - [realtime_monitor.py](#realtime_monitorpy)
  - [export_model.py](#export_modelpy)
  - [run_comparison.sh](#run_comparisonsh)
  - [run_in_docker.sh](#run_in_dockersh)
  - [setup_orin.sh](#setup_orinsh)
  - [Dockerfile.jetson](#dockerfilejetson)
  - [docker-compose.jetson.yml](#docker-composejetsonml)
- [Quick-Start Examples](#quick-start-examples)
- [Output Files Reference](#output-files-reference)
- [Docker / Jetson Orin Deployment](#docker--jetson-orin-deployment)

---

## Project Structure

```
aged_gender/
├── best.pt                    # Trained YOLO classification model
├── best_checkpoint.pth        # Trained MobileNetV3 GenderAge model
├── images/                    # Default image dataset folder
│
├── gender_age_model.py        # Model definition & loader (MobileNetV3)
├── gender_age_inference.py    # Standalone inference wrapper (MobileNetV3)
├── compare_models.py          # Benchmark YOLO vs GenderAge side-by-side
├── model_inference.py         # YOLO inference + performance report generator
├── realtime_monitor.py        # Real-time webcam/video monitor (YOLO)
├── export_model.py            # Export YOLO model → ONNX / TensorRT
│
├── run_comparison.sh          # Shell wrapper for compare_models.py
├── run_in_docker.sh           # Run scripts inside Docker container
├── setup_orin.sh              # Environment setup for NVIDIA Orin
│
├── Dockerfile.jetson          # Docker image for Jetson / Orin
├── docker-compose.jetson.yml  # Docker Compose for Jetson deployment
├── requirements.txt           # Python dependencies
├── ORIN_SETUP.md              # Orin-specific hardware setup guide
└── README_inference.md        # Legacy inference README (superseded by this file)
```

---

## Installation

```bash
# Create and activate a virtual environment (optional but recommended)
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Jetson / NVIDIA Orin**: Do **not** install `torch` from PyPI.  
> Use NVIDIA's pre-built wheels or the `l4t-ml` Docker container.  
> See [`ORIN_SETUP.md`](ORIN_SETUP.md) and [`setup_orin.sh`](setup_orin.sh).

---

## Age & Gender Classes

**Gender labels** (index → name):

| Index | Label |
|-------|-------|
| 0 | Female |
| 1 | Male |

**Age-group labels** (index → name → range):

| Index | Label | Age Range |
|-------|-------|-----------|
| 0 | Child | 0 – 16 |
| 1 | Young Adults | 17 – 30 |
| 2 | Middle-aged Adults | 31 – 45 |
| 3 | Old-aged Adults | 46+ |

**Combined class names** (YOLO-style, used across all scripts):

```
Female_Child  Female_YoungAdult  Female_MiddleAged  Female_OldAged
Male_Child    Male_YoungAdult    Male_MiddleAged    Male_OldAged
```

---

## File Declarations

### `gender_age_model.py`

**Purpose**: Defines the MobileNetV3-Small multi-task model and provides a convenience loader.

#### Constants

| Name | Type | Value / Description |
|------|------|---------------------|
| `GENDER_LABELS` | `list[str]` | `["Female", "Male"]` |
| `AGE_BINS` | `list[tuple]` | `[(0,16), (17,30), (31,45), (46,120)]` |
| `AGE_LABELS` | `list[str]` | Full age-group display strings |
| `AGE_SHORT_LABELS` | `list[str]` | `["Child", "YoungAdult", "MiddleAged", "OldAged"]` |

#### Class: `GenderAgeModel(nn.Module)`

Multi-task CNN built on a MobileNetV3-Small backbone.

| Component | Description |
|-----------|-------------|
| `backbone` | `timm` MobileNetV3-Small (feature extractor, `num_classes=0`) |
| `neck` | `Linear → BN → Hardswish → Dropout(0.25)` (256-dim) |
| `gender_head` | `Linear → BN → Hardswish → Dropout → Linear(2)` |
| `age_head` | `Linear → BN → Hardswish → Dropout → Linear(4)` |

**Constructor parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `"mobilenetv3_small_100"` | timm backbone identifier |
| `num_age_classes` | `4` | Number of age bins |
| `pretrained` | `True` | Load ImageNet weights for backbone |
| `img_size` | `224` | Input resolution (square) |

**Forward pass** `forward(x) → (gender_logits, age_logits)`:

- **Input**: `torch.Tensor` of shape `(N, 3, 224, 224)`, ImageNet-normalised
- **Output**: tuple of `(Tensor[N,2], Tensor[N,4])` — raw logits

#### Function: `load_gender_age_model(...) → GenderAgeModel`

Loads a `.pth` checkpoint created by the training pipeline.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `checkpoint_path` | *(required)* | Path to `best_checkpoint.pth` |
| `device` | `"cpu"` | `"cpu"` or `"cuda"` |
| `model_name` | `"mobilenetv3_small_100"` | Must match training config |
| `num_age_classes` | `4` | Number of age bins |
| `img_size` | `224` | Input resolution |

Returns a `GenderAgeModel` in `eval()` mode on the requested device.

---

### `gender_age_inference.py`

**Purpose**: Standalone CLI + importable inference wrapper for the MobileNetV3 GenderAge model.

#### Constants

| Name | Value |
|------|-------|
| `IMAGENET_MEAN` | `[0.485, 0.456, 0.406]` |
| `IMAGENET_STD` | `[0.229, 0.224, 0.225]` |
| `IMG_SIZE` | `224` |

#### Function: `preprocess(img_bgr) → torch.Tensor`

Converts a BGR `uint8` OpenCV image to an ImageNet-normalised `float32` tensor of shape `1×3×224×224`.

#### Class: `GenderAgeInference`

High-level inference wrapper.

**Constructor**: `GenderAgeInference(checkpoint_path, device="cpu")`

| Method | Signature | Returns |
|--------|-----------|---------|
| `predict_image` | `(img_input: str \| np.ndarray)` | `dict` — single prediction |
| `benchmark` | `(image_paths: list, num_images=100)` | `dict` — timing & memory stats |

**`predict_image` return keys**:

| Key | Type | Description |
|-----|------|-------------|
| `gender` | `str` | `"Female"` or `"Male"` |
| `gender_conf` | `float` | Gender prediction confidence (0–1) |
| `age_class` | `int` | Age-group index (0–3) |
| `age_label` | `str` | Full age-group label |
| `age_conf` | `float` | Age prediction confidence (0–1) |
| `combined_class` | `str` | e.g. `"Female_Child"` |
| `confidence` | `float` | Alias for `gender_conf` |
| `inference_time_ms` | `float` | Wall-clock inference time |
| `gender_probs` | `list[float]` | Full gender probability vector |
| `age_probs` | `list[float]` | Full age probability vector |

#### CLI Usage

```bash
# Single image
python3 gender_age_inference.py \
    --checkpoint best_checkpoint.pth \
    --input path/to/image.jpg

# Folder of images
python3 gender_age_inference.py \
    --checkpoint best_checkpoint.pth \
    --input ./images \
    --device cuda

# Save results to JSON
python3 gender_age_inference.py \
    --checkpoint best_checkpoint.pth \
    --input ./images \
    --output results.json
```

**CLI Arguments**:

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--checkpoint` | ✅ | — | Path to `best_checkpoint.pth` |
| `--input` | ✅ | — | Image file or folder |
| `--output` | ❌ | `None` | Save JSON results to this path |
| `--device` | ❌ | auto | `cuda` or `cpu` |
| `--num-images` | ❌ | `None` | Limit images processed |

---

### `compare_models.py`

**Purpose**: Benchmarks YOLO and GenderAge-MobileNetV3 side-by-side on the same dataset, producing performance tables, plots, CSV, and JSON.

#### Key Functions

| Function | Description |
|----------|-------------|
| `collect_images(root, recursive)` | Finds all image files under `root`; returns sorted `list[str]` |
| `save_annotated_image(img_bgr, label, conf, vis_dir, filename)` | Writes a prediction-annotated image to `vis_dir/<label>/filename` |
| `benchmark_yolo(model_path, image_paths, device, num_images, vis_dir)` | Runs YOLO on up to `num_images` images; returns stats dict |
| `benchmark_gender_age(checkpoint, image_paths, device, num_images, vis_dir)` | Runs GenderAge model; returns stats dict |
| `print_comparison_table(results)` | Prints a formatted table; returns `pd.DataFrame` |
| `generate_plots(results, output_dir, timestamp)` | Saves a 2×3 comparison plot PNG |
| `save_results(results, df, output_dir, timestamp)` | Writes CSV + JSON summary + per-model prediction JSONs |

#### CLI Usage

```bash
python3 compare_models.py \
    --model-yolo       best.pt \
    --model-gender-age best_checkpoint.pth \
    --images           ./images \
    --output           ./comparison_output \
    --device           cpu \
    --num-images       100
```

**CLI Arguments**:

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model-yolo` | ❌ | `None` | YOLO model path (`.pt` / `.onnx` / `.engine`) |
| `--model-gender-age` | ❌ | `None` | GenderAge checkpoint (`.pth`) |
| `--images` | ✅ | — | Image file or dataset folder |
| `--output` | ❌ | `./comparison_output` | Output directory |
| `--num-images` | ❌ | `100` | Max images per model |
| `--device` | ❌ | auto | `cuda` or `cpu` |
| `--recursive` | ❌ | `True` | Recurse into subdirectories |
| `--save-vis` | ❌ | `False` | Save annotated images |

> At least one of `--model-yolo` or `--model-gender-age` must be provided.

#### Outputs

| File | Description |
|------|-------------|
| `comparison_<ts>.csv` | Summary table |
| `comparison_detail_<ts>.json` | Per-model metrics (JSON) |
| `YOLO_predictions_<ts>.json` | Per-image YOLO predictions |
| `GenderAge-MobileNetV3_predictions_<ts>.json` | Per-image GenderAge predictions |
| `comparison_plots_<ts>.png` | 6-panel comparison chart |
| `YOLO_vis/` | *(optional)* Annotated YOLO images |
| `GenderAge_vis/` | *(optional)* Annotated GenderAge images |

---

### `model_inference.py`

**Purpose**: Detailed YOLO performance analyser. Runs single-image benchmarks, full-dataset inference, and generates JSON / Markdown / HTML reports with visualisations.

#### Class: `ModelPerformanceAnalyzer`

**Constructor**: `ModelPerformanceAnalyzer(model_path, device="cpu"|"cuda")`

| Method | Description |
|--------|-------------|
| `load_model()` | Loads the YOLO model; returns `True` on success |
| `get_system_info()` | Collects CPU, RAM, GPU info; returns `dict` |
| `benchmark_single_image(image_path, warmup=False)` | Single-image inference timing & memory |
| `run_inference_on_folder(folder_path, recursive, save_vis, output_dir)` | Full dataset inference; returns `list[dict]` |
| `generate_report(output_dir="./reports")` | Saves JSON + Markdown + HTML report + PNG visualisations |

**`benchmark_single_image` return keys**: `inference_time_ms`, `memory_used_mb`, `peak_memory_mb`, `predicted_class`, `confidence`, `top5_classes`, `top5_confidences`

#### CLI Usage

```bash
python3 model_inference.py \
    --model  best.pt \
    --input  ./images \
    --output ./reports \
    --device cuda
```

**CLI Arguments**:

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model` | ✅ | — | Model file (`.pt` / `.onnx` / `.engine`) |
| `--input` | ✅ | — | Image or folder |
| `--output` | ❌ | `./reports` | Output directory |
| `--device` | ❌ | auto | `cuda` or `cpu` |
| `--recursive` | ❌ | `False` | Recurse into subdirectories |
| `--save-vis` | ❌ | `False` | Save annotated images |
| `--format` | ❌ | `all` | Report format: `json`, `md`, `html`, `all` |

#### Outputs

| File | Description |
|------|-------------|
| `report_<ts>.json` | Full results as JSON |
| `report_<ts>.md` | Markdown performance report |
| `report_<ts>.html` | HTML performance report |
| `visualizations_<ts>.png` | 4-panel chart (class dist, confidence, KPIs, resource usage) |
| `processed_images_<ts>/` | *(optional)* Annotated images per class |

---

### `realtime_monitor.py`

**Purpose**: Opens a webcam or video file, runs the YOLO model on every frame, and overlays live FPS + system statistics.

#### Class: `RealTimeMonitor`

**Constructor**: `RealTimeMonitor(model_path, source=0, device="cpu"|"cuda")`

| Method | Description |
|--------|-------------|
| `load_model()` | Loads YOLO model |
| `get_system_stats()` | Returns `dict` with `cpu`, `ram`, `gpu`, `gpu_mem` (%) |
| `draw_stats(frame, stats, prediction)` | Burns stats overlay onto frame; returns frame |
| `process_frame(frame)` | Runs inference; returns prediction `dict` |
| `run()` | Main event loop (opens capture, handles keys, prints session summary on exit) |

**Keyboard controls** (during `run()`):

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot (`screenshot_<ts>.jpg`) |
| `p` | Pause / Resume |

#### CLI Usage

```bash
# Webcam (default source=0)
python3 realtime_monitor.py --model best.pt

# Video file
python3 realtime_monitor.py --model best.pt --source path/to/video.mp4

# Force CPU
python3 realtime_monitor.py --model best.pt --device cpu
```

**CLI Arguments**:

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model` | ✅ | — | Model file |
| `--source` | ❌ | `"0"` | Webcam index or video file path |
| `--device` | ❌ | auto | `cuda` or `cpu` |

---

### `export_model.py`

**Purpose**: Exports a YOLO `.pt` model to ONNX or TensorRT Engine format for deployment.

#### Function: `export_model(...)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | `"best.pt"` | Source YOLO model |
| `format` | `"onnx"` | `"onnx"` or `"engine"` |
| `imgsz` | `640` | Input image size |
| `dynamic` | `False` | Dynamic batch axes (ONNX) |
| `simplify` | `True` | Simplify ONNX graph |
| `half` | `False` | FP16 export (recommended for TensorRT) |

#### CLI Usage

```bash
# Export to ONNX
python3 export_model.py --model best.pt --format onnx

# Export to TensorRT Engine (FP16)
python3 export_model.py --model best.pt --format engine --half
```

**CLI Arguments**:

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model` | ❌ | `best.pt` | YOLO `.pt` model path |
| `--format` | ❌ | `onnx` | `onnx` or `engine` |
| `--imgsz` | ❌ | `640` | Image size for export |
| `--dynamic` | ❌ | `False` | Dynamic axes flag |
| `--half` | ❌ | `False` | FP16 half-precision |

---

### `run_comparison.sh`

**Purpose**: Convenience shell wrapper that calls `compare_models.py` with both model files, the `images/` directory, and `./comparison_output` as defaults.

#### Usage

```bash
chmod +x run_comparison.sh

# Default: cuda device, 100 images
./run_comparison.sh

# Override device
./run_comparison.sh --device cpu

# Quick test with 50 images
./run_comparison.sh --num-images 50

# Save annotated images
./run_comparison.sh --save-vis

# Full override
./run_comparison.sh --device cpu --num-images 200 --images /path/to/imgs --output /path/to/out
```

**Arguments passed through to `compare_models.py`**:

| Flag | Description |
|------|-------------|
| `--device <cuda\|cpu>` | Override inference device |
| `--num-images <N>` | Max images per model |
| `--images <path>` | Image directory |
| `--output <path>` | Output directory |
| `--save-vis` | Save annotated images |

---

### `run_in_docker.sh`

**Purpose**: Runs any project script inside the Jetson Docker container.

```bash
./run_in_docker.sh python3 compare_models.py --model-yolo best.pt --images ./images
```

---

### `setup_orin.sh`

**Purpose**: Sets up the Python environment on a fresh NVIDIA Orin device (installs dependencies from `requirements.txt`).

```bash
chmod +x setup_orin.sh && ./setup_orin.sh
```

---

### `Dockerfile.jetson`

**Purpose**: Docker image based on NVIDIA L4T / JetPack for running inference on Jetson Orin.

```bash
docker build -f Dockerfile.jetson -t aged-gender:jetson .
```

---

### `docker-compose.jetson.yml`

**Purpose**: Docker Compose configuration for Jetson Orin deployment with GPU passthrough.

```bash
docker compose -f docker-compose.jetson.yml up
```

---

## Quick-Start Examples

### 1 — Compare both models (recommended starting point)

```bash
./run_comparison.sh --device cpu --num-images 50
# Results in ./comparison_output/
```

### 2 — GenderAge inference only

```bash
# Single image
python3 gender_age_inference.py \
    --checkpoint best_checkpoint.pth \
    --input photo.jpg

# Folder, save JSON
python3 gender_age_inference.py \
    --checkpoint best_checkpoint.pth \
    --input ./images \
    --output results.json \
    --device cuda
```

### 3 — YOLO performance report

```bash
python3 model_inference.py \
    --model best.pt \
    --input ./images \
    --output ./reports \
    --device cuda \
    --save-vis
```

### 5 — Real-time webcam monitoring

```bash
python3 realtime_monitor.py --model best.pt --source 0
```

### 6 — Export YOLO to ONNX / TensorRT

```bash
python3 export_model.py --model best.pt --format onnx
python3 export_model.py --model best.pt --format engine --half
```

---

## Output Files Reference

| Script | Output Location | Key Files |
|--------|-----------------|-----------|
| `compare_models.py` | `./comparison_output/` | `comparison_<ts>.csv`, `comparison_plots_<ts>.png`, `*_predictions_<ts>.json` |
| `model_inference.py` | `./reports/` | `report_<ts>.json`, `report_<ts>.md`, `report_<ts>.html`, `visualizations_<ts>.png` |
| `gender_age_inference.py` | user-defined | `results.json` (optional) |
| `realtime_monitor.py` | current dir | `screenshot_<ts>.jpg` (on `s` key) |
| `export_model.py` | same dir as model | `best.onnx` or `best.engine` |

*(All timestamps `<ts>` are in `YYYYMMDD_HHMMSS` format.)*

---

## Docker / Jetson Orin Deployment

See [`ORIN_SETUP.md`](ORIN_SETUP.md) for full hardware configuration and [`setup_orin.sh`](setup_orin.sh) for automated environment setup.

```bash
# Build image
docker build -f Dockerfile.jetson -t aged-gender:jetson .

# Run with GPU
docker compose -f docker-compose.jetson.yml up

# Or run scripts directly in container
./run_in_docker.sh ./run_comparison.sh --device cuda
```

---

## Requirements

See [`requirements.txt`](requirements.txt) for the full list. Key packages:

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLO model loading and inference |
| `timm` | MobileNetV3 backbone (required for `gender_age_model.py`) |
| `opencv-python` | Image I/O and visualisation |
| `torch` | Deep learning backend |
| `numpy`, `pandas` | Numerical computation and result handling |
| `matplotlib`, `seaborn` | Plot generation |
| `psutil` | System resource monitoring |
| `tqdm` | Progress bars |
| `humanize` | Human-readable sizes in `model_inference.py` |
| `tabulate` | Nicer tables in `compare_models.py` (optional) |
| `onnx`, `onnxruntime-gpu` | ONNX model export and inference |
