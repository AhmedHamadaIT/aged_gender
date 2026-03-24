# Vision Pipeline API

The Vision Pipeline API is a FastAPI-based server for running multi-camera computer vision pipelines. It supports adding multiple RTSP streams, composing custom execution pipelines (e.g., object detection, age/gender estimation, mood detection), and streaming the visual results to clients via Server-Sent Events (SSE).

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.9+
- CUDA/cuDNN enabled environment (optional but recommended for GPU acceleration)

### Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the FastAPI server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 9000
   ```

*(Alternatively, you can run it via Docker Compose if standard deployment is configured).*

---

## 📖 API Endpoints Reference

The application lifecycle works as follows:
1. Register Cameras.
2. Setup the Pipeline Services.
3. Start the Pipeline processing.
4. Consume the Detection Stream (SSE).

---

### **Health Check**

#### `GET /`
Returns basic service information.

**Sample Response**
```json
{
  "service": "Vision Pipeline API",
  "version": "1.0.0"
}
```

---

### **Camera Management Routes**

#### `POST /cameras`
Configure one or multiple cameras.

**Sample Request**
```json
{
  "cameras": [
    {
      "id": "cam1",
      "url": "rtsp://username:password@10.0.0.5:554/stream1"
    }
  ]
}
```

**Sample Response**
```json
{
  "status": "configured",
  "cameras": {
    "cam1": "rtsp://username:password@10.0.0.5:554/stream1"
  }
}
```

#### `GET /cameras`
List all currently configured cameras.

**Sample Response**
```json
{
  "count": 1,
  "cameras": [
    {
      "id": "cam1",
      "url": "rtsp://username:password@10.0.0.5:554/stream1"
    }
  ]
}
```

#### `DELETE /cameras/{cam_id}`
Delete a configured camera.

**Sample Response**
```json
{
  "status": "removed",
  "camera_id": "cam1",
  "remaining": []
}
```

---

### **Detection Pipeline Routes**

#### `POST /detection/setup`
Configure which models/services will run in the pipeline.

**Sample Request**
```json
{
  "pipeline": ["detector", "age_gender", "mood"]
}
```

**Sample Response**
```json
{
  "status": "configured",
  "pipeline": ["detector", "age_gender", "mood"]
}
```

#### `POST /detection/start`
Start processing cameras using the configured pipeline. You can optionally specify a `camera_id` as a query parameter string. If not specified, it starts all configured cameras.

**Sample Request**
`POST /detection/start?camera_id=cam1`

**Sample Response**
```json
{
  "status": "started",
  "cameras": ["cam1"]
}
```

#### `POST /detection/stop`
Stop camera streams. You can optionally specify a `camera_id` as a query parameter string. If not specified, it stops all running cameras.

**Sample Request**
`POST /detection/stop?camera_id=cam1`

**Sample Response**
```json
{
  "status": "stopped",
  "cameras": ["cam1"]
}
```

#### `GET /detection/status`
Returns the operational status, FPS, and detection counts for all cameras.

**Sample Response**
```json
{
  "cameras": {
    "cam1": {
      "camera_id": "cam1",
      "rtsp_url": "rtsp://10.0.0.5:554/stream1",
      "running": true,
      "frame_count": 420,
      "fps": 28.5,
      "last_detections": 3,
      "total_detections": 1205,
      "uptime_seconds": 14.7,
      "error": null
    }
  }
}
```

#### `GET /detection/stream`
An SSE (Server-Sent Events) endpoint that yields one JSON payload per frame, combining inferences from all running cameras.

**Sample Response stream**
```json
data: {
  "camera_id": "main_room",
  "frame_count": 5,
  "timestamp": "2026-03-17T12:50:23+00:00",
  "data": {
    "detection": {
      "count": 1,
      "items": [{
        "bbox": [120, 80, 340, 420],
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.91,
        "center": [230, 250],
        "width": 220,
        "height": 340
      }]
    },
    "use_case": {
      "age_gender": [{
        "bbox": [120, 80, 340, 420],
        "gender": "Female",
        "age_group": "MiddleAged",
        "confidence": 0.87
      }],
      "mood": [{
        "bbox": [120, 80, 340, 420],
        "mood": "Happy",
        "confidence": 0.94
      }]
    }
  }
}

```

---

## 🛠️ Testing & cURL Examples

### Quick Start - Test with Image File

#### Test Single Image
```bash
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg" \
  -F "camera_id=test_camera_1"
```

#### Test with Result File Output
```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@/home/a7med/Downloads/f1.webp" \
  -F "camera_id=f1_test" \
  --output results.json

# View prettified response
cat results.json | jq '.'
```

#### Extract & Save Annotated Image from Response
```bash
# Save full response
response=$(curl -X POST "http://localhost:8000/process" \
  -F "file=@image.jpg" \
  -F "camera_id=test")

# Extract base64 frame and decode
echo "$response" | jq -r '.frame' | base64 -d > annotated_image.jpg

# View detections summary
echo "$response" | jq '.data.detection'
```

### Advanced Testing

#### Batch Process Multiple Images
```bash
mkdir -p results
for image in *.jpg *.png *.webp; do
  [ -f "$image" ] || continue
  
  echo "Processing: $image"
  curl -X POST "http://localhost:8000/process" \
    -F "file=@$image" \
    -F "camera_id=batch_$(date +%s%N)" \
    --output "results/$image.json"
done
```

#### Stream Processing
```bash
curl -X POST "http://localhost:8000/stream" \
  -F "file=@video.mp4" \
  -F "camera_id=video_stream" \
  -N  # No buffering, stream output
```

#### Test Mood Detection Only
```bash
curl -X POST "http://localhost:8000/mood" \
  -F "file=@face_image.jpg" | jq '.data.use_case.mood'
```

#### Test Age/Gender Detection Only
```bash
curl -X POST "http://localhost:8000/age-gender" \
  -F "file=@face_image.jpg" | jq '.data.use_case.age_gender'
```

#### Health Check
```bash
curl -X GET "http://localhost:8000/health" \
  -H "Content-Type: application/json"
```

### Response Processing Examples

#### Extract Detections with jq
```bash
# Get all detections
curl -s -X POST "http://localhost:8000/process" \
  -F "file=@image.jpg" \
  -F "camera_id=test" | jq '.data.detection.items[] | {class_name, confidence}'

# Filter high-confidence detections (>0.85)
curl -s -X POST "http://localhost:8000/process" \
  -F "file=@image.jpg" \
  -F "camera_id=test" | jq '.data.detection.items[] | select(.confidence > 0.85)'

# Get mood distribution
curl -s -X POST "http://localhost:8000/process" \
  -F "file=@image.jpg" \
  -F "camera_id=test" | jq '[.data.use_case.mood[] | .mood] | group_by(.) | map({mood: .[0], count: length})'
```

---

## 📊 Test Results

**Status**: ✅ All tests passed

### Tested Images
- **Sample Image** (zidane.jpg): 2 persons detected, 100% mood classification success
- **User Image** (f1.webp): 4 objects detected (3 persons + 1 chair), 100% age/gender classification

### Performance
- Average inference time: 2-3 seconds per image
- Detection confidence: 83-89%
- Age/Gender accuracy: 56-75% confidence
- Mood classification: 45-52% confidence

### Output Artifacts
- Annotated JPEG images with bounding boxes and labels
- Complete JSON stream data with all detection metadata
- Execution logs and timing information
- Base64-encoded frame data for transmission

**See detailed results**: [TEST_RESULTS.md](TEST_RESULTS.md)

---

## 📁 Project Structure

```
.
├── app.py                      # FastAPI main application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── TEST_RESULTS.md             # Detailed test documentation
├── .gitignore                  # Git ignore rules (models/ excluded)
├── models/                     # ML models (not tracked in git)
│   ├── yolov8n.pt
│   ├── best_aged_gender_6.onnx
│   └── best_mood.onnx
├── services/                   # Service modules
│   ├── detector.py
│   ├── age_gender.py
│   └── mood.py
├── scripts/                    # Testing and utility scripts
│   └── test_image_pipeline.py
├── logger/                     # Logging configuration
│   └── logger_config.py
└── outputs/                    # Test results directory
    ├── test_image/             # Sample image test results
    └── f1_test/                # F1.webp test results
```

---

## 🔄 Git Workflow

### Current Branch: `mood`
```bash
# View commit history
git log --oneline -10

# Check current changes
git status

# Commit new changes
git add .
git commit -m "Add test results and documentation"

# Push to remote
git push origin mood
```

### Model Files
Models are excluded from git tracking to reduce repository size:
- `best_aged_gender_6.onnx` (~85 MB)
- `best_mood.onnx` (~15 MB)
- `yolov8n.pt` (~25 MB)

Download separately or configure via environment variables.

---

## 📝 Notes

- All inference runs on CPU (CUDA disabled for compatibility)
- Models are loaded once at startup for performance
- SSE streaming allows real-time frame processing
- Bounding boxes include confidence scores and class labels
- Age/gender and mood predictions are per-face/object detected
