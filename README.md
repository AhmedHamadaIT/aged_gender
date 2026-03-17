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
