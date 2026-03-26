# Vision Pipeline API

The Vision Pipeline API is a FastAPI-based server for running multi-camera computer vision pipelines. It supports adding multiple RTSP streams, composing custom execution pipelines (e.g., object detection, age/gender estimation, mood detection), and streaming the visual results to clients via Server-Sent Events (SSE).

---

##  Setup & Run

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

##  API Endpoints Reference

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
  "pipeline": ["detector", "age_gender", "mood","ppe"]
}
```

**Sample Response**
```json
{
  "status": "configured",
  "pipeline": ["detector", "age_gender", "mood","ppe"]
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
      }],
      "ppe": [
      {
        "person_bbox": [
          677,
          305,
          850,
          523
        ],
        "count": 2,
        "items": [
          {
            "class_id": 2,
            "class_name": "gloves",
            "confidence": 0.9647,
            "x1": 742,
            "y1": 480,
            "x2": 805,
            "y2": 527
          },
          {
            "class_id": 1,
            "class_name": "hairnet",
            "confidence": 0.8999,
            "x1": 769,
            "y1": 312,
            "x2": 863,
            "y2": 374
          }
        ]
    }]
      }
    }
  }
```

---

## ✅ Testing & Real Result Examples

### Test Results Overview

#### Test 1: Sample Image (zidane.jpg)
- **Detections**: 2 Persons
- **Inference Time**: ~2 seconds
- **Confidence Scores**: 81.9% - 83.6%

#### Test 2: F1 Image (f1.webp) - Complete Test Results

**Input Image**: `f1.webp` (WebP format, 33 KB)
- **Local annotated version (provided)**: `/home/a7med/Downloads/f1a.png`

**Annotated Output Image with All Detections:**

![F1 Annotated Image with Detections - 500x281 pixels](outputs/f1_test/annotated.jpg)

**Visual Elements in Image:**
- Green bounding boxes around detected persons
- Blue bounding box around detected chair
- Labels showing: Class, Confidence %, Gender | Age Group, and Mood/Emotion
- All coordinates and detection metadata embedded

**Detection Summary:**
```
Total Objects Detected: 4

Person 1 (Top-Left Area)
├─ Bounding Box: [289, 71, 458, 257]
├─ Confidence: 89.01%
├─ Gender: Male
├─ Age Group: MiddleAged
├─ Mood: Angry
└─ Mood Confidence: 51.92%

Person 2 (Left Side)
├─ Bounding Box: [125, 70, 268, 220]
├─ Confidence: 84.15%
├─ Gender: Female
├─ Age Group: Senior
├─ Mood: Angry
└─ Mood Confidence: 45.12%

Person 3 (Center)
├─ Bounding Box: [209, 73, 301, 217]
├─ Confidence: 83.13%
├─ Gender: Male
├─ Age Group: Senior
├─ Mood: Happy
└─ Mood Confidence: 47.54%

Chair (Right Side)
├─ Bounding Box: [353, 136, 458, 259]
├─ Confidence: 52.47%
└─ Class ID: 56
```

**Performance Metrics:**

| Metric | Value |
|--------|-------|
| Total Processing Time | ~3 seconds |
| Image Resolution | 500×281 pixels |
| Number of Detections | 4 |
| Average Confidence | 82.9% |
| Age/Gender Confidence | 56-75% |
| Mood Detection Confidence | 45-52% |
| Annotated Image Size | 85 KB |
| Output JSON Size | 64 KB |

---

### Complete JSON Response Example (F1 Test)

**Raw API Response Structure:**
```json
{
  "camera_id": "f1_image",
  "frame_count": 1,
  "timestamp": "2026-03-24T09:29:31.395013Z",
  "frame": "<base64_encoded_jpeg_string_containing_annotated_image>",
  "data": {
    "detection": {
      "count": 4,
      "items": [
        {
          "bbox": [289, 71, 458, 257],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.8901,
          "center": [373, 164],
          "width": 169,
          "height": 186
        },
        {
          "bbox": [125, 70, 268, 220],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.8415,
          "center": [196, 145],
          "width": 143,
          "height": 150
        },
        {
          "bbox": [209, 73, 301, 217],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.8313,
          "center": [255, 145],
          "width": 92,
          "height": 144
        },
        {
          "bbox": [353, 136, 458, 259],
          "class_id": 56,
          "class_name": "chair",
          "confidence": 0.5247,
          "center": [405, 197],
          "width": 105,
          "height": 123
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {
          "bbox": [289, 71, 458, 257],
          "gender": "Male",
          "age_group": "MiddleAged",
          "confidence": 0.5693
        },
        {
          "bbox": [125, 70, 268, 220],
          "gender": "Female",
          "age_group": "Senior",
          "confidence": 0.5635
        },
        {
          "bbox": [209, 73, 301, 217],
          "gender": "Male",
          "age_group": "Senior",
          "confidence": 0.7525
        },
        {
          "bbox": [353, 136, 458, 259],
          "gender": "Male",
          "age_group": "Senior",
          "confidence": 0.5758
        }
      ],
      "mood": [
        {
          "bbox": [289, 71, 458, 257],
          "mood": "Angry",
          "confidence": 0.5192
        },
        {
          "bbox": [125, 70, 268, 220],
          "mood": "Angry",
          "confidence": 0.4512
        },
        {
          "bbox": [209, 73, 301, 217],
          "mood": "Happy",
          "confidence": 0.4754
        },
        {
          "bbox": [353, 136, 458, 259],
          "mood": "Angry",
          "confidence": 0.4556
        }
      ]
    }
  }
}
```

**JSON Field Reference:**
- **camera_id**: Identifier of the source camera or image
- **frame_count**: Frame number in stream (1 for single image)
- **timestamp**: ISO 8601 timestamp of processing
- **frame**: Base64-encoded JPEG containing annotated image (with all boxes, labels, confidence scores)
- **detection.count**: Total number of detected objects
- **bbox**: `[x1, y1, x2, y2]` - top-left (x1,y1) and bottom-right (x2,y2) pixel coordinates
- **center**: `[cx, cy]` - center point of bounding box
- **class_id**: COCO dataset class ID (0 = person, 56 = chair, etc.)
- **confidence**: Detection confidence score (0.0-1.0)
- **gender**: Predicted gender ("Male" or "Female")
- **age_group**: Predicted age category ("Young", "MiddleAged", "Senior", "Elderly")
- **mood**: Detected emotion ("Angry", "Happy", "Neutral")

---

## 🛠️ cURL Testing Examples

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

---

## SSE / SSH Stream Logs (Cashier + Mood + Age/Gender)

### Cashier-ready pipeline setup
```bash
curl -X POST "http://<jetson-ip>:9000/detection/setup" \
  -H "Content-Type: application/json" \
  -d '{"pipeline": ["detector", "age_gender", "mood", "cashier"]}'
```

### SSE live stream command
```bash
curl -N "http://<jetson-ip>:9000/detection/stream"
```

### SSH log tail command (server-side)
```bash
ssh <user>@<jetson-ip> "tail -f /path/to/repo/logger/app.log"
```

### Stream event source image (annotated)
The following stream event example is taken from an annotated frame generated in the production simulation run:
- Image: `outputs/cashier_production_sim/20260326T050929Z/annotated/frame_096720.jpg`

![Annotated Frame Used For Stream Example](outputs/cashier_production_sim/20260326T050929Z/annotated/frame_096720.jpg)

### SSE event shape with `cashier` in `use_case`
```json
data: {
  "camera_id": "main_room",
  "frame_count": 1,
  "timestamp": "2026-03-26T05:09:39.567737+00:00",
  "data": {
    "detection": {
      "count": 8,
      "items": [
        {
          "bbox": [597, 0, 1069, 291],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.9002,
          "center": [833, 145],
          "width": 472,
          "height": 291
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {
          "bbox": [597, 0, 1069, 291],
          "gender": "Female",
          "age_group": "Senior",
          "confidence": 0.5617
        }
      ],
      "mood": [
        {
          "bbox": [597, 0, 1069, 291],
          "mood": "Happy",
          "confidence": 0.4575
        }
      ],
      "cashier": {
        "cashier_zone": { "persons": 1, "drawers": 1, "cash": 3 },
        "customer_zone": { "persons": 1, "drawers": 0, "cash": 0 },
        "case_id": "N3",
        "severity": "NORMAL",
        "alerts": ["N3 EVENT: Transaction in progress"],
        "transaction": true,
        "frame_saved": true,
        "evidence_path": "outputs/.../evidence/normal/N3/cam_*.jpg",
        "frame_id": 1,
        "timestamp": "2026-03-26T05:09:39.257797+00:00"
      }
    }
  }
}
```

### Example Events Sourced From Annotated Frames (N3 / N5 / A2)
Each event below corresponds to an annotated image under:
`outputs/cashier_production_sim/20260326T050929Z/annotated/`

#### N3 example (frame_096750.jpg)
Annotated frame:
![Annotated Frame N3](outputs/cashier_production_sim/20260326T050929Z/annotated/frame_096750.jpg)

SSE event (shape):
```json
data: {
  "camera_id": "main_room",
  "frame_count": 2,
  "timestamp": "2026-03-26T05:09:39.740780+00:00",
  "frame": "<base64 JPEG from annotated corresponding frame>",
  "data": {
    "detection": {
      "count": 8,
      "items": [
        {
          "bbox": [
            608,
            0,
            1077,
            292
          ],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.9016,
          "center": [
            842,
            146
          ],
          "width": 469,
          "height": 292
        },
        {
          "bbox": [
            767,
            826,
            1290,
            1080
          ],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.8708,
          "center": [
            1028,
            953
          ],
          "width": 523,
          "height": 254
        },
        {
          "bbox": [
            843,
            687,
            1182,
            919
          ],
          "class_id": 1,
          "class_name": "drawer_open",
          "confidence": 0.8577,
          "center": [
            1012,
            803
          ],
          "width": 339,
          "height": 232
        },
        {
          "bbox": [
            980,
            702,
            1048,
            807
          ],
          "class_id": 2,
          "class_name": "cash",
          "confidence": 0.7952,
          "center": [
            1014,
            754
          ],
          "width": 68,
          "height": 105
        },
        {
          "bbox": [
            858,
            726,
            930,
            828
          ],
          "class_id": 2,
          "class_name": "cash",
          "confidence": 0.7949,
          "center": [
            894,
            777
          ],
          "width": 72,
          "height": 102
        },
        {
          "bbox": [
            1086,
            690,
            1163,
            804
          ],
          "class_id": 2,
          "class_name": "cash",
          "confidence": 0.7549,
          "center": [
            1124,
            747
          ],
          "width": 77,
          "height": 114
        },
        {
          "bbox": [
            1042,
            694,
            1102,
            798
          ],
          "class_id": 2,
          "class_name": "cash",
          "confidence": 0.4363,
          "center": [
            1072,
            746
          ],
          "width": 60,
          "height": 104
        },
        {
          "bbox": [
            994,
            697,
            1058,
            803
          ],
          "class_id": 2,
          "class_name": "cash",
          "confidence": 0.2973,
          "center": [
            1026,
            750
          ],
          "width": 64,
          "height": 106
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {
          "bbox": [
            608,
            0,
            1077,
            292
          ],
          "gender": "Female",
          "age_group": "Senior",
          "confidence": 0.5521
        },
        {
          "bbox": [
            767,
            826,
            1290,
            1080
          ],
          "gender": "Male",
          "age_group": "MiddleAged",
          "confidence": 0.656
        }
      ],
      "mood": [
        {
          "bbox": [
            608,
            0,
            1077,
            292
          ],
          "mood": "Happy",
          "confidence": 0.4224
        },
        {
          "bbox": [
            767,
            826,
            1290,
            1080
          ],
          "mood": "Happy",
          "confidence": 0.3664
        }
      ],
      "cashier": {
        "cashier_zone": {
          "persons": 1,
          "drawers": 1,
          "cash": 3
        },
        "customer_zone": {
          "persons": 1,
          "drawers": 0,
          "cash": 0
        },
        "case_id": "N3",
        "severity": "NORMAL",
        "alerts": [
          "N3 EVENT: Transaction in progress"
        ],
        "transaction": true,
        "frame_saved": true,
        "evidence_path": "/home/a7med/ml-server/outputs/cashier_production_sim/20260326T050929Z/evidence/normal/N3/cam_20260326T050939_702981.jpg",
        "frame_id": 2,
        "timestamp": "2026-03-26T05:09:39.693950+00:00",
        "cashier_persons": [
          {
            "bbox": [
              767,
              826,
              1290,
              1080
            ],
            "confidence": 0.8708,
            "zone": "ROI_CASHIER"
          }
        ]
      }
    }
  }
}
```

#### N5 example (frame_096810.jpg)
Annotated frame:
![Annotated Frame N5](outputs/cashier_production_sim/20260326T050929Z/annotated/frame_096810.jpg)

SSE event (shape):
```json
data: {
  "camera_id": "main_room",
  "frame_count": 4,
  "timestamp": "2026-03-26T05:09:40.092435+00:00",
  "frame": "<base64 JPEG from annotated corresponding frame>",
  "data": {
    "detection": {
      "count": 2,
      "items": [
        {
          "bbox": [
            752,
            0,
            1238,
            316
          ],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.9031,
          "center": [
            995,
            158
          ],
          "width": 486,
          "height": 316
        },
        {
          "bbox": [
            803,
            686,
            1319,
            1080
          ],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.8941,
          "center": [
            1061,
            883
          ],
          "width": 516,
          "height": 394
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {
          "bbox": [
            752,
            0,
            1238,
            316
          ],
          "gender": "Male",
          "age_group": "Senior",
          "confidence": 0.5167
        },
        {
          "bbox": [
            803,
            686,
            1319,
            1080
          ],
          "gender": "Male",
          "age_group": "Senior",
          "confidence": 0.5238
        }
      ],
      "mood": [
        {
          "bbox": [
            752,
            0,
            1238,
            316
          ],
          "mood": "Happy",
          "confidence": 0.4273
        },
        {
          "bbox": [
            803,
            686,
            1319,
            1080
          ],
          "mood": "Happy",
          "confidence": 0.4924
        }
      ],
      "cashier": {
        "cashier_zone": {
          "persons": 1,
          "drawers": 0,
          "cash": 0
        },
        "customer_zone": {
          "persons": 1,
          "drawers": 0,
          "cash": 0
        },
        "case_id": "N5",
        "severity": "NORMAL",
        "alerts": [],
        "transaction": false,
        "frame_saved": false,
        "evidence_path": null,
        "frame_id": 4,
        "timestamp": "2026-03-26T05:09:40.049337+00:00",
        "cashier_persons": [
          {
            "bbox": [
              803,
              686,
              1319,
              1080
            ],
            "confidence": 0.8941,
            "zone": "ROI_CASHIER"
          }
        ]
      }
    }
  }
}
```

#### A2 example (frame_097530.jpg)
Annotated frame:
![Annotated Frame A2](outputs/cashier_production_sim/20260326T050929Z/annotated/frame_097530.jpg)

SSE event (shape):
```json
data: {
  "camera_id": "main_room",
  "frame_count": 28,
  "timestamp": "2026-03-26T05:09:44.558313+00:00",
  "frame": "<base64 JPEG from annotated corresponding frame>",
  "data": {
    "detection": {
      "count": 8,
      "items": [
        {
          "bbox": [
            636,
            0,
            1159,
            329
          ],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.9005,
          "center": [
            897,
            164
          ],
          "width": 523,
          "height": 329
        },
        {
          "bbox": [
            803,
            875,
            1277,
            1080
          ],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.8751,
          "center": [
            1040,
            977
          ],
          "width": 474,
          "height": 205
        },
        {
          "bbox": [
            845,
            693,
            1187,
            918
          ],
          "class_id": 1,
          "class_name": "drawer_open",
          "confidence": 0.8661,
          "center": [
            1016,
            805
          ],
          "width": 342,
          "height": 225
        },
        {
          "bbox": [
            858,
            725,
            933,
            832
          ],
          "class_id": 2,
          "class_name": "cash",
          "confidence": 0.8498,
          "center": [
            895,
            778
          ],
          "width": 75,
          "height": 107
        },
        {
          "bbox": [
            1038,
            697,
            1106,
            804
          ],
          "class_id": 2,
          "class_name": "cash",
          "confidence": 0.8352,
          "center": [
            1072,
            750
          ],
          "width": 68,
          "height": 107
        },
        {
          "bbox": [
            974,
            705,
            1044,
            810
          ],
          "class_id": 2,
          "class_name": "cash",
          "confidence": 0.8256,
          "center": [
            1009,
            757
          ],
          "width": 70,
          "height": 105
        },
        {
          "bbox": [
            1100,
            690,
            1167,
            799
          ],
          "class_id": 2,
          "class_name": "cash",
          "confidence": 0.7996,
          "center": [
            1133,
            744
          ],
          "width": 67,
          "height": 109
        },
        {
          "bbox": [
            991,
            699,
            1061,
            816
          ],
          "class_id": 2,
          "class_name": "cash",
          "confidence": 0.2027,
          "center": [
            1026,
            757
          ],
          "width": 70,
          "height": 117
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {
          "bbox": [
            636,
            0,
            1159,
            329
          ],
          "gender": "Male",
          "age_group": "Senior",
          "confidence": 0.5673
        },
        {
          "bbox": [
            803,
            875,
            1277,
            1080
          ],
          "gender": "Male",
          "age_group": "Senior",
          "confidence": 0.7524
        }
      ],
      "mood": [
        {
          "bbox": [
            636,
            0,
            1159,
            329
          ],
          "mood": "Happy",
          "confidence": 0.4712
        },
        {
          "bbox": [
            803,
            875,
            1277,
            1080
          ],
          "mood": "Happy",
          "confidence": 0.4951
        }
      ],
      "cashier": {
        "cashier_zone": {
          "persons": 1,
          "drawers": 1,
          "cash": 4
        },
        "customer_zone": {
          "persons": 0,
          "drawers": 0,
          "cash": 0
        },
        "case_id": "A2",
        "severity": "ALERT",
        "alerts": [
          "A2 ALERT: Unexpected person in cashier zone"
        ],
        "transaction": false,
        "frame_saved": false,
        "evidence_path": null,
        "frame_id": 28,
        "timestamp": "2026-03-26T05:09:44.499561+00:00",
        "cashier_persons": [
          {
            "bbox": [
              803,
              875,
              1277,
              1080
            ],
            "confidence": 0.8751,
            "zone": "ROI_CASHIER"
          }
        ]
      }
    }
  }
}
```

### Production simulation artifacts (latest run)
- `outputs/cashier_production_sim/20260326T050929Z/annotated_sequence.gif`
- `outputs/cashier_production_sim/20260326T050929Z/stream.jsonl`
- `outputs/cashier_production_sim/20260326T050929Z/summary.json`
- `outputs/cashier_production_sim/20260326T050929Z/annotated/`

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

## 🎯 Models & Services

### 1. YOLO v8 Nano (Object Detection)
- **Model**: `models/yolov8n.pt` (or `models/yolov8n.engine`)
- **Framework**: PyTorch/TensorRT
- **Purpose**: Person & object detection with bounding boxes
- **Input**: RGB images (any size)
- **Output**: Bounding boxes, class IDs, confidence scores
- **Supported Classes**: 80 COCO classes (persons, chairs, tables, etc.)

### 2. Age/Gender Classification (ONNX)
- **Model**: `models/best_aged_gender_6.onnx`
- **Framework**: ONNX Runtime
- **Gender Classes**: `[Female, Male]`
- **Age Groups**: `[Young, MiddleAged, Senior, Elderly]`
- **Input**: Face crops (224×224 pixels)
- **Output**: Class predictions + confidence scores

### 3. Mood/Emotion Detection (ONNX)
- **Model**: `models/best_mood.onnx`
- **Framework**: ONNX Runtime
- **Classes**: `[Angry, Happy, Neutral]`
- **Input**: Face crops (128×128 pixels)
- **Output**: Mood class + confidence score

### 4. PPE Detection (ONNX)
- **Model**: `models/best_ppe.onnx`
- **Framework**: ONNX Runtime
- **Classes**: `[mask, hairnet, gloves]`
- **Input**: person crops (224×224 pixels)
- **Output**: PPE class + confidence score
---

## 📁 Project Structure

```
.
├── app.py                      # FastAPI main application
├── requirements.txt            # Python dependencies
├── README.md                   # This file (complete documentation)
├── .gitignore                  # Git ignore rules (models/ excluded)
├── models/                     # ML models (not tracked in git)
│   ├── yolov8n.pt             # YOLO v8 Nano (~25 MB)
│   ├── best_ppe.onnx           # PPE ONNX (~38 MB)
│   ├── best_aged_gender_6.onnx # Age/Gender ONNX (~85 MB)
│   └── best_mood.onnx          # Mood/Emotion ONNX (~15 MB)
├── services/                   # Service modules
│   ├── detector.py             # YOLO detection service
│   ├── age_gender.py           # Age/Gender classification
|   ├── ppe.py                  # PPE detection
│   └── mood.py                 # Mood/Emotion detection
├── scripts/                    # Testing and utility scripts
│   └── test_image_pipeline.py  # Image inference testing script
├── logger/                     # Logging configuration
│   └── logger_config.py        # Logger setup
└── outputs/                    # Test results directory
    ├── test_image/             # Sample image test results
    │   ├── annotated.jpg       # Annotated image with boxes
    │   ├── stream_data.json    # Full inference results
    │   └── run.log             # Execution logs
    └── f1_test/                # F1.webp test results
        ├── annotated.jpg       # 500×281px annotated JPEG
        ├── stream_data.json    # Complete detection/classification data
        └── run.log             # Pipeline execution log
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
git commit -m "Description of changes"

# Push to remote
git push origin mood
```

### Model Files
Models are excluded from git tracking to reduce repository size:
- `best_aged_gender_6.onnx` (~85 MB)
- `best_mood.onnx` (~15 MB)
- `best_ppe.onnx` (~38 MB)

- `yolov8n.pt` (~25 MB)

Download separately or configure via environment variables.

---

## ⚙️ Configuration

### Model Paths
```bash
# Environment variables
export YOLO_MODEL="./models/yolov8n.pt"
export AGE_GENDER_MODEL="./models/best_aged_gender_6.onnx"
export MOOD_MODEL="./models/best_mood.onnx"
export PPE_MODEL="./models/best_PPE.onnx"

```

### Detection Thresholds
- **YOLO Confidence**: 0.35 (configurable)
- **Face Detection Minimum Size**: 10×10 pixels
- **Mood Classification**: All 3 classes enabled

---


## 📋 API Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check / service info |
| `/process` | POST | Single image inference |
| `/stream` | POST | Video/stream processing |
| `/mood` | POST | Mood detection only |
| `/age-gender` | POST | Age/gender detection only |
| `/cameras` | POST | Configure cameras |
| `/cameras` | GET | List configured cameras |
| `/cameras/{cam_id}` | DELETE | Remove camera |
| `/detection/setup` | POST | Configure pipeline services |
| `/detection/start` | POST | Start processing |
| `/detection/stop` | POST | Stop processing |
| `/detection/status` | GET | Get operational status |
| `/detection/stream` | GET | SSE stream endpoint |
| `/docs` | GET | Interactive API documentation |

---

## 📞 Support & Documentation

For detailed API documentation and interactive testing:
```bash
# Access Swagger UI after starting server
http://localhost:8000/docs

# Access ReDoc documentation
http://localhost:8000/redoc
```

## Models file 
https://drive.google.com/drive/folders/1oAROlqkBo8C3rzTe4hAcS7abaIKC_Ugq?usp=drive_link 

