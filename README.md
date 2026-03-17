# 🎯 Vision Pipeline API — ML Server

A modular, multi-camera real-time computer vision server built on **FastAPI** + **YOLO** + **ONNX**.  
Each camera runs in its own process; all services are chained together through a shared context.

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          app.py  (FastAPI)                       │
│                                                                  │
│   POST /cameras          →  Camera Registry                     │
│   POST /detection/setup  →  Choose pipeline services            │
│   POST /detection/start  →  Spawn camera processes              │
│   GET  /detection/stream →  SSE stream of annotated frames      │
│   GET  /detection/status →  Check camera health                 │
│   POST /detection/stop   →  Graceful shutdown                   │
└────────────────────────┬────────────────────────────────────────┘
                         │  one OS process per camera
               ┌─────────▼──────────┐
               │   pipeline.py      │
               │   CameraPipeline   │
               │                    │
               │  stream.py         │   ← RTSP / video file
               │     ↓ frame        │
               │  DetectorService   │   ← YOLO person detection
               │     ↓ context      │
               │  AgeGenderService  │   ← ONNX age + gender
               │     ↓ context      │
               │  [MoodService]     │   ← ONNX mood (plug-in)
               │     ↓ context      │
               │  [CounterService]  │   ← in/out line counting (plug-in)
               │                    │
               │  → result_queue    │   ← SSE events to clients
               └────────────────────┘
```

### Context Object (passed between services)

Every service **reads from** and **writes to** a single `context` dict:

```python
context = {
    "data": {
        "frame"    : np.ndarray,          # current BGR frame (annotated in-place)
        "detection": {
            "items": List[Detection],     # DetectorService writes this
            "count": int
        },
        "use_case" : {
            "age_gender": List[AgeGenderResult],  # AgeGenderService writes this
            "mood"      : List[MoodResult],       # MoodService writes this (future)
            "counter"   : {...},                  # CounterService writes this (future)
        }
    }
}
```

---

## 🔍 How the Pipeline Works — Full Walkthrough

This section explains **exactly** what happens from the moment you start the server to the moment a JSON event arrives in your browser, step by step.

---

### Phase 1 — Server Startup (`app.py`)

```
uvicorn app:app --host 0.0.0.0 --port 9000
```

When the server starts, FastAPI fires the `startup` event:

```python
@app.on_event("startup")
def startup():
    detection.set_pipeline(build_pipeline)
```

`build_pipeline` is a **factory function** stored inside the `DetectionResource` singleton.  
It is not called yet — it is just registered so it can be called later when you hit `POST /detection/start`.

---

### Phase 2 — Registering Cameras (`POST /cameras`)

```bash
curl -X POST http://localhost:9000/cameras \
  -d '{"cameras": [{"id": "cam1", "url": "rtsp://192.168.1.10/stream"}]}'
```

`CameraRegistry` (in `apis/cameras.py`) stores this in a plain Python dict:

```python
self._cameras = {"cam1": "rtsp://192.168.1.10/stream"}
```

Nothing runs yet. The registry is just an in-memory address book of cameras.

---

### Phase 3 — Choosing Services (`POST /detection/setup`)

```bash
curl -X POST http://localhost:9000/detection/setup \
  -d '{"pipeline": ["detector", "age_gender"]}'
```

`DetectionResource.on_setup()` validates the names against `REGISTRY` in `services/__init__.py`:

```python
REGISTRY = {
    "detector"  : DetectorService,
    "age_gender": AgeGenderService,
}
```

If the names are valid, they are saved as `self._pipeline_names = ["detector", "age_gender"]`.  
Still nothing is running — this is just configuration.

---

### Phase 4 — Starting (`POST /detection/start`)

```bash
curl -X POST http://localhost:9000/detection/start
```

For **each** registered camera, `DetectionResource._start()` does:

```python
process = multiprocessing.Process(
    target = build_pipeline,           # the factory function from Phase 1
    args   = (cam_id, rtsp_url,
               shared_state,           # Manager().dict()  ← cross-process state
               stop_event,             # Manager().Event() ← send "stop" signal
               result_queue,           # Manager().Queue() ← push frame results
               pipeline_names),
    daemon = True,
)
process.start()
```

Key points:
- **Each camera is a separate OS process.** This means cameras run truly in parallel on multiple CPU cores and GPU streams.
- `shared_state` is a `multiprocessing.Manager().dict()` — a special dict that lives in a separate manager process and is visible from all camera processes and from the main server process simultaneously.
- `result_queue` is a `multiprocessing.Manager().Queue()` — all camera processes push their frame results into the same queue; the main server drains it for SSE.
- `stop_event` is a `multiprocessing.Manager().Event()` — setting it from the main process tells the camera process to exit cleanly.

---

### Phase 5 — The Frame Loop (`pipeline.py` inside each camera process)

Once `build_pipeline` runs in the child process, it creates a `CameraPipeline` and calls `.run()`.

#### 5a — Instantiate services

```python
services = [cls() for cls in self._service_classes]
```

Each service class is instantiated **once** per process. This is where model files are loaded into memory (YOLO weights, ONNX sessions, etc.).

#### 5b — Open the video source (`stream.py`)

```python
for frame in frames(self.rtsp_url):
```

`frames()` is a generator in `stream.py`. It:
1. Detects whether the source is RTSP (`rtsp://...`) or a local file.
2. Opens it with OpenCV `VideoCapture`.
3. Yields one BGR `numpy` frame at a time, forever.
4. If RTSP drops, it **reconnects automatically** (up to 10 consecutive failures before retry).

#### 5c — Resize the frame

```python
resized_frame = resize(frame, self.width, self.height)
```

`utils.resize()` downscales to `WIDTH` (env default `1280px`), preserving aspect ratio if `HEIGHT=0`.  
This keeps inference fast without processing 4K frames unnecessarily.

#### 5d — Encode the **raw** frame as base64 (before annotation)

```python
_, buf    = cv2.imencode(".jpg", resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
frame_b64 = base64.b64encode(buf).decode("utf-8")
```

The clean frame is encoded **before** any boxes are drawn on it, so clients that want the raw image can decode `frame_b64` themselves.

#### 5e — Build the context dict

```python
context = {
    "data": {
        "frame"    : resized_frame.copy(),
        "detection": {},
        "use_case" : {},
    }
}
```

This is the **shared data bus** between all services. Every service reads from it and writes into it.

#### 5f — Run each service in order

```python
for service in services:
    context = service(context)
```

This is the **heart of the pipeline**. Services are called one by one:

| Step | Service | Reads | Writes |
|------|---------|-------|--------|
| 1 | `DetectorService` | `context["data"]["frame"]` | `context["data"]["detection"]` |
| 2 | `AgeGenderService` | `frame` + `detection.items` | `context["data"]["use_case"]["age_gender"]` |
| 3 | *(MoodService)* | `frame` + `detection.items` | `context["data"]["use_case"]["mood"]` |
| 4 | *(CounterService)* | `detection.items` | `context["data"]["use_case"]["counter"]` |

Because each service returns the mutated `context`, the next service automatically sees everything the previous one wrote.  
**Order matters** — `AgeGenderService` depends on `DetectorService` having already written `detection.items`.

#### 5g — The Detector in detail (`services/detector.py`)

```python
results = self.model.predict(frame, conf=self.conf, device=self.device, ...)
```

YOLO runs inference on the full frame. For each box above the confidence threshold:

```python
detections.append(Detection(x1, y1, x2, y2, class_id, class_name, confidence))
```

`Detection` is a dataclass with helper properties: `.bbox`, `.center`, `.width`, `.height`.

If `SAVE_OUTPUT=True`, YOLO bounding boxes are **drawn directly on `context["data"]["frame"]`** so that downstream services and the saved JPEG both show the boxes.

#### 5h — The Age/Gender classifier in detail (`services/age_gender.py`)

For each `Detection` in `detection.items`:

1. **Crop** the person bbox (with `PADDING` pixels of extra margin):
   ```python
   crop = frame[y1-pad : y2+pad, x1-pad : x2+pad]
   ```
2. **Preprocess** the crop:
   - Resize to `224×224`
   - Convert BGR → RGB
   - Normalize with ImageNet mean/std
   - Transpose `HWC → NCHW` → shape `(1, 3, 224, 224)`
3. **Run ONNX inference:**
   ```python
   outputs = self.sess.run(None, {self.inp_name: blob})
   # outputs[0] → gender_logits (1, 2)
   # outputs[1] → age_logits    (1, 4)
   ```
4. **Softmax** both outputs, pick `argmax`:
   - Gender: `["Female", "Male"]`
   - Age: `["Young", "MiddleAged", "Senior", "Elderly"]`
5. Append an `AgeGenderResult` to the results list.

The results list is written to `context["data"]["use_case"]["age_gender"]`.

#### 5i — Save the annotated frame to disk

```python
if self.save_output:
    save_frame(context["data"]["frame"], self.out_dir, frame_count)
```

`utils.save_frame()` writes a JPEG to `./outputs/<camera_id>/frame_000042.jpg`.  
At this point the frame has all boxes and labels drawn by every service.

#### 5j — Serialize and push to the result queue

```python
result = {
    "camera_id"  : self.camera_id,
    "frame_count": frame_count,
    "timestamp"  : datetime.utcnow().isoformat(),
    "frame"      : frame_b64,          # ← clean raw frame (encoded in 5d)
    "data": {
        "detection": {
            "count": N,
            "items": [det.to_dict() for det in detections],
        },
        "use_case": {
            "age_gender": [r.to_dict() for r in age_gender_results],
            # + mood, counter, etc. if those services are active
        },
    },
}
self.result_queue.put_nowait(result)   # never blocks — drops frame if queue full
```

If the queue is full the frame is silently dropped — this is intentional to prevent inference from being held up by a slow client.

#### 5k — Update shared state

```python
self.shared_state[self.camera_id] = {
    "running"       : True,
    "frame_count"   : frame_count,
    "fps"           : fps,
    "uptime_seconds": ...,
}
```

This data is what `GET /detection/status` returns.

---

### Phase 6 — Streaming to Clients (`GET /detection/stream`)

```python
@app.get("/detection/stream")
async def detection_stream():
    result_queue = detection.result_queue()

    async def event_generator():
        while True:
            try:
                result = result_queue.get_nowait()
                yield f"data: {json.dumps(result)}\n\n"
            except Exception:
                await asyncio.sleep(0.01)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

This is a **Server-Sent Events (SSE)** endpoint:
- It drains the `result_queue` as fast as new frames arrive.
- Each event is `data: {json}\n\n` — the standard SSE format.
- Multiple cameras all push into the **same queue**, so a single SSE connection delivers events from all running cameras interleaved.
- The `asyncio.sleep(0.01)` yields control to the event loop when no frame is ready, so the server stays responsive.

---

### Phase 7 — Stopping (`POST /detection/stop`)

```python
self._stop_events[cam_id].set()        # signal the camera process to exit
self._processes[cam_id].join(timeout=5)  # wait up to 5s for clean shutdown
```

The camera process checks `stop_event.is_set()` at the top of each frame loop iteration and breaks out cleanly.  
The `finally` block in `CameraPipeline.run()` always sets `shared_state["running"] = False`.

---

### Complete Frame Lifecycle (Sequence Diagram)

```
Client                   FastAPI (main proc)        Camera Process
  |                              |                         |
  |-- POST /detection/start ---> |                         |
  |                    spawn Process──────────────────────>|
  |                              |                  frames() loop
  |                              |                  DetectorService(context)
  |                              |                  AgeGenderService(context)
  |                              |                  result_queue.put(result)
  |                              |                         |
  |-- GET /detection/stream ---> |                         |
  |<-- SSE: data:{json} -------- |<-- queue.get_nowait()   |
  |<-- SSE: data:{json} -------- |                         |
  |                              |                  (next frame...)
  |-- POST /detection/stop ----& |                         |
  |                    stop_event.set()──────────────────> |
  |                              |                  break loop
  |                              |                  process exits
```

---

## 📁 File Map

| File/Folder | Role |
|---|---|
| `app.py` | FastAPI entry point, all HTTP routes, startup logic |
| `pipeline.py` | `CameraPipeline` runs in each camera subprocess |
| `stream.py` | RTSP / video file frame generator |
| `schemas.py` | Pydantic request/response models |
| `utils.py` | Drawing, resizing, frame saving helpers |
| `services/__init__.py` | **Service registry** — maps names → classes |
| `services/detector.py` | YOLO detection service |
| `services/age_gender.py` | ONNX age & gender classification service |
| `services/base.py` | Shared base (currently minimal) |
| `apis/cameras.py` | Camera CRUD endpoints + in-memory registry |
| `apis/detection.py` | Detection lifecycle (start/stop/status/stream) |
| `models/` | ONNX model files |
| `logger/` | Shared structured logger |
| `dockerfile` | Jetson-compatible container (dustynv PyTorch base) |
| `docker-compose.yml` | Single-service compose with GPU passthrough |

---

## 🔗 How Services Chain Together

`CameraPipeline.run()` (in `pipeline.py`) executes services **sequentially** on every frame:

```python
for service in services:          # e.g. [DetectorService, AgeGenderService]
    context = service(context)    # each service mutates and returns context
```

1. **`DetectorService`** — runs YOLO inference, populates `context["data"]["detection"]`
2. **`AgeGenderService`** — iterates over `detection.items`, crops each bbox, runs ONNX, writes `context["data"]["use_case"]["age_gender"]`
3. *(future)* **`MoodService`** — same pattern, writes `context["data"]["use_case"]["mood"]`
4. *(future)* **`CounterService`** — same pattern, writes `context["data"]["use_case"]["counter"]`

---

## 🤖 Existing Models

### 1 — YOLO Detector (`DetectorService`)
- **File:** configurable via `YOLO_MODEL` env var (default `yolov8n.pt`)
- **Framework:** Ultralytics YOLO
- **Reads:** raw frame
- **Writes:** `detection.items` (list of `Detection` dataclass objects)

### 2 — Age & Gender (`AgeGenderService`)
- **File:** `models/best_aged_gender_6.onnx`
- **Framework:** ONNX Runtime
- **Architecture:** Dual-head MobileNetV3
- **Input:** `(1, 3, 224, 224)` — ImageNet-normalized RGB crop
- **Outputs:**
  - `gender_logits (1, 2)` → `[Female, Male]`
  - `age_output    (1, 4)` → `[Young, MiddleAged, Senior, Elderly]`
- **Reads:** frame + `detection.items`
- **Writes:** `use_case["age_gender"]` (list of `AgeGenderResult` dataclass)

---

## 🆕 How to Add a New Service (General Pattern)

> Follow these 4 steps to add **any** new service (mood, counter, tracker, etc.)

### Step 1 — Create `services/your_service.py`

```python
"""
services/your_service.py
------------------------
MyService — describe what it does.

Reads  context["data"]["detection"]["items"]
Writes context["data"]["use_case"]["my_key"]
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

import onnxruntime as ort
import numpy as np


@dataclass
class MyResult:
    bbox      : tuple
    label     : str
    confidence: float

    def to_dict(self):
        return {"bbox": list(self.bbox), "label": self.label, "confidence": round(self.confidence, 4)}


class MyService:
    def __init__(self):
        model_path = os.getenv("MY_MODEL", "./models/my_model.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[MY_SERVICE] Model not found: {model_path}")

        self.sess     = ort.InferenceSession(model_path)
        self.inp_name = self.sess.get_inputs()[0].name
        self.save     = os.getenv("SAVE_OUTPUT", "True").lower() in ("true", "1", "yes")

        print(f"[MY_SERVICE] Ready — {model_path}")

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        # resize, normalize, and convert to NCHW
        ...

    def _predict(self, crop: np.ndarray):
        blob    = self._preprocess(crop)
        outputs = self.sess.run(None, {self.inp_name: blob})
        # parse outputs
        return label, confidence

    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        frame      = context["data"]["frame"]
        detections = context["data"]["detection"].get("items", [])
        results    = []

        for det in detections:
            # crop person bbox from frame
            crop = frame[det.y1:det.y2, det.x1:det.x2]
            label, conf = self._predict(crop)
            results.append(MyResult(bbox=det.bbox, label=label, confidence=conf))

        context["data"]["use_case"]["my_key"] = results

        if self.save:
            context["data"]["frame"] = self._draw(frame, results)

        return context                      # ← always return context!

    def _draw(self, frame, results):
        # draw your labels on frame
        ...
        return frame
```

### Step 2 — Register in `services/__init__.py`

```python
from .my_service import MyService

REGISTRY = {
    "detector"  : DetectorService,
    "age_gender": AgeGenderService,
    "my_service": MyService,          # ← add this line
}
```

### Step 3 — Set environment variable

Add to `.env`:

```env
MY_MODEL=./models/my_model.onnx
```

### Step 4 — Configure pipeline via API

```bash
curl -X POST http://localhost:9000/detection/setup \
  -H "Content-Type: application/json" \
  -d '{"pipeline": ["detector", "age_gender", "my_service"]}'
```

---

## 😊 Adding the Mood Model (`MoodService`)

The mood model (`best_mood.onnx`) is a 3-class YOLO-based emotion classifier, converted to ONNX.

### `services/mood.py`

```python
"""
services/mood.py
----------------
Mood / Emotion service using best_mood.onnx (ONNX export of YOLOv8 3-class model).

Classes:  0=Happy  1=Neutral  2=Sad   (or check model.names after export)

Reads  context["data"]["frame"]
       context["data"]["detection"]["items"]
Writes context["data"]["use_case"]["mood"]  — List[MoodResult]
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

import cv2
import numpy as np
import onnxruntime as ort


MOOD_LABELS = ["Happy", "Neutral", "Sad"]   # adjust if your model has different classes


@dataclass
class MoodResult:
    bbox      : tuple
    mood      : str
    confidence: float

    def to_dict(self):
        return {
            "bbox"      : list(self.bbox),
            "mood"      : self.mood,
            "confidence": round(self.confidence, 4),
        }


class MoodService:
    def __init__(self):
        model_path = os.getenv("MOOD_MODEL", "./models/best_mood.onnx")
        self.save  = os.getenv("SAVE_OUTPUT", "True").lower() in ("true", "1", "yes")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[MOOD] Model not found: {model_path}")

        self.sess     = ort.InferenceSession(model_path)
        self.inp_name = self.sess.get_inputs()[0].name
        inp_shape     = self.sess.get_inputs()[0].shape   # e.g. [1,3,64,64]
        self.img_size = (int(inp_shape[3]), int(inp_shape[2]))   # (W, H)

        print(f"[MOOD] Ready — input {self.img_size}, classes: {MOOD_LABELS}")

    # ── Preprocessing ──────────────────────────────────────────────────────
    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        img = cv2.resize(crop, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]   # HWC→NCHW
        return img

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    # ── Inference ──────────────────────────────────────────────────────────
    def _predict(self, crop: np.ndarray):
        if crop.size == 0:
            return "Unknown", 0.0
        blob    = self._preprocess(crop)
        outputs = self.sess.run(None, {self.inp_name: blob})
        probs   = self._softmax(outputs[0][0])   # shape (num_classes,)
        idx     = int(np.argmax(probs))
        return MOOD_LABELS[idx], float(probs[idx])

    # ── Service callable ───────────────────────────────────────────────────
    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        frame      = context["data"]["frame"]
        detections = context["data"]["detection"].get("items", [])
        results    = []

        for det in detections:
            crop        = frame[det.y1:det.y2, det.x1:det.x2]
            mood, conf  = self._predict(crop)
            results.append(MoodResult(bbox=det.bbox, mood=mood, confidence=conf))

        context["data"]["use_case"]["mood"] = results

        if self.save:
            context["data"]["frame"] = self._draw(frame, results)

        return context

    # ── Annotation ─────────────────────────────────────────────────────────
    def _draw(self, frame: np.ndarray, results: List[MoodResult]) -> np.ndarray:
        out = frame.copy()
        MOOD_COLORS = {
            "Happy"  : (0, 215, 255),
            "Neutral": (200, 200, 200),
            "Sad"    : (255, 100,  50),
        }
        for r in results:
            x1, y1, x2, y2 = r.bbox
            color = MOOD_COLORS.get(r.mood, (255, 255, 255))
            label = f"{r.mood} {r.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            # draw label strip above the bbox top line
            cv2.rectangle(out, (x1, y1 - th - 20), (x1 + tw + 4, y1 - 4), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return out
```

### Register the Mood Service

```python
# services/__init__.py
from .mood import MoodService

REGISTRY = {
    "detector"  : DetectorService,
    "age_gender": AgeGenderService,
    "mood"      : MoodService,        # ← add
}
```

### Add to `.env`

```env
MOOD_MODEL=./models/best_mood.onnx
```

### Start with mood enabled

```bash
# Setup pipeline
curl -X POST http://localhost:9000/detection/setup \
  -H "Content-Type: application/json" \
  -d '{"pipeline": ["detector", "age_gender", "mood"]}'

# Start
curl -X POST "http://localhost:9000/detection/start"
```

---

## 🚶 Adding In/Out People Counter (`CounterService`)

The counter uses a **virtual line** drawn across the frame. When a person's center point crosses the line between two frames, they are counted as IN or OUT depending on the direction of movement.

### `services/counter.py`

```python
"""
services/counter.py
-------------------
Virtual-line people counter.

Draws a horizontal (or vertical) line across the frame.
Tracks each detected person's center point using a simple
dict keyed by detection position. When a person crosses the
line, increments IN or OUT based on direction.

Reads  context["data"]["detection"]["items"]
Writes context["data"]["use_case"]["counter"]
         {
           "in_count"  : int,
           "out_count" : int,
           "net"       : int,   # in - out
         }
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple


# ── Config from environment ──────────────────
# LINE_POSITION: 0.0-1.0 fraction of frame height (horizontal line)
# LINE_DIRECTION: "horizontal" or "vertical"
LINE_POSITION  = float(os.getenv("LINE_POSITION",  "0.5"))
LINE_DIRECTION = os.getenv("LINE_DIRECTION", "horizontal")


@dataclass
class CounterResult:
    in_count : int
    out_count: int
    net      : int

    def to_dict(self):
        return {"in_count": self.in_count, "out_count": self.out_count, "net": self.net}


class CounterService:
    def __init__(self):
        self.save      = os.getenv("SAVE_OUTPUT", "True").lower() in ("true", "1", "yes")
        self.in_count  = 0
        self.out_count = 0
        # {track_id: last_side}   — "above" | "below" (horizontal) or "left" | "right" (vertical)
        self._prev_side: Dict[int, str] = {}
        print(f"[COUNTER] Ready — line at {LINE_POSITION:.0%} ({LINE_DIRECTION})")

    # ── Side of line ──────────────────────────
    def _side(self, cx: int, cy: int, line_px: int) -> str:
        if LINE_DIRECTION == "horizontal":
            return "above" if cy < line_px else "below"
        else:
            return "left" if cx < line_px else "right"

    # ── Generate a stable ID for each detection ──
    # (replace with a real tracker for production)
    def _det_id(self, det) -> int:
        # Simple: use rounded center as proxy ID (good enough without a tracker)
        cx, cy = det.center
        return hash((round(cx, -1), round(cy, -1)))   # bucket to 10px grid

    # ── Service callable ──────────────────────
    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        frame      = context["data"]["frame"]
        detections = context["data"]["detection"].get("items", [])

        h, w = frame.shape[:2]
        line_px = int((h if LINE_DIRECTION == "horizontal" else w) * LINE_POSITION)

        current_ids = set()

        for det in detections:
            cx, cy  = det.center
            did     = self._det_id(det)
            side    = self._side(cx, cy, line_px)
            current_ids.add(did)

            if did in self._prev_side:
                prev = self._prev_side[did]
                if LINE_DIRECTION == "horizontal":
                    if prev == "above" and side == "below":
                        self.in_count += 1          # top→bottom = IN
                    elif prev == "below" and side == "above":
                        self.out_count += 1         # bottom→top = OUT
                else:
                    if prev == "left" and side == "right":
                        self.in_count += 1
                    elif prev == "right" and side == "left":
                        self.out_count += 1

            self._prev_side[did] = side

        # Clean up stale IDs
        for did in list(self._prev_side):
            if did not in current_ids:
                del self._prev_side[did]

        result = CounterResult(
            in_count  = self.in_count,
            out_count = self.out_count,
            net       = self.in_count - self.out_count,
        )

        context["data"]["use_case"]["counter"] = result

        if self.save:
            context["data"]["frame"] = self._draw(frame, line_px, result)

        return context

    # ── Annotation ─────────────────────────────
    def _draw(self, frame: np.ndarray, line_px: int, result: CounterResult) -> np.ndarray:
        out    = frame.copy()
        h, w   = out.shape[:2]
        color  = (0, 200, 255)

        if LINE_DIRECTION == "horizontal":
            cv2.line(out, (0, line_px), (w, line_px), color, 2)
        else:
            cv2.line(out, (line_px, 0), (line_px, h), color, 2)

        # Overlay counters
        labels = [
            f"IN : {result.in_count}",
            f"OUT: {result.out_count}",
            f"NET: {result.net}",
        ]
        y = 60
        for text in labels:
            cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        color, 1, cv2.LINE_AA)
            y += 30

        return out
```

### Register the Counter Service

```python
# services/__init__.py
from .counter import CounterService

REGISTRY = {
    "detector"  : DetectorService,
    "age_gender": AgeGenderService,
    "mood"      : MoodService,
    "counter"   : CounterService,   # ← add
}
```

### `.env` settings for counter

```env
LINE_POSITION=0.5        # 50% = centre of frame
LINE_DIRECTION=horizontal
```

### Enable via API

```bash
curl -X POST http://localhost:9000/detection/setup \
  -H "Content-Type: application/json" \
  -d '{"pipeline": ["detector", "counter"]}'
```

Counter results appear in every SSE frame event:

```json
{
  "data": {
    "use_case": {
      "counter": {"in_count": 12, "out_count": 7, "net": 5}
    }
  }
}
```

> **Tip:** For accurate production counting, replace `_det_id()` with a real multi-object tracker (e.g. `ByteTrack` from `ultralytics` or `norfair`). The counter service stores state per-process — use `multiprocessing.Manager().dict()` if you need counters to survive restarts or to aggregate across cameras.

---

## 🌐 Complete API Reference

### Camera Management

| Method | Route | Body | Description |
|--------|-------|------|-------------|
| `POST` | `/cameras` | `{"cameras": [{"id": "cam1", "url": "rtsp://..."}]}` | Add cameras |
| `GET` | `/cameras` | — | List all cameras |
| `DELETE` | `/cameras/{cam_id}` | — | Remove a camera |

### Detection Lifecycle

| Method | Route | Body | Description |
|--------|-------|------|-------------|
| `POST` | `/detection/setup` | `{"pipeline": ["detector", "age_gender"]}` | Choose services |
| `POST` | `/detection/start` | `?camera_id=cam1` (optional) | Start processing |
| `POST` | `/detection/stop` | `?camera_id=cam1` (optional) | Stop processing |
| `GET` | `/detection/status` | — | Camera health + FPS |
| `GET` | `/detection/stream` | — | SSE stream of frame events |

### SSE Event Format

```json
{
  "camera_id"  : "cam1",
  "frame_count": 42,
  "timestamp"  : "2026-03-15T01:00:00.000",
  "frame"      : "<base64 JPEG>",
  "data": {
    "detection": {
      "count": 2,
      "items": [
        {"bbox": [100, 200, 300, 400], "class_id": 0, "class_name": "person", "confidence": 0.92, "center": [200, 300], "width": 200, "height": 200}
      ]
    },
    "use_case": {
      "age_gender": [{"bbox": [...], "gender": "Male", "age_group": "Young", "confidence": 0.87}],
      "mood"      : [{"bbox": [...], "mood": "Happy", "confidence": 0.76}],
      "counter"   : {"in_count": 12, "out_count": 7, "net": 5}
    }
  }
}
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `YOLO_MODEL` | `yolov8n.pt` | YOLO model file path |
| `AGE_GENDER_MODEL` | `./models/best_aged_gender_6.onnx` | Age/gender ONNX model |
| `MOOD_MODEL` | `./models/best_mood.onnx` | Mood ONNX model |
| `CONF_THRESHOLD` | `0.35` | YOLO detection confidence |
| `FILTER_CLASSES` | *(all)* | Comma-separated class IDs to keep (e.g. `0` for person only) |
| `DEVICE` | `0` | CUDA device index or `cpu` |
| `SAVE_OUTPUT` | `True` | Save annotated frames to disk |
| `OUTPUT_DIR` | `./outputs` | Directory for saved frames |
| `WIDTH` | `1280` | Frame resize width |
| `HEIGHT` | `0` | Frame resize height (0 = keep aspect ratio) |
| `USE_STREAM` | `True` | Use RTSP; `False` = use local video file |
| `CAMERA_1_URL` | *(empty)* | Default RTSP URL for `stream.py` |
| `INPUT_VIDEO` | `./videos/sample.mp4` | Local video fallback path |
| `AGE_GENDER_PADDING` | `10` | Pixel padding around person bbox before classification |
| `LINE_POSITION` | `0.5` | Line position fraction for CounterService |
| `LINE_DIRECTION` | `horizontal` | `horizontal` or `vertical` for CounterService |

---

## 🚀 Quick Start

### Option A — Docker (Jetson / NVIDIA GPU)

```bash
# Build and start
docker compose up -d --build

# Tail logs
docker compose logs -f

# Open shell
docker compose exec yolo-detect bash
```

### Option B — Local (with Python 3.10+)

```bash
pip install fastapi uvicorn ultralytics onnxruntime opencv-python-headless python-dotenv

uvicorn app:app --host 0.0.0.0 --port 9000 --reload
```

### Typical Workflow

```bash
# 1. Add camera
curl -X POST http://localhost:9000/cameras \
  -H "Content-Type: application/json" \
  -d '{"cameras": [{"id": "cam1", "url": "rtsp://your-camera/stream"}]}'

# 2. Choose pipeline
curl -X POST http://localhost:9000/detection/setup \
  -H "Content-Type: application/json" \
  -d '{"pipeline": ["detector", "age_gender", "mood", "counter"]}'

# 3. Start
curl -X POST http://localhost:9000/detection/start

# 4. Subscribe to live stream (SSE)
curl -N http://localhost:9000/detection/stream

# 5. Check status
curl http://localhost:9000/detection/status

# 6. Stop
curl -X POST http://localhost:9000/detection/stop
```

---

## 🗂️ Adding a New Pipeline Service — Checklist

```
[ ] Create services/<name>.py
      - Define a Result dataclass with to_dict()
      - Define a Service class with __init__, __call__, _draw
      - __call__ must: read context → mutate → return context
[ ] Import and register in services/__init__.py
[ ] Add model path env var to .env
[ ] Restart server / container
[ ] POST /detection/setup with updated pipeline list
```
