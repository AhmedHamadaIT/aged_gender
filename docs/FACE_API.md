# Face Recognition API Documentation

This module provides a robust face recognition system for the `ml-server`, backed by InsightFace and FAISS. It supports library management, stranger detection, and real-time video stream recognition.

## Architecture Components

The system is split into four primary components:

1. **API Router (`apis/face_lib.py`)**: Exposes REST endpoints for managing face libraries, people, strangers, and single-image recognition queries.
2. **Face Store (`services/face_store.py`)**: A thread-safe, FAISS-backed face embedding and metadata store. It manages distinct libraries as well as a separate surveillance index for unknown (stranger) faces.
3. **Face Engine (`services/face_engine.py`)**: An `InsightFace` wrapper that extracts 512-d embeddings, bounding boxes, pose estimations (yaw/pitch), and quality heuristics.
4. **Task Pipeline (`services/face_recognition.py`)**: A task processor designed to be executed per-frame against a real-time stream. It implements the `FaceRecognitionTask` running attendance / surveillance tracking over track IDs.

---

## API Endpoints

All routes are prefixed with `/api/face`.

### Library Management

Libraries are collections of registered people/faces used to scope search operations.

- **`POST /api/face/lib`**
  - **Description**: Create a new face library.
  - **Form Data**: `lib_id` (int), `name` (str)
  - **Returns**: `{ "status": "created", "library": ... }`

- **`GET /api/face/lib`**
  - **Description**: List all libraries.
  - **Returns**: `{ "libraries": [ ... ] }`

- **`GET /api/face/lib/{lib_id}`**
  - **Description**: Get details regarding a specific library, including registered people.

- **`DELETE /api/face/lib/{lib_id}`**
  - **Description**: Delete a face library and all its associated person data.

### Person Management

Add or remove registered identities from a specified library.

- **`POST /api/face/lib/{lib_id}/persons`**
  - **Description**: Add a new person with multiple images to a library.
  - **Form/Multipart Data**: `person_id` (int), `name` (str), `images` (List[UploadFile])
  - **Behavior**: Detects faces in all uploaded images and stores their embeddings in the FAISS index.

- **`GET /api/face/lib/{lib_id}/persons`**
  - **Description**: List all people in a specific library.

- **`DELETE /api/face/lib/{lib_id}/persons/{person_id}`**
  - **Description**: Remove a person from the library and rebuild the FAISS index.

### One-Shot Recognition

- **`POST /api/face/recognize`**
  - **Description**: Recognize faces in a provided image against registered libraries.
  - **Form Data**: `image` (UploadFile), `lib_ids` (str, comma-separated ints or "-1" for all), `threshold` (int, default=70), `top_k` (int, default=5).
  - **Returns**: Array of detected faces and their top library matches.

### Stranger Management

Strangers are unknown faces detected during surveillance that do not match existing library members. 

- **`GET /api/face/strangers`**
  - **Description**: List strangers detected by the pipeline. Supports pagination via `limit` and `offset` queries.

- **`POST /api/face/strangers/search`**
  - **Description**: Reverse search an uploaded image against the stranger index.
  - **Form Data**: `image` (UploadFile), `top_k` (int, default=5), `threshold` (int, default=30)

- **`DELETE /api/face/strangers`**
  - **Description**: Clear the stranger index and delete all stranger captures.

---

## Usage: Pipeline Configuration

To use the Face Recognition engine in live video processing streams, setup a `FaceRecognitionTask` by configuring it via POST `/api/tasks`.

### Example Task Payload
```json
{
    "taskId": 100,
    "taskName": "Attendance",
    "algorithmType": "FACE",
    "channelId": 1,
    "enable": true,
    "threshold": 70,
    "libIds": "-1",
    "enableStranger": true,
    "detailConfig": {
        "facePixelSize": 60,
        "qualityThreshold": 60,
        "yawThreshold": 35,
        "pitchThreshold": 25,
        "failCount": 2
    },
    "validWeekday": ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY"],
    "validStartTime": 0,
    "validEndTime": 86400000
}
```

### Parameter Explanations
* `threshold`: The similarity score (0-100) required to recognize a face against registered libraries.
* `libIds`: Comma-separated list of library IDs to search against. Pass `"-1"` to search all libraries.
* `enableStranger`: If true, faces that fail recognition `failCount` times will be added to the stranger store.
* `facePixelSize`: Minimum allowed width/height of the face bounding box in pixels.
* `qualityThreshold`: Minimum estimated image quality score (0-100), calculated using detection confidence, size relative to frame, and sharpness variance.
* `yawThreshold` / `pitchThreshold`: Maximum allowed rotation angle (in degrees). Skips heavily profile faces.

## Data Storage

The Face DB components generate data on-disk with the following layout (controlled by `FACE_STORAGE_DIR`, default: `./data/face`):

```text
data/face/
├── libraries/
│   └── lib_{id}/
│       ├── index.faiss
│       ├── metadata.json
│       └── faces/
│           └── person_{id}_{idx}.jpg
└── surveillance/
    ├── strangers.faiss
    ├── strangers_meta.json
    └── faces/
```
