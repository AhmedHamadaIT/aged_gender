# Face Recognition API Documentation

This module provides a robust face recognition system for the `ml-server`, backed by InsightFace and FAISS. It supports library management, stranger detection, and real-time video stream recognition.

**HTTP cURL quick reference (aligned with the running app):** [API_USAGE.md §16 — Face recognition library](./API_USAGE.md#16-face-recognition-library-apiface).

## Architecture Components

The system is split into four primary components:

1. **API Router (`apis/face_lib.py`)**: Exposes REST endpoints for managing face libraries, people, strangers, and single-image recognition queries.
2. **Face Store (`services/face_store.py`)**: A thread-safe, FAISS-backed face embedding and metadata store. Uses `IndexIDMap(IndexFlatIP)` for cosine similarity search with incremental add/remove support. Manages distinct libraries as well as a separate surveillance index for unknown (stranger) faces.
3. **Face Engine (`services/face_engine.py`)**: An `InsightFace` wrapper that runs the full face pipeline: RetinaFace detection → 5-point landmark alignment (similarity transform) → ArcFace 512-d embedding extraction. Runs once on the full frame for efficiency.
4. **Task Pipeline (`services/face_recognition.py`)**: A task processor executed per-frame against a real-time stream. InsightFace detects faces on the full frame and associates them to YOLO-tracked persons via IoU overlap. Implements attendance / surveillance tracking.

### Face Detection Pipeline (per frame)

```
Frame from FrameBus
    │
    ├── YOLO BoT-SORT → person bounding boxes + track IDs
    │
    ├── InsightFace (single inference on full frame):
    │   ├── RetinaFace → face bboxes + 5-point landmarks
    │   ├── Similarity-transform alignment → 112×112 aligned face
    │   └── ArcFace → 512-d normalised embedding
    │
    ├── Face-to-Person association (IoU / containment)
    │
    └── FAISS cosine search → identity match or stranger
```

### Embedding Index Design

- **Index type**: `IndexIDMap(IndexFlatIP)` — exhaustive inner-product search on L2-normalised vectors (equivalent to cosine similarity), wrapped with custom int64 IDs.
- **FAISS ID scheme**: `person_id * 10000 + embedding_index` — enables `remove_ids()` for O(1) person deletion without full index rebuild.
- **Multi-embedding voting**: When a person has multiple enrolled images, the best (highest) cosine score across all their embeddings is used for matching.
- **Incremental updates**: Adding/removing persons updates the FAISS index in-place. No rebuild required.

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
  - **Behavior**: Detects faces in all uploaded images and stores their embeddings in the FAISS index. Each image produces one 512-d embedding. Multiple images per person improve matching robustness (multi-embedding voting).

- **`GET /api/face/lib/{lib_id}/persons`**
  - **Description**: List all people in a specific library.

- **`DELETE /api/face/lib/{lib_id}/persons/{person_id}`**
  - **Description**: Remove a person from the library. Uses FAISS `remove_ids()` — no full index rebuild needed.

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
│       ├── index.faiss        ← IndexIDMap(IndexFlatIP) with custom int64 IDs
│       ├── metadata.json      ← persons, embedding indices, face image paths
│       └── faces/
│           └── person_{id}_{idx}.jpg
└── surveillance/
    ├── strangers.faiss
    ├── strangers_meta.json
    └── faces/
```

### How Incremental Updates Work

- **Adding a person**: Embeddings are added to the FAISS index with deterministic IDs (`person_id * 10000 + emb_idx`). Metadata JSON is updated. No rebuild.
- **Removing a person**: FAISS `remove_ids()` deletes the embeddings by ID. Face images are deleted from disk. No rebuild.
- **Startup**: All libraries and indices are loaded from disk. The ID-to-person mapping is rebuilt from metadata.

### How Similarity Thresholding Works

1. Query embedding is L2-normalised.
2. FAISS inner-product search returns cosine similarities (0.0–1.0).
3. Scores are scaled to 0–100 range.
4. Per-person max score is computed (multi-embedding voting).
5. Results below the threshold are filtered out.
6. Top-k results are returned sorted by score.
