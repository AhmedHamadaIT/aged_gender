# Adding a New Task Service

This guide walks through every file you need to touch to add a brand new task (e.g. `CROWD_DENSITY`, `LOITERING`, `UNIFORM_CHECK`) to the ML server.

**See also:** [VISION_PIPELINE_README.md](./VISION_PIPELINE_README.md) — tests, cURL/SSH, and cashier **`data`** / cases. [SERVICE_TEST.md](./SERVICE_TEST.md) redirects there.

---

## Concepts to understand first

| Term | What it is |
|---|---|
| **FrameBus** | One process per camera. Captures frames, runs YOLO BoT-SORT, and fans `payload` dicts out to every task queue. |
| **Task Worker** | One process per task. Reads from its queue, calls your task class, pushes events to the SSE result queue. |
| **Task Class** | Your code. Receives a `payload`, returns a `list` of event dicts. |
| **Task Registry** | `services/__init__.py` — maps `algorithmType` string → your class. |
| **TaskConfig** | `apis/tasks.py` — defines the schema the backend sends when registering a task. |

---

## What your task class receives (`payload`)

Every frame, `task_worker.py` calls `your_task(payload)`. The payload dict has:

```python
{
    "camera_id" : str,               # e.g. "1"
    "frame_id"  : int,               # monotonically increasing frame counter
    "timestamp" : str,               # ISO 8601 UTC, e.g. "2026-04-05T10:00:01.123456"
    "frame_b64" : str,               # base64-encoded JPEG of the raw (pre-annotation) frame
    "frame"     : np.ndarray,        # BGR frame, shape (H, W, 3)
    "detection" : {
        "count" : int,               # number of tracked detections in this frame
        "items" : List[Detection],   # see Detection dataclass below
    },
}
```

### `Detection` dataclass (`services/detector.py`)

```python
@dataclass
class Detection:
    x1        : int          # bounding box top-left x
    y1        : int          # bounding box top-left y
    x2        : int          # bounding box bottom-right x
    y2        : int          # bounding box bottom-right y
    class_id  : int          # YOLO class index (COCO: 0 = person)
    class_name: str          # e.g. "person"
    confidence: float        # 0.0 – 1.0
    track_id  : int          # BoT-SORT track ID; -1 if no track assigned yet

    # Convenience properties
    @property
    def bbox(self) -> tuple:       # (x1, y1, x2, y2)
    def center(self) -> tuple:     # ((x1+x2)//2, (y1+y2)//2)
    def width(self) -> int:        # x2 - x1
    def height(self) -> int:       # y2 - y1
```

**Important:** `track_id == -1` means BoT-SORT has not yet assigned a stable ID (happens on the very first frame). Always filter these out if your logic depends on tracking continuity.

---

## Step-by-step: adding a new task

### Step 1 — Create `services/your_task.py`

You can place the task class in a **dedicated module** or **next to a related service** in the same file (this repo keeps **`CashierDrawerTask`** and **`CashierService`** together in [`services/cashier.py`](../services/cashier.py), registered as `CASHIER_BOX_OPEN`).

Your task is a class with:
- `__init__(self, task_config: dict)` — receives the full task config from the backend
- `__call__(self, payload: dict) -> list` — called every frame; returns a list of event dicts (empty list = nothing happened this frame)

```python
# services/loitering.py
import os, json, hashlib, time
from datetime import datetime, timezone

class LoiteringTask:

    def __init__(self, task_config: dict):
        self.task_id    = task_config["taskId"]
        self.task_name  = task_config["taskName"]
        self.channel_id = task_config["channelId"]
        self.threshold  = task_config.get("threshold", 50) / 100.0
        self.enable     = task_config.get("enable", True)

        detail = task_config.get("detailConfig", {})
        # Pull any custom fields your task needs from detailConfig
        self.dwell_seconds = detail.get("dwellSeconds", 30)

        # Parse zone / line geometry from areaPosition if needed
        self.zones = self._parse_zones(task_config.get("areaPosition", "[]"))

        # Schedule
        _WEEKDAY_MAP = {"MONDAY":0,"TUESDAY":1,"WEDNESDAY":2,"THURSDAY":3,
                        "FRIDAY":4,"SATURDAY":5,"SUNDAY":6}
        raw_days = task_config.get("validWeekday", list(_WEEKDAY_MAP.keys()))
        self.valid_weekdays = {_WEEKDAY_MAP[d] for d in raw_days if d in _WEEKDAY_MAP}
        self.valid_start_ms = task_config.get("validStartTime", 0)
        self.valid_end_ms   = task_config.get("validEndTime", 86400000)

        # Per-track state: {track_id: first_seen_timestamp}
        self._first_seen: dict = {}

        # Storage
        self._events_dir = os.getenv("EVENTS_DIR", "/local/storage/events")
        os.makedirs(self._events_dir, exist_ok=True)
        self._jsonl_path = os.path.join(self._events_dir, f"task_{self.task_id}.jsonl")

        print(f"[Loitering/{self.task_id}] Ready — dwell={self.dwell_seconds}s")

    def __call__(self, payload: dict) -> list:
        """Called every frame. Return a list of event dicts or []."""
        if not self.enable or not self._in_schedule():
            return []

        frame     = payload["frame"]
        detection = payload["detection"]
        now       = time.time()
        events    = []

        persons = [
            d for d in detection.get("items", [])
            if d.class_name == "person"
            and d.confidence >= self.threshold
            and d.track_id != -1
        ]

        active_ids = set()

        for det in persons:
            active_ids.add(det.track_id)

            # Record first time we saw this track
            if det.track_id not in self._first_seen:
                self._first_seen[det.track_id] = now

            dwell = now - self._first_seen[det.track_id]
            if dwell >= self.dwell_seconds:
                event = self._build_event(det, dwell, payload["timestamp"])
                self._persist(event)
                events.append(event)
                # Reset so it doesn't fire every frame after threshold
                self._first_seen[det.track_id] = now

        # Clean up tracks no longer in frame
        self._first_seen = {k: v for k, v in self._first_seen.items() if k in active_ids}

        return events

    def _build_event(self, det, dwell_seconds: float, timestamp: str) -> dict:
        now_ms   = int(time.time() * 1000)
        event_id = hashlib.md5(f"{self.task_id}_{det.track_id}_{now_ms}".encode()).hexdigest()
        x1, y1, x2, y2 = det.bbox
        return {
            "eventId"     : event_id,
            "eventType"   : "LOITERING",
            "timestamp"   : now_ms,
            "timestampUTC": datetime.fromtimestamp(now_ms/1000, tz=timezone.utc).isoformat().replace("+00:00","Z"),
            "taskId"      : self.task_id,
            "taskName"    : self.task_name,
            "channelId"   : self.channel_id,
            "person": {
                "trackingId" : str(det.track_id),
                "boundingBox": {"x": x1, "y": y1, "width": x2-x1, "height": y2-y1},
                "dwellSeconds": round(dwell_seconds, 1),
                "confidence" : int(det.confidence * 100),
            },
        }

    def _persist(self, event: dict):
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def _in_schedule(self) -> bool:
        now = datetime.now()
        if now.weekday() not in self.valid_weekdays:
            return False
        ms_now = (now.hour * 3600 + now.minute * 60 + now.second) * 1000
        return self.valid_start_ms <= ms_now <= self.valid_end_ms

    @staticmethod
    def _parse_zones(area_position: str) -> list:
        try:
            return json.loads(area_position) if area_position else []
        except Exception as e:
            print(f"[Loitering] Failed to parse areaPosition: {e}")
            return []
```

---

### Step 2 — Register the class in `services/__init__.py`

```python
# Add your import
from .loitering import LoiteringTask          # <-- add this

TASK_REGISTRY = {
    "CROSS_LINE"           : CrossLineTask,
    "MASK_HAIRNET_CHEF_HAT": MaskHairnetChefHatTask,
    "LOITERING"            : LoiteringTask,   # <-- add this
}
```

---

### Step 3 — Register the `algorithmType` in `apis/tasks.py`

```python
class TaskRegistry:
    SUPPORTED = {
        "CROSS_LINE",
        "MASK_HAIRNET_CHEF_HAT",
        "LOITERING",             # <-- add this
    }
```

---

### Step 4 — Add any custom `detailConfig` fields to `DetailConfig` (optional)

If your task needs custom config fields (beyond the common `threshold`, `areaPosition`, `validWeekday`, etc.), add them to the `DetailConfig` model in `apis/tasks.py`:

```python
class DetailConfig(BaseModel):
    # CROSS_LINE
    enableAttrDetect: bool      = False
    enableReid      : bool      = False
    # MASK_HAIRNET_CHEF_HAT
    alarmType       : List[str] = []
    # CASHIER_BOX_OPEN
    drawerOpenLimit : int       = 30
    serviceWaitLimit: int       = 30
    enableStaffList : bool      = False
    staffIds        : List[int] = []
    # LOITERING
    dwellSeconds    : int       = 30    # <-- add this
```

---

## Checklist

- [ ] `services/your_task.py` — task class with `__init__(task_config)` and `__call__(payload) -> list`
- [ ] `services/__init__.py` — imported and added to `TASK_REGISTRY`
- [ ] `apis/tasks.py → TaskRegistry.SUPPORTED` — `algorithmType` string added
- [ ] `apis/tasks.py → DetailConfig` — any new fields added (if needed)

That's it. The task worker, FrameBus, and SSE stream pick it up automatically.

---

## Rules and conventions

| Rule | Why |
|---|---|
| `__call__` must always return a `list` (even if empty `[]`) | `task_worker.py` iterates the return value — `None` will crash the worker |
| Never block inside `__call__` | The task worker has no timeout — a hanging task stalls that queue permanently |
| Filter `track_id == -1` if you use tracking | BoT-SORT takes 1–2 frames to assign stable IDs; `-1` means "not yet tracked" |
| Persist evidence yourself | The task worker does not save anything — persistence is the task's responsibility |
| Use `put_nowait` semantics | The result queue drops silently if full — don't rely on every event making it to SSE; always write to JSONL too |
| Keep `__init__` fast | It runs inside the worker process after `fork()` — loading large models here is fine, but keep it focused |

---

## Using another service inside your task

You can call any existing service from inside your task class. Import it in `__init__` (not at module level) so it loads inside the worker process, not the main process:

```python
def __init__(self, task_config: dict):
    ...
    # Load age/gender only if needed
    if task_config.get("detailConfig", {}).get("enableAttrDetect", False):
        from services.age_gender import AgeGenderService
        self._age_gender = AgeGenderService()
```

Call it by building the standard context dict:

```python
context = {
    "data": {
        "frame"    : frame,
        "detection": {"items": [det], "count": 1},
        "use_case" : {},
    }
}
context = self._age_gender(context)
results = context["data"]["use_case"].get("age_gender", [])
```

Available services:

| Class | Import | `use_case` key | Output |
|---|---|---|---|
| `AgeGenderService` | `services.age_gender` | `age_gender` | List of results with `.gender`, `.age_group` |
| `PPEService` | `services.ppe` | `ppe` | List of results with `.items` (list of `{class_name, confidence}`) |
| `MoodService` | `services.mood` | `mood` | List of results with mood label |
