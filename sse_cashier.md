# Cashier SSE stream reference (backend handoff)

Server-Sent Events from the vision pipeline (all cameras multiplexed on one connection).

**Endpoint:** `http://<jetson-ip>:9000/detection/stream`

**Note:** Each JSON payload may also include a top-level `"frame"` key (base64 JPEG). It is omitted in the examples below for readability.

---

## Case summary (quick reference)

| case_id | severity | condition |
|---------|----------|-----------|
| N1 | NORMAL | Idle — nothing happening |
| N2 | NORMAL | Cashier only, no customer |
| N3 | NORMAL | Active transaction |
| N4 | NORMAL | Staff handover (two persons at register) |
| N5 | NORMAL | Customer in customer zone, no open drawer in cashier zone |
| N6 | NORMAL | Drawer open, no cash (card / float check) |
| A1 | ALERT or CRITICAL | Unattended open drawer |
| A2 | ALERT | Unexpected person in cashier zone |
| A3 | CRITICAL | Cash + open drawer, no cashier (unguarded register) |
| A4 | CRITICAL | Unauthorised person at open register with cash |
| A5 | ALERT | Customer waiting beyond threshold without cashier |
| A6 | ALERT | Drawer open beyond threshold |
| A7 | ALERT | Cash in customer zone, no cashier |

**Timers (A5 / A6):** `customer_wait_max_seconds` → **A5** (customer wait in customer zone); `drawer_open_max_seconds` → **A6** (drawer open too long). Full **thresholds → logic → output** table: [`docs/CASHIER_BOX_OPEN.md`](docs/CASHIER_BOX_OPEN.md) Part III, section **A5 / A6: thresholds → logic → output**. **Per-frame** open-drawer tally: `summary.cashier_zone.drawers`. **`drawer_count` API:** `≠` frames with drawer open; `=` count of logged **`triggered`** events — see *Drawer metrics* there (Part III).

---

## How to connect

Disable buffering so lines arrive as the server emits them:

```bash
curl -N "http://<jetson-ip>:9000/detection/stream"
```

**Message format:** each event is one line starting with `data: `, followed by a single JSON object, then a blank line:

```text
data: { ...JSON payload... }

```

---

## HTTP error payloads for media/routes

Cashier media/evidence HTTP endpoints return standardized error JSON:

```json
{
  "status": "error",
  "error": {
    "code": "404",
    "message": "Cashier evidence not found.",
    "detail": "cashier_cam_01"
  }
}
```

For debugging over SSH or terminal, include headers:

```bash
curl -si "http://<jetson-ip>:9000/cashier/media/cashier_cam_01/latest/gif"
curl -si "http://<jetson-ip>:9000/cashier/evidence/missing.jpg"
```

---

## Event-level media (GIF vs JPEG)

The multiplexed stream (`GET /detection/stream`) carries **per-frame** cashier results under `data.use_case.cashier.summary`. **JPEG evidence** and **GIF clips** are produced by different mechanisms in [`services/cashier.py`](services/cashier.py).

### Where each artifact appears

| Media | When | Where |
|--------|------|--------|
| **JPEG** | When a frame is persisted as annotated evidence (`frame_saved: true`) | `data.use_case.cashier.summary.evidence_path` (relative path string). Also served under the cashier **Media** HTTP routes (see below). |
| **GIF** | After an **alert/critical “event session” ends** — the active `case_id` changes, so buffered pre/post frames are compiled | Written asynchronously to the evidence tree; **`gif_path`** is attached in the internal event log (`events.jsonl` under the evidence dir), not in every per-frame `summary`. Fetch via **`GET /cashier/media/.../gif`** or listen on **`GET /cashier/stream/{camera_id}`** (event types include `gif_ready` per [`apis/cashier.py`](apis/cashier.py); your build may rely on HTTP polling if `gif_ready` is not emitted yet). |

**JPEG (per frame, when saved):**

- Emitted in the **same** JSON as other cashier fields when `summary.frame_saved` is true.
- Typical triggers: first **debounced** frame of an **ALERT** or **CRITICAL** case (starts buffering for a possible GIF), or **N3** with `transaction: true` (saves a still; does **not** start the GIF buffer — see below).

**GIF (after event ends):**

- Only when an alert/critical session was **started** (`_start_event` in code) and later **resolves** because the evaluated `case_id` **changes** (`_check_resolved` → `_compile_gif`). Requires `evidence.save_gif: true` in config and optional **`imageio`** for encoding.
- Buffer sizes / FPS per case are defined in `_GIF_BUDGET` in [`services/cashier.py`](services/cashier.py) (keys **A1–A7** only).

### Per-case: JPEG vs GIF

| case_id | JPEG (`evidence_path`) | GIF |
|---------|-------------------------|-----|
| **N1** | **No** in default logic (idle) | **No** — NORMAL; no alert session |
| **N2** | **No** — `severity` is NORMAL; alert saver does not run | **No** |
| **N3** | **Yes**, when `transaction` is true (dedicated branch saves a still) | **No** — saves JPEG + log but does **not** call `_start_event` |
| **N4** | **No** (same as N2) | **No** |
| **N5** | **No** (same as N2) | **No** |
| **N6** | **No** — informational alerts may appear, but severity stays **NORMAL** | **No** |
| **A1–A7** | **Yes**, on debounced alert/critical frames while the case is active (`frame_saved`) | **Yes**, when that alert case **ends** (transition to another case), subject to `save_gif` and successful compile |

**Summary:** **GIFs are only produced for alert/critical lifecycles (A1–A7)** with a defined `_GIF_BUDGET`. **NORMAL cases (N1–N6) do not** run the GIF compiler from this path. **N3** is special: may emit **JPEG** for transactions, **not** a session GIF.

### cURL — fetch evidence after the fact

Replace `<jetson-ip>` and `cashier_cam_01` with your host and camera id. Evidence lives under `CASHIER_EVIDENCE_DIR` (default `./evidence/cashier`).

**Latest JPEG / GIF for a camera (newest file on disk):**

```bash
curl -sO "http://<jetson-ip>:9000/cashier/media/cashier_cam_01/latest/jpg"
curl -sO "http://<jetson-ip>:9000/cashier/media/cashier_cam_01/latest/gif"
```

**Specific event** — `event_id` matches cashier meta, e.g. `A2_20260329_135421_001` (from `GET /cashier/events` or logged `event_id`):

```bash
curl -sO "http://<jetson-ip>:9000/cashier/media/cashier_cam_01/event/A2_20260329_135421_001/jpg"
curl -sO "http://<jetson-ip>:9000/cashier/media/cashier_cam_01/event/A2_20260329_135421_001/gif"
```

**Per-camera SSE** (includes `frame`, `alert`, and documented `gif_ready` event names):

```bash
curl -N "http://<jetson-ip>:9000/cashier/stream/cashier_cam_01"
```

**Alerts-only stream** (suppresses `frame`, keeps alerts / `gif_ready`):

```bash
curl -N "http://<jetson-ip>:9000/cashier/stream/cashier_cam_01/only"
```

More HTTP routes (`/cashier/events`, `/cashier/evidence`, zone setup) are in [`docs/CASHIER_BOX_OPEN.md`](docs/CASHIER_BOX_OPEN.md) Part III.

---

## Recorded batch output

### Local cashier-YOLO batch (run `20260329T135320_10106`)

Per-frame JSON is in [`outputs/cashier_test/20260329T135320_10106/stream.jsonl`](outputs/cashier_test/20260329T135320_10106/stream.jsonl) (73 frames, `multiple_persons` images). Case counts in that file: **N5** 16, **N3** 21, **A2** 13, **N6** 11, **N2** 10, **A4** 2. Last record: `901.jpg`, **A2**, `frame_count` 73.

**Batch-only top-level fields** (not present on every live `/detection/stream` event): `input` (path/name), `outputs.annotated_image`, and sometimes `camera_id` / `frame_count` when the writer mirrors the live envelope.

**Batch / merged-detector fields** under `data.detection`: when the pipeline uses the cashier YOLO head (`Person` / `Drawer_Open` / `Cash`), you may see `cashier_model_items` and `cashier_model_count` alongside person-only `items` / `count`. Live SSE from RTSP uses the same `CashierService` output under `data.use_case.cashier`; the extra `cashier_model_*` keys appear when the batch or gateway attaches the full cashier detector output for debugging.

**Tail this batch log over SSH**

```bash
ssh <user>@<jetson-ip> "tail -f /path/to/ml-server/outputs/cashier_test/20260329T135320_10106/stream.jsonl"
```

### First 70 frames batch (updated ROIs, run `outputs/test_70`)

Batch executed on the first 70 images from `/mnt/01DA3A868F4FC7D0/frames_with_people/multiple_persons` with updated polygon ROIs in `config/cashier_zones.yaml`.

- Artifacts: `outputs/test_70/*.jpg` (annotated frames) and `outputs/test_70/*.json` (per-frame payloads)
- Count: **70 JPEG + 70 JSON**
- First frame: `1000.jpg` / `1000.json`
- Last frame: `1146.jpg` / `1146.json`

Updated normalized polygons used in this run:

- `ROI_CUSTOMER`: `[[0.32031, 0.28472], [0.64297, 0.25556], [0.63516, 0.00000], [0.31172, 0.00000]]`
- `ROI_CASHIER`: `[[0.32969, 0.54861], [0.65000, 0.52778], [0.69297, 0.99444], [0.33516, 0.98750]]`

### Full-dataset validation run (`20260329T060741_7464`)

Aggregated metrics: [`outputs/cashier_test/20260329T060741_7464/summary.json`](outputs/cashier_test/20260329T060741_7464/summary.json).  
Per-frame JSON: [`stream.jsonl`](outputs/cashier_test/20260329T060741_7464/stream.jsonl).

```bash
ssh <user>@<jetson-ip> "tail -f /path/to/ml-server/outputs/cashier_test/20260329T060741_7464/stream.jsonl"
```

**Application log (if your deployment writes here)**

```bash
ssh <user>@<jetson-ip> "tail -f /path/to/ml-server/logger/app.log"
```

### Real record: **N5** (first line, `20260329T135320_10106/stream.jsonl`)

`566.jpg` — multiple people; customer + cashier zones occupied; drawer closed; NORMAL.  
`detection.items` are persons only; `cashier.persons` omits `age_group` / `gender` / `mood` (join from `use_case.age_gender` / `mood` by `bbox`).

```json
{
  "input": {"path": ".../multiple_persons/566.jpg", "name": "566.jpg"},
  "outputs": {"annotated_image": "outputs/cashier_test/20260329T135320_10106/annotated/566.jpg"},
  "camera_id": "cashier_cam_01",
  "frame_count": 1,
  "timestamp": "2026-03-29T13:53:25.866347+00:00",
  "data": {
    "detection": {
      "count": 6,
      "items": [{"class_id": 0, "class_name": "Person", "bbox": [1378, 1093, 2208, 1797]}]
    },
    "use_case": {
      "cashier": {
        "persons": [
          {
            "person_bbox": [1378, 1093, 2208, 1797],
            "confidence": 0.8965,
            "zone": "ROI_CASHIER",
            "transaction": false,
            "items": {"drawers": [], "cash": []}
          }
        ],
        "summary": {
          "case_id": "N5",
          "severity": "NORMAL",
          "alerts": [],
          "cashier_zone": {"persons": 2, "drawers": 0, "cash": 0},
          "customer_zone": {"persons": 2, "drawers": 0, "cash": 0},
          "frame_id": 1,
          "cashier_persons": [
            {"bbox": [1378, 1093, 2208, 1797], "confidence": 0.8965, "zone": "ROI_CASHIER"}
          ]
        }
      }
    }
  }
}
```

*(Full line includes all six `detection.items`, parallel `age_gender` / `mood` arrays, and a second cashier-zone person.)*

### Real record: **A2** with drawer + cash (`896.jpg`, same batch)

Drawer and cash are linked to the cashier-zone person; `items.drawers` / `items.cash` use **`"confidence": null`** by design in [`CashierService`](services/cashier.py) (model scores are in `detection.cashier_model_items` when present).

```json
{
  "cashier": {
    "persons": [
      {
        "person_bbox": [1252, 1454, 2271, 1800],
        "zone": "ROI_CASHIER",
        "transaction": false,
        "items": {
          "drawers": [{"bbox": [1413, 1152, 1973, 1521], "confidence": null}],
          "cash": [
            {"bbox": [1430, 1182, 1545, 1360], "confidence": null},
            {"bbox": [1622, 1167, 1736, 1341], "confidence": null}
          ]
        }
      }
    ],
    "summary": {
      "case_id": "A2",
      "severity": "ALERT",
      "alerts": ["A2 ALERT: Unexpected person in cashier zone"],
      "cashier_zone": {"persons": 1, "drawers": 1, "cash": 5}
    }
  },
  "detection": {
    "cashier_model_items": [
      {"class_id": 1, "class_name": "Drawer_Open", "bbox": [1413, 1152, 1973, 1521], "confidence": 0.877},
      {"class_id": 2, "class_name": "Cash", "bbox": [1430, 1182, 1545, 1360], "confidence": 0.8841}
    ],
    "cashier_model_count": 9
  }
}
```

### Real record: **N6** (`900.jpg`, same batch)

Open drawer, no cash linked under the business rules; NORMAL with informational alert.

```json
{
  "cashier": {
    "persons": [
      {
        "person_bbox": [1349, 1229, 2320, 1800],
        "zone": "ROI_CASHIER",
        "items": {
          "drawers": [{"bbox": [1417, 1155, 1958, 1288], "confidence": null}],
          "cash": []
        }
      }
    ],
    "summary": {
      "case_id": "N6",
      "severity": "NORMAL",
      "alerts": ["N6 EVENT: Drawer open, no cash (card / float check)"],
      "cashier_zone": {"persons": 1, "drawers": 1, "cash": 0}
    }
  }
}
```

### Last frame in batch (`901.jpg`)

Same run, **A2**, `frame_count` 73 — end of the recorded pipeline for this `stream.jsonl`.

### Archive: **N2** / **A5** / **A2** (`20260329T060741_7464`)

Older full-dataset excerpts (720p cashier framing) remain illustrative for **N2**, **A5**, and multi-person **A2** without drawer/cash; see [`stream.jsonl`](outputs/cashier_test/20260329T060741_7464/stream.jsonl) and [`summary.json`](outputs/cashier_test/20260329T060741_7464/summary.json).

---

## Enrichment and matching rules

> Each `detection.items[]` element uses the detector schema: `bbox`, `class_id`, `class_name`, `confidence`, `center`, `width`, `height` (see [`Detection.to_dict()`](services/detector.py)). Class names may be lower (`person`) or title case (`Person`) depending on the weights file.
>
> When present, `detection.cashier_model_items[]` lists **Person**, **Drawer_Open**, and **Cash** from the cashier YOLO head (batch runs and merged pipelines). Join drawer/cash **scores** from there to `cashier.persons[].items` by matching integer `bbox` if you need confidences; [`CashierService`](services/cashier.py) sets **`items.*[].confidence` to `null`** and encodes association via IoU (`thresholds.proximity_iou`).
>
> `age_gender` and `mood` are parallel arrays keyed by person (and sometimes other) boxes. Match to `cashier.persons[].person_bbox` by **identical `bbox` / `person_bbox` coordinates** (same pixel box).
>
> The **synthetic** SSE examples further below show `age_group` / `gender` / `mood` duplicated on each `cashier.persons[]` entry for readability. **Production payloads** (see `20260329T135320_10106/stream.jsonl`) emit persons **without** those fields on `cashier.persons[]`; merge from `use_case.age_gender` / `use_case.mood` on the client if you want one object per person.

---

## NORMAL cases

### N1 — Idle

**Severity:** NORMAL  
**Condition:** No persons, drawers, or cash in ROIs; baseline state.

```sse
data: {
  "camera_id": "cashier_cam_01",
  "frame_count": 1204,
  "timestamp": "2026-03-26T08:12:03.441120+00:00",
  "data": {
    "detection": {"count": 0, "items": []},
    "use_case": {
      "age_gender": [],
      "mood": [],
      "cashier": {
        "persons": [],
        "summary": {
          "cashier_zone": {"persons": 0, "drawers": 0, "cash": 0},
          "customer_zone": {"persons": 0, "drawers": 0, "cash": 0},
          "case_id": "N1",
          "severity": "NORMAL",
          "alerts": [],
          "transaction": false,
          "frame_saved": false,
          "evidence_path": null,
          "frame_id": 1204,
          "timestamp": "2026-03-26T08:12:03.115900+00:00",
          "cashier_persons": []
        }
      }
    }
  }
}
```

### N2 — Cashier only

**Severity:** NORMAL  
**Condition:** One staff member in cashier zone; drawer closed; no customer in customer zone.

```sse
data: {
  "camera_id": "cashier_cam_01",
  "frame_count": 1450,
  "timestamp": "2026-03-26T08:12:18.902331+00:00",
  "data": {
    "detection": {
      "count": 1,
      "items": [
        {
          "bbox": [540, 165, 790, 598],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.88,
          "center": [665, 381],
          "width": 250,
          "height": 433
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {"bbox": [540, 165, 790, 598], "gender": "Female", "age_group": "MiddleAged", "confidence": 0.72}
      ],
      "mood": [
        {"bbox": [540, 165, 790, 598], "mood": "Neutral", "confidence": 0.81}
      ],
      "cashier": {
        "persons": [
          {
            "person_bbox": [540, 165, 790, 598],
            "confidence": 0.88,
            "zone": "ROI_CASHIER",
            "transaction": false,
            "age_group": "MiddleAged",
            "gender": "Female",
            "mood": "Neutral",
            "items": {"drawers": [], "cash": []}
          }
        ],
        "summary": {
          "cashier_zone": {"persons": 1, "drawers": 0, "cash": 0},
          "customer_zone": {"persons": 0, "drawers": 0, "cash": 0},
          "case_id": "N2",
          "severity": "NORMAL",
          "alerts": [],
          "transaction": false,
          "frame_saved": false,
          "evidence_path": null,
          "frame_id": 1450,
          "timestamp": "2026-03-26T08:12:18.551002+00:00",
          "cashier_persons": [
            {"bbox": [540, 165, 790, 598], "confidence": 0.88, "zone": "ROI_CASHIER"}
          ]
        }
      }
    }
  }
}
```

### N3 — Active transaction

**Severity:** NORMAL  
**Condition:** Cashier + customer + open drawer + cash/near-person rules satisfied.

```sse
data: {
  "camera_id": "cashier_cam_01",
  "frame_count": 1882,
  "timestamp": "2026-03-26T08:13:01.224418+00:00",
  "data": {
    "detection": {
      "count": 2,
      "items": [
        {
          "bbox": [502, 158, 768, 612],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.91,
          "center": [635, 385],
          "width": 266,
          "height": 454
        },
        {
          "bbox": [892, 172, 1148, 638],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.86,
          "center": [1020, 405],
          "width": 256,
          "height": 466
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {"bbox": [502, 158, 768, 612], "gender": "Male", "age_group": "Senior", "confidence": 0.68},
        {"bbox": [892, 172, 1148, 638], "gender": "Female", "age_group": "Young", "confidence": 0.77}
      ],
      "mood": [
        {"bbox": [502, 158, 768, 612], "mood": "Happy", "confidence": 0.89},
        {"bbox": [892, 172, 1148, 638], "mood": "Neutral", "confidence": 0.76}
      ],
      "cashier": {
        "persons": [
          {
            "person_bbox": [502, 158, 768, 612],
            "confidence": 0.91,
            "zone": "ROI_CASHIER",
            "transaction": true,
            "age_group": "Senior",
            "gender": "Male",
            "mood": "Happy",
            "items": {
              "drawers": [{"bbox": [588, 455, 712, 582], "confidence": 0.84}],
              "cash": [{"bbox": [618, 498, 682, 552], "confidence": 0.79}]
            }
          },
          {
            "person_bbox": [892, 172, 1148, 638],
            "confidence": 0.86,
            "zone": "ROI_CUSTOMER",
            "transaction": false,
            "age_group": "Young",
            "gender": "Female",
            "mood": "Neutral",
            "items": {"drawers": [], "cash": []}
          }
        ],
        "summary": {
          "cashier_zone": {"persons": 1, "drawers": 1, "cash": 3},
          "customer_zone": {"persons": 1, "drawers": 0, "cash": 0},
          "case_id": "N3",
          "severity": "NORMAL",
          "alerts": ["N3 EVENT: Transaction in progress"],
          "transaction": true,
          "frame_saved": true,
          "evidence_path": "outputs/evidence/normal/N3/cashier_cam_01_20260326_081300.jpg",
          "frame_id": 1882,
          "timestamp": "2026-03-26T08:12:59.881200+00:00",
          "cashier_persons": [
            {"bbox": [502, 158, 768, 612], "confidence": 0.91, "zone": "ROI_CASHIER"}
          ]
        }
      }
    }
  }
}
```

### N4 — Staff handover

**Severity:** NORMAL  
**Condition:** Two staff in cashier zone with drawer context (supervisor / handover).

```sse
data: {
  "camera_id": "cashier_cam_01",
  "frame_count": 2105,
  "timestamp": "2026-03-26T08:13:22.778009+00:00",
  "data": {
    "detection": {
      "count": 2,
      "items": [
        {
          "bbox": [480, 150, 735, 605],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.89,
          "center": [607, 377],
          "width": 255,
          "height": 455
        },
        {
          "bbox": [560, 175, 805, 598],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.85,
          "center": [682, 386],
          "width": 245,
          "height": 423
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {"bbox": [480, 150, 735, 605], "gender": "Male", "age_group": "Elderly", "confidence": 0.61},
        {"bbox": [560, 175, 805, 598], "gender": "Female", "age_group": "MiddleAged", "confidence": 0.74}
      ],
      "mood": [
        {"bbox": [480, 150, 735, 605], "mood": "Neutral", "confidence": 0.82},
        {"bbox": [560, 175, 805, 598], "mood": "Neutral", "confidence": 0.71}
      ],
      "cashier": {
        "persons": [
          {
            "person_bbox": [480, 150, 735, 605],
            "confidence": 0.89,
            "zone": "ROI_CASHIER",
            "transaction": false,
            "age_group": "Elderly",
            "gender": "Male",
            "mood": "Neutral",
            "items": {
              "drawers": [{"bbox": [575, 448, 698, 575], "confidence": 0.81}],
              "cash": []
            }
          },
          {
            "person_bbox": [560, 175, 805, 598],
            "confidence": 0.85,
            "zone": "ROI_CASHIER",
            "transaction": false,
            "age_group": "MiddleAged",
            "gender": "Female",
            "mood": "Neutral",
            "items": {"drawers": [], "cash": []}
          }
        ],
        "summary": {
          "cashier_zone": {"persons": 2, "drawers": 1, "cash": 0},
          "customer_zone": {"persons": 0, "drawers": 0, "cash": 0},
          "case_id": "N4",
          "severity": "NORMAL",
          "alerts": ["N4 EVENT: Staff handover / supervisor at register"],
          "transaction": false,
          "frame_saved": false,
          "evidence_path": null,
          "frame_id": 2105,
          "timestamp": "2026-03-26T08:13:22.401118+00:00",
          "cashier_persons": [
            {"bbox": [480, 150, 735, 605], "confidence": 0.89, "zone": "ROI_CASHIER"},
            {"bbox": [560, 175, 805, 598], "confidence": 0.85, "zone": "ROI_CASHIER"}
          ]
        }
      }
    }
  }
}
```

### N5 — Customer present, drawer closed

**Severity:** NORMAL  
**Condition:** Customer in customer zone; no open drawer in cashier zone.

```sse
data: {
  "camera_id": "cashier_cam_01",
  "frame_count": 2340,
  "timestamp": "2026-03-26T08:13:55.112883+00:00",
  "data": {
    "detection": {
      "count": 1,
      "items": [
        {
          "bbox": [905, 188, 1165, 655],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.83,
          "center": [1035, 421],
          "width": 260,
          "height": 467
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {"bbox": [905, 188, 1165, 655], "gender": "Male", "age_group": "Young", "confidence": 0.66}
      ],
      "mood": [
        {"bbox": [905, 188, 1165, 655], "mood": "Happy", "confidence": 0.77}
      ],
      "cashier": {
        "persons": [
          {
            "person_bbox": [905, 188, 1165, 655],
            "confidence": 0.83,
            "zone": "ROI_CUSTOMER",
            "transaction": false,
            "age_group": "Young",
            "gender": "Male",
            "mood": "Happy",
            "items": {"drawers": [], "cash": []}
          }
        ],
        "summary": {
          "cashier_zone": {"persons": 0, "drawers": 0, "cash": 0},
          "customer_zone": {"persons": 1, "drawers": 0, "cash": 0},
          "case_id": "N5",
          "severity": "NORMAL",
          "alerts": [],
          "transaction": false,
          "frame_saved": false,
          "evidence_path": null,
          "frame_id": 2340,
          "timestamp": "2026-03-26T08:13:54.730045+00:00",
          "cashier_persons": []
        }
      }
    }
  }
}
```

### N6 — Drawer open, no cash

**Severity:** NORMAL  
**Condition:** Drawer open with staff present; no cash detected (card / float).

```sse
data: {
  "camera_id": "cashier_cam_01",
  "frame_count": 2511,
  "timestamp": "2026-03-26T08:14:12.556701+00:00",
  "data": {
    "detection": {
      "count": 1,
      "items": [
        {
          "bbox": [518, 160, 778, 608],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.9,
          "center": [648, 384],
          "width": 260,
          "height": 448
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {"bbox": [518, 160, 778, 608], "gender": "Female", "age_group": "Senior", "confidence": 0.7}
      ],
      "mood": [
        {"bbox": [518, 160, 778, 608], "mood": "Neutral", "confidence": 0.84}
      ],
      "cashier": {
        "persons": [
          {
            "person_bbox": [518, 160, 778, 608],
            "confidence": 0.9,
            "zone": "ROI_CASHIER",
            "transaction": false,
            "age_group": "Senior",
            "gender": "Female",
            "mood": "Neutral",
            "items": {
              "drawers": [{"bbox": [595, 462, 718, 588], "confidence": 0.78}],
              "cash": []
            }
          }
        ],
        "summary": {
          "cashier_zone": {"persons": 1, "drawers": 1, "cash": 0},
          "customer_zone": {"persons": 0, "drawers": 0, "cash": 0},
          "case_id": "N6",
          "severity": "NORMAL",
          "alerts": ["N6 EVENT: Drawer open, no cash (card / float check)"],
          "transaction": false,
          "frame_saved": false,
          "evidence_path": null,
          "frame_id": 2511,
          "timestamp": "2026-03-26T08:14:12.198334+00:00",
          "cashier_persons": [
            {"bbox": [518, 160, 778, 608], "confidence": 0.9, "zone": "ROI_CASHIER"}
          ]
        }
      }
    }
  }
}
```

---

## ALERT and CRITICAL cases

### A1 — Unattended open drawer (CRITICAL, customer present)

**Severity:** CRITICAL  
**Condition:** Drawer open in cashier zone; no staff in cashier zone; customer present (escalated variant).

```sse
data: {
  "camera_id": "cashier_cam_01",
  "frame_count": 3022,
  "timestamp": "2026-03-26T08:15:40.009221+00:00",
  "data": {
    "detection": {
      "count": 1,
      "items": [
        {
          "bbox": [918, 195, 1175, 648],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.81,
          "center": [1046, 421],
          "width": 257,
          "height": 453
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {"bbox": [918, 195, 1175, 648], "gender": "Female", "age_group": "MiddleAged", "confidence": 0.58}
      ],
      "mood": [
        {"bbox": [918, 195, 1175, 648], "mood": "Angry", "confidence": 0.62}
      ],
      "cashier": {
        "persons": [
          {
            "person_bbox": [918, 195, 1175, 648],
            "confidence": 0.81,
            "zone": "ROI_CUSTOMER",
            "transaction": false,
            "age_group": "MiddleAged",
            "gender": "Female",
            "mood": "Angry",
            "items": {"drawers": [], "cash": []}
          }
        ],
        "summary": {
          "cashier_zone": {"persons": 0, "drawers": 1, "cash": 0},
          "customer_zone": {"persons": 1, "drawers": 0, "cash": 0},
          "case_id": "A1",
          "severity": "CRITICAL",
          "alerts": ["A1 CRITICAL: Unattended open drawer — customer present"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/evidence/critical/A1/cashier_cam_01_20260326_081538.jpg",
          "frame_id": 3022,
          "timestamp": "2026-03-26T08:15:39.640512+00:00",
          "cashier_persons": []
        }
      }
    }
  }
}
```

### A2 — Unexpected person in cashier zone

**Severity:** ALERT  
**Condition:** Cashier zone occupied with drawer/cash context that does not match a normal transaction pattern (fallback rule).

```sse
data: {
  "camera_id": "cashier_cam_01",
  "frame_count": 3188,
  "timestamp": "2026-03-26T08:16:05.331884+00:00",
  "data": {
    "detection": {
      "count": 1,
      "items": [
        {
          "bbox": [120, 200, 380, 628],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.77,
          "center": [250, 414],
          "width": 260,
          "height": 428
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {"bbox": [120, 200, 380, 628], "gender": "Male", "age_group": "Young", "confidence": 0.55}
      ],
      "mood": [
        {"bbox": [120, 200, 380, 628], "mood": "Angry", "confidence": 0.58}
      ],
      "cashier": {
        "persons": [
          {
            "person_bbox": [120, 200, 380, 628],
            "confidence": 0.77,
            "zone": "ROI_CASHIER",
            "transaction": false,
            "age_group": "Young",
            "gender": "Male",
            "mood": "Angry",
            "items": {
              "drawers": [{"bbox": [602, 460, 725, 585], "confidence": 0.73}],
              "cash": [{"bbox": [640, 505, 695, 548], "confidence": 0.76}]
            }
          }
        ],
        "summary": {
          "cashier_zone": {"persons": 1, "drawers": 1, "cash": 2},
          "customer_zone": {"persons": 0, "drawers": 0, "cash": 0},
          "case_id": "A2",
          "severity": "ALERT",
          "alerts": ["A2 ALERT: Unexpected person in cashier zone"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/evidence/alert/A2/cashier_cam_01_20260326_081604.jpg",
          "frame_id": 3188,
          "timestamp": "2026-03-26T08:16:04.982110+00:00",
          "cashier_persons": [
            {"bbox": [120, 200, 380, 628], "confidence": 0.77, "zone": "ROI_CASHIER"}
          ]
        }
      }
    }
  }
}
```

### A3 — Unguarded register (CRITICAL)

**Severity:** CRITICAL  
**Condition:** Open drawer and cash visible; **no person** in cashier zone (`cashier.persons` empty; `detection.count` 0 after person filter).

```sse
data: {
  "camera_id": "cashier_cam_01",
  "frame_count": 3401,
  "timestamp": "2026-03-26T08:16:33.884102+00:00",
  "data": {
    "detection": {"count": 0, "items": []},
    "use_case": {
      "age_gender": [],
      "mood": [],
      "cashier": {
        "persons": [],
        "summary": {
          "cashier_zone": {"persons": 0, "drawers": 1, "cash": 2},
          "customer_zone": {"persons": 0, "drawers": 0, "cash": 0},
          "case_id": "A3",
          "severity": "CRITICAL",
          "alerts": ["A3 CRITICAL: Cash + open drawer — register unguarded"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/evidence/critical/A3/cashier_cam_01_20260326_081632.jpg",
          "frame_id": 3401,
          "timestamp": "2026-03-26T08:16:33.501887+00:00",
          "cashier_persons": []
        }
      }
    }
  }
}
```

### A4 — Unauthorised at register with cash

**Severity:** CRITICAL  
**Condition:** Person in cashier zone not sufficiently overlapping the open drawer while cash is present.

```sse
data: {
  "camera_id": "cashier_cam_01",
  "frame_count": 3588,
  "timestamp": "2026-03-26T08:17:01.772009+00:00",
  "data": {
    "detection": {
      "count": 1,
      "items": [
        {
          "bbox": [95, 185, 340, 598],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.84,
          "center": [217, 391],
          "width": 245,
          "height": 413
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {"bbox": [95, 185, 340, 598], "gender": "Male", "age_group": "Elderly", "confidence": 0.52}
      ],
      "mood": [
        {"bbox": [95, 185, 340, 598], "mood": "Angry", "confidence": 0.66}
      ],
      "cashier": {
        "persons": [
          {
            "person_bbox": [95, 185, 340, 598],
            "confidence": 0.84,
            "zone": "ROI_CASHIER",
            "transaction": false,
            "age_group": "Elderly",
            "gender": "Male",
            "mood": "Angry",
            "items": {
              "drawers": [{"bbox": [610, 468, 732, 592], "confidence": 0.8}],
              "cash": [{"bbox": [648, 512, 702, 558], "confidence": 0.74}]
            }
          }
        ],
        "summary": {
          "cashier_zone": {"persons": 1, "drawers": 1, "cash": 2},
          "customer_zone": {"persons": 0, "drawers": 0, "cash": 0},
          "case_id": "A4",
          "severity": "CRITICAL",
          "alerts": ["A4 CRITICAL: Unauthorised person at open register with cash"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/evidence/critical/A4/cashier_cam_01_20260326_081700.jpg",
          "frame_id": 3588,
          "timestamp": "2026-03-26T08:17:01.330221+00:00",
          "cashier_persons": [
            {"bbox": [95, 185, 340, 598], "confidence": 0.84, "zone": "ROI_CASHIER"}
          ]
        }
      }
    }
  }
}
```

### A5 — Customer waiting (timer)

**Severity:** ALERT  
**Condition:** Customer in customer zone; no cashier; wait duration exceeds configured threshold (default 30s per [`config/cashier_zones.yaml`](config/cashier_zones.yaml) `customer_wait_max_seconds`).

```sse
data: {
  "camera_id": "cashier_cam_01",
  "frame_count": 3844,
  "timestamp": "2026-03-26T08:17:42.551120+00:00",
  "data": {
    "detection": {
      "count": 1,
      "items": [
        {
          "bbox": [900, 190, 1158, 642],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.87,
          "center": [1029, 416],
          "width": 258,
          "height": 452
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {"bbox": [900, 190, 1158, 642], "gender": "Female", "age_group": "Senior", "confidence": 0.64}
      ],
      "mood": [
        {"bbox": [900, 190, 1158, 642], "mood": "Angry", "confidence": 0.71}
      ],
      "cashier": {
        "persons": [
          {
            "person_bbox": [900, 190, 1158, 642],
            "confidence": 0.87,
            "zone": "ROI_CUSTOMER",
            "transaction": false,
            "age_group": "Senior",
            "gender": "Female",
            "mood": "Angry",
            "items": {"drawers": [], "cash": []}
          }
        ],
        "summary": {
          "cashier_zone": {"persons": 0, "drawers": 0, "cash": 0},
          "customer_zone": {"persons": 1, "drawers": 0, "cash": 0},
          "case_id": "A5",
          "severity": "ALERT",
          "alerts": ["A5 ALERT: Customer waiting 34s — no cashier present"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/evidence/alert/A5/cashier_cam_01_20260326_081741.jpg",
          "frame_id": 3844,
          "timestamp": "2026-03-26T08:17:42.180334+00:00",
          "cashier_persons": []
        }
      }
    }
  }
}
```

### A6 — Drawer open too long

**Severity:** ALERT  
**Condition:** Drawer open longer than `drawer_open_max_seconds` (default 30s).

```sse
data: {
  "camera_id": "cashier_cam_01",
  "frame_count": 4012,
  "timestamp": "2026-03-26T08:18:10.993441+00:00",
  "data": {
    "detection": {
      "count": 1,
      "items": [
        {
          "bbox": [530, 168, 785, 602],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.9,
          "center": [657, 385],
          "width": 255,
          "height": 434
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {"bbox": [530, 168, 785, 602], "gender": "Male", "age_group": "MiddleAged", "confidence": 0.69}
      ],
      "mood": [
        {"bbox": [530, 168, 785, 602], "mood": "Neutral", "confidence": 0.79}
      ],
      "cashier": {
        "persons": [
          {
            "person_bbox": [530, 168, 785, 602],
            "confidence": 0.9,
            "zone": "ROI_CASHIER",
            "transaction": false,
            "age_group": "MiddleAged",
            "gender": "Male",
            "mood": "Neutral",
            "items": {
              "drawers": [{"bbox": [600, 455, 722, 580], "confidence": 0.77}],
              "cash": []
            }
          }
        ],
        "summary": {
          "cashier_zone": {"persons": 1, "drawers": 1, "cash": 0},
          "customer_zone": {"persons": 0, "drawers": 0, "cash": 0},
          "case_id": "A6",
          "severity": "ALERT",
          "alerts": ["A6 ALERT: Drawer open 35s (limit 30s)"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/evidence/alert/A6/cashier_cam_01_20260326_081809.jpg",
          "frame_id": 4012,
          "timestamp": "2026-03-26T08:18:10.612008+00:00",
          "cashier_persons": [
            {"bbox": [530, 168, 785, 602], "confidence": 0.9, "zone": "ROI_CASHIER"}
          ]
        }
      }
    }
  }
}
```

### A7 — Cash in customer zone

**Severity:** ALERT  
**Condition:** Cash detected in customer zone while cashier zone has no staff and no open drawer.

```sse
data: {
  "camera_id": "cashier_cam_01",
  "frame_count": 4190,
  "timestamp": "2026-03-26T08:18:38.220118+00:00",
  "data": {
    "detection": {
      "count": 1,
      "items": [
        {
          "bbox": [940, 205, 1188, 668],
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.79,
          "center": [1064, 436],
          "width": 248,
          "height": 463
        }
      ]
    },
    "use_case": {
      "age_gender": [
        {"bbox": [940, 205, 1188, 668], "gender": "Male", "age_group": "Young", "confidence": 0.6}
      ],
      "mood": [
        {"bbox": [940, 205, 1188, 668], "mood": "Angry", "confidence": 0.55}
      ],
      "cashier": {
        "persons": [
          {
            "person_bbox": [940, 205, 1188, 668],
            "confidence": 0.79,
            "zone": "ROI_CUSTOMER",
            "transaction": false,
            "age_group": "Young",
            "gender": "Male",
            "mood": "Angry",
            "items": {
              "drawers": [],
              "cash": [{"bbox": [1020, 520, 1088, 572], "confidence": 0.72}]
            }
          }
        ],
        "summary": {
          "cashier_zone": {"persons": 0, "drawers": 0, "cash": 0},
          "customer_zone": {"persons": 1, "drawers": 0, "cash": 1},
          "case_id": "A7",
          "severity": "ALERT",
          "alerts": ["A7 ALERT: Cash in customer zone — no cashier present"],
          "transaction": false,
          "frame_saved": true,
          "evidence_path": "outputs/evidence/alert/A7/cashier_cam_01_20260326_081837.jpg",
          "frame_id": 4190,
          "timestamp": "2026-03-26T08:18:37.889334+00:00",
          "cashier_persons": []
        }
      }
    }
  }
}
```

---

## Stream processing (Python)

Install a client library (for example [`sseclient-py`](https://pypi.org/project/sseclient-py/)):

```bash
pip install sseclient-py requests
```

Example consumer (merge `age_group` / `gender` / `mood` onto each person if you want the same shape as the samples above):

```python
import json
import requests
from sseclient import SSEClient


def stream_cashier(host: str) -> None:
    url = f"http://{host}:9000/detection/stream"
    response = requests.get(url, stream=True, timeout=None)
    client = SSEClient(response)
    for event in client.events():
        if not event.data:
            continue
        payload = json.loads(event.data)
        uc = payload.get("data", {}).get("use_case", {})
        cashier = uc.get("cashier") or {}
        summary = cashier.get("summary") or {}
        case_id = summary.get("case_id")
        severity = summary.get("severity")
        alerts = summary.get("alerts") or []
        persons = cashier.get("persons") or []

        ag_by_bbox = {tuple(e["bbox"]): e for e in uc.get("age_gender") or []}
        mood_by_bbox = {tuple(e["bbox"]): e for e in uc.get("mood") or []}

        for p in persons:
            bb = tuple(p.get("person_bbox") or [])
            ag = ag_by_bbox.get(bb, {})
            md = mood_by_bbox.get(bb, {})
            print(
                f"[{case_id}] {severity} | zone={p.get('zone')} | "
                f"age={ag.get('age_group') or p.get('age_group')} | "
                f"gender={ag.get('gender') or p.get('gender')} | "
                f"mood={md.get('mood') or p.get('mood')} | "
                f"transaction={p.get('transaction')}"
            )
        if severity in ("ALERT", "CRITICAL"):
            print("ALERT:", alerts)


if __name__ == "__main__":
    stream_cashier("<jetson-ip>")
```

---

## SSH log tail (operations)

**Batch JSONL (same payloads as SSE bodies, without `data:`)**

```bash
ssh <user>@<jetson-ip> "tail -f /path/to/ml-server/outputs/cashier_test/20260329T135320_10106/stream.jsonl"
```

```bash
ssh <user>@<jetson-ip> "tail -f /path/to/ml-server/outputs/cashier_test/20260329T060741_7464/stream.jsonl"
```

**Application log**

```bash
ssh <user>@<jetson-ip> "tail -f /path/to/ml-server/logger/app.log"
```

---

## Implementation check (code vs this batch)

| Observation in `20260329T135320_10106/stream.jsonl` | Source |
|------------------------------------------------------|--------|
| `cashier.persons[].items.drawers/cash[].confidence` is `null` | [`CashierService._build_person_entries`](services/cashier.py) — only `bbox` is stored on linked items; model score is not copied. |
| `detection.cashier_model_items` / `cashier_model_count` | Written by [`scripts/test_cashier_batch.py`](scripts/test_cashier_batch.py) when the pre-cashier detector output includes drawer/cash classes; use these arrays if you need raw confidences. |
| No `age_group` / `gender` on `cashier.persons[]` | Matches current `PersonEntry` serialization in [`services/cashier.py`](services/cashier.py); merge from `use_case.age_gender` / `mood`. |
| A2 alert text: `Unexpected person in cashier zone` | Emitted by case logic in [`services/cashier.py`](services/cashier.py) (wording may differ from older archived samples that mentioned “multiple persons” only). |

---

## Related HTTP API

Zone geometry, thresholds, and extra cashier REST routes (`/cashier/status`, `/cashier/events`, per-camera SSE) are described in OpenAPI at `/docs` and in [`docs/CASHIER_BOX_OPEN.md`](docs/CASHIER_BOX_OPEN.md) Part III (including **Cashier ROI shapes, thresholds, and end of pipeline**). **JPEG vs GIF** and media `curl` examples: [Event-level media (GIF vs JPEG)](#event-level-media-gif-vs-jpeg) above.
