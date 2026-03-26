# Vision Pipeline API - Cashier Drawer Integration Reference

Version 1.0 - March 26, 2026

## Overview
This service runs inside the Vision Pipeline API and monitors cashier drawer events using ROI zones:
`ROI_CASHIER` (staff/register side) and `ROI_CUSTOMER` (customer side).

The cashier logic consumes the detector output (persons/drawers/cash) and produces:
`case_id` (N1-N6, A1-A7), `severity` (NORMAL/ALERT/CRITICAL), `alerts`, plus optional evidence frames/gifs.

## Base URL
Use the same base as the Vision Pipeline API:
`http://<jetson-ip>:9000`

## Cashier API Endpoints

### `GET /cashier/status`
Returns the latest cashier state per active camera.

```bash
curl -s "http://<jetson-ip>:9000/cashier/status" | jq .
```

### `GET /cashier/events`
Paginated event log (in-memory) with optional filters.

Query params:
- `severity`: `NORMAL | ALERT | CRITICAL`
- `case_id`: `N1-N6` or `A1-A7`
- `camera_id`: camera label used by the pipeline
- `limit`: default `100` (min 1, max 1000)
- `offset`: default `0`

```bash
curl -s "http://<jetson-ip>:9000/cashier/events?severity=ALERT&case_id=A6&camera_id=cam&limit=50&offset=0" | jq .
```

### `DELETE /cashier/events`
Clears the in-memory event log.

```bash
curl -s -X DELETE "http://<jetson-ip>:9000/cashier/events" | jq .
```

### `GET /cashier/evidence`
Lists saved evidence images (annotated JPEGs) under `CASHIER_EVIDENCE_DIR`.

Optional query params:
- `severity`: `NORMAL | ALERT | CRITICAL`
- `case_id`: `N3 / A1 / A3 ...`
- `limit`: default `50`

```bash
curl -s "http://<jetson-ip>:9000/cashier/evidence?case_id=N3&limit=20" | jq .
```

### `GET /cashier/evidence/{file_path}`
Downloads one evidence JPEG.

Important: `file_path` is relative to `CASHIER_EVIDENCE_DIR` (default: `./evidence/cashier`).

Example (download the evidence file created for `frame_096750.jpg` in this run):

```bash
curl -s -o N3_evidence.jpg \
  "http://<jetson-ip>:9000/cashier/evidence/normal/N3/cam_20260326T045443_072402.jpg"
```

### `GET /cashier/zones`
Returns current zone config loaded from `CASHIER_CONFIG` (default: `./config/cashier_zones.yaml`).

```bash
curl -s "http://<jetson-ip>:9000/cashier/zones" | jq .
```

### `POST /cashier/zones`
Update zone polygons and thresholds live (no restart required).

Request body shape (`ZoneConfigRequest`):
- `ROI_CASHIER`: zone polygon/rectangle with normalized points (0.0-1.0), plus `active`
- `ROI_CUSTOMER`: same
- `thresholds`: cashier timing/proximity thresholds

Example (keep rectangles and only override thresholds):

```bash
curl -s -X POST "http://<jetson-ip>:9000/cashier/zones" \
  -H "Content-Type: application/json" \
  -d '{
    "thresholds": {
      "drawer_open_max_seconds": 25,
      "customer_wait_max_seconds": 40,
      "proximity_iou": 0.05,
      "config_reload_interval": 60
    }
  }' | jq .
```

### `POST /cashier/zones/reset`
Resets zones to the built-in defaults (left=cashier/register, right=customer).

```bash
curl -s -X POST "http://<jetson-ip>:9000/cashier/zones/reset" | jq .
```

## Cashier Cases (N1-N6, A1-A7)

### Severity legend
- `NORMAL`: no alert
- `ALERT`: abnormal but not "critical"
- `CRITICAL`: high-confidence violation

### Case definitions
| case_id | Meaning (high-level) | Severity | transaction |
|---|---|---|---|
| `N1` | Idle register | NORMAL | false |
| `N2` | Cashier on duty, no drawer open | NORMAL | false |
| `N3` | Transaction in progress (cashier + open drawer + customer + cash, nearby) | NORMAL | true |
| `N4` | Staff handover / supervisor: 2 staff near a drawer | NORMAL | false |
| `N5` | Customer waiting too long without drawers (no cashier start yet) | NORMAL | false |
| `N6` | Drawer open, no cash (float/card check style) | NORMAL | false |
| `A1` | Unattended open drawer (no cashier in cashier zone) | ALERT or CRITICAL* | false |
| `A2` | Unexpected person in cashier zone (other rules didn't match) | ALERT | false |
| `A3` | Cash + open drawer, no cashier present (theft signature) | CRITICAL | false |
| `A4` | Person near open drawer w/ cash but NOT proximate to drawer (intruder) | CRITICAL | false |
| `A5` | Customer waiting too long without cashier | ALERT | false |
| `A6` | Drawer open too long | ALERT | false |
| `A7` | Cash in customer zone while no cashier present | ALERT | false |

\* `A1` becomes `CRITICAL` if there is at least one customer person in `ROI_CUSTOMER`; otherwise it is `ALERT`.

### Nearby / proximity logic (used by several cases)
The cashier service uses:
- drawer proximity threshold: `proximity_iou` (default `0.05`)
- "nearby" means IoU between any person bbox and any drawer bbox is >= threshold

## ROI Zone Format (what `/cashier/zones` expects)
Zone polygons use normalized coordinates:
- `points`: `[{x: 0.0-1.0, y: 0.0-1.0}, ...]`
- `shape`: `rectangle` or `polygon`
- `active`: if `false`, the zone is disabled

The default config in `config/cashier_zones.yaml` is polygon-based.

## Image Annotations Example (`frame_096750.jpg`)
Input:
- `"/home/a7med/Documents/all_original_frames/Cashier Drawer/others/frame_096750.jpg"`

Annotated evidence produced by the cashier service for this frame (`case_id=N3`):

![Cashier Evidence N3](evidence/cashier/normal/N3/cam_20260326T045443_072402.jpg)

Related SSE-like event example from annotated simulation frame:
- Stream file: `outputs/cashier_production_sim/20260326T050929Z/stream.jsonl`
- Annotated frame: `outputs/cashier_production_sim/20260326T050929Z/annotated/frame_096720.jpg`

![Annotated Simulation Frame (SSE Source)](outputs/cashier_production_sim/20260326T050929Z/annotated/frame_096720.jpg)

### Cashier output for this frame
Evidence metadata:
- `case_id`: `N3`
- `severity`: `NORMAL`
- `alerts`: `N3 EVENT: Transaction in progress`
- `cashier_zone`: `persons=1, drawers=1, cash=3`
- `customer_zone`: `persons=1, drawers=0, cash=0`

### Bounding-box annotations (detector to ROI assignment)
Frame size: `1920x1080`

Class mapping used by the cashier model:
- `class_id=0` `Person`
- `class_id=1` `Drawer_Open`
- `class_id=2` `Cash`

| class | class_id | confidence | bbox `[x1,y1,x2,y2]` | ROI zone |
|---|---:|---:|---|---|
| Person | 0 | 0.9016 | `[608, 0, 1077, 292]` | `ROI_CUSTOMER` |
| Person | 0 | 0.8708 | `[767, 826, 1290, 1080]` | `ROI_CASHIER` |
| Drawer_Open | 1 | 0.8577 | `[843, 687, 1182, 919]` | `ROI_CASHIER` |
| Cash | 2 | 0.7952 | `[980, 702, 1048, 807]` | `ROI_CASHIER` |
| Cash | 2 | 0.7949 | `[858, 726, 930, 828]` | `ROI_CASHIER` |
| Cash | 2 | 0.7549 | `[1086, 690, 1163, 804]` | `ROI_CASHIER` |
| Cash | 2 | 0.4363 | `[1042, 694, 1102, 798]` | `ROI_CASHIER` |
| Cash | 2 | 0.2973 | `[994, 697, 1058, 803]` | `ROI_CASHIER` |

### Age/Gender + Mood (per detected person bbox)
Age/Gender:
| bbox | gender | age_group | confidence |
|---|---|---|---:|
| `[608, 0, 1077, 292]` | Female | Senior | 0.5511 |
| `[767, 826, 1290, 1080]` | Male | Senior | 0.5518 |

Mood:
| bbox | mood | confidence |
|---|---|---:|
| `[608, 0, 1077, 292]` | Happy | 0.4345 |
| `[767, 826, 1290, 1080]` | Neutral | 0.5613 |

## Notes / Integration Gotchas
1. The cashier service depends on detector `class_id` mapping matching the cashier model. Expected mapping: `0=Person`, `1=Drawer_Open`, `2=Cash`.
2. In your environment, ensure `YOLO_MODEL` is set to the cashier-capable model (`best_cashier.onnx`) or configure the pipeline so the detector produces those classes.

## Add / Update Zones (Rectangle)
Example: update both zones as rectangles (normalized coordinates 0.0-1.0). For `shape=rectangle`, provide exactly 2 points: top-left and bottom-right.

```bash
curl -s -X POST "http://<jetson-ip>:9000/cashier/zones" \
  -H "Content-Type: application/json" \
  -d '{
    "ROI_CASHIER": {
      "shape": "rectangle",
      "points": [{"x": 0.00, "y": 0.00}, {"x": 0.45, "y": 1.00}],
      "active": true
    },
    "ROI_CUSTOMER": {
      "shape": "rectangle",
      "points": [{"x": 0.55, "y": 0.00}, {"x": 1.00, "y": 1.00}],
      "active": true
    },
    "thresholds": {
      "drawer_open_max_seconds": 30,
      "customer_wait_max_seconds": 30,
      "proximity_iou": 0.05,
      "config_reload_interval": 60
    }
  }' | jq .
```

## Add / Update Zones (Polygon)
Example: update cashier zones using polygons (normalized coordinates 0.0-1.0).

```bash
curl -s -X POST "http://<jetson-ip>:9000/cashier/zones" \
  -H "Content-Type: application/json" \
  -d '{
    "ROI_CASHIER": {
      "shape": "polygon",
      "points": [
        {"x": 0.30, "y": 0.20},
        {"x": 0.70, "y": 0.20},
        {"x": 0.78, "y": 1.00},
        {"x": 0.22, "y": 1.00}
      ],
      "active": true
    },
    "ROI_CUSTOMER": {
      "shape": "polygon",
      "points": [
        {"x": 0.55, "y": 0.00},
        {"x": 1.00, "y": 0.00},
        {"x": 1.00, "y": 1.00},
        {"x": 0.50, "y": 1.00}
      ],
      "active": true
    }
  }' | jq .
```

## Production-like Sequential Test (Images -> SSE-like stream + GIF)
This is the same simulation used in the latest run: process consecutive images from the same camera location and generate:
- annotated images per frame
- `stream.jsonl` with one SSE-like JSON event per frame
- `annotated_sequence.gif`

### Run command
```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ) && export RUN_ID && ./.venv/bin/python - <<'PY'
import os, json, glob
from pathlib import Path
from datetime import datetime, timezone
import cv2, imageio

run_id = os.environ["RUN_ID"]
base_in = "/home/a7med/Documents/all_original_frames/Cashier Drawer/others"
all_frames = sorted(glob.glob(f"{base_in}/frame_*.jpg"))
start_name = "frame_096720.jpg"
idx = next(i for i, p in enumerate(all_frames) if p.endswith(start_name))
frames = all_frames[idx:idx+30]

out_root = Path(f"./outputs/cashier_production_sim/{run_id}")
annot_dir = out_root / "annotated"
evidence_dir = out_root / "evidence"
annot_dir.mkdir(parents=True, exist_ok=True)

os.environ["YOLO_MODEL"] = str(Path("./models/best_cashier.onnx").resolve())
os.environ["CONF_THRESHOLD"] = "0.2"
os.environ["FILTER_CLASSES"] = ""
os.environ["DEVICE"] = "cpu"
os.environ["SAVE_OUTPUT"] = "True"
os.environ["CASHIER_EVIDENCE_DIR"] = str(evidence_dir.resolve())

from services.detector import DetectorService
from services.cashier import CashierService
from services.age_gender import AgeGenderService
from services.mood import MoodService

detector = DetectorService()
cashier = CashierService()
age_gender = AgeGenderService()
mood = MoodService()

gif_images, cases = [], []
jsonl_path = out_root / "stream.jsonl"
with jsonl_path.open("w") as jf:
    for n, img_path in enumerate(frames, start=1):
        frame = cv2.imread(img_path)
        context = {"data": {"frame": frame, "detection": {}, "use_case": {}}}
        context = detector(context)
        all_dets = context["data"]["detection"].get("items", [])
        context = cashier(context)
        person_dets = [d for d in all_dets if str(getattr(d, "class_name", "")).lower() == "person" or getattr(d, "class_id", None) == 0]
        context["data"]["detection"] = {"items": person_dets, "count": len(person_dets)}
        context = age_gender(context)
        context = mood(context)

        out_img = annot_dir / Path(img_path).name
        cv2.imwrite(str(out_img), context["data"]["frame"], [cv2.IMWRITE_JPEG_QUALITY, 92])
        gif_images.append(cv2.cvtColor(context["data"]["frame"], cv2.COLOR_BGR2RGB))

        det_items = []
        for d in all_dets:
            x1, y1, x2, y2 = d.bbox
            det_items.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": d.class_id,
                "class_name": str(d.class_name).lower(),
                "confidence": round(float(d.confidence), 4),
                "center": [int((x1+x2)/2), int((y1+y2)/2)],
                "width": int(x2-x1),
                "height": int(y2-y1),
            })

        event = {
            "camera_id": "main_room",
            "frame_count": n,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "detection": {"count": len(det_items), "items": det_items},
                "use_case": {
                    "age_gender": [r.to_dict() for r in context["data"]["use_case"].get("age_gender", [])],
                    "mood": [r.to_dict() for r in context["data"]["use_case"].get("mood", [])],
                    "cashier": context["data"]["use_case"].get("cashier", {}),
                },
            },
        }
        jf.write(json.dumps(event) + "\n")
        cash = event["data"]["use_case"]["cashier"]
        cases.append({
            "frame": Path(img_path).name,
            "case_id": cash.get("case_id"),
            "severity": cash.get("severity"),
            "transaction": cash.get("transaction"),
        })

gif_path = out_root / "annotated_sequence.gif"
imageio.mimsave(gif_path, gif_images, fps=8, loop=0)

(out_root / "summary.json").write_text(json.dumps({
    "run_id": run_id,
    "frame_count": len(cases),
    "input_start": Path(frames[0]).name,
    "input_end": Path(frames[-1]).name,
    "gif_path": str(gif_path),
    "stream_jsonl": str(jsonl_path),
    "annotated_dir": str(annot_dir),
    "cases": cases
}, indent=2))

print("DONE", out_root)
PY
```

### Outputs from the latest simulation run
- `outputs/cashier_production_sim/20260326T050929Z/annotated_sequence.gif`
- `outputs/cashier_production_sim/20260326T050929Z/stream.jsonl`
- `outputs/cashier_production_sim/20260326T050929Z/summary.json`
- `outputs/cashier_production_sim/20260326T050929Z/annotated/`

