# Tasks service (`/api/tasks`)

## Description

Registers **analytics tasks**: algorithm type, target camera (`channelId`), thresholds, schedule windows, and algorithm-specific JSON (`areaPosition`, `detailConfig`). Only **enabled** tasks are started when you call `POST /detection/start`.

Supported `algorithmType` values (from `apis/tasks.py` `TaskRegistry.SUPPORTED`):

| `algorithmType` | Description | Guide |
|---|---|---|
| `CROSS_LINE` | Line crossing / counting with direction tracking | [cross_line.md](./cross_line.md) |
| `MASK_HAIRNET_CHEF_HAT` | PPE / headwear compliance (mask, hairnet, chef hat) | [mask_hairnet_chef_hat.md](./mask_hairnet_chef_hat.md) |
| `PHONE_USAGE` | Phone-in-hand usage inside configurable zones | [phone_usage.md](./phone_usage.md) |
| `CASHIER_BOX_OPEN` | Cashier drawer open / zone dwell monitoring | [cashier.md](./cashier.md) |
| `FACE` | Face recognition — known persons + stranger detection | `services/face_recognition.py` |

Evidence metadata for all task types follows **ML Image Contract V2** (structured `captureImage` / `sceneImage` on task events). See [ml_image_v2.md](./ml_image_v2.md).

Each algorithm uses the **same** HTTP surface (`/cameras`, `/api/tasks`, `/detection/*`, `/detection/stream`). There are no extra paths only for cross-line or PPE; the guides above document **config and event behaviour** per algorithm.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/tasks` | Create or upsert a task (body includes `taskId`) |
| GET | `/api/tasks` | List all tasks |
| GET | `/api/tasks/{task_id}` | Get one task by numeric id |
| PUT | `/api/tasks/{task_id}` | Update; body `taskId` must match URL |
| DELETE | `/api/tasks/{task_id}` | Delete a task |

## curl — create cross-line task

```bash
export BASE="http://localhost:9000"

curl -sS -X POST "${BASE}/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 10,
    "taskName": "entrance_line",
    "algorithmType": "CROSS_LINE",
    "channelId": "1",
    "enable": true,
    "threshold": 60,
    "areaPosition": "[{\"line_id\":\"1\",\"line_name\":\"Entrance\",\"point\":[{\"x\":100,\"y\":400},{\"x\":900,\"y\":400}],\"direction\":1}]",
    "detailConfig": {
      "enableAttrDetect": false,
      "enableReid": false,
      "alarmType": []
    },
    "validWeekday": ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"],
    "validStartTime": 0,
    "validEndTime": 86400000
  }'
```

## curl — list all tasks

```bash
curl -sS "${BASE}/api/tasks"
```

## curl — get one task

```bash
curl -sS "${BASE}/api/tasks/10"
```

## curl — update task

```bash
curl -sS -X PUT "${BASE}/api/tasks/10" \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": 10,
    "taskName": "entrance_line",
    "algorithmType": "CROSS_LINE",
    "channelId": "1",
    "enable": true,
    "threshold": 70,
    "areaPosition": "[{\"line_id\":\"1\",\"line_name\":\"Entrance\",\"point\":[{\"x\":100,\"y\":400},{\"x\":900,\"y\":400}],\"direction\":1}]",
    "detailConfig": {"enableAttrDetect": false, "enableReid": false, "alarmType": []},
    "validWeekday": ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"],
    "validStartTime": 0,
    "validEndTime": 86400000
  }'
```

## curl — delete task

```bash
curl -sS -X DELETE "${BASE}/api/tasks/10"
```

## `detailConfig` fields (schema defaults)

| Field | Used by | Notes |
|-------|---------|--------|
| `enableAttrDetect`, `enableReid` | `CROSS_LINE` | Optional attributes / ReID |
| `alarmType` | `MASK_HAIRNET_CHEF_HAT` | Violation list: `no_mask`, `no_hat`, `no_chef_hat` |
| `channelName`, `deviceSN` | `MASK_HAIRNET_CHEF_HAT` | Optional Eyego `data.channelName` / `data.deviceSN` |
| `drawerOpenLimit`, `serviceWaitLimit`, `enableStaffList`, `staffIds` | `CASHIER_BOX_OPEN` | Drawer / service timing and staff list |

## Edge device note

Keep `taskName` **unique** if you plan to use the WebSocket alias `WS /tasks/{task_name}/live`; duplicate names return HTTP 409 when resolving the camera.
