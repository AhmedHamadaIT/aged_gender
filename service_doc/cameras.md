# Cameras service

## Description

In-memory registry mapping **camera id** → **RTSP (or stream) URL**. Task definitions reference cameras by `channelId`, which must match the camera `id` registered here.

Cameras can be added or removed while the server runs; starting detection validates that each task’s `channelId` has a registered camera URL.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/cameras` | Add or update one or more cameras |
| GET | `/cameras` | List all cameras |
| DELETE | `/cameras/{cam_id}` | Remove a camera by id |

## Request schema (POST)

- `cameras`: array of `{ "id": string, "url": string }`, minimum length 1

## curl — register cameras

```bash
export BASE="http://localhost:9000"

curl -sS -X POST "${BASE}/cameras" \
  -H "Content-Type: application/json" \
  -d '{
    "cameras": [
      {"id": "1", "url": "rtsp://192.168.1.10/stream"},
      {"id": "2", "url": "rtsp://192.168.1.11/stream"}
    ]
  }'
```

## curl — list

```bash
curl -sS "${BASE}/cameras"
```

## curl — delete

```bash
curl -sS -X DELETE "${BASE}/cameras/1"
```

## Edge device note

The **edge device** only needs to send the RTSP URL that the **ML server** can reach. If the camera is on a private VLAN, ensure routing/firewall allows the ML server host to pull that RTSP URL.
