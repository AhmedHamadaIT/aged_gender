# Frontend guide: live camera WebSocket (JPEG)

This document is for dashboard / UI engineers consuming **`WS /cameras/{camera_id}/live`** (or **`WS /tasks/{task_name}/live`**). It explains what you see in DevTools, how to render efficiently, and how to reduce glitches and memory growth when many tiles are open.

For protocol basics and CLI tests, see [`../service_doc/websockets.md`](../service_doc/websockets.md) and the longer examples in [`API_USAGE.md`](./API_USAGE.md#11-live-stream-websocket-camerasidlive).

---

## What you are seeing

### Gray blocks, smearing, macroblocking

The ML server receives **RTSP (often HEVC / 4K)** and produces **annotated JPEG** frames for the browser. **Packet loss or decoder stress** upstream can still produce ugly frames occasionally; the backend applies heuristics and FFmpeg tuning to limit that. **The UI cannot fix corrupted bitstreams**, but it can:

- Avoid **piling up work** (decode + layout) on every single message.
- **Drop** obviously invalid payloads before assigning `img.src`.
- **Reconnect cleanly** so stale buffers and tab throttling do not make things worse.

If artifacts persist with a healthy network, involve the **ops / video** side (substream, bitrate, IDR interval, TCP RTSP).

### Many `blob:http://…` rows in the Network panel

That is **normal** when you use `URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" }))` and set `img.src`. Each assignment can show as a blob resource. It is **not** the same as HTTP polling; traffic still goes over the **single WebSocket**.

What **is** a problem is creating a **new** object URL on every message **without** revoking the previous one: memory grows and the tab gets slower. Always **revoke** the last URL when you replace it (see patterns below).

---

## Server contract (binary path)

| Item | Behavior |
|------|------------|
| Default message body | **Raw JPEG** bytes (`FF D8 …`). |
| `binaryType` | Set **`ws.binaryType = "arraybuffer"`** (or `"blob"`) so `onmessage` receives binary. |
| Backpressure | Server may **drop** frames if the client does not read fast enough (`WS_SEND_TIMEOUT_MS`, default 50 ms). You may see **jumps in motion**, not slow motion. |
| Per-socket FPS cap | Server may cap send rate with **`WS_MAX_FPS`** (default 15). Coordinate with **`REDIS_LIVE_FPS`** so expectations match. |
| Optional sequence prefix | If ops set **`WS_INCLUDE_SEQ_HEADER=true`**, each message is **`4` bytes little-endian `uint32` sequence** + **JPEG**. Default is **off** so existing clients keep working. |

**JPEG sanity check** (default wire format):

```javascript
function looksLikeJpeg(u8) {
  return u8 && u8.length >= 3 && u8[0] === 0xff && u8[1] === 0xd8 && u8[2] === 0xff;
}
```

If `WS_INCLUDE_SEQ_HEADER` is enabled, strip the first 4 bytes before checking / decoding:

```javascript
function parseLiveFrameMessage(buf) {
  const u8 = new Uint8Array(buf);
  const WITH_SEQ = false; // set true when server runs WS_INCLUDE_SEQ_HEADER=true
  if (!WITH_SEQ) return u8;
  if (u8.length < 5) return null;
  const seq = new DataView(buf).getUint32(0, true);
  const jpeg = u8.subarray(4);
  return { seq, jpeg };
}
```

Use **`seq`** to detect gaps (`seq - lastSeq > 1`) and optionally show a small “catching up” indicator.

---

## Recommended rendering patterns

### 1. One `Image` + rAF + revoke (simple, good enough for many tiles)

Queue **only the latest** frame; render **once per animation frame**; revoke the previous object URL.

```javascript
function attachLivePreview(wsUrl, imgEl) {
  let ws;
  let objectUrl = null;
  let pending = null;
  let rafScheduled = false;

  function paint() {
    rafScheduled = false;
    if (pending == null) return;
    const u8 = pending instanceof Uint8Array ? pending : new Uint8Array(pending);
    pending = null;
    if (!looksLikeJpeg(u8)) return;

    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(new Blob([u8], { type: "image/jpeg" }));
    imgEl.src = objectUrl;
  }

  function connect() {
    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";
    ws.onmessage = (ev) => {
      pending = ev.data;
      if (!rafScheduled) {
        rafScheduled = true;
        requestAnimationFrame(paint);
      }
    };
    ws.onerror = () => ws.close();
    return ws;
  }

  connect();
  return () => {
    try { ws && ws.close(); } catch (_) {}
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  };
}
```

This reduces work when the server bursts frames and avoids retaining every intermediate blob.

### 2. `createImageBitmap` (often less janky than `<img>` for frequent updates)

Decodes off the main thread more cleanly in many browsers:

```javascript
async function paintBitmap(canvas, imageDataBuf) {
  const u8 = new Uint8Array(imageDataBuf);
  if (!looksLikeJpeg(u8)) return;
  const bmp = await createImageBitmap(new Blob([u8], { type: "image/jpeg" }));
  const ctx = canvas.getContext("2d");
  if (canvas.width !== bmp.width || canvas.height !== bmp.height) {
    canvas.width = bmp.width;
    canvas.height = bmp.height;
  }
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
}
```

Use the same **“latest frame wins”** queue + **rAF** as above so you do not await decode for stale frames.

### 3. Avoid per-frame `Object.assign` / React state for the raw buffer

Do **not** store every JPEG in React state or Redux. Keep the socket handler **outside** render hot paths, write to a ref, and drive a **single** rAF loop or a dedicated preview component with `useLayoutEffect` + ref.

---

## Reconnection and `last_seq`

The server **does not** reconnect the browser for you. Use **exponential backoff** (e.g. 1s → 2s → … cap 30s) and show status text.

Optional: after reconnect, if you tracked **`seq`** (with `WS_INCLUDE_SEQ_HEADER` or from SSE / metrics), open:

```text
ws://host/cameras/{id}/live?last_seq=12345
```

The server replays a **short ring** of recent frames with `seq > last_seq` (best-effort). Helps dashboards recover a bit smoother.

---

## Many cameras (e.g. eight tiles)

- **One WebSocket per camera** (required by design).
- Use **rAF + latest-frame** per tile (patterns above).
- Prefer **fixed layout** sizes so resizing does not trigger reflow on every frame.
- Consider **`WS_MAX_FPS`** on the server plus a **single** `IntersectionObserver` to **pause** connections for tiles off-screen (close WebSocket or stop processing until visible again) — biggest win for CPU and battery.

---

## TLS and proxies

- Use **`wss://`** in production behind HTTPS.
- Ensure reverse proxies **disable buffering** for WebSocket upgrade paths and allow **long-lived** connections.

---

## Quick checklist

| Check | Why |
|-------|-----|
| `ws.binaryType = "arraybuffer"` | Correct binary handling. |
| Revoke previous `blob:` URL | Prevents memory leak and DevTools noise. |
| rAF + “latest only” queue | Matches bursty frames; keeps UI smooth. |
| JPEG magic bytes before decode | Skip garbage if a proxy corrupts a message once. |
| Exponential backoff on `onclose` | Avoid reconnect storms. |
| Optional `?last_seq=` + seq header | Smarter recovery when enabled server-side. |

---

## Who to ping when debugging

| Symptom | Likely layer |
|---------|----------------|
| Valid JPEG but blocky / gray **walls** in scene | RTSP / camera / network (HEVC loss). Ops + server ingest tuning. |
| Stalls then bursts | Backpressure + client too slow; reduce work per frame or lower resolution server-side. |
| Memory climbs in Chrome | Missing `URL.revokeObjectURL` or holding buffers in state. |
| Disconnects every ~60s | Proxy idle timeout; increase keepalive / ping. |

This repo’s server uses uvicorn WebSocket ping settings in Docker Compose; align proxy timeouts with that documentation.
