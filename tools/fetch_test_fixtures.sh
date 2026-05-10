#!/usr/bin/env bash
# Generate minimal synthetic MP4 fixtures for integration tests (no network).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/tests/fixtures/videos"
mkdir -p "$OUT"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found; install ffmpeg to generate fixtures." >&2
  exit 1
fi

gen() {
  local name="$1"
  local seconds="${2:-8}"
  local label="$3"
  if [[ -f "$OUT/$name" ]] && [[ "${FORCE:-0}" != "1" ]]; then
    echo "exists: $OUT/$name (set FORCE=1 to regenerate)"
    return 0
  fi
  ffmpeg -y -f lavfi -i "testsrc=duration=${seconds}:size=640x480:rate=15" \
    -vf "drawtext=text='${label}':fontsize=24:fontcolor=white:x=20:y=20" \
    -pix_fmt yuv420p -c:v libx264 "$OUT/$name" </dev/null >/dev/null 2>&1
  echo "wrote $OUT/$name"
}

gen lobby_cross_line.mp4 10 "cross_line"
gen stream_torture.mp4 6 "torture"
gen cashier_drawer.mp4 8 "cashier"
gen ppe_violation.mp4 8 "ppe"
gen ppe_compliant.mp4 8 "ok"
gen phone_usage.mp4 8 "phone"
gen face_enroll.mp4 6 "enroll"
gen face_match.mp4 6 "match"

echo "$OUT/*.mp4" | tr ' ' '\n' > "$ROOT/tests/fixtures/mediamtx_streams.txt"
echo "Done. RTSP paths (after compose): rtsp://127.0.0.1:8554/lobby_cross_line etc."
