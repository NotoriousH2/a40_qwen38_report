#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../config.env"
PID_FILE="$HERE/../llama-server.pid"
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  PID="$(cat "$PID_FILE")"; echo "[stop] pid $PID 종료 중"
  kill -TERM "$PID" 2>/dev/null || true
  for _ in $(seq 1 20); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  kill -0 "$PID" 2>/dev/null && kill -KILL "$PID" 2>/dev/null || true
else
  echo "[stop] 기록된 프로세스 없음"
fi
rm -f "$PID_FILE"
sleep 1
curl -sf --max-time 3 "http://$HOST:$PORT/health" >/dev/null 2>&1 \
  && { echo "[stop] 경고: 아직 $PORT 응답 중"; exit 1; } || echo "[stop] 포트 $PORT 해제됨"
