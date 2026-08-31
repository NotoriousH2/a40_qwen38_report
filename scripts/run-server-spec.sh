#!/usr/bin/env bash
#
# 원본 스펙 재현용 — 최초 요구사항에 적힌 옵션을 한 글자도 바꾸지 않은 구성입니다.
# 성능은 run-server-dev.sh 가 더 낫습니다(docs/04). 재현·대조용으로만 쓰세요.
#   차이: KV q8_0, -np 1, -c 131072, 캐시 재사용 없음
#
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../config.env"
BIN="$LLAMACPP_DIR/build/bin/llama-server"
TARGET="$MODEL_DIR/$TARGET_FILE"; DRAFT="$MODEL_DIR/$DRAFT_FILE"
LOG="$HERE/../llama-server.log"; PID_FILE="$HERE/../llama-server.pid"
[ -x "$BIN" ] || { echo "빌드 없음. build-llamacpp.sh 먼저." >&2; exit 1; }
if curl -sf --max-time 3 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  echo "[spec] 이미 $PORT 응답 중 — 종료."; exit 0
fi
: > "$LOG"
nohup "$BIN" -m "$TARGET" -md "$DRAFT" \
  --spec-type draft-dflash --spec-draft-n-max 4 \
  -c 131072 -np 1 -ngl all -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-draft-type-k f16 --spec-draft-type-v f16 \
  -b 2048 -ub 256 --jinja --metrics --reasoning-preserve \
  --host "$HOST" --port "$PORT" >>"$LOG" 2>&1 &
PID=$!; echo "$PID" > "$PID_FILE"; disown "$PID" 2>/dev/null || true
for _ in $(seq 1 300); do
  curl -sf --max-time 3 "http://$HOST:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$PID" 2>/dev/null || { echo "기동 실패:"; tail -30 "$LOG"; exit 1; }
  sleep 2
done
echo "[spec] 서빙: http://$HOST:$PORT/v1"
