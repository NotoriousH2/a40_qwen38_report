#!/usr/bin/env bash
#
# 권장 프로파일 — 이 저장소의 모든 측정을 근거로 고른 최종 설정입니다.
#
#   Q3_K_XL + DFlash2(n-max 4) + f16 KV + 프리픽스 캐시 재사용 + 128K 슬롯 2개
#
# 왜 이 값인지는 docs/04, docs/05, docs/06 을 보세요. 요약하면:
#   f16 KV        : draft acceptance 를 0.56 → 0.66 으로 올려 오히려 빨라집니다 (docs/04)
#   n-max 4       : 6, 8 로 올리면 acceptance 가 무너져 더 느려집니다 (docs/04)
#   -ub 256       : 512~2048 로 올려도 prefill 이 전혀 안 빨라집니다 (docs/04)
#   --cache-reuse : 64K 재프롬프트가 70초 → 0.7초 (docs/05, 개발 워크플로의 핵심)
#   -np 2         : 슬롯당 128K, 두 세션이 각자 프리픽스 캐시를 유지
#
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../config.env"

BIN="$LLAMACPP_DIR/build/bin/llama-server"
TARGET="${TARGET:-$MODEL_DIR/$TARGET_FILE}"
DRAFT="${DRAFT:-$MODEL_DIR/$DRAFT_FILE}"
CTX="${CTX:-262144}"; NP="${NP:-2}"        # 슬롯당 CTX/NP = 131072
LOG="${LOG:-$HERE/../llama-server.log}"
PID_FILE="$HERE/../llama-server.pid"

log(){ printf '\033[1;36m[dev]\033[0m %s\n' "$*"; }
die(){ printf '\033[1;31m[dev] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[ -x "$BIN" ]    || die "빌드가 없습니다. scripts/build-llamacpp.sh 를 먼저 실행하세요."
[ -f "$TARGET" ] || die "타깃 모델이 없습니다. scripts/download-models.sh 를 먼저 실행하세요."
[ -f "$DRAFT" ]  || die "드래프터가 없습니다. scripts/download-models.sh 를 먼저 실행하세요."

if curl -sf --max-time 3 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  log "이미 $PORT 에서 응답 중입니다 — 종료."; exit 0
fi
mkdir -p "$SLOT_SAVE_PATH"

ARGS=(
  -m "$TARGET" -md "$DRAFT"
  --spec-type draft-dflash --spec-draft-n-max 4
  --spec-draft-type-k f16 --spec-draft-type-v f16
  -c "$CTX" -np "$NP" -ngl all -fa on
  --cache-type-k f16 --cache-type-v f16
  --cache-reuse 256
  --slot-save-path "$SLOT_SAVE_PATH"
  -b 2048 -ub 256
  --jinja --metrics --reasoning-preserve
  --host "$HOST" --port "$PORT"
)
: > "$LOG"
log "기동: ctx=$CTX np=$NP (슬롯당 $((CTX/NP)))  →  http://$HOST:$PORT"
nohup "$BIN" "${ARGS[@]}" >>"$LOG" 2>&1 &
PID=$!; echo "$PID" > "$PID_FILE"; disown "$PID" 2>/dev/null || true
for _ in $(seq 1 300); do
  curl -sf --max-time 3 "http://$HOST:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$PID" 2>/dev/null || die "기동 중 종료되었습니다:
$(tail -n 30 "$LOG")"
  sleep 2
done
curl -sf --max-time 3 "http://$HOST:$PORT/health" >/dev/null 2>&1 || die "health 실패 — $LOG 확인"
log "서빙 : http://$HOST:$PORT/v1 (OpenAI 호환)"
log "검증 : scripts/verify.sh"
log "중지 : scripts/stop-server.sh"
