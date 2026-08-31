#!/usr/bin/env bash
#
# 상태를 감지해 필요한 것만 수행하고 권장 프로파일로 서버를 띄웁니다.
#
#   이미 빌드/모델이 있는 머신  →  약 40초 (전부 건너뛰고 기동만)
#   완전히 새 머신              →  약 20~40분 (CUDA 빌드 15분 + GGUF 14.3GB)
#
# 사용법:  ./scripts/bootstrap.sh
#
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../config.env"

step(){ printf '\n\033[1;35m▶ %s\033[0m\n' "$*"; }

step "1/4  GPU 확인"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "nvidia-smi 를 찾을 수 없습니다." >&2; exit 1; }
echo "  주의: config.env 의 CUDA_ARCH=$CUDA_ARCH 가 이 GPU와 맞는지 확인하세요 (A40=86)."

step "2/4  llama.cpp (DFlash2, PR #$LLAMACPP_PR) 빌드"
"$HERE/build-llamacpp.sh"

step "3/4  GGUF 준비"
"$HERE/download-models.sh"

step "4/4  권장 프로파일로 기동"
"$HERE/run-server-dev.sh"

printf '\n\033[1;32m준비 완료.\033[0m  검증하려면:  %s\n' "$HERE/verify.sh"
