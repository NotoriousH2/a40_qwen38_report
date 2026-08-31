#!/usr/bin/env bash
# llama.cpp를 PR #27342 (DFlash2) head 에서 CUDA Release 로 빌드합니다.
# 주의: PR 은 master 가 아니라 xsn/dflash2 브랜치로 머지되었습니다.
#       따라서 master 를 빌드하면 DFlash2 가 들어오지 않습니다.
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../config.env"

if [ -x "$LLAMACPP_DIR/build/bin/llama-server" ]; then
  echo "[build] 이미 빌드됨: $LLAMACPP_DIR/build/bin/llama-server"
  exit 0
fi
command -v nvcc >/dev/null || { echo "[build] CUDA toolkit(nvcc)이 필요합니다" >&2; exit 1; }

mkdir -p "$(dirname "$LLAMACPP_DIR")"
if [ ! -d "$LLAMACPP_DIR/.git" ]; then
  echo "[build] clone 중…"
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$LLAMACPP_DIR"
fi
cd "$LLAMACPP_DIR"
git fetch origin "pull/$LLAMACPP_PR/head:dflash2" || true
git checkout dflash2
git rev-parse HEAD | grep -q "^$LLAMACPP_SHA" \
  || echo "[build] 경고: HEAD 가 검증된 sha($LLAMACPP_SHA)와 다릅니다. PR 이 갱신되었을 수 있습니다."

# LLAMA_CURL=OFF: libcurl-dev 가 없어도 빌드되게. 모델은 별도로 받으므로 무방.
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DLLAMA_CURL=OFF
cmake --build build --config Release -j "$(nproc)" \
      --target llama-server llama-cli llama-bench llama-perplexity
echo "[build] 완료: $LLAMACPP_DIR/build/bin/llama-server"
