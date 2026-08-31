#!/usr/bin/env bash
# GGUF 다운로드. 이미 있으면 건너뜁니다. ALT=1 이면 비교용 Q4_K_XL 도 받습니다.
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../config.env"
mkdir -p "$MODEL_DIR"

PY="${PY:-python3}"
$PY - "$MODEL_DIR" "${ALT:-0}" <<'PYEOF'
import os, sys
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import hf_hub_download
out, alt = sys.argv[1], sys.argv[2] == "1"
jobs = [("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q3_K_XL.gguf"),
        ("incoai/Qwen3.8-27B-DFlash2-GGUF", "Qwen3.8-27B-DFlash2-Q4_K_M.gguf")]
if alt:
    jobs.append(("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_XL.gguf"))
for repo, fn in jobs:
    if os.path.exists(os.path.join(out, fn)):
        print("skip (이미 있음):", fn); continue
    print("downloading:", fn, flush=True)
    hf_hub_download(repo, fn, local_dir=out)
print("완료")
PYEOF
