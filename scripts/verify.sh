#!/usr/bin/env bash
# 서버가 제대로 떴는지 7항목 검증. run-server-*.sh 이후에 실행하세요.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../config.env"
BASE="http://$HOST:$PORT"
PY="${PY:-python3}"
echo "=== 1. /health ==="; curl -s -w "  (HTTP %{http_code})\n" "$BASE/health"
echo "=== 2. /v1/models ==="
curl -s "$BASE/v1/models" | $PY -c "
import json,sys,os
d=json.load(sys.stdin)['data'][0]
print('  model :', os.path.basename(d['id']))
print('  n_ctx :', d['meta']['n_ctx'], '(슬롯당)')
print('  params:', d['meta']['n_params'])"
echo "=== 3. 슬롯 ==="; curl -s "$BASE/slots" | $PY -c "import json,sys; print('  슬롯 수:', len(json.load(sys.stdin)))"
echo "=== 4. VRAM ==="; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/  /'
$PY - "$BASE" <<'PYEOF'
import json, sys, urllib.request, hashlib
BASE=sys.argv[1]
def api(p,b,t=1200):
    return json.load(urllib.request.urlopen(urllib.request.Request(BASE+p,
        data=json.dumps(b).encode(),headers={"Content-Type":"application/json"}),timeout=t))
print("=== 5. 384토큰 생성 x3 + DFlash2 acceptance ===")
for i in range(3):
    r=api("/completion",{"prompt":"Explain how a hash table works, including collision handling.",
        "n_predict":384,"ignore_eos":True,"temperature":0,"cache_prompt":False})
    t=r["timings"]; dn=t.get("draft_n") or 0; da=t.get("draft_n_accepted") or 0
    print(f"  run{i+1}: {t['predicted_per_second']:.2f} tok/s"
          + (f"  acceptance={da/dn:.3f} ({da}/{dn})" if dn else "  (투기 디코딩 없음)"))
def chat(temp, seed=None):
    b={"model":"m","messages":[{"role":"user","content":"Name a coffee shop and give a one-line slogan."}],
       "max_tokens":60,"temperature":temp,"chat_template_kwargs":{"enable_thinking":False}}
    if seed is not None: b["seed"]=seed
    return api("/v1/chat/completions",b)["choices"][0]["message"]["content"]
h=lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]
print("=== 6. temperature=0 재현성 ===")
o=[chat(0.0) for _ in range(3)]
print("  sha:", [h(x) for x in o], "→ 전부 동일:", len(set(o))==1)
print("=== 7. temperature=1.0 샘플링 ===")
s=[chat(1.0) for _ in range(4)]; sd=[chat(1.0,seed=1234) for _ in range(2)]
print("  seed 미지정 고유 출력:", len(set(s)), "/ 4  (달라야 정상)")
print("  seed=1234 전부 동일:", len(set(sd))==1, " (같아야 정상)")
PYEOF
