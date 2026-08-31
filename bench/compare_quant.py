import json,urllib.request,glob,os
BASE="http://127.0.0.1:8080"
def api(p,b,t=3000):
    return json.load(urllib.request.urlopen(urllib.request.Request(BASE+p,
        data=json.dumps(b).encode(),headers={"Content-Type":"application/json"}),timeout=t))
def ntok(s): return len(api("/tokenize",{"content":s})["tokens"])
m=json.load(urllib.request.urlopen(BASE+"/v1/models"))
print("model :", os.path.basename(m["data"][0]["id"]))
print("n_ctx :", m["data"][0]["meta"]["n_ctx"], "| slots:",
      len(json.load(urllib.request.urlopen(BASE+"/slots"))))
# 짧은 컨텍스트 384토큰 (이전 스윕과 동일 조건)
print("\n[짧은 ctx] 384토큰 x3")
SHORT="Explain how a hash table works, including collision handling, in clear technical prose."
for i in range(3):
    r=api("/completion",{"prompt":SHORT,"n_predict":384,"ignore_eos":True,
                         "temperature":0,"cache_prompt":False})
    t=r["timings"]; acc=(t.get("draft_n_accepted") or 0)/(t.get("draft_n") or 1)
    print(f"  run{i+1}: decode={t['predicted_per_second']:.2f} t/s  acc={acc:.3f}")
# 60K 실제 코드
SRC="/root/build/llama.cpp-dflash2"
files=sorted(glob.glob(f"{SRC}/src/*.cpp")+glob.glob(f"{SRC}/src/*.h")+glob.glob(f"{SRC}/common/*.cpp"))
buf=""
for f in files:
    buf+=f"\n// ===== {os.path.relpath(f,SRC)} =====\n"+open(f,errors="ignore").read()
    if len(buf)>260000: break
lo,hi=0,len(buf)
while lo<hi:
    mid=(lo+hi)//2
    if ntok(buf[:mid])<60000: lo=mid+1
    else: hi=mid
ctx=buf[:lo]
print(f"\n[60K 실제 코드] ctx={ntok(ctx)} 토큰")
for i,q in enumerate(["\n// TASK: Explain the slot lifecycle.\n",
                      "\n// TASK: Explain how KV cache is allocated.\n",
                      "\n// TASK: Explain error handling on overflow.\n"]):
    r=api("/completion",{"prompt":ctx+q,"n_predict":256,"ignore_eos":True,
                         "temperature":0,"cache_prompt":True})
    t=r["timings"]; acc=(t.get("draft_n_accepted") or 0)/(t.get("draft_n") or 1)
    print(f"  turn{i+1}: prompt_n={t['prompt_n']:>6}  prefill={t['prompt_ms']/1000:>6.1f}s"
          f"  decode={t['predicted_per_second']:.2f} t/s  acc={acc:.3f}")
