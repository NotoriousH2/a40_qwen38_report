#!/usr/bin/env python3
"""실제 C++ 소스 ~64K 토큰을 컨텍스트로 넣고 현실적인 acceptance/decode 측정."""
import glob, json, os, signal, subprocess, time, urllib.request
BIN="/root/build/llama.cpp-dflash2/build/bin/llama-server"
DRAFT="/root/models/gguf/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
Q3="/root/models/gguf/Qwen3.8-27B-UD-Q3_K_XL.gguf"
PORT=8081; BASE=f"http://127.0.0.1:{PORT}"
SWEEP=os.path.dirname(os.path.abspath(__file__))
SRC="/root/build/llama.cpp-dflash2"

def api(p,b,t=3000):
    return json.load(urllib.request.urlopen(urllib.request.Request(BASE+p,
        data=json.dumps(b).encode(),headers={"Content-Type":"application/json"}),timeout=t))
def ntok(s): return len(api("/tokenize",{"content":s})["tokens"])

def build_ctx(target):
    files=sorted(glob.glob(f"{SRC}/src/*.cpp")+glob.glob(f"{SRC}/src/*.h")
                +glob.glob(f"{SRC}/common/*.cpp")+glob.glob(f"{SRC}/tools/server/*.cpp"))
    buf=[]; tot=0
    for f in files:
        try: txt=open(f,encoding="utf-8",errors="ignore").read()
        except Exception: continue
        chunk=f"\n// ===== FILE: {os.path.relpath(f,SRC)} =====\n"+txt
        n=ntok(chunk[:20000])*(len(chunk)/min(len(chunk),20000))
        buf.append(chunk); tot+=n
        if tot>target*1.15: break
    s="".join(buf)
    # 정확히 자르기
    lo,hi=0,len(s)
    while lo<hi:
        mid=(lo+hi)//2
        if ntok(s[:mid])<target: lo=mid+1
        else: hi=mid
    return s[:lo]

def start(cfg):
    args=[BIN,"-m",Q3,"-c","131072","-np","1","-ngl","all","-fa","on",
          "--cache-type-k",cfg["kv"],"--cache-type-v",cfg["kv"],
          "-b","2048","-ub","256","--jinja","--metrics","--reasoning-preserve",
          "--cache-reuse","256","--host","127.0.0.1","--port",str(PORT)]
    if cfg["spec"]:
        args+=["-md",DRAFT,"--spec-type","draft-dflash","--spec-draft-n-max",str(cfg.get("nmax",4)),
               "--spec-draft-type-k","f16","--spec-draft-type-v","f16"]
    log=open(f"{SWEEP}/rc_{cfg['name']}.log","w")
    p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
    for _ in range(600):
        if p.poll() is not None: return None,"exited"
        try: urllib.request.urlopen(BASE+"/health",timeout=3).read(); return p,None
        except Exception: time.sleep(1)
    p.kill(); return None,"timeout"
def stop(p):
    if p is None: return
    p.send_signal(signal.SIGTERM)
    try: p.wait(timeout=40)
    except subprocess.TimeoutExpired: p.kill(); p.wait()
    time.sleep(3)

QS=["\n// TASK: Explain in prose how the server handles a request that exceeds the context window.\n",
    "\n// TASK: Describe the role of the slot abstraction and how prompt cache reuse works.\n",
    "\n// TASK: Summarize what happens during model loading and KV cache allocation.\n"]

CONFIGS=[dict(name="spec_f16",kv="f16",spec=True),
         dict(name="spec_q8", kv="q8_0",spec=True),
         dict(name="nospec_f16",kv="f16",spec=False)]
res=[]
ctx=None
for cfg in CONFIGS:
    print(f"\n### {cfg['name']}",flush=True)
    p,err=start(cfg)
    if err: print("  FAILED",err,flush=True); continue
    try:
        if ctx is None:
            ctx=build_ctx(64000); print(f"  실제 코드 컨텍스트 {ntok(ctx)} 토큰",flush=True)
        rec=[]
        for i,q in enumerate(QS):
            r=api("/completion",{"prompt":ctx+q,"n_predict":256,"ignore_eos":True,
                                 "temperature":0,"cache_prompt":True})
            t=r["timings"]
            acc=(t.get("draft_n_accepted") or 0)/(t.get("draft_n") or 1)
            rec.append({"prefill_s":round(t.get("prompt_ms",0)/1000,1),
                        "prompt_n":t.get("prompt_n"),
                        "decode_tps":round(t.get("predicted_per_second",0),2),
                        "acc":round(acc,3)})
            print(f"   Q{i+1}: prompt_n={t.get('prompt_n'):>6} prefill={rec[-1]['prefill_s']:>5}s "
                  f"decode={rec[-1]['decode_tps']:>6.2f} t/s  acc={acc:.3f}",flush=True)
        res.append({"name":cfg["name"],"runs":rec})
    except Exception as e: print("  ERROR",repr(e),flush=True)
    finally: stop(p)
json.dump(res,open(f"{SWEEP}/results_realcode.json","w"),indent=1)
print("\nWROTE results_realcode.json")
