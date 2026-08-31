#!/usr/bin/env python3
"""장문 컨텍스트 정밀 측정: 프롬프트 길이를 토크나이저로 정확히 맞추고,
   ignore_eos + n_predict 고정으로 decode tok/s를 안정적으로 측정."""
import json, os, signal, subprocess, time, urllib.request

BIN   = "/root/build/llama.cpp-dflash2/build/bin/llama-server"
DRAFT = "/root/models/gguf/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
Q3    = "/root/models/gguf/Qwen3.8-27B-UD-Q3_K_XL.gguf"
PORT  = 8081; BASE=f"http://127.0.0.1:{PORT}"
SWEEP = os.path.dirname(os.path.abspath(__file__))
NPRED = 256

def vram(): return int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.used",
    "--format=csv,noheader,nounits"]).decode().strip().splitlines()[0])

def api(path, body, timeout=3000):
    req=urllib.request.Request(BASE+path, data=json.dumps(body).encode(),
                               headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req,timeout=timeout))

def ntok(text): return len(api("/tokenize",{"content":text})["tokens"])

def complete(prompt, npred=NPRED, cache=True):
    r=api("/completion",{"prompt":prompt,"n_predict":npred,"ignore_eos":True,
                         "temperature":0,"cache_prompt":cache})
    return r["timings"], r.get("tokens_evaluated")

LINE=("// module {t}: the scheduler reconciles pending work items against the lease "
      "table and emits a compaction plan for tier {t}.\n")
def make(tag, target):
    line=LINE.format(t=tag); per=ntok(line)
    reps=max(1,int(target/per))
    return line*reps

def start(cfg):
    args=[BIN,"-m",Q3,"-c","131072","-np","1","-ngl","all","-fa","on",
          "--cache-type-k",cfg["kv"],"--cache-type-v",cfg["kv"],
          "-b","2048","-ub","256","--jinja","--metrics","--reasoning-preserve",
          "--cache-reuse","256","--host","127.0.0.1","--port",str(PORT)]
    if cfg["spec"]:
        args+=["-md",DRAFT,"--spec-type","draft-dflash","--spec-draft-n-max","4",
               "--spec-draft-type-k","f16","--spec-draft-type-v","f16"]
    log=open(f"{SWEEP}/l2_{cfg['name']}.log","w")
    p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
    for _ in range(600):
        if p.poll() is not None: return None,f"exited rc={p.returncode}"
        try: urllib.request.urlopen(BASE+"/health",timeout=3).read(); return p,None
        except Exception: time.sleep(1)
    p.kill(); return None,"health timeout"

def stop(p):
    if p is None: return
    p.send_signal(signal.SIGTERM)
    try: p.wait(timeout=40)
    except subprocess.TimeoutExpired: p.kill(); p.wait()
    time.sleep(3)

def run(cfg):
    print(f"\n### {cfg['name']} kv={cfg['kv']} spec={cfg['spec']}",flush=True)
    p,err=start(cfg)
    if err:
        tail=open(f"{SWEEP}/l2_{cfg['name']}.log").read()[-1500:]
        print("  FAILED:",err,"\n",tail,flush=True); return {"name":cfg["name"],"failed":err}
    res={"name":cfg["name"],"cfg":cfg,"vram_loaded":vram(),"points":{}}
    try:
        for tag,target in [("8K",8000),("32K",32000),("64K",64000),("110K",110000)]:
            pr=make(tag,target)
            t,ev=complete(pr)
            res["points"][tag]={"prompt_n":t.get("prompt_n"),
                "prefill_tps":round(t.get("prompt_per_second",0),1),
                "prefill_s":round(t.get("prompt_ms",0)/1000,1),
                "decode_tps":round(t.get("predicted_per_second",0),2),
                "draft_n":t.get("draft_n"),"draft_acc":t.get("draft_n_accepted"),
                "vram":vram()}
            d=res["points"][tag]
            acc=(d["draft_acc"]/d["draft_n"]) if d.get("draft_n") else 0
            print(f"   {tag:5} n={d['prompt_n']:>6}  prefill {d['prefill_tps']:>6.1f} t/s "
                  f"({d['prefill_s']:>5.1f}s)  decode {d['decode_tps']:>5.2f} t/s  acc={acc:.3f}",flush=True)
        # 개발 패턴: 같은 64K 프리픽스 재사용
        base=make("64K",64000)
        t1,_=complete(base+"\n// Q1: name one invariant.\n",64)
        t2,_=complete(base+"\n// Q2: name one failure mode.\n",64)
        res["reuse"]={"t1_prompt_n":t1.get("prompt_n"),"t1_prefill_s":round(t1.get("prompt_ms",0)/1000,1),
                      "t2_prompt_n":t2.get("prompt_n"),"t2_prefill_s":round(t2.get("prompt_ms",0)/1000,1)}
        r=res["reuse"]
        print(f"   재사용: 1회차 n={r['t1_prompt_n']} {r['t1_prefill_s']}s "
              f"→ 2회차 n={r['t2_prompt_n']} {r['t2_prefill_s']}s",flush=True)
    except Exception as e:
        res["error"]=repr(e); print("  ERROR",repr(e),flush=True)
    finally: stop(p)
    return res

CONFIGS=[
 dict(name="spec_f16",   kv="f16",  spec=True),
 dict(name="spec_q8",    kv="q8_0", spec=True),
 dict(name="spec_q4",    kv="q4_0", spec=True),
 dict(name="nospec_f16", kv="f16",  spec=False),
]
if __name__=="__main__":
    out=f"{SWEEP}/results_longctx2.json"
    json.dump([run(c) for c in CONFIGS],open(out,"w"),indent=1)
    print("\nWROTE",out)
