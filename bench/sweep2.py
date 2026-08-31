#!/usr/bin/env python3
"""DFlash2 / llama.cpp 설정 스윕. 각 설정마다 서버를 새로 띄우고 동일 프롬프트로 측정."""
import json, os, signal, subprocess, sys, time, urllib.request

BIN   = "/root/build/llama.cpp-dflash2/build/bin/llama-server"
DRAFT = "/root/models/gguf/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
Q3    = "/root/models/gguf/Qwen3.8-27B-UD-Q3_K_XL.gguf"
Q4    = "/root/models/gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf"
PORT  = 8081                      # 스윕 전용 포트 (8080/8000 과 충돌 방지)
BASE  = f"http://127.0.0.1:{PORT}"
SWEEP = os.path.dirname(os.path.abspath(__file__))

SHORT = "Explain how a hash table works, including collision handling, in clear technical prose."
UNIT  = "Unified memory bandwidth bounds decode throughput on edge accelerators today. "
LONG  = UNIT*2600 + "\nReply with the single word: OK"

def vram():
    o = subprocess.check_output(["nvidia-smi","--query-gpu=memory.used",
                                 "--format=csv,noheader,nounits"]).decode().strip()
    return int(o.splitlines()[0])

def post(path, body, timeout=1800):
    req = urllib.request.Request(BASE+path, data=json.dumps(body).encode(),
                                 headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def chat(msg, max_tokens, temp=0.0):
    b={"model":"m","messages":[{"role":"user","content":msg}],"max_tokens":max_tokens,
       "temperature":temp,"chat_template_kwargs":{"enable_thinking":False}}
    t=time.time(); r=post("/v1/chat/completions", b); return r, time.time()-t

def start(cfg):
    args=[BIN,"-m",cfg["model"],"-c",str(cfg["ctx"]),"-np","1","-ngl","all","-fa","on",
          "--cache-type-k",cfg["kv"],"--cache-type-v",cfg["kv"],
          "-b",str(cfg["b"]),"-ub",str(cfg["ub"]),
          "--jinja","--metrics","--reasoning-preserve",
          "--host","127.0.0.1","--port",str(PORT)]
    if cfg["spec"]:
        args += ["-md",DRAFT,"--spec-type","draft-dflash",
                 "--spec-draft-n-max",str(cfg["nmax"]),
                 "--spec-draft-type-k","f16","--spec-draft-type-v","f16"]
    log=open(f"{SWEEP}/{cfg['name']}.log","w")
    p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
    for _ in range(600):
        if p.poll() is not None:
            return None, f"server exited rc={p.returncode}"
        try:
            urllib.request.urlopen(BASE+"/health",timeout=3).read(); return p, None
        except Exception: time.sleep(1)
    p.kill(); return None, "health timeout"

def stop(p):
    if p is None: return
    p.send_signal(signal.SIGTERM)
    try: p.wait(timeout=30)
    except subprocess.TimeoutExpired: p.kill(); p.wait()
    time.sleep(3)

def run(cfg):
    print(f"\n### {cfg['name']}  {cfg}", flush=True)
    p,err = start(cfg)
    if err:
        tail=open(f"{SWEEP}/{cfg['name']}.log").read()[-1200:]
        print(f"  FAILED: {err}\n{tail}", flush=True)
        return {"name":cfg["name"],"cfg":cfg,"failed":err,"log_tail":tail}
    res={"name":cfg["name"],"cfg":cfg}
    try:
        res["vram_loaded_mib"]=vram()
        # 짧은 컨텍스트 디코딩 x3
        tps=[]; dn=[]; da=[]
        for _ in range(3):
            r,dt = chat(SHORT,384)
            t=r.get("timings",{})
            tps.append(t.get("predicted_per_second"))
            dn.append(t.get("draft_n") or 0); da.append(t.get("draft_n_accepted") or 0)
        res["decode_tps_short"]=tps
        res["decode_tps_short_med"]=sorted(tps)[1]
        res["draft_n"]=dn[0]; res["draft_acc"]=da[0]
        res["accept_rate"]=(da[0]/dn[0]) if dn[0] else None
        # 31K prefill + 장문 컨텍스트 디코딩
        r,dt = chat(LONG,128)
        t=r.get("timings",{})
        res["prefill_n"]=t.get("prompt_n"); res["prefill_tps"]=t.get("prompt_per_second")
        res["decode_tps_long"]=t.get("predicted_per_second")
        res["vram_peak_mib"]=vram()
        # per-position acceptance
        m=urllib.request.urlopen(BASE+"/metrics",timeout=30).read().decode()
        pos={}
        for line in m.splitlines():
            if "accepted_tokens_per_pos_total" in line and not line.startswith("#"):
                k=line.split('position="')[1].split('"')[0]; pos[k]=float(line.split()[-1])
            if line.startswith("llamacpp:spec_decode_num_drafts_total"): res["drafts_total"]=float(line.split()[-1])
            if line.startswith("llamacpp:spec_decode_num_draft_tokens_total"): res["draft_tok_total"]=float(line.split()[-1])
            if line.startswith("llamacpp:spec_decode_num_accepted_tokens_total"): res["acc_tok_total"]=float(line.split()[-1])
        res["per_pos"]=pos
        print("  ", json.dumps({k:v for k,v in res.items() if k not in("cfg",)}), flush=True)
    except Exception as e:
        res["error"]=repr(e); print("  ERROR",e, flush=True)
    finally:
        stop(p)
    return res

CONFIGS=[
 dict(name="I_n4_f16kv",   model=Q3, ctx=131072, kv="f16",  b=2048, ub=256, spec=True,  nmax=4),
 dict(name="J_n3",         model=Q3, ctx=131072, kv="q8_0", b=2048, ub=256, spec=True,  nmax=3),
 dict(name="K_n2",         model=Q3, ctx=131072, kv="q8_0", b=2048, ub=256, spec=True,  nmax=2),
 dict(name="L_nospec_f16", model=Q3, ctx=131072, kv="f16",  b=2048, ub=256, spec=False, nmax=0),
 dict(name="M_q4_n4",      model=Q4, ctx=131072, kv="q8_0", b=2048, ub=256, spec=True,  nmax=4),
 dict(name="N_q4_n4_f16",  model=Q4, ctx=131072, kv="f16",  b=2048, ub=256, spec=True,  nmax=4),
]
if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="stage2":
        best=json.load(open(f"{SWEEP}/best.json"))
        CONFIGS=[dict(best, name="H_q4kxl", model=Q4)]
        out=f"{SWEEP}/results_stage2.json"
    else:
        out=f"{SWEEP}/results2.json"
    results=[run(c) for c in CONFIGS]
    json.dump(results, open(out,"w"), indent=1)
    print("\nWROTE", out)
