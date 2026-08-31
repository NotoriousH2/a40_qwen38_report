# 02. vLLM + NVFP4 — 원본 세팅을 A40에 그대로 올리기

원본: [MiaAI-Lab/Qwen3.8-27B-NVFP4-RTX-5090](https://github.com/MiaAI-Lab/Qwen3.8-27B-NVFP4-RTX-5090)
(사본은 `vendor/vllm-nvfp4-rtx5090/`, 출처는 같은 폴더의 `ATTRIBUTION.md`)

목표는 **옵션을 바꾸지 않고** 그대로 뜨는지 확인하는 것이었습니다. 결과적으로 vLLM 인자는
하나도 바꾸지 않았고, 고친 것은 스크립트의 환경 감지 로직 한 곳뿐입니다.

## 모델

`RadixArk/Qwen3.8-27B-NVFP4` (21.95 GB). ModelOpt NVFP4 양자화, arch `Qwen3_5ForConditionalGeneration`.

구조상 특징이 성능 해석에 중요합니다.

- 64층 중 `full_attention_interval: 4` → **16층만 full attention**, 나머지는 linear attention(게이트 델타넷)
- full attention 은 KV heads 4 × head_dim **256**
- `mtp_num_hidden_layers: 1` (MTP 투기 디코딩용)
- 비전 타워 포함(VLM)이지만 이번 기동에서는 멀티모달 입력이 비활성 상태로 떴습니다
- 양자화는 MIXED_PRECISION: MLP 는 NVFP4(group 16), linear_attn 은 FP8

## 부팅 실패 1회 — Python 헤더 ABI 불일치

첫 시도는 이렇게 죽었습니다.

```
SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats
  at triton/compiler/compiler.py:468 in _init_handles
```

원인은 원본 `start.sh` 의 이 판별입니다.

```bash
if ! printf '#include <Python.h>\n' | /usr/bin/gcc -E - >/dev/null 2>&1; then
  # python3-dev 가 없다고 판단 → uv 로 CPython 3.14 를 받아 헤더를 -I 로 붙임
```

이 컨테이너에는 `/usr/include/python3.12/Python.h` 가 **멀쩡히 있습니다.** 다만 gcc 기본 탐색
경로가 아니라 `-I` 없이는 못 찾을 뿐입니다. 스크립트는 "없다"고 오판해 **CPython 3.14 헤더**를
붙였고, Triton JIT 의 C shim 이 3.14 헤더로 컴파일된 채 **3.12 인터프리터**에서 로드되어
ABI가 어긋났습니다.

**고친 방법** — `~/.local/bin/cc-triton` 래퍼가 실행 중인 인터프리터와 같은 3.12 헤더를 먼저
찾도록 바꾸고 Triton 캐시를 비웠습니다.

```bash
#!/usr/bin/env bash
for d in /usr/include/python3.12 "$HOME"/.uv-pythons/*/include/python3.1*; do
  [ -f "$d/Python.h" ] && exec /usr/bin/gcc -I"$d" "$@"
done
exec /usr/bin/gcc "$@"
```

```bash
rm -rf /root/.triton/cache ~/.cache/vllm
```

**이건 RTX 5090이 아니라서 나는 문제가 아닙니다.** 파이썬 헤더가 비표준 경로에 있는
컨테이너라면 어디서든 재현됩니다.

## A40에서 실제로 선택된 커널

두 번째 시도에서 그대로 떴습니다. 로그가 폴백 경로를 명확히 보여줍니다.

```
Selected MarlinFP8ScaledMMLinearKernel for ModelOptFp8LinearMethod
Using MarlinNvFp4LinearKernel for NVFP4 GEMM
WARNING Your GPU does not have native support for FP4 computation ...
        Weight-only FP4 compression will be used leveraging the Marlin kernel.
Using TURBOQUANT attention backend out of potential backends: ['TURBOQUANT']
GPU KV cache size: 286,466 tokens
Maximum concurrency for 262,144 tokens per request: 1.09x
```

- **NVFP4 → Marlin W4A16 폴백.** Blackwell 의 W4A4 커널 대신 4bit 가중치를 매 GEMM 마다
  bf16 으로 역양자화합니다. 따라서 메모리는 아끼지만 **연산 부하가 늘어납니다.**
- **TurboQuant 4bit KV 는 그대로 동작.** Triton 커널 기반이라 아키텍처 제약이 없습니다.
- 원본의 PR #40914 백포트 패치도 그대로 적용되었습니다(anchor 일치).

## 측정 결과

| 항목 | A40 (측정) | 원본 주장 (RTX 5090) |
|---|---|---|
| 단일 스트림 디코딩 | **53~76 tok/s** (886토큰 장문 53.4) | ~160 tok/s |
| 31K prefill | 23.2초 (≈1.3K tok/s) | — |
| 출력 붕괴(garble) | 없음 (한/영, 코드, thinking 모드 정상) | 0/15 |
| VRAM | 28.9 / 46.0 GB | 32GB 카드 기준 |

속도가 절반인 이유는 두 가지입니다. Marlin 역양자화로 **연산 병목**이 생기고,
메모리 대역폭이 **696 GB/s 대 1792 GB/s** 로 차이납니다.

## 이 구성의 한계 (llama.cpp 로 넘어간 이유)

- `--kv-cache-memory-bytes 5905580032` (5.5 GiB) 는 32GB 카드용 핀입니다.
  A40에서는 로그가 `Initial free memory 44.16 GiB, reserved 5.5 GiB` 라고 찍으며
  **17 GB 를 놀립니다.**
- `--max-num-seqs 1` 로 동시성이 1에 묶여 있습니다.
- `enable_prefix_caching=False` — 개발 워크플로에 치명적입니다.

부팅 로그 원본: `results/logs/vllm-nvfp4-boot.log`
