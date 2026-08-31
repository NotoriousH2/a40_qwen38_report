# A40 한 장으로 Qwen3.8-27B 서빙하기

RTX 5090(32GB, sm_120)용으로 공개된 세팅을 **NVIDIA A40(48GB, sm_86)** 에 옮기고,
거기서 출발해 개발용으로 쓸 만한 구성까지 찾아간 기록입니다.
2026-08-31 하루 동안의 실험이고, 모든 수치는 이 저장소의 `results/` 에 원본이 들어 있습니다.

---

## 결론부터

**최종 구성은 llama.cpp + Qwen3.8-27B-UD-Q3_K_XL + DFlash2 투기 디코딩입니다.**
128K 컨텍스트 슬롯 2개를 VRAM 31.5GB 로 돌리면서, 짧은 대화는 70 tok/s,
64K 코드베이스를 물린 상태에서도 65 tok/s 가 나옵니다.

```bash
git clone https://github.com/notorioush2/a40_qwen38_report.git
cd a40_qwen38_report
./scripts/bootstrap.sh          # 빌드·모델이 이미 있으면 40초, 새 머신이면 20~40분
./scripts/verify.sh             # 7항목 검증
```

`bootstrap.sh` 는 상태를 보고 필요한 것만 합니다. 빌드가 있으면 건너뛰고, GGUF 가 있으면 건너뛰고,
서버가 이미 떠 있으면 아무것도 하지 않습니다. 경로와 포트는 전부 `config.env` 한 곳에 있습니다.

| | 값 |
|---|---|
| 모델 | `unsloth/Qwen3.8-27B-GGUF` → `Qwen3.8-27B-UD-Q3_K_XL.gguf` (13.1 GB) |
| 드래프터 | `incoai/Qwen3.8-27B-DFlash2-GGUF` → `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (1.14 GB) |
| 런타임 | llama.cpp PR [#27342](https://github.com/ggml-org/llama.cpp/pull/27342) head (`2f3923b`, build 10513), CUDA |
| 컨텍스트 | 슬롯당 131,072 × 2 슬롯 |
| VRAM | 31.5 / 46.0 GB |
| API | OpenAI 호환, `http://127.0.0.1:8080/v1` |

---

## 이 실험에서 배운 것

숫자만 보면 놓치는 것들이 있어서, 흐름을 짧게 적어 둡니다.
자세한 근거는 각 문서로 연결됩니다.

### 1. NVFP4 세팅은 A40에서 "돌긴 돌았다"

원본 저장소는 RTX 5090의 FP4 텐서코어를 전제로 짜여 있습니다. A40에는 그 하드웨어가 없습니다.
그런데 vLLM이 알아서 Marlin W4A16 커널로 폴백해서, 옵션을 **한 줄도 바꾸지 않고** 그대로 떴습니다.
대신 속도는 절반 수준(53~76 tok/s vs 원본 주장 160 tok/s)이었습니다.

정작 발목을 잡은 건 GPU가 아니라 파이썬이었습니다. 원본 스크립트가 `python3-dev` 유무를
`gcc -E <Python.h>` 로 판별하는데, 이 컨테이너에는 헤더가 `/usr/include/python3.12/` 에 멀쩡히
있고 gcc 기본 탐색 경로에만 없었습니다. 스크립트는 "없다"고 판단해 uv로 CPython **3.14** 헤더를
붙였고, 3.12 인터프리터와 ABI가 어긋나 부팅이 죽었습니다. → [docs/02](docs/02-vllm-nvfp4.md)

### 2. 예상했던 튜닝 두 개가 전부 틀렸다

llama.cpp로 옮긴 뒤, 남는 VRAM 27GB를 어디에 쓸지 16개 조건을 재봤습니다.
제가 유력하다고 본 두 가지가 모두 빗나갔습니다.

- **`-ub` 를 올리면 prefill 이 빨라질 것이다** → 256/1024/2048에서 951/962/939 tok/s. 무변화.
  설정은 확실히 먹혔습니다(VRAM이 19.1→20.3→21.8GB로 비례 증가). 그냥 병목이 아니었습니다.
- **드래프터 `block_size` 가 8이니 `--spec-draft-n-max` 도 8로 올리면 이득일 것이다**
  → 4/6/8에서 67.0/61.4/58.0 tok/s. **올릴수록 느려집니다.** acceptance가 0.564→0.408로
  무너져서 verify 연산만 늘어납니다.

대신 예상 밖의 것이 이득이었습니다. **KV 캐시를 `q8_0`에서 `f16`으로 바꾸니 빨라졌습니다.**
메모리를 더 쓰는데 더 빠른 이유는, KV 양자화 오차가 드래프터와 타깃의 예측을 어긋나게 만들고
있었기 때문입니다. acceptance가 0.564 → 0.663으로 오르면서 속도와 품질이 같이 올라갔습니다.
→ [docs/04](docs/04-tuning-sweep.md)

### 3. 긴 컨텍스트의 병목은 decode가 아니라 prefill이다

개발용으로 64K 이상을 쓰려니 다른 그림이 나왔습니다.

```
컨텍스트    prefill        decode
   8K       8.0초          93 tok/s
  32K      33.2초          94 tok/s
  64K      75.2초          87 tok/s
 110K     149.2초          79 tok/s
```

decode는 110K에서도 멀쩡합니다. **문제는 64K를 한 번 넣는 데 75초**라는 것이고,
이건 튜닝으로 안 됩니다. 737~1116 tok/s는 A40에서 27B Q3 모델의 하드웨어 한계에 가깝습니다.

그래서 답은 "빠르게 만들기"가 아니라 **"두 번 하지 않기"** 였습니다.

```
같은 64K 프리픽스로 이어서 질문:
  1회차   prompt_n=60,009    69.3초
  2회차   prompt_n=   262     0.7초     ← 99배
```

`--cache-reuse 256` 이 프리픽스를 KV shifting으로 재사용합니다. 개발 워크플로는
"같은 코드베이스 + 바뀌는 질문"이라 이 패턴에 정확히 맞습니다.
**고정 컨텍스트를 프롬프트 앞쪽에, 바뀌는 질문을 뒤쪽에** 두는 것만 지키면 됩니다.
→ [docs/05](docs/05-long-context.md)

### 4. Q4로 올릴 이유가 없었다

남는 VRAM으로 양자화 등급을 올리는 게 자연스러워 보였지만, 재보니 손해였습니다.

```
              디코딩(64K 코드)   코드 perplexity   VRAM
Q3_K_XL         65.4 tok/s        1.6866          31.5 GB
Q4_K_XL         41.8 tok/s        1.6707          35.6 GB
                  -36%             -0.94%          +4.1 GB
```

**품질 0.94% 개선에 속도 36%를 지불하는 셈**입니다. 덤으로 Q4에서는 draft acceptance까지
떨어졌습니다(0.609 → 0.465). → [docs/06](docs/06-quant-decision.md)

### 5. 측정을 두 번 틀렸다

이 저장소에서 가장 값어치 있는 문서는 아마 [docs/07](docs/07-measurement-pitfalls.md)일 겁니다.
결론 표만 남기면 사라지는 맥락이라 따로 적었습니다.

- **초단문 생성으로 재면 투기 디코딩이 부당하게 불리하게 나옵니다.** 장문 프롬프트를
  `Reply with the single word: OK` 로 끝내는 바람에 생성이 2~3토큰뿐이었고, 스텝당 고정비만
  측정됐습니다. 이걸 근거로 "장문에서 DFlash2가 42% 느리다"는 **틀린 결론**을 냈다가,
  제대로 재니 오히려 **1.7~2배 빠르다**로 뒤집혔습니다.
- **반복적인 filler 텍스트를 쓰면 acceptance가 1.0으로 부풀어** 최선값만 보게 됩니다.
  실제 C++ 소스로 바꾸니 acceptance가 0.43~0.68로 현실적인 값이 나왔습니다.

---

## 저장소 구성

```
config.env             경로·포트·모델 파일명. 다른 머신으로 옮길 때 여기부터.
scripts/
  bootstrap.sh         ★ 상태 감지 → 필요한 것만 수행 후 기동
  run-server-dev.sh    ★ 권장 프로파일 (모든 측정의 결론)
  run-server-spec.sh   원본 스펙 무수정 재현 (대조용)
  verify.sh            7항목 검증
  build-llamacpp.sh / download-models.sh / stop-server.sh
bench/                 측정에 쓴 스크립트 원본 (재현용)
results/               조건별 원본 수치(JSON) + 서버 로그 + perplexity
docs/                  01~07 상세 기록
vendor/                MiaAI-Lab 원본 저장소 사본 (출처는 ATTRIBUTION.md)
```

| 문서 | 내용 |
|---|---|
| [01 환경](docs/01-environment.md) | 하드웨어·드라이버·디스크 배치 규칙 |
| [02 vLLM NVFP4](docs/02-vllm-nvfp4.md) | 원본 세팅 재현, Marlin 폴백, Python 헤더 버그 |
| [03 llama.cpp DFlash2](docs/03-llamacpp-dflash2.md) | PR head 빌드와 7항목 검증 |
| [04 튜닝 스윕](docs/04-tuning-sweep.md) | 16개 조건 전체 표 |
| [05 장문 컨텍스트](docs/05-long-context.md) | 길이별 곡선, 캐시 재사용 |
| [06 양자화 결정](docs/06-quant-decision.md) | Q3 vs Q4 + perplexity |
| [07 측정 함정](docs/07-measurement-pitfalls.md) | 틀린 측정과 정정 이력 |

## 재현 메타데이터

| 항목 | 값 |
|---|---|
| GPU | NVIDIA A40 46,068 MiB (sm_86) |
| 드라이버 | 580.159.03 |
| CUDA toolkit | 12.8.93 |
| llama.cpp | `2f3923bc81346046aa5765dda15fb28497f49ac6` (build 10513) |
| 빌드 옵션 | `-DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF` |
| vLLM (2장) | 0.27.1 + torch 2.13.0+cu130 |
| 측정일 | 2026-08-31 |

> **다른 GPU로 옮길 때**: `config.env` 의 `CUDA_ARCH` 를 반드시 바꾸세요(A40=86).
> 그리고 여기 수치는 전부 A40 기준입니다 — 대역폭이 다른 카드에서는
> 특히 KV 타입과 `--spec-draft-n-max` 의 최적값이 달라질 수 있습니다.
