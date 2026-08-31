# 03. llama.cpp + DFlash2 — 빌드와 검증

## PR #27342 는 master 에 없습니다

[PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342) "spec: add DFlash2 support
(local convolution + candidate selector)" 는 **merged 상태이지만 base 가 `xsn/dflash2`**
입니다. master 를 빌드하면 DFlash2 가 들어오지 않습니다.

```bash
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git fetch origin pull/27342/head:dflash2
git checkout dflash2
# HEAD = 2f3923bc81346046aa5765dda15fb28497f49ac6  (head repo: z-lab/llama.cpp-fork:dflash2)
```

DFlash2 소스 존재 확인: `src/models/dflash.cpp`, `common/speculative.cpp` 등.

## 빌드

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF
cmake --build build --config Release -j 32 \
      --target llama-server llama-cli llama-bench llama-perplexity
```

- `CMAKE_CUDA_ARCHITECTURES=86` — A40. **다른 GPU면 반드시 바꿔야 합니다.**
- `LLAMA_CURL=OFF` — `libcurl-dev` 부재. 모델은 별도로 받으므로 무방.
- 소요 시간 약 15분 (`-j 32`). CUDA flash-attention 템플릿 인스턴스가 대부분을 차지합니다.

결과: `version: 0.1.2-dev (build 10513, commit 2f3923bc8)`

## 옵션 존재 여부를 먼저 대조했습니다

요구된 옵션이 이 빌드에 실재하는지 `--help` 로 확인한 뒤 실행했습니다. 전부 존재했습니다.

```
--spec-type   none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,
              ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache
--spec-draft-n-max N        (default: 3)
--spec-draft-type-k/-v TYPE
-ngl, --n-gpu-layers N      exact number, 'auto', or 'all'
-fa,  --flash-attn [on|off|auto]
```

## DFlash2 로딩 확인

```
common_speculative_init_result: loading draft model 'Qwen3.8-27B-DFlash2-Q4_K_M.gguf'
common_speculative_impl_draft_dflash: adding speculative implementation 'draft-dflash'
common_speculative_impl_draft_dflash: - n_max=4, n_min=0, p_min=0.00
common_speculative_impl_draft_dflash: - block_size=8, mask_token_id=248070,
                                       n_extract=5, sample_from_anchor=true
```

기동 로그 앞부분에 나오는 아래 두 줄은 **무해합니다.** 메모리 피팅 단계에서만 나오고
이후 draft 모델은 정상 로드됩니다.

```
E llama_init_from_model: failed to initialize the context: dflash requires ctx_other
  to be set (this warning is normal during memory fitting)
W srv load_model: [spec] failed to measure draft model memory
```

## 7항목 검증 결과 (최초 스펙 구성, 128K / `-np 1` / q8_0 KV)

| # | 항목 | 결과 |
|---|---|---|
| 1 | `/health` | `{"status":"ok"}`, HTTP 200 |
| 2 | `/v1/models` 의 `n_ctx` | **131072** (`n_ctx_train` 262144, `n_params` 27.3B) |
| 3 | VRAM | 19,165 / 46,068 MiB |
| 4 | 384토큰 생성 ×3 | 58.99 / 63.70 / 65.59 tok/s |
| 5 | DFlash2 acceptance | 1410 draft 중 795 채택 = **56.4%**, draft당 2.25토큰 |
| 6 | temperature=0 재현성 | 4회 sha256 완전 일치 (`b5286faa9c2e15ed`) |
| 7 | temperature=1.0 샘플링 | seed 미지정 6/6 고유, seed=1234 3/3 동일 |

acceptance 의 위치별 분포:

```
position 0 → 300/354 (84.7%)
position 1 → 219/354 (61.9%)
position 2 → 165/354 (46.6%)
position 3 → 111/354 (31.4%)
```

이 감쇠 곡선을 보고 "n-max 를 늘리면 이득"이라고 예상했지만, docs/04 에서 틀린 것으로
드러납니다.

> 이 구성을 재현하려면 `scripts/run-server-spec.sh`. 성능은 `run-server-dev.sh` 가 더 낫습니다.
