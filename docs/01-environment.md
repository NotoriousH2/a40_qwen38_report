# 01. 환경

## 하드웨어

| 항목 | 값 |
|---|---|
| GPU | NVIDIA A40, 46,068 MiB, Compute Capability **8.6 (Ampere)** |
| 드라이버 | 580.159.03 (CUDA 13.0 지원) |
| CUDA toolkit | 12.8.93 (`/usr/local/cuda`) |
| CPU | 96 코어 |
| RAM | 503 GB |
| 로컬 디스크 | 400 GB (overlay), 387 GB 여유 |

A40에 **FP4/FP8 텐서코어가 없다**는 점이 이 실험 전체를 관통하는 제약입니다.
원본 세팅이 전제한 RTX 5090은 sm_120이고 NVFP4 W4A4 커널을 하드웨어로 지원합니다.

## 스토리지 배치 규칙

작업 디렉터리 `/workspace` 는 **FUSE 네트워크 스토리지**입니다. 대량 I/O를 하면 안 됩니다.
그래서 아래 원칙을 지켰습니다.

| 대상 | 위치 | 이유 |
|---|---|---|
| 모델 GGUF (14.3 GB) | `/root/models/gguf` | 로컬 디스크 |
| llama.cpp 빌드 | `/root/build/llama.cpp-dflash2` | 로컬 디스크 |
| vLLM venv (8 GB) | `/root/venvs/qwen38-nvfp4` | 로컬 디스크 |
| HF 캐시 | `/tmp/hf` | 로컬 디스크 |
| 스크립트·문서·결과 | `/workspace/lab/...` | 작아서 무방 |

vLLM 저장소는 `.venv` 를 스크립트 위치에 만들도록 하드코딩돼 있어서,
`/root/venvs/...` 로 **심볼릭 링크**를 걸어 우회했습니다.

## 사전 설치 상태

- llama-server 가 `/opt/llama.cpp` 에 별도로 설치돼 있었습니다. **이 실험은 그것을 건드리지 않고**
  `/root/build/llama.cpp-dflash2` 에 별도 빌드했습니다.
- `libcurl-dev` 가 없어 llama.cpp 빌드 시 `-DLLAMA_CURL=OFF` 를 썼습니다.
  모델은 별도로 받으므로 영향 없습니다.
- `python3-dev` 헤더는 `/usr/include/python3.12/` 에 존재하지만 gcc 기본 탐색 경로에는
  없습니다. 이것이 docs/02 의 버그 원인입니다.
