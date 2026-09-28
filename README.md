# V100 Inference Notes

Measured notes on serving large MoE language models on Volta (sm_70) GPUs.

## Scope

- Single reference workstation; every number is a measurement from that configuration, not a general benchmark.
- Format: problem statement -> solution -> measured results -> negative results.
- No support is provided; verify everything on your own hardware.

## Reference configuration

| Component | Specification |
|---|---|
| Compute GPUs | 3 x Tesla V100 32 GB (sm_70) |
| Display GPU | 1 x low-power card, excluded from compute (see enumeration notes) |
| Workstation | Dell Precision T7910 class, dual Xeon |
| OS / toolchain | Ubuntu (console); CUDA 12.8; nvcc 12.8.93 |
| Model focus | Qwen3.8-Flash-Next (qwen4exp architecture), GSQ-RCO GGUF quantizations; Ornith-1.5-35B-A3B Q8_0 |

## Documents

| File | Topic |
|---|---|
| [docs/01-mtp-speculative-decode-sm70.md](docs/01-mtp-speculative-decode-sm70.md) | MTP speculative decoding for qwen4exp: loader/export fixes, recipe, measured results |
| [docs/02-exllamav3-sm70.md](docs/02-exllamav3-sm70.md) | exllamav3 on sm_70: patches, performance trajectory, known failures |
| [docs/03-benchmarking-methodology.md](docs/03-benchmarking-methodology.md) | Measurement protocol: contamination detection, token-weighted evaluation |
| [docs/04-workstation-config-tables.md](docs/04-workstation-config-tables.md) | Configuration tables: KV, micro-batch, device enumeration, expected fallbacks |

## Patches

| File | Target |
|---|---|
| [patches/llama.cpp/qwen4exp-mtp-stack.patch](patches/llama.cpp/qwen4exp-mtp-stack.patch) | llama.cpp qwen4exp lineage @ f3f1a8f27: NextN/MTP support + detached-head loader |
| [patches/exllamav3/sm70-all-local-changes.patch](patches/exllamav3/sm70-all-local-changes.patch) | exllamav3: aggregate sm_70 changes (GEMV / GEMV-int8 / GEMM CU, dispatch, MoE fan) |
| [patches/exllamav3/sm70-half-merge.patch](patches/exllamav3/sm70-half-merge.patch) | exllamav3: half-integer bitrate plumbing (subset) |
| [patches/exllamav3/sm70-gemv-scratch-gate.patch](patches/exllamav3/sm70-gemv-scratch-gate.patch) | exllamav3: GEMV scratch sizing + dispatch gate (subset) |

## Headline results (same 3x V100 configuration)

| Setup | Decode t/s @2.4k fill | Decode t/s @19.5k fill |
|---|---|---|
| llama.cpp, no speculation (control, same binary) | 38.97 | -- |
| llama.cpp + MTP draft head | 56.84 | 47.20 |
| llama.cpp + MTP, production shape | 55.15 | 48.90 |
| exllamav3, best sm_70 build | 16.15 @2k | 14.32 |

Patches retain upstream attribution; use under the upstream licenses (llama.cpp and exllamav3 are MIT-licensed). All numbers measured with the protocol described in docs/03.
