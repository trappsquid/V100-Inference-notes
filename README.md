# V100 Inference Notes

Measured notes from a 3 x Tesla V100 (sm_70) workstation running large language models locally: which models run, how they are configured, what the hardware can and cannot do, and how results were verified.

**Current stack (2026-10-05):** the :8081 slot is running the **Strata engine** (doc 07) with Qwen3.8-Flash-Next -- currently the abliterated Q8_0 spill test (vision on, 2 batch slots), after the uncensored IQ4_XS and GSQ-RCO IQ3_XXS Strata deployments. The llama.cpp and exllamav3 material in this repository documents the alternate stacks (swapped in on demand; one model at a time).

## The machine

| Component | Detail |
|---|---|
| Workstation | Dell Precision T7910, dual socket |
| Compute GPUs | 3 x Tesla V100 32 GB (PG500-216), PCIe Gen3 x16, no NVLink |
| Display GPU | Quadro P400 -- excluded from all serving |
| CPU | 2 x Intel Xeon E5-2697A v4 (16C / 32T each; 2 NUMA nodes) |
| RAM | ~188 GiB ECC DDR4 (two sockets, ~94 GiB per NUMA node) |
| Storage | SATA SSDs; models on a 932 GB NTFS volume + a 400 GB ext4 LVM volume |
| CUDA | 12.8 (ceiling for sm_70: CUDA <= 12.x; CUDA 13 removed Volta) |
| Serving | Strata engine for Qwen3.8-Flash-Next (doc 07, patches/strata/) -- the current :8081 stack. Alternate stacks: llama.cpp fork builds (master + qwen4exp MTP stack, see patches/) and exllamav3 fork experiments. One model at a time on :8081. |

## Models in rotation

Snapshot 2026-10-05. Decode rates without a fill depth are in-service log samples (see doc 05); campaign numbers follow the protocol in doc 03. Prompt read: `a -> b` = short -> long fill ramp where measured; single values are samples; `--` = not recorded.

| Model | Quant | Size | Context / slots | Spec decode | Decode t/s | Prompt read t/s | Status |
|---|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next abliterated (Strata lane) | Q8_0 | ~189 GB | 262144 / 2 slots | MTP head (--spec 4) | 72.5 @2.4k; 66.0 @20k fill | 458 -> 1,578 | **served now** -- spill test (doc 07) |
| Qwen3.8-Flash-Next Uncensored (Strata lane) | IQ4_XS | 92 GB | 262144 / 2 slots | MTP head (--spec 4) | 85.3 @2.5k; ~82 @18k fill | 707 -> 1,772 | Strata deployment, 2026-10-04 |
| Qwen3.8-Flash-Next GSQ-RCO (Strata lane) | IQ3_XXS + MTP head | ~107 GB | 262144 / 2 slots | MTP head (--spec 4) | 77.6 @2.0k; 73.9 @17k fill | 656 -> 1,158 | first Strata deployment, 2026-10-04 |
| Qwen3.8-Flash-Next (GSQ-RCO) | IQ3_XXS + MTP head | ~107 GB | 262144 / np2 | draft-mtp | 55.2 @2.4k; 48.9 @19.5k fill | 339-425 | llama.cpp lane (idle; Strata holds :8081 since 2026-10-04) |
| Qwen3.8-35B-A3B-Distill | Q8_0 | 36 GB | 262144 / np2 | draft-mtp (in-model head) | ~144 (sample) | ~544 | active, vision |
| Ornith-1.5-35B-A3B | Q8_0 | 36 GB | 480000 / np3 (160k/slot) | none (-22% measured) | ~83-88 | 615 -> 1,118 (ub ladder) | active, vision |
| ThinkingCap-Qwen3.8-27B | Q8_0 + DFlash2 draft | 28 GB | 262144 / np2 | draft-dflash + ngram-map-k | ~46 (sample) | ~432 (990-tok sample) | active, vision |
| Qwythos-9B (1M ctx) | Q8_0 | 9.2 GB | 786432 / np3 (262k/slot) | draft-mtp | ~65 (sample) | ~2,114 | newest lane |
| Qwen3.8-Flash-Next Uncensored | IQ4_XS | 92 GB | 524288 / np2 | draft-mtp (p-min 0.75) | -- | -- | legacy llama.cpp lane |
| DeepSeek-V4-Flash | Q4_K, experts on CPU | ~364 GB | 524288 / np2 | none | ~16 | ~105 | parked |

## Documents

| File | Topic |
|---|---|
| [docs/00-hardware-platform.md](docs/00-hardware-platform.md) | Platform: dual socket, NUMA, PCIe topology, storage |
| [docs/01-mtp-speculative-decode-sm70.md](docs/01-mtp-speculative-decode-sm70.md) | MTP speculative decoding for qwen4exp: loader/export fixes, recipe, measured results |
| [docs/02-exllamav3-sm70.md](docs/02-exllamav3-sm70.md) | exllamav3 on sm_70: patches, performance trajectory, known failures |
| [docs/03-benchmarking-methodology.md](docs/03-benchmarking-methodology.md) | Measurement protocol: contamination detection, token-weighted evaluation |
| [docs/04-workstation-config-tables.md](docs/04-workstation-config-tables.md) | Config tables: KV, micro-batch, device enumeration, expected fallbacks |
| [docs/05-models.md](docs/05-models.md) | Per-model serving configurations (sanitized startup configs) and results |
| [docs/06-serving-playbook.md](docs/06-serving-playbook.md) | Flag reference and decisions (model-agnostic) |
| [docs/07-strata-sm70.md](docs/07-strata-sm70.md) | Strata engine on sm_70 (current stack): setup, measured results, PLE n-gram table findings, limits |

## Patches

| File | Target |
|---|---|
| [patches/llama.cpp/qwen4exp-mtp-stack.patch](patches/llama.cpp/qwen4exp-mtp-stack.patch) | llama.cpp qwen4exp lineage @ f3f1a8f27: NextN/MTP support + detached-head loader |
| [patches/exllamav3/sm70-all-local-changes.patch](patches/exllamav3/sm70-all-local-changes.patch) | exllamav3: aggregate sm_70 changes (GEMV / GEMV-int8 / GEMM CU, dispatch, MoE fan) |
| [patches/exllamav3/sm70-half-merge.patch](patches/exllamav3/sm70-half-merge.patch) | exllamav3: half-integer bitrate plumbing (subset) |
| [patches/exllamav3/sm70-gemv-scratch-gate.patch](patches/exllamav3/sm70-gemv-scratch-gate.patch) | exllamav3: GEMV scratch sizing + dispatch gate (subset) |
| [patches/strata/v0.1.39-split-metadata-shard.patch](patches/strata/v0.1.39-split-metadata-shard.patch) | Strata v0.1.39: validate the architecture only on a split's metadata shard |
| [patches/strata/v0.1.39-ple-q8_0-rows.patch](patches/strata/v0.1.39-ple-q8_0-rows.patch) | Strata v0.1.39: Q8_0 PLE n-gram rows (170 B) on both readers |

## Headline results

Strata lane (current stack; doc 07) -- depth is stated per row:

| Setup | Decode t/s @ short fill | Decode t/s @ long fill | Prompt read t/s |
|---|---|---|---|
| Strata, uncensored IQ4_XS, all-resident (fills 2.5k / 18k) | 85.3 | 81.8 | 707 -> 1,772 |
| Strata, GSQ IQ3_XXS, all-resident (fills 2.0k / 17k) | 77.6 | 73.9 | 656 -> 1,158 |
| Strata, abliterated Q8_0, 67% expert residency -- served now (fills 2.4k / 20k) | 72.5 | 66.0 | 458 -> 1,578 |

llama.cpp / exllamav3 stacks:

| Setup | Decode t/s @2.4k fill | Decode t/s @19.5k fill |
|---|---|---|
| llama.cpp, no speculation (control, same binary) | 38.97 | -- |
| llama.cpp + MTP draft head (bench shape) | 56.84 | 47.20 |
| llama.cpp + MTP (production shape) | 55.15 | 48.90 |
| exllamav3, best sm_70 build | 16.15 @2k | 14.32 |

Prefill: 339-425 t/s on the llama.cpp arms (doc 01; speculation affects generation only); exllamav3 prefill not recorded.

Patches retain upstream attribution; use under the upstream licenses (llama.cpp, exllamav3 and Strata are MIT-licensed). All numbers are single-machine measurements from the reference configuration, per the protocol in docs/03.
