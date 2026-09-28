# exllamav3 on sm_70 (Volta): enablement notes and performance trajectory

## Problem

exllamav3 does not target sm_70. On this configuration the first stable sm_70 build delivered ~5.96 t/s pure decode at ~19.5k context -- far below what the memory bandwidth of 3 x V100 suggests. Tensor-parallel execution crashes (see failures below).

## Changes applied

Aggregate patch: `patches/exllamav3/sm70-all-local-changes.patch` (superset of the two focused subset patches).

| File | Change |
|---|---|
| `exllamav3_ext/quant/exl3_gemm.cu` | route `half_k` through the GEMV try-launch path; sm_70 kernel selection plumbing |
| `exllamav3_ext/quant/exl3_gemv.cu` | K 5-8 coverage via sm_70 kernel selection (`exl3_gemv_sm70_select_kernel`); half-rate dispatch (`exl3_gemv_select_kernel_half`, declined below cc 8) |
| `exllamav3_ext/quant/exl3_gemv_int8.cu` | int8 path adjustments for the above |
| `modules/quant/exl3.py` | scratch sized for the max-M launch (m=8 row scratch; avoids an IMA at tiled m>1); dispatch gate: rows <= 8 use GEMV on cc < 8, larger M falls through to reconstruct + hgemm |
| `modules/block_sparse_mlp.py` | sm_70 multi-matrix fan (`FNX_MOE_MULTI=1`, guarded to the non-Ampere path, batch 1): gate/up fan one token row through the selected expert matrices in a single cooperative launch, activation on stacked rows, down fan, single weighted sum; numerics validated in-tree (gate/up exact, down relmax 3.2e-4) |

Subsets:

- `patches/exllamav3/sm70-half-merge.patch` -- half-integer bitrate (1.5/2.5/3.5 bpw, mul1 codebook) plumbing through GEMM/GEMV/GEMM-int8. Declines below cc 8 (no sm_70 half-rate kernels exist).
- `patches/exllamav3/sm70-gemv-scratch-gate.patch` -- the `exl3.py` scratch sizing + dispatch gate changes only.

## Environment

- exllamav3 fork (`sm70-upstream-candidate`); Python 3.12 venv; torch 2.10.0+cu128; CUDA 12.8.
- 3 x Tesla V100 32 GB; same workstation as the rest of this repository.

## Performance trajectory (pure decode, measured)

| Step | t/s @ ~19.5k |
|---|---|
| first stable sm_70 build | 5.96 |
| incremental fixes | 7.56 |
| + MTP draft | 8.24 |
| incremental fixes | 9.17 |
| + `no_reconstruct=False` | 10.58 |
| + multi-fan (`FNX_MOE_MULTI`) | **14.32** (16.15 @ ~2k) |

Intermediate steps were iterations on the same kernels; the surviving delta is the four patch areas above.

## Negative results / known failures

- **Tensor-parallel (expert-parallel) eager execution on sm_70 crashes**: `IndexError` in `block_sparse_mlp.py` (~line 1431) under the NCCL path. Unresolved; parked.
- **MTP under the multi-fan configuration is net-negative at depth**: 10.03 t/s vs 14.32 t/s (MTP off) at ~19.5k.
- Exllamav3 remains ~3.4x slower than the llama.cpp MTP stack on identical hardware at equal context (14.32 vs 48.90 t/s @ ~19.5k). The llama.cpp path is the one kept in production; this work is kept for quant-format access (EXL3).
