# MTP speculative decoding for qwen4exp on sm_70

## Problem

Qwen3.8-Flash-Next (qwen4exp architecture) ships a single-layer multi-token-prediction (MTP) head that can drive speculative decoding. Two independent blockers prevented loading and running a detached MTP head GGUF:

1. **No detached-head load path in the loader.**
   - `check_tensor_dims: tensor 'blk.0.hc_attn_norm.weight' not found` -- the loader probes for trunk tensors that a head-only file does not contain.
2. **Unstable head export naming across GGUF exports.**
   - `done_getting_tensors: wrong number of tensors` -- older exports use `shared_head_norm` style names; the working export keeps the model-level output mixer (`output_hc_*`) and loads as 37 tensors.

## Solution

Applied stack (llama.cpp qwen4exp fork lineage, base `f3f1a8f27`):

| Step | Change | Source |
|---|---|---|
| 1 | NextN/MTP draft head support: gguf-py tensor registration, model load path, GGUF export | upstream PR #27836 lineage (3 commits) |
| 2 | Detached qwen4exp MTP head loading (+ hipCUB top-k on HIP) | commit `4125c1393` (credited upstream: crusaderky `a82a58a`; PR #26592) |

Merge notes (conflict resolution as applied):

- `src/models/qwen4exp.cpp`: skip the per-layer-embedding (PLE) block for detached heads, gated by `mtp_only` -- a detached head describes no PLE table and carries none.
- `tests/test-backend-ops.cpp`: keep the newer base version of the `test_top_k` cases.

The combined diff `f3f1a8f27..HEAD` is attached: `patches/llama.cpp/qwen4exp-mtp-stack.patch` (976 lines).

Head files:

- Working: 37-tensor export of the MTP head that keeps the model-level output mixer (Q8_0, ~4.14 GB, used for bench arms; Q4_K_M of the same head used for full-context production).
- Failing: older 34/35-tensor exports (`shared_head_norm` naming) -> wrong-tensor-count at load.

## Reproduction

1. Base tree at `f3f1a8f27` (qwen4exp fork lineage).
2. Apply `patches/llama.cpp/qwen4exp-mtp-stack.patch`.
3. Build for sm_70 (e.g. `-DCMAKE_CUDA_ARCHITECTURES=70`), CUDA 12.x.
4. Serve (measured configuration):

```
llama-server -m <model-IQ3_XXS-00001-of-00002.gguf> --alias main
  -ngl 999 -ts 1,1,1 -sm layer
  -c 262144 -np 2 -b 4096 -ub 4096
  -fa on -ctk f16 -ctv f16
  --lazy-mode on --load-mode none
  -md <mtp-head-Q4_K_M.gguf>
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.0
  --jinja --chat-template-file <template.jinja>
  --host 0.0.0.0 --port 8081
```

Startup notes:

- On this VRAM budget the server logs `failed to allocate compute buffers -- retrying without pipeline parallelism` and then loads healthy. This is the expected fallback, not a crash.
- Verify the head loads as 37 tensors and the log reports the draft model before measuring.

## Results (measured)

Fill = prompt tokens; ~300 generated tokens per arm; temperature 0; `ignore_eos`; fresh evaluations only (see docs/03).

| Arm | 2.4k fill | 19.5k fill |
|---|---|---|
| no speculation (control) | 38.97 t/s | -- |
| draft-mtp | 56.84 t/s | 47.20 t/s |
| draft-mtp + ngram-mod chain | 56.82 t/s | 48.88 t/s |
| production shape (np2 / -c 262144 / ub 4096) | 55.15 t/s | 48.90 t/s |

- Prompt processing (prefill): 339-356 t/s production shape; up to ~425 t/s on plain text.
- Acceptance: ~70-76 %; mean accepted length ~3.1-3.3 per verify window on probe-style arms; per-position acceptance ~0.83 / 0.72 / 0.63.
- Token-weighted production traffic: mean accepted length ~2.6 -- probe-style gains can wash out on real traffic mixes; see docs/03 for implications.
- The ngram-mod chain produced 0 additional drafts on summarization-style traffic (inert here; kept for other workloads).

## Open items / negative results

- Draft gating implemented but untested here: `--spec-draft-p-min` (draft chain stops when the draft head's top-token probability is below the threshold) and window size 4. Candidate arm: `--spec-draft-n-max 4 --spec-draft-p-min 0.5`.
- Gains are larger at shallow context; always verify at the target context depth, not short-context warm states.
