# Serving playbook: flags and decisions (model-agnostic)

Distilled from the launchers in [doc 05](05-models.md) and results in this repository. Target hardware: 3 x V100 32 GB (96 GB VRAM total), dual-socket host, one model at a time per port.

**Sections:** [1. Device selection](#1-device-selection) · [2. Attention & KV cache](#2-attention-and-kv-cache) · [3. Batch sizes](#3-batch-sizes) · [4. Context & slots](#4-context-and-slots) · [5. Speculative decoding](#5-speculative-decoding-decision-table) · [6. Reasoning flags](#6-reasoning-flags) · [7. Memory & load modes](#7-memory-and-load-modes) · [8. Ops pattern](#8-ops-pattern) · [9. Sampling presets](#9-sampling-presets-used) · [10. Vision](#10-vision-mmproj) · [11. Prefix caching](#11-prefix-caching-and-lan-clients) · [12. Glossary](#12-glossary-what-every-flag-in-the-launchers-does)

## 1. Device selection

- Keep the display GPU out of compute. Pin explicitly, e.g. `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2,3`; verify with `llama-server --list-devices`.
- Layer split (`-ts 1,1,1 -sm layer`) is the default winner on PCIe-only V100s. Tensor split is slow on 3 GPUs (allreduce crosses PCIe/inter-socket links).
- Alternative: `-dev CUDA0,CUDA1,CUDA2 -devd ...` combined with `-ot <tensor pattern>=<device>` when specific tensors must land on specific GPUs (see doc 05 section 6).

## 2. Attention and KV cache

- `-fa on`: works on sm_70 with f16 KV; keep it on.
- `-ctk f16 -ctv f16`: standard here. A legacy lane used `q8_0` KV to save VRAM; on Volta there is no fast int8 compute path, so treat q8_0 KV as a space-saving fallback, not a speedup.
- Speculative draft models keep their own caches: `--spec-draft-type-k f16 --spec-draft-type-v f16`.

## 3. Batch sizes

- Dense-ish 27-35B Q8 at np2/np3: `-b 2048 -ub 512` (VRAM-lean) up to `-ub 4096`.
- Large-context MoE: `-ub 4096` is the knee for prefill on this box; prefill scales 615 -> 1118 t/s from ub 512 -> 8192 at +~11 GB/card compute scratch at ub 8192 (ladder in doc 04).
- Time-to-first-token, 49k-token prompt: 86.6 s (ub 512) -> 50.8 s (ub 4096).

## 4. Context and slots

- `-c` is total; per-slot = `-c / -np`. Shapes used here: 262144/np2, 480000/np3 (160k/slot), 786432/np3 (262k/slot).
- KV total scales with `-c`; compute scratch scales with per-slot context and `ub`. Budget tables in doc 04.

## 5. Speculative decoding decision table

| Type | Needs | Notes |
|---|---|---|
| `draft-mtp` | MTP head -- separate `-md` file or in-model head | Best decoded-side win at low-mid depth (+45% at 2.4k for Qwen3.8-Flash-Next); turned off for Ornith (net -22% at n_max 3) |
| `draft-dflash` (+ `ngram-map-k`) | draft GGUF (`-md`) | Used for ThinkingCap-27B (n_max 7) |
| `ngram-*` | nothing | ngram-mod produced 0 drafts on this traffic; ngram-map-k chains with dflash |

- Knobs: `--spec-draft-n-max` (3 typical; 6-7 when drafts are strong), `--spec-draft-n-min 1`, `--spec-draft-p-min` for conservative gating (a legacy lane uses 0.75).
- Verify token-weighted (doc 03): probe-style acceptance overstates production gains.

## 6. Reasoning flags

Pattern used by all thinking models here:

```
--reasoning on --reasoning-effort <low|medium|xhigh> --reasoning-preserve --reasoning-format deepseek
[--reasoning-budget N --reasoning-budget-message "..."]
```

## 7. Memory and load modes

- `--load-mode dio` (O_DIRECT) standard; `mmap` for smaller models; `none` when the model is expected hot in page cache.
- `--lazy-mode`: reads large embedding tables on demand -- ~474 MB/s cold from disk, RAM-speed on repeat. Budget system RAM accordingly (~188 GiB here; ~123 GiB observed in page cache at steady state).
- Oversized MoE: `-ot exps=CPU` + `--numa distribute` + `--moe-cache on` (DeepSeek-V4-Flash reference in doc 05). CPU-side MoE is DDR4-bandwidth-bound: ~105 t/s prefill / ~16 t/s decode observed.
- Launchers set `-fit off` and size VRAM explicitly with the flags above.

## 8. Ops pattern

- One model per port; swap = stop old instance, start new instance, wait for `/health` (60-90 s load for these sizes), then verify VRAM and the `n_slots` line in the log.
- Give every model instance a distinct unit name -- a stale shared name can make a health-wait latch onto a different instance (doc 03).
- `--metrics` on every instance for token/latency counters.

## 9. Sampling presets used

| Model family | temp | top-p | top-k | extra |
|---|---|---|---|---|
| Qwen3.8-Flash-Next (thinking) | 1.0 | 0.95 | 20 | -- |
| Ornith-1.5-35B-A3B | 1.0 | 0.95 | 20 | -- |
| Qwen3.8-35B-A3B-Distill | 0.6 | 0.95 | 20 | -- |
| Qwythos-9B | 0.6 | 0.95 | 20 | repeat-penalty 1.05 |
| ThinkingCap-27B | 1.0 | 0.95 | 20 | effort xhigh |

Values come from model cards / GGUF metadata where present; launchers pass them explicitly.

## 10. Vision (mmproj)

- Load `--mmproj <projector.gguf>`; f16/BF16 projectors add ~1.1 GB per instance.
- `--image-min-tokens 1024` used across launchers.
- Verified working on sm_70 (clip-style projector on V100); keep the display GPU out of the split as usual.

## 11. Prefix caching and LAN clients

- Prompt caching is ON by default (`--cache-prompt`); the cache is per slot and hits require a byte-stable prompt prefix (system prompt + tool schemas + history matching from position 0).
- Instrumentation: response `timings.cache_n` = tokens served from cache, `timings.prompt_n` = tokens actually evaluated. Example on the reference box: identical follow-up request reused 422 / 426 tokens (wall 1.82 s -> 0.14 s).
- Prompt-cache budget: `--cache-ram` (upstream default 8192 MiB; launchers here set 65536 MiB = 64 GiB; host RAM is ~188 GiB). Idle slots park in this host-RAM cache (`--cache-idle-slots`, default on) and are restored for later tasks; a larger budget retains more prefixes across clients.
- Eviction is size-based LRU, not a timeout: nothing expires by clock; entries drop only when new saves exceed the budget (`removing oldest entry`). At the previous 16384 MiB budget this bit continuously: saved states carry up to 32 context checkpoints per slot (~130-150 MiB each at ~20k ctx, up to 362 MiB at depth), so single entries reach multi-GB (1.1-5.5 GiB eviction events observed). Symptom: a resume after a 22-min pause reused only 7123/21600 tokens (~7.5k tokens re-prefilled, ~21 s), then 2-3 catch-up turns; in-flow turns on the same session reused 17-19k tokens per turn.
- Fix deployed 2026-09-29 on all launchers: `--cache-ram 65536` plus checkpoint tuning `--ctx-checkpoints 16 --checkpoint-min-step 16384` (halves checkpoint count and creation rate; 16 x 16384 = 262144 >= the 131072 per-slot ctx).
- `--cache-reuse` (interior-chunk reuse via KV shifting; upstream default 0): **unavailable under multimodal** -- with an mmproj loaded the server force-disables `cache_reuse` and `ctx_shift` at startup. These launchers run vision, so it stays unset; text-only runs may add `--cache-reuse 256` (value: reuse of non-prefix shared chunks, e.g. clients that truncate mid-history).
- Context overflow: `--context-shift` is disabled by default (and under multimodal); clients manage conversation length themselves; the per-slot `-c` value is the hard budget.
- Cold start: caches are empty after a restart or model swap. A fixed known prefix can be pushed with a small prewarm script after restart so the first real turn skips cold prefill.
- Optional: `--slot-save-path` plus `/slots` API `save`/`restore` persist individual slot caches to disk (manual; not enabled here).
- Already on everywhere: SSE keep-alive pings (`--sse-ping-interval 30`), `--metrics`, `/slots` monitoring, `return_progress` streaming, `--timeout` 3600 s.

## 12. Glossary: what every flag in the launchers does

One-line meanings; deeper notes in the sections above.

| Flag | Meaning |
|---|---|
| `-m <file>` | model file (the first shard of a split model). |
| `-md <file>` | draft model for speculative decoding. |
| `-ngl N` / `-ngld N` | layers offloaded to GPU: main model / draft. |
| `-ts 1,1,1` / `-sm layer` | tensor split across devices / split mode; `layer` runs whole layers per GPU. |
| `-dev` / `-devd` | explicit device lists for the main model / draft (§1). |
| `-c N` / `-np N` | total context / parallel slots (per-slot context = `-c / -np`). |
| `-b N` / `-ub N` | logical batch / microbatch (`-ub` drives prompt chunking; ladder in doc 04). |
| `-fa on` | flash attention (§2). |
| `-ctk` / `-ctv` | KV cache data type for K / V (§2). |
| `-t N` / `-tb N` | CPU threads / threads for batch work. |
| `--alias main` | the model name the API lists and answers to. |
| `--host` / `--port` | bind address and port (`0.0.0.0:8081` is the fleet convention). |
| `--metrics` | Prometheus counters on `/metrics` (§8). |
| `--numa distribute` | spread work across both CPU sockets' memory (§7). |
| `--lazy-mode on` | read large tables on demand from disk instead of pinning them (§7). |
| `--load-mode dio\|mmap\|none` | how model data is read: O_DIRECT / memory-map / assume page-cache hot (§7). |
| `-fit off` | disable automatic VRAM fitting; size everything explicitly (§7). |
| `--no-host` | do not place model buffers in host RAM. |
| `--jinja` | use the chat template shipped inside the GGUF. |
| `--chat-template-file <f>` | override the template with a file. |
| `--reasoning ...` | thinking controls (§6). |
| `--reasoning-budget N` / `--reasoning-budget-message` | cap thinking length / the message shown when it is hit. |
| `--temp` / `--top-p` / `--top-k` | sampling (§9). |
| `--repeat-penalty F` / `--n-predict N` | repetition penalty / maximum generated tokens. |
| `-ot '<pattern>=<device>'` | force tensors onto specific devices (§1, doc 05 §6). |
| `--mmproj <f>` / `--image-min-tokens N` | vision projector / image token budget (§10). |
| `--spec-type <t>` | speculative stage type: `draft-mtp`, `draft-dflash,ngram-map-k`, ... (§5). |
| `--spec-draft-n-max` / `-n-min` / `-p-min` | draft length bounds / confidence gate (§5). |
| `--spec-draft-type-k` / `-v` | KV cache types for the draft's own cache. |
| `-lv N` | log verbosity. |
| env `LLAMA_ATTN_ROT_DISABLE=1` | required for the QSA indexer on this architecture. |
