# Strata (sm_70): Qwen3.8-Flash-Next on 3 x V100

Third engine on the reference workstation, alongside the llama.cpp fork (docs 01, 05) and exllamav3 (doc 02). Strata ([Niko1221/Strata](https://github.com/Niko1221/Strata)) is a dedicated engine for Qwen3.8-Flash-Next: MoE experts live across VRAM and host RAM, and the model's 51B-parameter PLE n-gram table streams from SSD a few rows per token. OpenAI- and Anthropic-compatible APIs, an optional image encoder, and MTP speculative decode with the model's own draft head. Verified on engine **v0.1.39**; sm_70 runs via the experimental build path (`STRATA_EXPERIMENTAL_SM60=1`, CUDA 12.x -- CUDA 13 dropped Volta).

Deployments on this box, in order: GSQ-RCO IQ3_XXS (2026-10-04, 2- and 3-card), uncensored IQ4_XS (2026-10-04), abliterated Q8_0 (2026-10-05, spill test -- served live at write time, vision on, 2 slots).

Fill depths are stated with every number; protocol per doc 03. One model at a time on :8081, same as the other lanes.

**Sections:** [Results](#results) · [Serving posture](#serving-posture) · [Flag reference](#flag-reference-every-field-in-the-startup-configs) · [PLE n-gram table](#ple-n-gram-table-read-path-and-encodings) · [Negative results and limits](#negative-results-and-limits) · [Patches](#patches)

## Results

| Model / quant | Expert residency | Decode @ short | Decode @ long | Prompt read | Spec accept |
|---|---|---|---|---|---|
| GSQ-RCO IQ3_XXS, 2 cards | 100% (24,576) | 77.6 @2.0k | 73.9 @17k | 656 -> 1,158 t/s | 76-80% |
| GSQ-RCO IQ3_XXS, 3 cards | 100% | 76.1 @2.0k | 73.0 @17k | 643 -> 1,168 t/s | 76-80% |
| Uncensored IQ4_XS (single file) | 100% | 85.3-86.8 @2.5k | ~82 @18k | 707 -> 1,772 t/s | ~76% |
| Abliterated Q8_0 (6 shards) | 67% (16,450) | 72.5-73.8 @2.4k | 66.0-66.2 @20k | 458 -> 1,578 t/s | 74-78% |

For scale, the llama.cpp fork's production lane for the same model family on the same cards runs 55.15 t/s @2.4k fill / 48.90 @19.5k, prefill 339-425 t/s (doc 05). Strata's prompt read is 1.9-4.2x that; decode is above it at every depth tested.

Notes:

- The Q8_0 row is the currently served file and a deliberate spill-condition test: its experts (~120 GiB) exceed the ~75 GiB 3-card budget, so the engine filled 16,450 / 24,576 expert slots (5,524 / 5,631 / 5,295 per card; ~446 MiB VRAM spare) and served the rest from disk/RAM. At the tested depths the non-resident set barely engaged -- 99.1-99.3% expert-cache hit, ~0.0% fetched over PCIe -- because the profile-ranked fill captured the hot set. The decode delta vs the all-resident IQ4_XS (-15% / -19%) is the quant's ~2x expert bytes, not spill I/O.
- The same model under a 49%-residency cap (92.9% hit) ran 73.6 vs 85.3 t/s (-13.7%): spill cost appears only when the working set actually rotates.
- Both Q8_0 figures required two local engine fixes (patches/strata/): split-shard metadata validation and Q8_0 PLE rows. Stock v0.1.39 refuses both.
- Vision: red-circle smoke, 1.3-3.4 s depending on model/load; the encoder warms at 1024 image tokens.
- Concurrency: 2 batch slots on IQ3_XXS run 37-41 t/s per stream (75-82 aggregate) at 100% expert-cache hit.
- Boot: ~50-60 s (IQ3_XXS, all-resident) to ~3-5 min (single-file IQ4_XS / 6-shard Q8_0, first fill). IQ3_XXS re-fill is ~4 GiB/s.

## Serving posture

```
strata --serve --pack <pack-dir>
  --native <shard1.gguf> [--ple-gguf <shard2.gguf>]
  --gpu 1,2,3 --layer-split <a,b>
  --kv int8 --kv-resident 32768 --max-context 262144
  --spec 4 --spec-min-p 0.5 --mtp <mtp-dir>
  --batch 2 --batch-groups 2 --trim-stage-weights
  --vision --vram-reserve-mib 700
```

Per-model sanitized startup configs (all three deployments, with the runner and every flag): [doc 05 §8](05-models.md#8-strata-lane-separate-engine----current-8081-stack).

- **Device mask**: `--gpu 1,2,3` under `CUDA_DEVICE_ORDER=PCI_BUS_ID`. The display GPU (P400, sm_61) enumerates last and must stay out of the mask -- on any engine, the same masking rule as doc 06.
- **Layer split**: pinned explicitly (`--trim-stage-weights` requires it); byte-balanced by summing per-layer `*_exps` tensor bytes from the GGUF header (`22,45` for the 2-shard IQ3_XXS layout; `14,31` for the single-file IQ4_XS and the 6-shard Q8_0).
- **KV**: int8 with 32,768 resident cells, the rest streams to RAM; 262144 total context (native).
- **Environment** (both on v0.1.39 Linux): `STRATA_NO_LARGEPAGES=1` -- host-arena load otherwise crawls for 10+ min with no I/O (upstream #771). `STRATA_VERIFY_ALL_RESIDENT=0` -- required for 2-slot batch on an all-resident split; without it the first concurrent request kills the engine (upstream #776; ~5% solo-decode cost).
- **Setup notes**: `STRATA_EXPERIMENTAL_SM60=1 ./setup.sh --check` first; build needs cmake >= 3.24 (the venv one -- a system cmake 3.22 fails), CUDA 12.8. Heavy steps run under `systemd-run --user` on this box. `--gguf-dir <dir>` reuses local GGUFs without re-downloading.

## Flag reference (every field in the startup configs)

Meanings from the engine's own help and docs (v0.1.39); the why behind the choices is in "Serving posture" above. The JSON fields map 1:1 onto engine flags (`serve/server.py` assembles them).

| Field / flag | What it does |
|---|---|
| `--pack <dir>` | the pack directory: the model staged by setup (weights in the engine's layout, tokenizer, per-layer tables). |
| `--native <shard.gguf>` | load the model's own GGUF shard(s) directly and turn on the native pinned-CUDA tensor paths; for split models this is shard 1 (repeat the flag for more shards). |
| `--ple-gguf <shard.gguf>` | the shard holding the PLE n-gram table (`per_layer_token_embd.weight`) -- usually the second shard. The engine warns that without the table layer 1's PLE is silently skipped and every number downstream changes; check the boot log reports PLE on. |
| `--expert-profile <file>` | pre-load the VRAM expert tier from a recorded `profile.bin` (built by `tools/make_profile.py`) instead of admitting experts on first use; the ranked fill is why 67% residency still hits 99%+. |
| `--expert-cache auto` | keep expert blobs resident in VRAM and compute their rows on the GPU; `auto` sizes the cache to each card (16,450 of 24,576 slots in the Q8 deployment; 100% in the all-resident ones). |
| `--prefill auto` | batched prompt processing in chunks; `auto` = the largest chunk (up to 8192) whose buffers the expert cache can lend. |
| `--spec 4` | speculation window: the MTP layer drafts up to 4 tokens per check and one full pass checks them (engine default 3; acceptance 74-80% here). |
| `--spec-min-p 0.5` | how sure the draft layer must be to add another guess to a check (calibratable; 0.5 here). |
| `--mtp <dir>` | the MTP draft-layer data the engine loads for speculation (the `rt` runtime form prepared by setup). |
| `--max-context 262144` | KV/state capacity in tokens. |
| `--kv int8` | KV storage: int8 codes + fp16 scale per 64 values -- half the VRAM of fp16 (engine default fp16). |
| `--kv-resident 32768` | KV streaming: keep 32,768 cells of each QSA layer in VRAM (min 20,480) and the whole K/V in pinned RAM; the freed VRAM goes to expert slots; contexts of <= 32k tokens are not streamed at all. |
| `--batch 2` | serve (opt-in): up to 2 requests decode together in batch slots (2..8), each slot with its own session. |
| `--batch-groups 2` | with a layer split: the slots pipeline through the GPUs in G groups. |
| `--trim-stage-weights` | with an explicit `--layer-split`: each GPU loads only its own layers' dense weights. |
| `--vision` | serve takes images too; the `vision` block names the encoder binary (`strata-vision`), the `mmproj` projector and the image-token budget (`max_tokens`). |
| `--vram-reserve-mib 700` | VRAM in MiB the engine leaves free for other programs (engine default 700). |
| `gpu: [1,2,3]` | the cards to use -- PCI-bus order on this box; the display GPU stays out of the mask (see the device-mask rule above). |
| `layer_split: "14,31"` | the layer index each stage starts at (`0-13 / 14-30 / 31-47`; two boundaries, three cards), picked by byte-balancing each stage's weights from the GGUF header (`22,45` for the 2-shard IQ3_XXS). Required by `--trim-stage-weights`. |
| `fit_max_tokens: true` | shorten a `max_tokens` that does not fit the context instead of answering 400 (server flag `--fit-max-tokens`). |
| `aliases: ["main"]` | the model name(s) the server lists and answers to -- the fleet convention is `main`. |
| `tokenizer: <pack>/tokenizer` | the pack's tokenizer directory (falls back to a byte tokenizer if absent). |
| `lib_dirs` | extra library search paths for the engine process (CUDA 12.8 `bin`/`lib64`). |
| `exe` / `cwd` / `log` | the engine binary / its working directory / the serve log. |
| `gpus_asked: true` | setup bookkeeping: the device list was chosen explicitly. |
| env `STRATA_NO_LARGEPAGES=1` | skip host large-page setup (v0.1.39 load fix -- without it the load stalls with no I/O). |
| env `STRATA_VERIFY_ALL_RESIDENT=0` | skip the all-resident verification that otherwise kills 2-slot batch on an all-resident split. |

## PLE n-gram table: read path and encodings

The table (28.8-54.4 GB in the encodings seen here) is read randomly, a few rows per token, from SSD. Two questions were tested: read mode, and encoding.

**Read mode.** `--ple-io direct` (default) = unbuffered, row-sized reads by a dedicated I/O thread (256 outstanding reads, prefetch, SSD keep-awake). A/B on the 54.4 GB Q8_0 table:

| mode | decode @2.4k | decode @20k | TTFT @2.4k |
|---|---|---|---|
| direct | 73.8 / 69.9 | 66.2 / 66.1 | 5.2 s |
| mmap | 40.2 / 51.5 | 48.4 / 55.8 | 18.1 s |

mmap faults synchronously on the token thread (random rows in a file far larger than RAM): -15% to -45% decode, fills 3-4x slower. Keep `direct`. `--ple-io ram` (lock the whole table in RAM) does not fit this box for these tables: ~47 GiB free with the engine resident vs a 54.4 GB table. Not tested.

**Encodings.** The reader accepts IQ4_NL (90 B/row), Q5_0 (110 B), Q8_0 (170 B) and FP8 E4M3 (160 B). Per-row relative L2 error against the checkpoint's BF16 table (the source of record; upstream numbers, PR #651): **Q8_0 0.53% / FP8 2.66% / IQ4_NL 7.59%**. A Q8_0 table is therefore the closest readable encoding, not merely the one that boots. An FP8 table ran ~+2-4% decode vs IQ4_NL on one upstream rig (PR #291) -- i.e. encoding is a fidelity choice, not a performance lever. A full-precision BF16 table (102 GB) exists upstream as a fidelity opt-in (PR #464 -> #586): its speed differs by +/-2% (noise).

## Negative results and limits

- `--ple-io mmap`: slower (table above); use the default.
- `--ple-io ram`: infeasible for these table sizes on a ~188 GiB box with the engine resident.
- Q6_K expert layers are not in the engine's supported gate/up/down format lists -- boot refuses and names the layer.
- A new quant family can hit several format gates in sequence (split metadata -> PLE row encoding): budget a patch + rebuild cycle, not a bare boot.
- Multi-GPU Linux was upstream-untested at v0.1.39 ("reports welcome"); the 2- and 3-card configurations here work as shipped.

## Patches

| File | Change |
|---|---|
| [patches/strata/v0.1.39-split-metadata-shard.patch](../patches/strata/v0.1.39-split-metadata-shard.patch) | `NativeDense::load` validates the architecture only on the metadata shard of a split (writers may reduce later shards' metadata to `split.*` while leaving `general.architecture` in every shard) |
| [patches/strata/v0.1.39-ple-q8_0-rows.patch](../patches/strata/v0.1.39-ple-q8_0-rows.patch) | Q8_0 PLE rows (170 B) through the existing `dequantize_q8_0` on both readers; `PLE_ROW_BYTES_MAX` 160 -> 170 |

Both against v0.1.39; MIT, as the upstream engine. Upstream equivalents were in review at write time (#651 and #865 for Q8_0 tables; #864 for a resident-expert buffer rotation). Engine numbers in this doc are single-machine, from the reference configuration; upstream numbers are attributed by PR number.
