# Models: tested serving configurations

Snapshot: 2026-10. Served one at a time on port 8081 from the reference workstation (see README); display GPU excluded via `CUDA_VISIBLE_DEVICES`; layer split across the three V100s unless noted. Flag blocks are the actual serving configurations with local paths collapsed. Sections 1-7 are the llama.cpp lanes; section 8 is the Strata lane -- the current :8081 stack. Flag meanings: [doc 06 §12](06-serving-playbook.md#12-glossary-what-every-flag-in-the-launchers-does) (llama.cpp) and [doc 07](07-strata-sm70.md) (Strata).

Decode rates marked "(sample)" are the last completed task in that model's server log at write time -- fill depths vary; campaign numbers follow the protocol in doc 03.

**Sections:** [1. GSQ-RCO](#1-qwen38-flash-next-gsq-rco----llamacpp-daily-driver-lane) · [2. Distill](#2-qwen38-35b-a3b-distill----q8_0--vision--mtp) · [3. Ornith](#3-ornith-15-35b-a3b----q8_0--vision) · [4. ThinkingCap](#4-thinkingcap-qwen38-27b----q8_0--dflash2-draft--vision) · [5. Qwythos](#5-qwythos-9b-claude-mythos-5-1m-context-line----q8_0--mtp) · [6. Uncensored IQ4_XS (legacy)](#6-qwen38-flash-next-uncensored----iq4_xs-legacy-lane) · [7. DeepSeek-V4](#7-deepseek-v4-flash----q4_k-moe-experts-on-cpu-parked) · [8. Strata lane](#8-strata-lane-separate-engine----current-8081-stack)

## Common pattern

```
llama-server -m <model.gguf> --alias main
  -ngl 999 -ts 1,1,1 -sm layer
  -c <total> -np <slots> -b <batch> -ub <ub>
  -fa on -ctk f16 -ctv f16
  --host 0.0.0.0 --port 8081 --metrics
```

Shared conventions: `-ngl 999` (all layers on GPU), `-ts 1,1,1` (even 3-way layer split), f16 KV, `--metrics` on every instance, alias `main` for the API.

---

## 1. Qwen3.8-Flash-Next (GSQ-RCO) -- llama.cpp daily driver lane

- Files: IQ3_XXS 2-shard GGUF + BF16 mmproj (~107 GB total), MTP head GGUF (Q4_K_M, ~2.5 GB; Q8_0 head ~4.1 GB for reduced-context use).
- Engine: llama.cpp fork with qwen4exp MTP support (doc 01 / attached patch).
- Config:

```
llama-server -m <model-IQ3_XXS-00001-of-00002.gguf>
  --mmproj <mmproj-Qwen3.8-Flash-Next-BF16.gguf> --image-min-tokens 1024
  --alias main -ngl 999 -ts 1,1,1 -sm layer
  -c 262144 -np 2 -b 4096 -ub 4096
  -fa on -ctk f16 -ctv f16
  --lazy-mode on --load-mode none --numa distribute
  -md <mtp-head-Q4_K_M.gguf>
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.0
  --jinja --chat-template-file <qwen-fixed-v22.5.jinja>
  --reasoning on --reasoning-effort medium --reasoning-preserve --reasoning-format deepseek
  --temp 1.0 --top-p 0.95 --top-k 20
  --metrics
  --host 0.0.0.0 --port 8081
```

- Environment: `LLAMA_ATTN_ROT_DISABLE=1` (QSA indexer requirement on this architecture); no `-ot` override (the PLE tensor is lazy-host by construction).

Measured (campaign, doc 03 protocol; full arms in [doc 01](01-mtp-speculative-decode-sm70.md)):

| Arm | @2.4k fill | @19.5k fill |
|---|---|---|
| no speculation (control) | 38.97 | -- |
| + MTP draft | 56.84 | 47.20 |
| production shape | 55.15 | 48.90 |

Acceptance ~70-76%; mean accepted length ~3.1-3.3 probe / ~2.6 token-weighted production. Prefill 339-425 t/s. Notes: head must be a 37-tensor export; Q8_0 head only fits reduced context.

## 2. Qwen3.8-35B-A3B-Distill -- Q8_0 + vision + MTP

- Files: 36 GB Q8_0 + F16 mmproj (858 MB); MTP comes from the model's own head (no `-md`).
- Config:

```
    -m Qwen3.8-35B-A3B-Q8_0.gguf
    --mmproj mmproj-Qwen3.8-35B-A3B-F16.gguf --image-min-tokens 1024
    --alias main -ngl 999 -ts 1,1,1 -sm layer
    -c 262144 -np 2 -b 2048 -ub 512
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 1
    --spec-draft-type-k f16 --spec-draft-type-v f16
    -fa on -fit off -ctk f16 -ctv f16 -t 16 --load-mode dio
    --jinja --chat-template-file qwen-fixed-v22.jinja
    --reasoning on --reasoning-effort medium --reasoning-preserve --reasoning-format deepseek
    --temp 0.6 --top-p 0.95 --top-k 20
```

- Observed in service: ~143.9 t/s decode (666-token sample), prompt processing ~543.7 t/s (1564-token chunk).

## 3. Ornith-1.5-35B-A3B -- Q8_0 + vision

- Files: 36 GB CRACK Q8_0 + BF16 mmproj (861 MB).
- Config:

```
    -m Ornith-1.5-35B-A3B-CRACK-Q8_0.gguf
    --mmproj mmproj-Ornith-1.5-35B-BF16.gguf --image-min-tokens 1024
    --alias main -ngl 999 -ts 1,1,1 -sm layer
    -c 480000 -np 3 -b 4096 -ub 4096
    -fa on -ctk f16 -ctv f16
    -t 8 -tb 8 --load-mode dio
    --jinja --temp 1.0 --top-p 0.95 --top-k 20
```

- Observed in service: ~83-88 t/s decode; prefill ladder 615 -> 1118 t/s across ub 512 -> 8192 (doc 04).
- MTP measured net-negative on this model (-22% at n_max 3; parity at n_max 1) -- left off.

## 4. ThinkingCap-Qwen3.8-27B -- Q8_0 + DFlash2 draft + vision

- Files: 28 GB Q8_0 + DFlash2 draft (2 GB) + F16 mmproj.
- Config:

```
    -m ThinkingCap-Qwen3.8-27B-Q8_0.gguf -md Qwen3.8-27B-DFlash2-Q8_0.gguf
    --mmproj mmproj-ThinkingCap-Qwen3.8-27B-f16.gguf --image-min-tokens 1024
    --alias main -ngl 999 -ts 1,1,1 -sm layer
    -c 262144 -np 2 -b 2048 -ub 512
    --spec-type draft-dflash,ngram-map-k --spec-draft-n-max 7 --spec-draft-n-min 1
    --spec-draft-type-k f16 --spec-draft-type-v f16
    -fa on -fit off -ctk f16 -ctv f16 -t 16 --load-mode dio
    --jinja --chat-template-file qwen-fixed-v22.jinja
    --reasoning on --reasoning-effort xhigh --reasoning-preserve --reasoning-format deepseek
    --temp 1.0 --top-p 0.95 --top-k 20
```

- Observed in service: ~45.7 t/s decode (81-token sample); prompt processing ~432 t/s (990-token sample), ~760-830 t/s at ~13k fills. Effort preset: xhigh.

## 5. Qwythos-9B (Claude-Mythos-5, 1M-context line) -- Q8_0 + MTP

- Files: 9.2 GB Q8_0 + F16 mmproj (876 MB).
- Config:

```
    -m Qwythos-9B-Claude-Mythos-5-1M-MTP-Q8_0.gguf
    --mmproj mmproj-Qwythos-9B-Claude-Mythos-5-1M-F16.gguf --image-min-tokens 1024
    --alias main -ngl 999 -ts 1,1,1 -sm layer
    -c 786432 -np 3 -b 2048 -ub 4096
    --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-n-min 1
    --spec-draft-type-k f16 --spec-draft-type-v f16
    -fa on -fit off -ctk f16 -ctv f16 -t 8 -tb 8 --load-mode mmap
    --jinja --reasoning on --reasoning-format deepseek
    --temp 0.6 --top-p 0.95 --top-k 20 --repeat-penalty 1.05 --n-predict 16384
```

- Observed in service: ~65.3 t/s decode (337-token sample); prompt processing ~2114 t/s.

## 6. Qwen3.8-Flash-Next Uncensored -- IQ4_XS (legacy lane)

- Older fork build; shows two patterns not used elsewhere here: q8_0 KV cache and per-tensor placement.

```
    -m Qwen3.8-Flash-Next-Uncensored.IQ4_XS.gguf -md mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf
    --mmproj Qwen3.8-Flash-Next-Uncensored.mmproj-f16.gguf --image-min-tokens 1024
    --alias main --host 0.0.0.0 --port 8081 --numa distribute
    -dev CUDA0,CUDA1,CUDA2 -devd CUDA0,CUDA1,CUDA2 -ngld 999
    --split-mode layer --load-mode none --no-host --fit off
    -ot 'per_layer_token_embd.weight=CPU,v\.blk\..*=CUDA2'
    -fa on -ctk q8_0 -ctv q8_0 -c 524288 -np 2
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75
    --reasoning on --reasoning-effort low --reasoning-preserve
    --reasoning-budget 16384 --reasoning-format deepseek
```

- Superseded by the master-build daily (doc 01); kept for the placement / q8_0-KV patterns.
- Prompt processing: not recorded.

## 7. DeepSeek-V4-Flash -- Q4_K, MoE experts on CPU (parked)

- Reference configuration for oversized MoE with CPU offload on this platform.

```
    -m DeepSeek-V4-Flash-Q4_K-0731-abliterated.gguf --alias main
    -c 524288 -np 2 -ngl 999 -ot exps=CPU -b 4096 -ub 4096 -t 16 -tb 16
    -ctk f16 -ctv f16 --flash-attn on --moe-cache on --numa distribute
    --reasoning on --reasoning-preserve --reasoning-budget 4000 --reasoning-format deepseek
```

- Observed in service: ~15.9 t/s decode, ~104.9 t/s prompt processing (experts on CPU, DDR4-bandwidth-bound; see doc 00).
- Parked -- model files relocated; launcher retained as reference.

## 8. Strata lane (separate engine) -- current :8081 stack

A second serving stack for the same Qwen3.8-Flash-Next family, holder of :8081 since 2026-10-04: Strata keeps experts across VRAM + RAM and streams the PLE n-gram table from SSD. Full setup, posture and PLE findings in [doc 07](07-strata-sm70.md). Deployments, newest first:

| Model / quant | Size | Decode | Prompt read | Notes |
|---|---|---|---|---|
| [Abliterated Q8_0 (6 shards)](#81-abliterated-q8_0----strata-huihui-q8_0json) | ~189 GB | 72.5 @2.4k; 66.0 @20k fill | 458 -> 1,578 | served now (spill test, 67% residency); needs [patches/strata/](../patches/strata/) |
| [Uncensored IQ4_XS (single file)](#82-uncensored-iq4_xs----strata-orca-iq4_xsjson) | 92 GB | 85.3 @2.5k; ~82 @18k fill | 707 -> 1,772 | all-resident; fastest decode measured on this box |
| [GSQ-RCO IQ3_XXS (2 shards)](#83-gsq-rco-iq3_xxs----strata-iq3_xxsjson) | ~107 GB | 77.6 @2.0k; 73.9 @17k fill | 656 -> 1,158 | first Strata deployment, 2026-10-04 |

For scale, the llama.cpp production lane above runs 55.2 @2.4k fill / 48.9 @19.5k.

Startup (engine v0.1.39 + `patches/strata/`; `<strata-repo>` = engine checkout, `<strata-data>` = pack/MTP data dir, `<models-dir>` = GGUF root):

```
cd <strata-repo>
export STRATA_NO_LARGEPAGES=1        # host-arena load fix on v0.1.39
export STRATA_VERIFY_ALL_RESIDENT=0  # 2-slot batch on an all-resident split
exec .venv/bin/python serve/server.py --engine strata --config <config>.json --port 8081 --open
```

Config per deployment (sanitized; the three files differ only in the pack/model paths, `--ple-gguf`, `layer_split`, `model_name`, `log`, and the vision block):

### 8.1 Abliterated Q8_0 -- `strata-huihui-q8_0.json`

```json
{
 "exe": "<strata-repo>/engine/strata",
 "args": [
  "--pack", "<strata-data>/packs/huihui-q8_0",
  "--native", "<models-dir>/Huihui-Qwen3.8-Flash-Next-abliterated/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf",
  "--expert-profile", "<strata-repo>/data/expert-profile.bin",
  "--expert-cache", "auto",
  "--prefill", "auto",
  "--spec", "4",
  "--spec-min-p", "0.5",
  "--mtp", "<strata-data>/mtp/rt",
  "--max-context", "262144",
  "--kv", "int8",
  "--kv-resident", "32768",
  "--batch", "2",
  "--batch-groups", "2",
  "--trim-stage-weights",
  "--vision",
  "--vram-reserve-mib", "700"
 ],
 "cwd": "<strata-repo>",
 "tokenizer": "<strata-data>/packs/huihui-q8_0/tokenizer",
 "model_name": "qwen3.8-flash-next-abliterated-q8_0",
 "log": "<strata-repo>/strata-huihui-q8_0.log",
 "lib_dirs": ["/usr/local/cuda-12.8/bin", "/usr/local/cuda-12.8/lib64"],
 "port": 8081,
 "host": "0.0.0.0",
 "fit_max_tokens": true,
 "aliases": ["main"],
 "gpu": [1, 2, 3],
 "gpus_asked": true,
 "layer_split": "14,31",
 "vision": {
  "exe": "<strata-repo>/engine/strata-vision",
  "mmproj": "<models-dir>/Huihui-Qwen3.8-Flash-Next-abliterated/mmproj-model-bf16.gguf",
  "model": "<models-dir>/Huihui-Qwen3.8-Flash-Next-abliterated/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf",
  "gpu": true,
  "max_tokens": 1024
 }
}
```

### 8.2 Uncensored IQ4_XS -- `strata-orca-iq4_xs.json`

```json
{
 "exe": "<strata-repo>/engine/strata",
 "args": [
  "--pack", "<strata-data>/packs/orca-iq4_xs",
  "--native", "<models-dir>/Qwen3.8-Flash-Next-Uncensored.IQ4_XS.gguf",
  "--ple-gguf", "<models-dir>/Qwen3.8-Flash-Next-Uncensored.IQ4_XS.gguf",
  "--expert-profile", "<strata-repo>/data/expert-profile.bin",
  "--expert-cache", "auto",
  "--prefill", "auto",
  "--spec", "4",
  "--spec-min-p", "0.5",
  "--mtp", "<strata-data>/mtp/rt",
  "--max-context", "262144",
  "--kv", "int8",
  "--kv-resident", "32768",
  "--batch", "2",
  "--batch-groups", "2",
  "--trim-stage-weights",
  "--vision",
  "--vram-reserve-mib", "700"
 ],
 "cwd": "<strata-repo>",
 "tokenizer": "<strata-data>/packs/orca-iq4_xs/tokenizer",
 "model_name": "qwen3.8-flash-next-uncensored-iq4_xs",
 "log": "<strata-repo>/strata-orca-iq4_xs.log",
 "lib_dirs": ["/usr/local/cuda-12.8/bin", "/usr/local/cuda-12.8/lib64"],
 "port": 8081,
 "host": "0.0.0.0",
 "fit_max_tokens": true,
 "aliases": ["main"],
 "gpu": [1, 2, 3],
 "gpus_asked": true,
 "layer_split": "14,31",
 "vision": {
  "exe": "<strata-repo>/engine/strata-vision",
  "mmproj": "<models-dir>/Qwen3.8-Flash-Next-Uncensored/Qwen3.8-Flash-Next-Uncensored.mmproj-f16.gguf",
  "model": "<models-dir>/Qwen3.8-Flash-Next-Uncensored.IQ4_XS.gguf",
  "gpu": true,
  "max_tokens": 1024
 }
}
```

### 8.3 GSQ-RCO IQ3_XXS -- `strata-iq3_xxs.json`

```json
{
 "exe": "<strata-repo>/engine/strata",
 "args": [
  "--pack", "<strata-data>/packs/iq3_xxs",
  "--native", "<models-dir>/Qwen3.8-Flash-Next-GSQ-RCO/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
  "--ple-gguf", "<models-dir>/Qwen3.8-Flash-Next-GSQ-RCO/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
  "--expert-profile", "<strata-repo>/data/expert-profile.bin",
  "--expert-cache", "auto",
  "--prefill", "auto",
  "--spec", "4",
  "--spec-min-p", "0.5",
  "--mtp", "<strata-data>/mtp/rt",
  "--max-context", "262144",
  "--kv", "int8",
  "--kv-resident", "32768",
  "--batch", "2",
  "--batch-groups", "2",
  "--trim-stage-weights",
  "--vision",
  "--vram-reserve-mib", "700"
 ],
 "cwd": "<strata-repo>",
 "tokenizer": "<strata-data>/packs/iq3_xxs/tokenizer",
 "model_name": "qwen3.8-flash-next-iq3_xxs",
 "log": "<strata-repo>/strata-iq3_xxs.log",
 "lib_dirs": ["/usr/local/cuda-12.8/bin", "/usr/local/cuda-12.8/lib64"],
 "port": 8081,
 "host": "0.0.0.0",
 "fit_max_tokens": true,
 "aliases": ["main"],
 "gpu": [1, 2, 3],
 "gpus_asked": true,
 "layer_split": "22,45",
 "vision": {
  "exe": "<strata-repo>/engine/strata-vision",
  "mmproj": "<models-dir>/Qwen3.8-Flash-Next-GSQ-RCO/mmproj-Qwen3.8-Flash-Next-BF16.gguf",
  "model": "<models-dir>/Qwen3.8-Flash-Next-GSQ-RCO/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
  "gpu": true,
  "max_tokens": 1024
 }
}
```
