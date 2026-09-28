# Models: tested serving configurations

Snapshot: 2026-09. Served one at a time on port 8081 from the reference workstation (see README); display GPU excluded via `CUDA_VISIBLE_DEVICES`; layer split across the three V100s unless noted. Flag blocks are the actual serving configurations with local paths collapsed.

Decode rates marked "(sample)" are the last completed task in that model's server log at write time -- fill depths vary; campaign numbers follow the protocol in doc 03.

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

## 1. Qwen3.8-Flash-Next (GSQ-RCO) -- daily driver

- Files: IQ3_XXS 2-shard GGUF + BF16 mmproj (~107 GB total), MTP head GGUF (Q4_K_M, ~2.5 GB; Q8_0 head ~4.1 GB for reduced-context use).
- Engine: llama.cpp fork with qwen4exp MTP support (doc 01 / attached patch).
- Config: `-c 262144 -np 2 -b 4096 -ub 4096`, `--load-mode none`, `--numa distribute`, `--spec-type draft-mtp --spec-draft-n-max 3`, vision via mmproj, reasoning medium, temp 1.0 / top-p 0.95 / top-k 20.

Measured (campaign, doc 03 protocol):

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

- Observed in service: ~45.7 t/s decode (81-token sample). Effort preset: xhigh.

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
