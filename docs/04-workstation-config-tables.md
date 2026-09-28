# Workstation configuration tables (3 x V100, layer split)

All numbers measured on the reference configuration (see README).

## GPU device enumeration

- With a display GPU present, `nvidia-smi` indices and CUDA indices can disagree (observed: display card = `nvidia-smi` index 0, but enumerated *last* by CUDA).
- Pin compute devices explicitly by PCI bus order, e.g. `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2,3` (indices depend on your board).
- Verify the actual device list with `llama-server --list-devices` before trusting any split.

## CUDA version ceiling

- CUDA 13 removed sm_70. Toolchain must stay at CUDA <= 12.x (12.8 verified here).

## KV cache vs total context (f16 KV, 3 GPUs, layer split)

| total `-c` | KV per card |
|---|---|
| 786432 | 8448 MiB |
| 524288 | 5634 MiB |
| 196608 | 2112 MiB |
| 262144 (np2) | ~2.8 GiB |

Per-slot context = `-c / -np`. KV total scales with `-c`; compute scratch scales with *per-slot* context and `ub`:

- Example: `-c 349696 -np 2` (174848 tokens/slot) with ub 4096 -> ~9763 MiB/card compute scratch; ub 2048 -> ~9320 MiB/card.

## Micro-batch (`ub`) ladder -- prompt processing (35B-A3B Q8_0, np3 x 160k, ~24k-token prompt)

| ub | prompt t/s | compute/card | VRAM free after load |
|---|---|---|---|
| 512 | 615 | 1142 MiB | ~14 GB |
| 2048 | 955 | 2966 MiB | ~13 GB |
| 4096 | 1093 | 5932 MiB | ~11 GB |
| 8192 | 1118 | 11859 MiB | 4-6 GB |

- Decode on 16-token samples was flat within noise across the ladder (86.2 -> 78.8 t/s); treat such single-sample spreads as noise.
- Time-to-first-token example, 49,380-token prompt: 86.6 s (ub 512) -> 50.8 s (ub 4096).

## Expected log behaviors (not errors)

- `failed to allocate compute buffers -- retrying without pipeline parallelism`: pipeline-parallel buffer allocation does not fit; the server falls back and loads healthy. Expected on tight configs.
- Long prompt processing produces no client-visible output unless progress events are requested; the server emits an SSE comment ping every 30 s by default (`--sse-ping-interval`).

## VRAM budgeting rules (observed)

1. KV cache total is a function of total `-c`, independent of `-np`.
2. Compute scratch follows per-slot context and `ub`, not total `-c`.
3. Keep the display GPU out of the split; keep >= ~1 GB headroom per card.
4. A vision projector (mmproj) adds ~1.1 GB per loaded instance (observed).
5. On-demand loading of large embedding tables (`--lazy-mode`) trades VRAM for disk reads: ~474 MB/s cold from the source volume on this workstation.
