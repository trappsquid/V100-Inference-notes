# Hardware platform: dual-socket V100 workstation

The reference system for the numbers in this repository. Values are as observed on the machine (source command noted), not spec-sheet claims.

## Summary

| Component | Detail | Source |
|---|---|---|
| Workstation | Dell Precision T7910 (dual-socket tower) | -- |
| CPU | 2 x Intel Xeon E5-2697A v4 @ 2.60 GHz (16C / 32T each; up to 3.6 GHz; 40 MiB L3 each; AVX2) | `lscpu` |
| NUMA | 2 nodes; node0 = CPUs 0-15,32-47; node1 = CPUs 16-31,48-63; local distance 10 / remote 21 | `numactl --hardware` |
| RAM | ~188 GiB total (~94 GiB per node); ECC DDR4 (platform generation supports up to DDR4-2400) | `numactl`, `dmidecode` |
| GPUs | 3 x Tesla V100 32 GB (sm_70) + 1 x Quadro P400 2 GB (display) | `nvidia-smi` |
| GPU links | all PCIe Gen3 x16 (max); no NVLink | `nvidia-smi -q`, `topo -m` |
| Storage | 3 x SATA-class SSD | `lsblk` |
| CUDA | 12.8 / nvcc 12.8.93 | -- |

## NUMA layout and GPU affinity

`nvidia-smi topo -m` (abridged; T = traffic type between GPUs, CPU/NUMA affinity of each GPU):

| | GPU0 | GPU1 | GPU2 | GPU3 | CPU affinity | NUMA node |
|---|---|---|---|---|---|---|
| GPU0 (P400) | X | PHB | PHB | SYS | 0-15,32-47 | 0 |
| GPU1 (V100) | PHB | X | PHB | SYS | 0-15,32-47 | 0 |
| GPU2 (V100) | PHB | PHB | X | SYS | 0-15,32-47 | 0 |
| GPU3 (V100) | SYS | SYS | SYS | X | 16-31,48-63 | 1 |

Observed facts:

- GPUs 0-2 (PCI bus 02/03/04) attach to NUMA node 0; GPU3 (PCI bus A1) attaches to node 1 -- the slot wiring feeds from both sockets.
- GPU3 <-> GPU1/2 traffic crosses the inter-socket link (SYS). With `-sm layer` the pipeline crosses sockets once per forward pass; layer split tolerates this well. Tensor-parallel allreduce across SYS/PHB is slow (doc 06) -- layer split is the default for good reason.
- Inter-node protocol distance 21 vs local 10: expect roughly half bandwidth for cross-socket transfers on this generation.

## Memory (DDR4) and inference

- ~188 GiB ECC DDR4 across two sockets. Addressable per node: ~94 GiB.
- Page cache matters: ~123 GiB observed in page cache at steady state (model pages). Repeat model loads and `--lazy-mode` tensor reads are effectively RAM-speed; cold reads from the model volume run ~474 MB/s.
- CPU-side MoE offload (`-ot exps=CPU`) runs at DDR4-bandwidth limits: reference numbers ~105 t/s prefill / ~16 t/s decode for DeepSeek-V4-Flash with experts on CPU (doc 05).
- NUMA strategies exposed by llama.cpp: `--numa distribute | isolate | numactl | mirror`. Launchers here use `distribute` when CPU work is involved.
- When CPU threads are used, keep them on one node where possible (numactl/CPU affinity options exist for the main and draft models) -- cross-socket thread migration costs more than it saves on this generation.

## GPU capability ceiling (why sm_70 is called out everywhere)

- Volta = fp16 tensor cores; no bf16, no tf32. Kernels requiring Ampere+ semantics cannot run here, and CUDA 13 dropped sm_70 entirely -- the toolchain is pinned at CUDA <= 12.x (12.8 verified).
- No NVLink on these PCIe cards; 3-GPU communication runs over PCIe Gen3.
- A display GPU is present: pin compute devices explicitly and verify with `llama-server --list-devices` (details in docs/04).

## Storage

| Volume | Size | Filesystem | Use |
|---|---|---|---|
| /mnt/models | 932 GB | NTFS | primary GGUF store (~663 GB used) |
| /models | 400 GB | ext4 (LVM) | EXL3 pack + secondary GGUF store |
| / | 100 GB | ext4 (LVM) | OS |

- All model volumes are SATA-class SSDs; maximum observed cold sequential read ~474 MB/s.
- Model loading pages through the disk cache; system RAM is the cheapest optimization on this platform (repeat loads become nearly free).
