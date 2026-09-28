# Benchmarking methodology (single shared workstation)

## Problem

On a shared workstation, naive benchmarks produce invalid or misleading numbers. Failure modes observed:

1. **Warm-cache contamination** -- a "new" measurement silently reuses a prefix cache from a previous run.
2. **Cross-run attachment** -- a bench script waiting on a named service attaches to a *different* server instance started on the same port.
3. **Probe-vs-production divergence** -- acceptance-rate probes overstate speculative-decoding gains relative to real traffic.
4. **Short-context warmups quoted as steady-state numbers.**

## Protocol

1. **Report fill depth and generation length with every number.** A token rate at 2k context is not comparable to 19.5k. Never quote short-context warm results.
2. **Verify fresh evaluation.** The server reports prompt-eval and eval token counts per request. A valid arm shows `eval ~= fill` (every prompt token evaluated). `eval << fill` = prefix-cache hit -> discard the arm.
3. **Serialize bench jobs.** One server instance per port; stop prior instances before starting the next. Shared unit/service names are a hazard: a health-wait can latch onto a later instance with a warm cache.
4. **Fresh process per arm.** VRAM state, KV caches, and speculation state must start cold. Check health and speculative-decode lines in the log before measuring.
5. **Include a control arm.** Measure the no-speculation baseline on the *same binary* in the same session as the speculative arm.
6. **Evaluate speculative decoding token-weighted.** Probe-style arms showed sizable gains that washed out on production token mixes in some windows. Measure with production-like traffic before adopting.
7. **Read the outputs.** Coherence failures (repetition, truncation) do not show up in tok/s; preview generated text on every new configuration.
8. **Log the exact flags.** A number without its full command line is not reproducible.

## Concrete tells

| Observation | Verdict |
|---|---|
| `eval=4` against `fill=21076` | invalid -- prefix-cache hit |
| `eval=21076` against `fill=21076` | valid fresh evaluation |
| decode rate quoted without fill depth | not comparable -- re-measure |
| gain present in probe, absent in production mix | do not adopt on probe evidence alone |
| server start without draft-model line in log | speculation was not active -- arm is a control in disguise |
