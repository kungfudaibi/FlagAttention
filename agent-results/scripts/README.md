# SAGE-002 / SAGE-003 / SAGE-004 / SAGE-005 / SAGE-006 scripts

All authoritative SAGE-003/004/005 scripts target container **GCU 5** (PCI
`0000:83:00.0`, serial `A0D5460520864`) and must be run with the CPU pinning
from the task:

```bash
bash agent-results/scripts/preflight_gcu5.sh
env PYTHONPATH=/opt/flagtree:$PWD/src taskset -c 160-191 /flagos/bin/python -u <script>
```

The device ordinal is not portable between containers — SAGE-002 ran on this
same physical card as "GCU 1". The scripts therefore identify the device by
**PCI address** and refuse to run when efsmi maps that address to a different
ordinal. The SAGE-002 scripts below still hard-code DEV 1 because that is what
their recorded raw data used; they are kept unchanged for audit.

## Authoritative (produce the numbers in `agent-results/SAGE-003.md`)

| Script | What it measures |
|---|---|
| `sage003_ab.py` | Proof that `F.scaled_dot_product_attention` is a host fallback and that `aten::_scaled_dot_product_flash_attention` is the GCU kernel; then 3 alternating A/B rounds of the Sage full call versus that device op on identical inputs; stage attribution (quantizer with the SAGE-003 launch vs the previous default launch, attention alone); cold compile in a child process with an empty `TRITON_CACHE_DIR`; accuracy, finite values, peak memory; raw per-sample JSONL. Exit 4 on wrong device, 3 on numeric/sample failure, 5 on changed dispatch. |
| `sage003_attn_sweep.py` | Sweeps the attention kernel's launch knobs (`block_m`, `block_n`, `num_warps`, `num_stages`) on **pre-quantized** inputs, so quantization is excluded and the kernel gets its most favourable setting; validates every candidate against the device op before timing it. Exit 3 if any candidate leaves the numeric gate. |
| `sage003_summarize.py` | Regenerates every table in the report from `agent-results/raw/sage003_ab_*.jsonl`, plus the launch sweep, the bimodality breakdown, and the exit-code controls; exits nonzero when a decision record is missing, not PASS, or an exit code disagrees. |
| `preflight_gcu5.sh` | Refuses unless efsmi DEV 5 is PCI `0000:83:00.0` with the assigned serial; prints used MiB. |
| `sage004_ab.py` | The SAGE-004 formal **full-call** A/B: exact legacy Sage (pre-SAGE-003 4-warp/3-stage quantizer) versus candidate Sage (shipped quantizer, same attention kernel) on identical FP16 Q/K/V; 3 alternating ABBA rounds, per-sample rows, host wall clock and GCU events, cold compile with an empty `TRITON_CACHE_DIR`, peak memory per candidate; gates the output **and** the INT8 quantization intermediates. Exit 4 wrong device, 3 numeric/sample/scope failure, 5 changed dispatch. Replaces the seeded fail-closed placeholder. |
| `sage004_profile.py` | Localises the SAGE-004 attention-kernel bottleneck by **single-site ablation** of the shipped kernel on pre-quantized inputs (`full`, per-stage removals, both-dots ceiling, QK-matmul-only floor) and reports TFLOPS against the tile FLOPs. Never mixed into the full-call A/B. Exit 4 wrong device, 3 if an ablation fails. |

## SAGE-005 scripts (recorded on GCU 5)

All three pin GCU 5 by PCI address *and* serial through `efsmi` before the first
device action, and abort otherwise. That guard is not decorative: the ordinal is
container-local, and the three SAGE tasks each ran on a different host.

| Script | What it measures |
|---|---|
| `sage005_micro.py` | Four probes into where the fused-SDPA gap lives. `--probe resident` runs pure `tl.dot` with both operands resident (the compute-only ceiling). `--probe tilemajor` adds a per-iteration tile load in the shipped kernel's layout, so the delta is the cost of the tile feed. `--probe slope` benchmarks the shipped kernel across query lengths and reports **microseconds per KV iteration** with launch/prologue/epilogue differenced out, for single-fragment ablations of the kernel; a companion `ablation_artifact` record proves the dot ops survived the ablation. `--probe epilogue` prices the softmax primitives separately. `--probe` accepts a comma-separated list. |
| `sage005_dispatch.py` | Proves which op actually ran, per path: the device fused op, the repository `F.sdpa` host fallback, the Sage full call and the attention-only kernel. Records the CPU op chain, host-copy count, device ops, and a GCU-event latency. Residency is decided by device-event time because the Kineto GCU activity stream is **unreliable on this image** (identical repeated profiles return either the full runtime list or only two `aten::empty` records); that flakiness is itself recorded rather than papered over. |
| `sage005_cold_memory.py` | Cold compile in a child process with its own empty `TRITON_CACHE_DIR` (the parent would reuse the on-disk cache and report a warm number), plus peak device MiB per path with allocator counters reset immediately before the call. |

### SAGE-006: evidence repair for the SAGE-005 review

SAGE-005 quoted several numbers whose generating script only printed a median to
stdout. The review was right that no per-sample JSONL existed for them. SAGE-006
does not pretend otherwise; it separates four provenance levels:

| level | meaning | where |
| --- | --- | --- |
| A | per-sample JSONL committed during SAGE-005 | `raw/sage005_*.jsonl` |
| B | per-sample JSONL produced by a SAGE-006 re-run | `raw/sage006_recheck_*.jsonl` |
| C | verbatim stdout recovered from the SAGE-005 session log | `raw/sage005_recovered/*.log` |
| D | not recoverable | none — every claim fell into A, B or C |

| Script | What it measures |
|---|---|
| `sage005_guard.py` | Shared GCU-5 guard (PCI **and** serial), event timer, JSONL writer, repo-root resolution. Imported by the recheck script; not run directly. |
| `sage005_recheck.py` | Short re-runs of exactly the claims that lacked raw data: `--probe fp16` (pure-FP16 Triton vs native), `fuse` (the two rejected row-max rewrites plus bitwise equality), `flags` (`predequantize_k` x `tile_major_predequant_k` x `qk_scale_before_dot`), `stages` (`num_stages` 1-4, row-major and tile-major, plain vs manual prefetch), `waves` (16-192 CTAs at fixed work, plus `block_m`), `bw` (copy/read at 2-32 KiB blocks), `marginal` (single-wave slope; only `nCTA <= 24` enters the fit). `--probe` takes a comma-separated list, `--jsonl` is required, and every run writes an `env` record first. |
| `sage006_artifacts.py` | Recomputes every IR and disassembly number from the committed artefacts, with no device: op-level counts per IR level, async call sites and how many LLVM ops precede their `tops_wait_dte`, symbol sizes, instruction/opcode histograms per `flash_attn_fwd_kernel` instantiation, and SHA-256 of the vendor library, fatbin, kernel and disassembly. `--ir`, `--vendor`, `--sha256`, `--jsonl`. |
| `recovered/*.py` | The 49 `/tmp` probe scripts exactly as SAGE-005 ran them (three of them are the fuse variants), plus `recovered/sage005_micro_variants/` with the nine files `sage005_micro.py` generated: three microbenchmark kernels (resident, tile-major, epilogue) and six ablation variants of the shipped kernel. Kept for audit; they hard-code `set_device(5)` and print rather than write JSONL. Use `sage005_recheck.py` for anything reproducible. |

`recovered/` deliberately keeps the original defects so the audit trail is
honest: `sage005_marg2.py` sweeps D128 to `T=4096`, which crosses into a second
wave and inflates the slope (see `SAGE-006.md` §3.4); `sage005_fuse.py` crashes
before printing its equality check; `sage005_waves.py` fails on a tuple
`.float()`. The recheck script fixes all three.

The SAGE-005 formal A/B reuses `sage004_ab.py` unchanged: legacy Sage as the
control, shipped Sage as the candidate, 3 alternating rounds. Note it needs
`--pairs 24` with `--min-samples 40`; at `--pairs 12` each round yields 24
samples and the run correctly fails its own sample-count gate (`status: FAIL`,
exit 3) rather than reporting an underpowered comparison.

## SAGE-002 scripts (recorded on GCU 1)

| Script | What it measures |
|---|---|
| `sage002_stage_timing.py` | Q/K quant, attention-on-prequantized, full call, exact SDPA, and the GCU-native fused op; 3 rounds x N synchronized samples, raw per-sample records, memory, accuracy; exit 3 on gate failure |
| `sage002_full_path_ab.py` | Exact-baseline vs candidate full call in ABBA pairs; every raw sample with pair/position; memory; numerical error; `--require-quiet` gate on device identity and foreign GCU 1 memory; exit 3 on any validity failure |
| `sage002_config_sweep.py` | Launch-config and feature-flag sweep, accuracy validated before timing |
| `sage002_baseline_dispatch.py` | Proves `F.scaled_dot_product_attention` is a CPU fallback here: CPU profiler names and per-call process CPU time vs the vendor op |
| `sage002_summarize.py` | Regenerates every table in the SAGE-002 report from `agent-results/raw/*.jsonl` |
| `preflight_gcu1.sh` | Refuses to run unless efsmi DEV 1 is PCI `0000:83:00.0`; prints used MiB |

## Supporting evidence

| Script | Purpose |
|---|---|
| `sage002_io_dtype_probe.py` | Which input/output dtype combinations compile and their accuracy |
| `sage002_two_device_probe.py` | Reproduces the one-process/two-GCU abort that motivates the test guard |
| `sage002_vendor_sweep.py` | Vendor fused attention vs `F.sdpa` vs Sage, with accuracy |
| `sage002_vendor_attn.py` | Direct vendor-op dispatch check |
| `sage002_sdpa_device_probe.py` | `F.sdpa` and the vendor op on both GCUs |

## Exploratory diagnostics (kept for audit, not authoritative)

`sage002_probe.py`, `sage002_diag.py`, `sage002_diag2.py`,
`sage002_cpu_probe.py`, `sage002_affinity_probe.py`, `sage002_bind_probe.py`,
`sage002_visibility_probe.py`, `sage002_device_compare.py`,
`sage002_device_attr.py`, `sage002_dtype_probe.py`, `sage002_placement.py`,
`sage002_loop.py`, `sage002_sat.py`, `sage002_matmul_diag.py`,
`sage002_bench_prefix.py`, `sage002_sdpa_compare.py`, `sage002_sdpa_repro.py`,
`sage002_profiler_probe.py`.

These were used to establish device attribution, the CPU fallback, and the
matmul/quant behavior before the authoritative harnesses existed. Their logs
are in `agent-results/raw/` because several of them recorded negative results
(e.g. that `TOPS_VISIBLE_DEVICES=1` masks the ordinal but not the physical
device) that would otherwise be re-investigated.
