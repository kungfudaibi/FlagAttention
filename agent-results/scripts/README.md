# SAGE-002 scripts

All scripts target container **GCU 1** (PCI `0000:83:00.0`) and must be run
with the device mask and CPU pinning:

```bash
bash agent-results/scripts/preflight_gcu1.sh
env PYTHONPATH=/opt/flagtree:$PWD/src taskset -c 160-191 /flagos/bin/python -u <script>
```

## Authoritative (produce the numbers in `agent-results/SAGE-002.md`)

| Script | What it measures |
|---|---|
| `sage002_stage_timing.py` | Q/K quant, attention-on-prequantized, full call, exact SDPA, and the GCU-native fused op; 3 rounds x N synchronized samples, raw per-sample records, memory, accuracy; exit 3 on gate failure |
| `sage002_full_path_ab.py` | Exact-baseline vs candidate full call in ABBA pairs; every raw sample with pair/position; memory; numerical error; `--require-quiet` gate on device identity and foreign GCU 1 memory; exit 3 on any validity failure |
| `sage002_config_sweep.py` | Launch-config and feature-flag sweep, accuracy validated before timing |
| `sage002_baseline_dispatch.py` | Proves `F.scaled_dot_product_attention` is a CPU fallback here: CPU profiler names and per-call process CPU time vs the vendor op |
| `sage002_summarize.py` | Regenerates every table in the report from `agent-results/raw/*.jsonl` |
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
