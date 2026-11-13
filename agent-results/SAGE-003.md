# SAGE-003: GCU device-fused SDPA baseline vs SageAttention

Status: **complete — no-go for a speed claim. One verified quantization
launch optimization landed; default routing unchanged.**

## Decision in one paragraph

On container GCU 5 (PCI `0000:83:00.0`) the only attention path that actually
runs on the GCU is `aten::_scaled_dot_product_flash_attention`. The
repository's `F.scaled_dot_product_attention` **falls back to the host**:
`aten::_scaled_dot_product_flash_attention_for_cpu`, `aten::_to_cpu`, four
`topsMemcpyAsync` PCIe copies, ~32 ms of host CPU per call at
`1x2x512x64`. Against the real device op the Enflame per-block INT8
SageAttention full call is **3.96x slower at `1x2x4096x64` and 4.53x slower at
`4x32x1024x128`** — i.e. it takes that many times the device op's latency
(medians of three alternating rounds, 10 synchronized samples each, FP16;
"fast mode" medians when a round is bimodal, which is the conservative choice
for Sage and is spelled out below). Accuracy is fine: cosine 0.99992,
relative L2 0.012, all values finite. The gap is not a bad launch configuration and not only quantization.
The quantizer was Triton's default 4-warp/3-stage launch and is 2.6-5.3x
faster with `num_warps=1, num_stages=1` with bitwise identical int8/scale
outputs — that one change is the optimization this task allows, and it shortened
the full call by 1.4-2.1x. It is not nearly enough: quantization is 15-28% of the
Sage call at these shapes, and **even with quantization deleted the attention
kernel alone is 3.2-3.8x the device fused op**, with every other launch
configuration worse (up to 17.7x). Default routing and the operator registry are untouched.

## What this task adds over SAGE-002

SAGE-002 concluded the same no-go but on D64 only, and it did not establish the
device baseline as rigorously as this task requires. SAGE-003:

- pins **GCU 5** by PCI address and refuses to run otherwise, instead of
  trusting a bare ordinal (the same card was "GCU 1" in SAGE-002's container);
- **proves the kernel path in-process** on every run — profiler op names in a
  child process plus per-call host CPU time — and fails the run (exit 5) if the
  baseline or the reference ever changes identity;
- covers **D64 and D128** with four formal and four additional shapes,
  FP16 and BF16 outputs, and partial/trailing blocks;
- separates **quantization error from kernel error** by comparing against the
  device op computed on the *dequantized* inputs;
- measures **cold compile** in a separate process with an empty
  `TRITON_CACHE_DIR`, separates `full` from per-stage scope, and records cache
  file counts before/after;
- measures **peak memory per candidate** instead of one shared number;
- lands **one** auditable optimization with a bitwise-equivalence test, and
  reports the no-go for speed.

## Exact revision, device, and scope

- Base commit `3c3634fba806df4e0713e7bce14f901f262bb1f1`; result commit
  **`17936d6`** on branch `agent/enflame-sage-sdpa-device-5`; clone
  `/workspace/agent-workspaces/sage-sdpa-device-5`. Working tree clean, nothing
  pushed or merged. (A commit cannot contain its own hash: `17936d6` holds the
  code, data, and this report; the follow-up commit on top only corrects this
  line.)
- Container **GCU 5**, `efsmi -L` DEV 5 = PCI **`0000:83:00.0`**, SN
  `A0D5460520864`, 42976 MiB total. DEV 6 (`0000:84:00.0`) and DEV 7
  (`0000:a3:00.0`) were never used.
- `torch.gcu.set_device(5)` is the first device action in every process; the
  current stream and every tensor are asserted to be `gcu:5` before any
  measurement (`runtime_identity` record, `device_identity_ok: true`).
- All device commands run under `taskset -c 160-191` with
  `PYTHONPATH=/opt/flagtree:$PWD/src` and `/flagos/bin/python`.
- Runtime: Python 3.12.13, PyTorch 2.10.0+cpu, torch_gcu
  2.10.0+3.7.20260408, Triton 3.6.0, FlagTree 0.7.0+enflame3.6. Full capture in
  `agent-results/raw/sage003_environment.txt`.
- GCU 5 reported `Used Size 0 MiB`, `GCU Usage 0.0%` before the formal runs;
  no foreign process held a GCU node.

## The baseline is a device kernel — and `F.sdpa` is not

Measured in-process on GCU 5, recorded in every run as `path_evidence`
(profiler collected in a throwaway child process, because this image's
profiler leaves a broken heap that aborts at exit; the child's exit status is
part of the gate):

| | `aten::_scaled_dot_product_flash_attention` | `F.scaled_dot_product_attention` |
|---|---|---|
| dispatcher op | `aten::_scaled_dot_product_flash_attention` | `aten::scaled_dot_product_attention` → `aten::_scaled_dot_product_flash_attention_for_cpu` |
| kernel | `GCU::_scaled_dot_product_flash_attention`, one `topsLaunchKernel` | none on the GCU |
| host copies | none | `aten::_to_cpu`, 4x `aten::copy_`, 4x `topsMemcpyAsync` |
| host CPU per call @`1x2x512x64` | **0.16-0.37 ms** | **31.7-32.0 ms** |
| verdict | GCU kernel | host fallback |

`sdpa_kernel(SDPBackend.FLASH_ATTENTION)` does not change this, `MATH` routes to
`aten::_scaled_dot_product_attention_math` (still host copies), and
`EFFICIENT_ATTENTION`/`CUDNN_ATTENTION` raise "No viable backend". Any speed
claim for Sage against `F.sdpa` on this image would be a claim against a PCIe
round trip, so the A/B below uses only the GCU op.

## Formal A/B: Sage full call vs device fused SDPA

Three rounds, 5 ABBA pairs per round (10 timed samples per scope), warm-up
excluded, one synchronized device-event pair around exactly one public call,
identical tensors inside a pair, leading candidate alternated per round.

### FP16 in, FP16 out — steady-state medians

| shape | device fused ms | Sage full ms | Sage/device | device peak MiB | Sage peak MiB |
|---|---|---|---|---|---|
| `1x2x4096x64` | 0.2900 | 1.1484 | **3.96x** | 9.03 | 10.00 |
| `4x32x1024x128` | 1.2149 | 5.5010 | **4.53x** | 288.52 | 320.04 |

Per-round, using each round's own device median, Sage/device is 3.94-4.05x at
D64 and 4.39-4.74x at D128. When a D128 round is bimodal the **fast mode** is
what is compared (5.33-5.70 ms); using the raw median instead would report
4.42-4.75x, never lower.

### FP16 in, BF16 out

| shape | device fused ms | Sage full ms | Sage/device | cosine | rel L2 |
|---|---|---|---|---|---|
| `1x2x4096x64` | 0.2894 | 1.1490 | **3.97x** | 0.99988 | 0.0121 |
| `4x32x1024x128` | 1.2079 | 5.7545 | **4.76x** | 0.99985 | 0.0124 |

BF16 per-round ratios are 3.93-3.99x (D64) and 4.44-4.69x (D128) on fast-mode
medians.

An additional FP16 sweep widens the shape coverage. Sage/device ranges
**3.22x-3.40x** (`1x2x2048x64`), **3.31x-3.40x** (`1x2x8192x64`),
**3.23x-3.40x** (`2x8x512x128`) and **3.39x-3.54x** (`4x32x2048x128`) on
fast-mode medians. The complete per-round tables are in
`agent-results/raw/sage003_tables.md`. Across every measured shape and both
output dtypes the ratio never drops below **3.2x**; there is no shape, dtype,
or round where Sage is competitive.

Peak memory tracks the tensors, not the kernel: Sage peaks 1.09-1.11x the device
op in every case (it additionally materializes the int8 Q/K and the per-block
scales). At `4x32x2048x128` that is 640.07 MiB vs 577.05 MiB.

### Accuracy and finite values

Gate: cosine >= 0.99, relative L2 <= 0.05, all values finite. Observed over all
formal and additional rounds:

| shape | cosine | relative L2 | max abs | finite |
|---|---|---|---|---|
| `1x2x4096x64` | 0.9999197-0.9999245 | 0.01165-0.01213 | 0.0027 | yes |
| `4x32x1024x128` | 0.9999293-0.9999326 | 0.01208-0.01227 | 0.0071 | yes |
| `1x2x2048x64` | 0.9999183-0.9999245 | 0.01169-0.01231 | 0.0042 | yes |
| `1x2x8192x64` | 0.9999250-0.9999309 | 0.01160-0.01245 | 0.0052 | yes |
| `2x8x512x128` | 0.9999301-0.9999340 | 0.01204-0.01211 | 0.0137 | yes |
| `4x32x2048x128` | 0.9999280-0.9999297 | 0.01204-0.01249 | 0.0094 | yes |

BF16 output is marginally worse (cosine 0.99985, relative L2 0.0125) purely
from output rounding; the comparison is done in the output dtype so the kernel
is not charged for the reference's FP16->BF16 conversion.

**Where the error comes from.** Sage is compared against the device op computed
on the *dequantized* Q/K, which isolates quantization from kernel arithmetic:

| stage | relative L2 |
|---|---|
| per-block INT8 quantization of Q | 0.0085-0.0087 |
| per-block INT8 quantization of K | 0.0083-0.0085 |
| device op on dequantized Q/K vs on original Q/K | 0.0120-0.0123 |
| Sage kernel vs device op on the same dequantized Q/K | 0.0008-0.0020 |
| Sage full call vs device op on original Q/K | 0.0117-0.0125 |

Quantization is essentially the entire error budget; the kernel itself
contributes ~0.1-0.2% relative L2. That is the expected trade for INT8 input
quantization and it passes the gate comfortably.

**Finite values, including the ugly cases.** Zeros, zero rows, and
`randn*8` inputs all stay finite. Saturating FP16 inputs (`65504` in Q and K)
produce NaN in *both* Sage and the device fused op — `65504^2` overflows FP16
accumulation before either kernel can help — so this is an input-domain limit,
not a Sage defect, and it is recorded rather than gated.

## Cold compile

Measured in a separate process per shape and per scope, with an empty
`TRITON_CACHE_DIR` (the child verifies `cache_files_at_start == 0` and the run
fails otherwise). `full` = baseline then the Sage full call; `stages` =
quantizer then attention alone.

| shape | scope | first call s | steady ms | compile overhead s |
|---|---|---|---|---|
| `1x2x4096x64` | full | 8.78 | 1.140 | 8.78 |
| `1x2x4096x64` | stages (quantizer) | 3.96 | 0.1896 | 3.96 |
| `1x2x4096x64` | stages (attention) | 4.84 | 1.025 | 4.84 |
| `4x32x1024x128` | full | 7.98 | 5.760 | 7.97 |
| `4x32x1024x128` | stages (quantizer) | 4.10 | 1.560 | 4.10 |
| `4x32x1024x128` | stages (attention) | 3.82 | 3.893 | 3.82 |

The first Sage call costs **8-9 s** of Triton compilation, split between the
quantizer (3.96-4.10 s) and the attention kernel (3.82-4.84 s; 24 cache entries
total). The
device fused op compiles nothing and its first call is 5-8 ms of one-time
runtime setup. For any short-lived workload this dominates the entire
comparison; it is reported separately and excluded from steady state.

## Where the time goes, and the one optimization

Stage attribution on the same tensors (medians of 7 synchronized samples per
round, diagnostic not gated):

| shape | quant (tuned) | quant (default launch) | speedup | attention | attention/device | quant share of Sage |
|---|---|---|---|---|---|---|
| `1x2x4096x64` | 0.169-0.175 | 0.609-0.654 | 3.48-3.86x | 1.034-1.041 | 3.57-3.64x | 15% |
| `4x32x1024x128` | 1.491-1.503 | 7.669-7.725 | 5.10-5.16x | 3.903-3.913 | 3.21-3.25x | 26-28% |

**Main overhead: the quantizer's launch shape, then the attention kernel.**
The per-block INT8 quantizer was launched with Triton's default 4 warps and 3
stages. Each program quantizes one 128-row (Q) or 64-row (K) block whose scale
is a block-wide max reduction; spreading one block over four warps makes those
warps serialise on the reduction, and the kernel is launch-bound rather than
bandwidth-bound at these sizes. Launching it with `num_warps=1, num_stages=1`
removes that traffic.

**The optimization** (`src/flag_attn/runtime/backend/_enflame/sage_attention/quant_per_block.py`):
the two kernel launches now pass `num_warps=_QUANT_NUM_WARPS` (1) and
`num_stages=_QUANT_NUM_STAGES` (1). Nothing else changed — same kernel, same
grid, same arithmetic, same dtype. The change is confined to the Enflame
override, so the generic CPU implementation is untouched and the operator
registry still resolves exactly as before.

**Verification that it is free:**

- bitwise identical int8 and scale outputs across `num_warps` in {1,2,4,8} and
  `num_stages` in {1,2,3}, on 13 shape/layout combinations including partial
  trailing blocks (`300`, `257`, `129` rows) and NHD layout;
- measured 3.48-5.16x faster on the formal shapes and 2.57-5.31x across all
  eight;
- `tests/flag_attn/test_sage_bottleneck.py::test_quant_launch_config_is_bitwise_equivalent`
  re-checks this on every test run;
- the A/B harness re-measures the old launch on every round and fails the run
  if the outputs differ.

**It is not enough.** The full call improved 1.44-2.12x but remains 3.9-4.7x the
device op; recombining the measured stages puts the previous launch at
1.44-1.48x (D64) and 1.58-2.12x (D128) of the tuned full call. The arithmetic bound is blunt: even if quantization were free the
attention kernel alone is **3.2-3.9x** the device op; even if the attention
kernel matched the device op exactly, quantization alone would still leave the
call at **0.59x-1.26x** the device op, i.e. up to 1.26x slower. Both stages have
to improve by an order of magnitude, and tuning cannot get there.

### The bimodal full call

The full call at D128 is **not** a smooth distribution: it alternates between
two latencies about 1.6-1.8x apart, and the slow mode is not a rare tail (up to
80% of calls in one round). Quantization and attention timed *separately* are
each rock steady (0.5% spread), so this only appears when both run in one
call. It is a device-side stall, not a counter artefact: an independent
wall-clock measurement with `synchronize()` on both sides reproduces it, and
the two clocks agree on 16 of 20 calls in a direct comparison. A resident
128 MiB scratch allocation has no effect, and pre-allocating the quantizer's
outputs has no effect, so it is not allocator segment growth.

| shape | dtype | round | fast mode ms | slow mode ms | ratio | slow-call share |
|---|---|---|---|---|---|---|
| `4x32x1024x128` | fp16 | 0 | 5.329 | 8.980 | 1.69x | 30% |
| `4x32x1024x128` | fp16 | 1 | 5.699 | 9.080 | 1.59x | 30% |
| `4x32x1024x128` | bf16 | 0 | 5.682 | 9.009 | 1.59x | 50% |
| `4x32x1024x128` | bf16 | 2 | 5.360 | 9.067 | 1.69x | 10% |
| `4x32x2048x128` | fp16 | 2 | 17.677 | 30.974 | 1.75x | 80% |
| `2x8x512x128` | fp16 | 1 | 0.362 | 0.467 | 1.29x | 50% |

This is handled honestly rather than averaged away: `summarize()` clusters on
the extremes and reports both modes, their ratio, and the slow-call share.
An earlier version anchored the slow threshold on the median, which reported
`slow_call_fraction 0.0` for a distribution that was 60% slow calls once the
slow mode became the median — that bug is fixed and pinned by
`test_summary_reports_bimodality_regardless_of_which_mode_dominates`, and the
device-side reality by
`test_full_call_latency_swing_is_device_side_not_counter_noise`. Every headline
Sage/device ratio above uses the **fast** mode when a round is bimodal, so the
reported slowdown is the smallest one the data supports.

**The attention kernel is not badly configured.** Sweeping the exposed knobs on
*pre-quantized* inputs — the most favourable possible setting, quantization
excluded entirely — every configuration is validated against the device op
before timing:

| shape | device ms | best config | best ms | best/device | worst/device |
|---|---|---|---|---|---|
| `1x2x4096x64` | 0.2763 | bm=128 bn=128 nw=1 ns=1 | 1.055 | 3.82x | 12.11x |
| `4x32x1024x128` | 1.1783 | bm=128 bn=128 nw=1 ns=1 | 3.904 | 3.31x | 17.71x |

`block_m=128, block_n=128, num_warps=1, num_stages=1` is already the default
choice and is the optimum; `block_n=64` costs 1.3-1.5x more, `block_m=64` costs
1.6-2.1x more, and `num_warps>=2` costs 1.2-5.4x more. There is no configuration
that closes a 3.3x gap.

## Reproduction

```bash
cd /workspace/agent-workspaces/sage-sdpa-device-5
bash agent-results/scripts/preflight_gcu5.sh

# formal A/B, both output dtypes, with cold compile
for dt in float16 bfloat16; do
  env PYTHONPATH=/opt/flagtree:$PWD/src taskset -c 160-191 /flagos/bin/python -u \
    agent-results/scripts/sage003_ab.py \
    --shapes 1x2x4096x64 4x32x1024x128 --output-dtype $dt \
    --rounds 3 --pairs 5 --warmup 5 --min-samples 10 --cold \
    --jsonl agent-results/raw/sage003_ab_$dt.jsonl \
    --summary agent-results/raw/sage003_ab_${dt}_summary.json
done

# wider shape coverage
env PYTHONPATH=/opt/flagtree:$PWD/src taskset -c 160-191 /flagos/bin/python -u \
  agent-results/scripts/sage003_ab.py \
  --shapes 1x2x2048x64 1x2x8192x64 2x8x512x128 4x32x2048x128 \
  --rounds 3 --pairs 5 --warmup 5 --min-samples 10 \
  --jsonl agent-results/raw/sage003_ab_shapes_float16.jsonl

# attention launch sweep on pre-quantized inputs
env PYTHONPATH=/opt/flagtree:$PWD/src:agent-results/scripts \
  taskset -c 160-191 /flagos/bin/python -u \
  agent-results/scripts/sage003_attn_sweep.py \
  --jsonl agent-results/raw/sage003_attn_sweep.jsonl

# tests
env SAGE002_TEST_DEVICE=gcu:5 PYTHONPATH=/opt/flagtree:$PWD/src \
  taskset -c 160-191 /flagos/bin/python -m pytest \
  tests/flag_attn/test_sage_attention.py tests/flag_attn/test_sage_bottleneck.py -q

# regenerate every table in this report
/flagos/bin/python agent-results/scripts/sage003_summarize.py
```

`sage003_summarize.py` exits nonzero if a decision record is missing, is not
PASS, or a recorded exit code does not match expectation, so the tables cannot
be regenerated from a stale or failed run.

## Exit codes

| run | exit | meaning |
|---|---|---|
| `sage003_ab_float16` | 0 | PASS |
| `sage003_ab_bfloat16` | 0 | PASS |
| `sage003_ab_shapes_float16` | 0 | PASS |
| `sage003_attn_sweep` | 0 | PASS, every candidate inside the numeric gate |
| `tests` (both Sage files, GCU 5) | 0 | 36 passed |
| `tests` (bottleneck alone, GCU 5) | 0 | 27 passed |
| negative control: cosine gate set impossible | 3 | numeric failure detected |
| negative control: `--min-samples 99` | 3 | missing samples detected |
| negative control: `--expect-index 6` | 4 | wrong device detected, no GCU touched |
| control: `preflight_gcu5.sh` | 0 | DEV 5 is PCI `0000:83:00.0`, SN matches |

The harness also exits 2 when no GCU is visible and 5 when the baseline
dispatch identity changes; those paths are covered by
`dispatch_violations()` and the device guard rather than by a live negative
run, because triggering them would require changing the image or moving the
card. The wrong-device control is real: it runs with `--expect-index 6` and is
refused before any tensor is created.

## Changes in this commit

| file | change |
|---|---|
| `src/flag_attn/runtime/backend/_enflame/sage_attention/quant_per_block.py` | launcher passes `num_warps=1, num_stages=1` for both quantizer calls; bitwise identical outputs |
| `agent-results/scripts/sage003_ab.py` | new: device guard, dispatch proof, 3-round alternating A/B, stage attribution, cold compile, memory, accuracy, fail-closed exits |
| `agent-results/scripts/sage003_attn_sweep.py` | new: attention launch sweep on pre-quantized inputs |
| `agent-results/scripts/sage003_summarize.py` | new: regenerates this report's tables from raw JSONL |
| `agent-results/scripts/preflight_gcu5.sh` | new: refuses unless efsmi DEV 5 is PCI `0000:83:00.0` |
| `agent-results/scripts/README.md` | documents the SAGE-003 scripts and the PCI-based device identity |
| `tests/flag_attn/test_sage_bottleneck.py` | +10 tests: device identity, D64/D128 vs device fused op in FP16 and BF16, quant launch-config bitwise equivalence, harness fail-closed paths (wrong device, missing samples), and the two bimodality tests |
| `tests/flag_attn/test_sage_attention.py` | pins tensors to the assigned device instead of the bare backend name (which resolved to GCU 0 and made the GCU 5 tests skip) |
| `agent-results/raw/sage003_*` | environment, JSONL per-sample records, exit codes, tables |

**Not changed:** default routing, the operator registry, the attention kernel,
the generic CPU implementation, and any other clone. Nothing was pushed or
merged.

## Conclusion

SageAttention on this Enflame GCU is **not faster** than the device fused
attention op and cannot be made faster by launch tuning. The honest result is:

1. The repository baseline `F.scaled_dot_product_attention` is a host fallback
   on this image and must never be used as a device baseline.
2. Against the real device op, Sage is 3.2-4.8x slower across D64 and D128,
   FP16 and BF16, eight shapes, every round.
3. Quantization is 9-54% of the Sage call; the one available optimization
   makes it 2.6-5.3x faster at zero numerical cost, and is kept because it is
   verified, local, and free — not because it changes the verdict.
4. The attention kernel itself, with quantization excluded, is 3.3-3.8x the
   device op at its optimal launch configuration, and every other
   configuration is worse (up to 17.7x).
5. The D128 full call is bimodal (two modes 1.6-1.8x apart); ratios use the
   fast mode, so the reported gap is the most favourable reading.
6. Default routing stays as it is.
