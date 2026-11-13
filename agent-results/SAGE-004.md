# SAGE-004: GCU SageAttention attention-kernel bottleneck

Status: **complete — no-go for a kernel/compile-level speed claim. The shipped
kernel is kept unchanged; default routing is unchanged.**

## Decision in one paragraph

The bottleneck is **not** matrix throughput and **not** a launch configuration.
On container GCU 5 it is the **online-softmax bookkeeping wrapped around the two
matmuls**, dominated by the row-max reduction that still runs on FP32 score
tiles. The QK matmul on its own reaches **72 TFLOPS at D64 and 86 TFLOPS at
D128**, but with the softmax chain restored the kernel runs at **8.3 and
17.6 TFLOPS**. Deleting *only* the softmax bookkeeping — keeping both matmuls
and both operand loads — removes **36.6%** of the kernel at `1x2x4096x64` and
**29.0%** at `4x32x1024x128`. The single largest item is the online-softmax row
max (**24.0% / 19.8%**), then the accumulator rescale (7.8% / 8.9%) and the
probability row sum (10.0% / 7.0%). That is a hard structural floor: even a
hypothetical free softmax would leave the kernel at a **2.36-2.48x** device
fused-SDPA cost, so no softmax-side change can close the gap. I tried the one
numerically-free compile-level lever (a cheaper FP16 row max) and it was
**slower**, and its sub-ULP perturbation would also have broken bitwise
equality with the legacy call for no gain. I therefore kept **no** experimental
change. The only code change is test/harness infrastructure.

The formal full-call A/B is still reported in full because it is the required
performance comparison: candidate Sage (the SAGE-003 launch-tuned quantizer plus
the unchanged attention kernel) is **1.38x faster at `1x2x4096x64` and 2.16x
faster at `4x32x1024x128`** than the exact legacy Sage call, with **bitwise
identical outputs** — that gain is SAGE-003's quantizer, re-confirmed here, not
a new attention-kernel result.

## What this task adds over SAGE-003

SAGE-003 proved the device path, landed the quantizer launch fix, and reported
that "the attention kernel with quantization deleted is 3.2-3.8x the device
op". It did not say *where inside the kernel* that time goes, and it never put
the legacy full call and a candidate full call into one alternating A/B. SAGE-004:

- **localises the kernel bottleneck by ablation**, not by config sweeps: each
  variant is a single textual replacement in a copy of the shipped kernel,
  asserted to occur exactly once, so it measures "the shipped kernel minus one
  stage". Both a `both_dots` ceiling (both matmuls kept, all softmax deleted) and
  a `qk_matmul_only` floor (QK matmul only) are measured.
- reports the QK matmul in **TFLOPS against the tile's own FLOPs**, separating
  "the matmul is slow" from "the surface around the matmul is slow".
- runs the **formal full-call A/B** required by the card: exact legacy Sage
  (pre-SAGE-003 4-warp/3-stage quantizer) versus candidate Sage, same FP16 Q/K/V,
  same output dtype and shape, **three alternating rounds, per-sample rows, host
  wall clock and GCU events**, cold compile with an empty `TRITON_CACHE_DIR`,
  and peak device memory per candidate.
- checks the output **and the INT8 quantization intermediates** for finiteness,
  max abs error, relative L2 and cosine, and re-runs the existing Sage suite.

## Exact revision, device, and scope

- Base commit **`8473113cf505af02d3673ddd0279bb32c490689d`** (the accepted
  SAGE-003 head); branch `agent/enflame-sage-attn-kernel-gcu5`; clone
  `/workspace/agent-workspaces/sage-attn-kernel-gcu5`. Nothing pushed or merged.
- Result commit **`5cf5b51713825c8330548f4397919dc4ce99ee1a`**
  (`5cf5b51`) on `agent/enflame-sage-attn-kernel-gcu5`. It carries the two
  scripts, the extended tests, all raw JSONL/exits/environment and this report.
  A commit cannot contain its own hash, so the follow-up commit on top only adds
  `agent-results/raw/sage004_commit.txt` with that hash; no measured artifact
  changes in it.
- Container **GCU 5**, `efsmi -L` DEV 5 = PCI **`0000:83:00.0`**, SN
  `A0D5460520864`, 42976 MiB. DEV 6 (`0000:84:00.0`) and DEV 7
  (`0000:a3:00.0`) were never touched; no other clone or the main repo was
  modified.
- `torch.gcu.set_device(5)` is the first device action in every process; the
  current stream and every tensor are asserted to be `gcu:5` before any
  measurement. `efsmi` reported `Used Size 0 MiB` at pre-flight.
- All device commands run under `taskset -c 160-191` with
  `PYTHONPATH=/opt/flagtree:$PWD/src` and `/flagos/bin/python`.
- Runtime: Python 3.12.13, PyTorch 2.10.0+cpu, torch_gcu
  2.10.0+3.7.20260408, Triton 3.6.0, FlagTree 0.7.0+enflame3.6. Full capture in
  `agent-results/raw/sage004_environment.txt`.
- Device shape: GCU 5 reports **2 multi-processors, 24 SIPs, 12 max threads per
  MP**. This matters for reading the numbers: the grid is small (32 CTAs at D64,
  1024 at D128) and the kernel is one warp wide, so per-CTA serial work — the
  softmax chain — is not hidden by occupancy.

## The bottleneck: softmax bookkeeping, not matrix math

`agent-results/scripts/sage004_profile.py`, raw data
`agent-results/raw/sage004_profile.jsonl`, exit 0
(`agent-results/raw/sage004_profile.exit`). Pre-quantized inputs, so this
isolates the attention kernel and is **never** mixed into the full-call A/B.

Each row is the shipped kernel with exactly one stage removed. "Saves" is the
fraction of the full kernel that stage costs.

| stage removed | D64 `1x2x4096x64` | D128 `4x32x1024x128` |
|---|---|---|
| `both_dots` (both matmuls kept, **all** softmax removed) | **36.6%** | **29.0%** |
| row max `tl.max(qk, 1)` | **24.0%** | **19.8%** |
| probability row sum `tl.sum(p, 1)` | 10.0% | 7.0% |
| accumulator rescale `acc * alpha` | 7.8% | 8.9% |
| `exp2` | 4.1% | 3.6% |
| PV matmul `tl.dot(p, v, acc)` | 11.8% | 7.2% |
| `qk_matmul_only` (QK matmul only) | 88.4% | 79.5% |

Throughput on the same tiles (ideal FLOPs of the tile, not the whole call):

| variant | D64 TFLOPS | D128 TFLOPS |
|---|---|---|
| full kernel | 8.3 | 17.6 |
| both matmuls, no softmax | 13.1 | 24.9 |
| **QK matmul only** | **71.9** | **85.9** |

Read together: the QK matmul alone is **near the device's practical ceiling**
(71.9 / 85.9 TFLOPS), so the matrix path is not the problem. Adding the PV
matmul and the operand loads costs a lot (`both_dots` is only 13.1 / 24.9
TFLOPS), and adding the softmax chain on top costs another **36.6% / 29.0%**.
The row max is the single biggest item and the only one still computed on the
**FP32** score tile, which is why it is expensive relative to `exp2` (which
operates after the tile is narrowed).

**The decisive number is the ceiling.** `both_dots` keeps both matmuls and all
loads and deletes every softmax operation. It is **2.48x** the device fused op at
D64 and **2.36x** at D128. So even a *free* softmax cannot make this kernel
competitive: the remaining PV-matmul-plus-load structure is already 2.4x the
device fused op. That is why this task stops rather than grinding the softmax.

### The one change I tried, and why it is not kept

The row max is the largest single item, so the evidence-supported move was to
run it on FP16 (narrowing the score tile first, exactly as the kernel already
does before `exp2`), cutting the FP32 reduction width. Measured on the real
kernel at the same shapes:

| shape | baseline | FP16 row max | result |
|---|---|---|---|
| `1x2x4096x64` | 1.0314 ms | 1.0868 ms | **5% slower** |
| `4x32x1024x128` | 3.9007 ms | 4.1294 ms | **6% slower** |

It is slower, so the narrowing pass costs more than the FP32 reduction it
replaces. It also perturbs the output at ~1e-4 (sub-ULP), which would break
bitwise equality with the legacy call while buying nothing. **Not kept.**

I also checked the remaining structural knobs that are not block/warp/stage
sweeps, and all are dead ends on this backend:

- **`vector_length` (compile-level vector width)**: the only valid value is 512.
  128/256 fail to compile (`vector.load op requires 3 indices`); 1024/2048 fail
  at link (`relocation ... cannot be used against symbol ...; recompile with
  -fPIC`).
- **`redundant_sip` scheduling**: in this backend's launcher it is implemented
  for `gcu400`/`gcu410` only and is inert on `gcu300`, which is what GCU 5
  reports.
- **`first_tile_d128` fast path**: ablation is inside noise (1.0411 → 1.0340 ms
  at D64, 3.8879 → 3.8796 ms at D128; `agent-results/raw/sage004_pv_probe.log`).
- **INT8 `tl.dot`**: the kernel source already records that INT8 `tl.dot` does
  not lower to an efficient matrix path on this S60 backend, which is why the
  shipped kernel converts to FP16 around the dot. That is not a new finding and
  not something to "fix" blindly.

## Formal full-call A/B: exact legacy Sage vs candidate Sage

`agent-results/scripts/sage004_ab.py`. This is the performance comparison the
card asks for: complete calls including Q/K quantization, attention and output,
on identical externally supplied FP16 Q/K/V, same output dtype and shape.

- **legacy Sage full call** = the pre-SAGE-003 quantizer (Triton default
  4 warps / 3 stages) + the attention kernel.
- **candidate Sage full call** = the shipped SAGE-003 quantizer (1 warp /
  1 stage) + the same attention kernel.
- 3 rounds, 5 ABBA pairs per round (10 timed samples per scope), warm-up
  excluded, one synchronized event pair plus host wall clock around exactly one
  public call, leading candidate alternated per round, per-sample rows in the
  JSONL. A bimodal candidate is scored on its **fast mode** (the most favourable
  reading for the candidate).

Commands (exit codes recorded beside each artifact):

```
sage004_ab.py --shapes 1x2x4096x64 4x32x1024x128 --rounds 3 --pairs 5 \
  --warmup 5 --min-samples 10 --cold \
  --jsonl agent-results/raw/sage004_ab_float16.jsonl \
  --summary agent-results/raw/sage004_ab_float16_summary.json      # exit 0
# same with --output-dtype bfloat16 -> sage004_ab_bfloat16.*       # exit 0
```

### FP16 in, FP16 out

| shape | legacy Sage ms | candidate Sage ms | candidate/legacy | per-round range | rounds faster |
|---|---|---|---|---|---|
| `1x2x4096x64` | 1.5616 | **1.1343** | **0.726x** | 0.716-0.732 | 3/3 |
| `4x32x1024x128` | 11.5106 | **5.3230** | **0.462x** | 0.459-0.464 | 3/3 |

### FP16 in, BF16 out

| shape | legacy Sage ms | candidate Sage ms | candidate/legacy | per-round range | rounds faster |
|---|---|---|---|---|---|
| `1x2x4096x64` | 1.5633 | **1.1408** | **0.726x** | 0.716-0.730 | 3/3 |
| `4x32x1024x128` | 11.5298 | **5.3217** | **0.463x** | 0.459-0.481 | 3/3 |

Host wall clock agrees with the event numbers (e.g. D128 round 0: legacy
11.590 ms event / 11.729 ms wall; candidate 5.325 ms event / 5.466 ms wall), and
host CPU per call is 0.20-0.28 ms for both, i.e. launch-bound on the host and
device-bound in the measurement.

### Where the 1.38-2.16x comes from

Per-round stage attribution, same tensors, diagnostic not gated:

| shape | quantizer speedup | quantizer share of full call | attention share | attention / device SDPA |
|---|---|---|---|---|
| `1x2x4096x64` | 3.09-3.78x | 11-12% | 64-67% | 2.27-2.56x |
| `4x32x1024x128` | 3.95-5.14x | 13-17% | 34% | 2.91-2.95x |

The gain is entirely SAGE-003's quantizer launch fix; the attention kernel is
unchanged and remains **2.3-3.0x** the device fused op. The pre-quantized
attention-only timing is reported separately and never enters the ratio.

### Accuracy and finite values

Gate: cosine >= 0.99, relative L2 <= 0.05, all values finite, for every
comparison and every round. Observed over all formal rounds (both output
dtypes):

| comparison | cosine | relative L2 | max abs | finite |
|---|---|---|---|---|
| candidate vs device fused SDPA | 0.99992 (FP16) / 0.99985 (BF16) | 0.0117-0.0123 | 0.0023-0.0071 | yes |
| legacy vs device fused SDPA | identical to candidate | 0.0117-0.0123 | same | yes |
| candidate vs legacy | **1.0000000** | **0.0** | **0.0** | yes |
| candidate vs `F.sdpa` (host reference) | 0.99992 / 0.99985 | 0.0117-0.0123 | — | yes |

**The candidate output is bitwise identical to the legacy output**
(`torch.equal`, max abs difference exactly 0.0) at both shapes and both output
dtypes. The quantization intermediates are also bitwise identical:

| intermediate | bitwise equal | max abs | finite |
|---|---|---|---|
| `q_int8`, `k_int8`, `q_scale`, `k_scale` | **yes** | 0.0 | yes |

So there is no numerical regression of any kind — the candidate is not merely
within tolerance, it is the same tensor. This is also pinned as a test
(`test_sage004_candidate_full_call_is_bitwise_legacy`).

### Peak device memory

Measured per candidate with their own `reset_peak_memory_stats`:

| shape | legacy peak | candidate peak | device fused peak |
|---|---|---|---|
| `1x2x4096x64` | 11.00 MiB | 11.00 MiB | 10.03 MiB |
| `4x32x1024x128` | 352.04 MiB | 352.04 MiB | 320.52 MiB |

Candidate and legacy peak identically; the device fused op uses 9-10% less
because it never materializes INT8 Q/K or the per-block scales.

### Cold compile

Separate process per shape and per candidate, empty `TRITON_CACHE_DIR`, child
verifies `cache_files_at_start == 0` and the run fails otherwise.

| shape | scope | first call | steady ms | compile overhead |
|---|---|---|---|---|
| `1x2x4096x64` | device fused | 5.12 ms | 0.2893 | 4.8 ms |
| `1x2x4096x64` | Sage full | 8834.1 ms | 1.140 | **8832.9 ms** |
| `4x32x1024x128` | device fused | 8.07 ms | 1.1970 | 6.9 ms |
| `4x32x1024x128` | Sage full | 7932.8 ms | 5.323 | **7927.5 ms** |

The first Sage call costs 7.9-8.8 s of Triton compilation; the device fused op
compiles nothing. That dominates any short-lived workload and is excluded from
steady state, exactly as in SAGE-003.

## The GCU fused-SDPA reference is a device path (checked every run)

`path_evidence` runs in-process on every invocation and fails the run (exit 5) if
either identity changes. The only device attention path is
`aten::_scaled_dot_product_flash_attention` (`GCU::_scaled_dot_product_flash_attention`,
one `topsLaunchKernel`). The repository `F.scaled_dot_product_attention` is still
the **host fallback** (`aten::_scaled_dot_product_flash_attention_for_cpu`,
`aten::_to_cpu`, PCIe copies, ~32 ms host CPU per call at `1x2x512x64`). A CPU
fallback is never used as a device baseline; the harness asserts the device op
is a GCU kernel *and* that `F.sdpa` still falls back, so a future change to
either cannot silently move the goalposts. It is used only as a competitiveness
reference, never as the A/B leader.

## Tests and regression

`tests/flag_attn/test_sage_bottleneck.py` extended with six SAGE-004 tests:

- `test_sage004_profile_pins_one_replacement_per_variant` — every ablation anchor
  must still occur exactly once in the shipped kernel, so the profile cannot
  silently describe code that is not the kernel; asserts `both_dots` keeps the PV
  matmul and `qk_matmul_only` does not.
- `test_sage004_ab_prefers_fast_mode_and_never_scores_a_cpu_fallback` — the A/B
  scores a bimodal candidate on its fast mode, and `dispatch_violations` rejects
  a host-fallback "device baseline" and a device-native `F.sdpa`.
- `test_sage004_ab_fails_closed_on_wrong_device` — exit 4 without touching a GCU.
- `test_sage004_ab_fails_on_missing_samples` — exit 3 rather than a fast-looking
  run on too few samples.
- `test_sage004_candidate_full_call_is_bitwise_legacy` (D64 and D128) — the
  shipped full call must equal the legacy full call bitwise; this is what makes
  "no numerical regression" checkable rather than asserted.

Full suite (`SAGE002_TEST_DEVICE=gcu:5`, `taskset -c 160-191`):

```
pytest tests/flag_attn/test_sage_attention.py tests/flag_attn/test_sage_bottleneck.py -q
42 passed, 1 warning in 57.62s                                       # exit 0
```

The six SAGE-004 tests all run (not skipped): `-k sage004` → `6 passed,
27 deselected`. Negative controls exercised explicitly: wrong device → exit 4;
too few samples → exit 3.

## Exit codes

| artifact | exit |
|---|---|
| `sage004_profile.exit` | 0 |
| `sage004_ab_float16.exit` | 0 |
| `sage004_ab_bfloat16.exit` | 0 |
| `sage004_tests.exit` | 0 |
| wrong-device negative control | 4 (expected) |
| missing-samples negative control | 3 (expected) |

## Changed files

- `agent-results/scripts/sage004_ab.py` — the seeded fail-closed placeholder
  replaced by the formal full-call A/B described above.
- `agent-results/scripts/sage004_profile.py` — reproducible kernel bottleneck
  ablation.
- `tests/flag_attn/test_sage_bottleneck.py` — +180 lines, six SAGE-004 tests.
- `agent-results/raw/sage004_*` — raw JSONL, summaries, exits, environment log,
  probe log.
- **No kernel, operator, routing, or registry change.** The default route and the
  operator registry are exactly as SAGE-003 left them.

## Why this is a negative result, and what I did not test

The card says: if there is no explainable, verifiable change, stop and submit a
negative result. That is the case here, on evidence:

1. The QK matmul is already at 72/86 TFLOPS — there is no matrix-throughput win
   to take.
2. The softmax bookkeeping is 36.6%/29.0% of the kernel, and the `both_dots`
   ceiling shows a *free* softmax still leaves the kernel at **2.36-2.48x** the
   device fused op. The gap is structural in the PV-matmul-plus-load path, not a
   removable softmax cost.
3. The one numerically-free lever on the biggest item (FP16 row max) measured
   **5-6% slower** and would have broken bitwise equality for nothing.

Untested / out of scope, stated so the negative result is not over-read:

- **A different algorithm** (e.g. a flash-attention-style kernel that keeps the
  running max in a cheaper domain and fuses the rescale into the PV matmul, or a
  two-pass softmax with a separate row-statistics kernel). These are kernel
  rewrites, not a "single evidence-supported change", and the `both_dots` ceiling
  says they start 2.4x behind. Not attempted.
- **BF16/FP16 mixed score tiles for the row max**, or moving the reduction to a
  different SIP layout. The FP16 attempt is the cheap version of this and lost;
  the rest needs backend work in the Triton-GCU lowering, not in this kernel.
- **Output dtypes other than FP16 and BF16** (the kernel cannot mix BF16 V with
  its FP16 probability tile), **backward pass**, **causal/masked attention**,
  **GQA with unequal Q/KV head counts**, and **shapes outside the two required
  plus `1x2x512x64` used in the smokes** are not measured.
- **Multi-tenant / concurrent-device behaviour** was not measured; GCU 5 was
  empty at pre-flight and no other GCU was touched.
- **`num_warps`, `num_stages`, `block_m`, `block_n` were deliberately not swept**
  — SAGE-003 already covered that space and the card forbids repeating it.

## Bottom line

- The attention bottleneck is the **online-softmax chain around the matmuls**,
  above all the FP32 row max, not matrix throughput and not launch tuning.
- A **hard floor of 2.36-2.48x** the device fused op survives even deleting all
  softmax, so no softmax-side change can make this kernel competitive; that is
  the explainable, verifiable reason to stop.
- The **complete-call** comparison the card requires passes cleanly: candidate
  Sage is **1.38x / 2.16x** faster than legacy Sage with **bitwise identical**
  outputs and quantization intermediates — that is SAGE-003's quantizer,
  re-confirmed, not a new kernel win.
- **No experimental change is kept**, and **default routing is unchanged**.
