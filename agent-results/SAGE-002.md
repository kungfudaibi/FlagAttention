# SAGE-002: Enflame SageAttention bottleneck decision

Status: **complete — no-go for a speed claim. No kernel or routing change made.**

## Decision in one paragraph

The Enflame per-block INT8 SageAttention full call is **8.6–14.1x faster than
the repository's literal baseline** `F.scaled_dot_product_attention` on every
measured shape, and it passes the accuracy gate. That comparison is not a
kernel win: on this image `F.scaled_dot_product_attention` never reaches the
GCU. It falls back to `aten::_scaled_dot_product_flash_attention_for_cpu`,
copies Q/K/V to the host and burns 30–3,200 ms of CPU per call. Against the
GCU-native fused attention op (`aten::_scaled_dot_product_flash_attention`),
which is the only fair on-device baseline, the Sage full call is **4.4–9.2x
slower**, and even deleting quantization entirely would still leave the
attention kernel **2.4–3.7x slower** than that baseline. Quantization is the
largest single stage anyway (41% of the full call at T4096, 66% at
`4x32x1024x128`). No launch-configuration change closes the gap, so per the
task rule the optimization stops here with a reproducible no-go rather than a
speculative kernel edit. Default routing is untouched; `src/` and
`benchmark/` have no diff against the base commit.

## Exact revision, device, and scope

- Base commit `ad29b18c9171ad700129e38371bc4dde745d7357`, branch
  `agent/enflame-sage-bottleneck`, clone
  `/workspace/agent-workspaces/sage-bottleneck`.
- Container **GCU 1**, `efsmi -L` DEV 1 = PCI **`0000:83:00.0`**, SN
  `A0D5460520864`. GCU 0 (DEV 0, PCI `63:00.0`) was never used by these runs.
- All device commands pinned with `taskset -c 160-191`; every run preceded by
  `agent-results/scripts/preflight_gcu1.sh`, which refuses unless efsmi DEV 1
  is that PCI address.
- Runtime: `/flagos/bin/python` with `PYTHONPATH=/opt/flagtree:$PWD/src`,
  Python 3.12.13, PyTorch 2.10.0+cpu, torch_gcu 2.10.0+3.7.20260408, Triton
  3.6.0, FlagTree 3.6.0.
- Timing method everywhere: one synchronized device-event pair around exactly
  one public call (`start.record(); fn(); end.record(); synchronize()`),
  warmup excluded. Process CPU time is sampled **before** the synchronizing
  wait, so launch cost and host fallback are visible without charging
  spin-wait to the kernel.

## Selected shapes and dtype

The repository records no production workload target, so the SAGE-001
long-sequence shapes are used **provisionally**, plus the repository
benchmark's own workload family as the business-relevant secondary shape:

| Role | Shape | Dtype |
|---|---|---|
| Provisional primary | `[B,H,T,D] = [1,2,T,64]`, `T in {512,1024,2048,4096}` | FP16 in/out |
| Provisional long | `[1,2,8192,64]` | FP16 in/out |
| Business (repo benchmark) | `[4,32,1024,128]`, `[4,32,2048,128]` | FP16 in/out |
| Extra head dim | `[1,2,4096,64]` output BF16 | FP16 in / BF16 out |

Accuracy gate (repository Sage benchmark thresholds, candidate vs baseline on
the same input): cosine `>= 0.99`, relative L2 `<= 0.05`, all outputs finite.
Max-abs is recorded but not gated, because FP16 output rounding dominates it.

## Finding 1: the repository SDPA baseline is a CPU fallback

`agent-results/scripts/sage002_baseline_dispatch.py`,
raw `agent-results/raw/sage002_baseline_dispatch.jsonl`:

| T | `F.sdpa` event ms | `F.sdpa` CPU ms | vendor op event ms | vendor CPU ms | cosine |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.933 | 30.0 | 0.0784 | 0.00 | 0.999997 |
| 1024 | 4.504 | 140.0 | 0.0775 | 0.00 | 0.999994 |
| 2048 | 6.653 | 220.0 | 0.1320 | 0.00 | 0.999994 |
| 4096 | 14.864 | 480.0 | 0.2817 | 0.00 | 0.999993 |

The CPU profiler for `F.sdpa` shows
`GCU::wrapper_CompositeImplicitAutograd_gcu__scaled_dot_product_attention` →
`aten::_to_cpu` / `aten::_to_copy` (4 copies) →
`aten::_scaled_dot_product_flash_attention_for_cpu`; the vendor op profiles as
`GCU::_scaled_dot_product_flash_attention` with zero host CPU time. Timing
scales with the number of pinned CPUs (T4096: 1 CPU 203.9 ms, 2 CPUs 168.9,
4 CPUs 86.8, 8 CPUs 50.2, 32 CPUs 14.9), which is host-parallel behavior, not
device behavior. `torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)`
and `TORCH_GCU_IGNORE_TOPSATEN_STATUS=1` do not change it.

Consequence: SAGE-001's "SDPA" column (0.048–0.263 ms) was not a GCU SDPA
measurement either. It is not used as a baseline here.

## Finding 2: stage breakdown (Q/K quant, attention, full call)

`agent-results/scripts/sage002_stage_timing.py`, raw
`agent-results/raw/sage002_stage_final.jsonl` (3 rounds x 20 synchronized
samples per scope, medians of round medians; FP16):

| Shape | quant ms | attention ms | full ms | exact SDPA ms | device SDPA ms | quant share |
|---|---:|---:|---:|---:|---:|---:|
| 1x2x512x64 | 0.1578 | 0.1583 | 0.2382 | 0.8054 | 0.0684 | 66.3% |
| 1x2x4096x64 | 0.6494 | 1.0485 | 1.5927 | 14.9497 | 0.2816 | 40.8% |

Per-round full-call medians at T4096: 1.5931 / 1.5927 / 1.5915 ms
(stdev 0.010–0.012 ms), so the split is stable, not an average of drift.

Second breakdown at the business shape, `sage002_stage_d128.jsonl`
(3 rounds x 15 samples):

| Shape | quant ms | attention ms | full ms | exact SDPA ms | device SDPA ms | quant share |
|---|---:|---:|---:|---:|---:|---:|
| 4x32x1024x128 | 7.8858 | 3.9100 | 11.8633 | 131.6719 | 1.1914 | 66.5% |

Quantization dominates at head_dim 128 and still costs 41% at the longest
D64 sequence. `full` is not the sum of the two isolated scopes because the
isolated `quant` scope allocates and the full call reuses tensors; the
decomposition is reported as measured, not normalized.

## Finding 3: exact-baseline vs candidate alternating A/B

`agent-results/scripts/sage002_full_path_ab.py` runs ABBA pairs (baseline,
candidate, candidate, baseline) per round, stores every raw sample with its
pair and position, and exits nonzero if any validity check fails (device
identity, quiet gate, dtype, shape, finiteness, accuracy, sample count, and a
guard that the exact baseline has not silently become a device kernel).

Formal-intent run `sage002_ab_final.jsonl`: 5 rounds x 10 pairs,
`--require-quiet`, both shapes in one process. The quiet gate passed with
`gcu1_foreign_mib_at_startup = 0.0` and no other GCU node holder; the host was
not fully idle (load average 25.6 on 256 CPUs, GDN2-002 finished on GCU 0), so
these remain **agent-time** numbers and the orchestrator's post-turn quiet-host
run is the formal one.

| Shape | candidate ms | exact SDPA ms | device SDPA ms | vs exact | vs device | cosine | rel L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1x2x4096x64 | 1.7434 | 15.0731 | 0.3884 | 8.65x | **0.22x** | 0.999924 | 0.01165 |
| 4x32x1024x128 | 11.6935 | 123.3442 | 1.3010 | 10.55x | **0.11x** | 0.999931 | 0.01207 |

Per-round candidate medians: T4096 1.7139 / 1.7540 / 1.7434 / 1.7049 /
1.7493 ms; D128 12.0641 / 12.0633 / 11.6935 / 11.6314 / 11.6427 ms.
Every round reported `failures: []`.

Supporting runs (2 rounds x 5 pairs unless noted), same harness:

| Shape | candidate ms | exact SDPA ms | device SDPA ms | vs exact | vs device | cosine |
|---|---:|---:|---:|---:|---:|---:|
| 1x2x8192x64 | 4.4305 | 54.5076 | 0.9198 | 12.30x | 0.21x | 0.999925 |
| 4x32x2048x128 | 30.3290 | 426.1974 | 5.1784 | 14.05x | 0.17x | 0.999927 |
| 4x32x1024x128 (3x10) | 11.6844 | 125.2088 | 1.3050 | 10.72x | 0.11x | 0.999931 |

Memory (process view, `torch.gcu`): peak allocated 16.0 MiB at `1x2x4096x64`,
468 MiB at `4x32x1024x128`; reserved 38 MiB / 496 MiB; free device memory
unchanged across each round (41.7 GiB of 41.8 GiB), so no leak per round.

## Finding 4: no configuration closes the gap

`sage002_config_sweep.py` / `sage002_config_sweep.jsonl` exercised every
launch knob and feature flag the Enflame kernel exposes, 15 samples each,
accuracy validated before timing. All configurations passed the gate
(cosine 0.99992–0.99993).

| Config | `1x2x4096x64` ms | vs device | `4x32x1024x128` ms | vs device |
|---|---:|---:|---:|---:|
| default | 1.602 | 0.178x | 11.752 | 0.100x |
| num_warps=2 | 1.815 | 0.157x | 18.681 | 0.063x |
| num_warps=4 | 3.910 | 0.073x | 28.601 | 0.041x |
| block_m=64 | 2.268 | 0.126x | 15.868 | 0.074x |
| block_n=64 | 1.960 | 0.145x | 13.842 | 0.085x |
| block_m=64,block_n=64 | 2.823 | 0.101x | 17.690 | 0.066x |
| num_stages=2 | 1.598 | 0.178x | 11.715 | 0.100x |
| prefetch_v=False | 1.611 | 0.177x | 12.067 | 0.097x |
| prefetch_k_next=False | 1.599 | 0.178x | 11.901 | 0.098x |
| predequantize_k=False | 1.602 | 0.178x | 11.728 | 0.100x |
| qk_scale_before_dot=False | 1.937 | 0.147x | 11.925 | 0.098x |

The best configuration in each shape (`num_stages=2`) differs from the
default by 0.2–0.3%, i.e. noise. The defaults are already the best of the
exposed space, so there is no opt-in change with a plausible net benefit.

## Ceiling argument (why this is a stop, not a tuning gap)

Even if quantization were free, the attention kernel alone must beat the
GCU-native fused op: measured ceiling `device_sdpa / attention` is
**0.42x** (T512), **0.34x** (T1024), **0.39x** (T2048), **0.27x** (T4096),
**0.30x** (`4x32x1024x128`). Simultaneously, quantization alone already costs
more than twice the entire fused baseline at T4096 (0.649 vs 0.282 ms). No
combination of the existing stages can be net-positive against the on-device
fused path, so the honest outcome is a no-go.

## Pass / fail / untested

| Item | Status | Evidence |
|---|---|---|
| GCU 1 identity (PCI `0000:83:00.0`) enforced | **PASS** | `preflight_gcu1.sh` in every run |
| Stage split quant / attention / full, 3 rounds, raw samples | **PASS** | `sage002_stage_final.jsonl`, `sage002_stage_d128.jsonl` |
| Full call vs exact SDPA, alternating A/B, 3+ rounds | **PASS** (8.65x/10.55x faster than the CPU-fallback baseline) | `sage002_ab_final.jsonl` |
| Full call vs GCU-native fused attention | **FAIL** (0.11–0.23x, i.e. 4.4–9.2x slower) | same files |
| Accuracy gate, FP16 in/out | **PASS** cosine 0.99992–0.99993, rel L2 0.0116–0.0125 | stage + A/B rounds |
| Accuracy gate, FP16 in / BF16 out | **PASS** cosine 0.999922, rel L2 0.01223 | `agent-results/raw/sage002_io_dtype_probe.log` |
| BF16 **input** Q/K/V | **FAIL (unsupported)** — Triton `CompilationError: Both operands must be same dtype. Got fp16 and bf16` | same probe log |
| Kernel correctness tests on same tensors/dtype | **PASS** 12 passed | `sage002_tests_new.log` |
| Existing Sage test file | **PASS** 9 passed | `sage002_tests_existing.log` |
| Combined test session on one device | **PASS** 9 passed, 12 skipped (skip is the mixed-GCU guard) | `sage002_tests_both.log` |
| One optional config change with net benefit | **FAIL — none exists**; best is 0.2–0.3% ≈ noise | `sage002_config_sweep.jsonl` |
| Default routing / `src/` / `benchmark/` unchanged | **PASS** empty `git diff` on both paths | `git diff --stat` |
| Formal quiet-host A/B | **NOT RUN by agent** (orchestrator runs it after both turns) | quiet gate passed, host load 25.6 |
| Harness fails closed when a validity check breaks | **PASS** | negative control with cosine gate 0.999999 exits 3 (`sage002_ab_mode_checks.log`) |
| Both `--baseline` modes run correctly | **PASS** | `device` and `exact` single-baseline runs exit 0 |

## Limits and caveats

1. These are **agent-time** measurements taken while the host was shared and
   GDN2-002 was running on GCU 0 (it exited before the final run). GCU 1's
   foreign memory was 0 MiB at startup and no other process held a GCU node,
   but host load average was 25.6, so absolute values may shift on a quiet
   host. Ratios are the intended output, and the direction (candidate slower
   than the on-device fused op at every shape) is far outside the observed
   noise.
2. `F.scaled_dot_product_attention` is a CPU fallback in this image, so any
   "speedup vs SDPA" number from this branch, including SAGE-001's, compares a
   GCU kernel with a host path. It must not be read as a kernel speedup.
3. One process cannot hold contexts on both GCUs on this image: mixing
   `gcu:0` and `gcu:1` aborts with
   `topsMemcpyAsync ... topsErrorInvalidValue`. The new test file therefore
   skips itself when another GCU already has a context, instead of crashing
   the session.
4. `return_lse=False` returns an empty CPU tensor, not `None`; the tests assert
   that existing contract rather than changing it.
5. The kernel hardcodes FP16 for the probability tile and V, so BF16 inputs
   fail compilation. FP16 input with BF16 output works. No fallback was added;
   BF16 input support is out of scope for this decision.
6. Shapes are provisional/business-relevant by proxy (SAGE-001 long sequence
   and the repository benchmark family); no production trace was available.

## Deliverables

| Path | Purpose |
|---|---|
| `agent-results/scripts/sage002_stage_timing.py` | quant / attention / full / exact SDPA / device SDPA, 3 rounds, raw samples, memory, accuracy, nonzero exit on failure |
| `agent-results/scripts/sage002_full_path_ab.py` | exact-baseline vs candidate full-call ABBA A/B, raw per-sample records, memory, numerical error, quiet gate, nonzero exit on any validity failure |
| `agent-results/scripts/sage002_config_sweep.py` | launch-config and feature-flag sweep with accuracy validation |
| `agent-results/scripts/sage002_baseline_dispatch.py` | proves the SDPA baseline's CPU fallback via profiler and CPU time |
| `agent-results/scripts/sage002_summarize.py` | regenerates the tables above from `agent-results/raw/*.jsonl` |
| `agent-results/scripts/preflight_gcu1.sh` | PCI-identity preflight, refuses DEV 1 ≠ `0000:83:00.0` |
| `tests/flag_attn/test_sage_bottleneck.py` | same-input/same-dtype correctness vs SDPA, quant invariants, LSE contract, non-default kernel configs, registry override vs generic fallback |
| `agent-results/raw/` | every raw JSONL sample and log, including the failed and skipped attempts |

Reproduce the tables:

```bash
env PYTHONPATH=/opt/flagtree:$PWD/src /flagos/bin/python \
  agent-results/scripts/sage002_summarize.py agent-results/raw/sage002_*.jsonl
```

Reproduce one measurement:

```bash
bash agent-results/scripts/preflight_gcu1.sh
env PYTHONPATH=/opt/flagtree:$PWD/src taskset -c 160-191 /flagos/bin/python -u \
  agent-results/scripts/sage002_full_path_ab.py \
  --shapes 1,2,4096,64 4,32,1024,128 --rounds 5 --pairs 10 --warmup 5 \
  --require-quiet --out agent-results/raw/sage002_ab_formal.jsonl
env PYTHONPATH=/opt/flagtree:$PWD/src /flagos/bin/python -m pytest \
  tests/flag_attn/test_sage_bottleneck.py -q
```

## Recommendation

Do not enable SageAttention as a speed optimization on Enflame S60 for these
shapes, and do not change default routing. If the project needs a real speedup
here, the actionable direction is to route attention through
`aten::_scaled_dot_product_flash_attention` (0.28–1.30 ms, cosine 0.99999 vs
`F.sdpa`) or to make `F.scaled_dot_product_attention` dispatch to a GCU kernel
at all — both are outside this task's Sage-only scope and are recorded as
findings, not implemented. The Sage path remains available and correct; it is
simply not the fastest on this device.
