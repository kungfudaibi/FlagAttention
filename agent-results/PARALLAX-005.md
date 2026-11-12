# PARALLAX-005: fixed-D64 cold/steady A/B on GCU 1

## Identity and scope

| item | value |
| --- | --- |
| candidate SHA | `857cab96d64cf5d0e048c1b53096cd8580e2af8e` (report commit below adds only measurement files) |
| exact old-code baseline SHA | `70bdc5e3ed75e877445b9888250b9155523ef852` |
| branch / clone | `agent/enflame-parallax-steady-ab` in `/workspace/agent-workspaces/parallax-steady-ab` |
| baseline worktree | `.baseline-70bdc5`, detached at the baseline SHA; never edited |
| container device | GCU 1 only (host GCU 5); `efsmi` DUsed 0 % on both devices at start, no other performance process |
| CPU affinity | every GCU command under `taskset -c 160-191`; probe recorded `affinity_count=32`, first CPUs 160-162 |
| shape | B=2, T=256, H=2, HQ=8, D=64, G=4, dense causal, no window, no varlen |
| dtypes | FP16 and BF16, separate processes |
| seed / scale | `torch.manual_seed(42)`; `scale = 64**-0.5`; same tensor build for both routes |
| reference | `tests/flag_attn/test_parallax.py::_reference_parallax`, FP32 |
| thresholds | relative RMS < 0.005 (FP16), < 0.02 (BF16), output and dq/dr/dk/dv |
| runtime | `/flagos/bin/python`, `PYTHONPATH=<tree>/src:<tree>:/opt/flagtree` |
| time budget | start 05:41Z, hard stop 06:26Z (45 min); all device work finished by ~06:15Z |

`PARALLAX_FIXED_D64` defaults to off: `_fixed_d64_enabled()` returns False for unset, `0`, `false`, and empty, True only for `1`/`true`/`yes` (static check, exit 0). Candidate runs set it to `1`, baseline runs `0`.

## Measurement definitions

- **Cold**: fresh process, empty dedicated `TRITON_CACHE_DIR` (recorded `cache_files: 0`), one forward + backward device call, `timeout 600`, JIT/autotune probes installed. Every compile and autotune config is logged with wall time; a timeout (124) is never converted into a speedup.
- **Steady**: fresh process per route per round; path-probe call (probes on) then probes removed, 2 untimed warmups, then 5 samples. Each sample is one forward (with `torch.gcu.synchronize()` on both sides), a backward on that graph, and a full fresh forward+backward. Compilation never enters a sample: the steady cache is a copy of that route's own cold cache and the path-probe warmup absorbs autotune re-benchmarking.
- **A/B**: 3 rounds per dtype, each round baseline then candidate, 5 samples per metric per round. Each route runs in its own process from its own worktree with its own cache directory, so process startup, allocator state, and cache are not shared; both use identical tensors, seed, scale, and device.

## Actual kernel paths (probe, device 1 only)

| route | forward | backward launches | autotune configs on first call |
| --- | --- | --- | --- |
| baseline `70bdc5e` | `parallel_parallax_fwd_kernel` | `..._bwd_kernel_dqr_fused`, `..._bwd_kernel_dkv_grouped` | 30 |
| candidate `857cab9`, switch off | same as baseline by construction (gate is the only change) | same | not re-measured (cache-key miss under `timeout 200`, exit 124) |
| candidate `857cab9`, switch on | `parallel_parallax_fwd_kernel` | `..._bwd_kernel_preprocess`, `..._bwd_kernel_dqr_dense_aligned`, `..._bwd_kernel_dkv_dense_aligned`, `..._bwd_kernel_reduce_gqa` | 9 |

The probe asserted `flag_attn` resolved inside `.baseline-70bdc5` for baseline runs and inside the task clone for candidate runs, and recorded the worktree HEAD and `parallel.py` SHA-256 prefix (`51cf806459d5d043` baseline vs `7f27ce4597e3c5cb` candidate) in every log. The candidate genuinely selects fixed dense DQR/DKV; this is a launch record, not an inference from `use_dense_aligned`.

## Correctness (fresh cold processes, exit 0)

| dtype | route | o | dq | dr | dk | dv | max rel-RMS | result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 | baseline | 0.000369 | 0.000432 | 0.000355 | 0.000368 | 0.000323 | 0.000432 | 5/5 pass |
| FP16 | candidate fixed | 0.000369 | 0.000432 | 0.000355 | 0.000440 | 0.000401 | 0.000440 | 5/5 pass |
| BF16 | baseline | 0.002999 | 0.003275 | 0.002856 | 0.002949 | 0.002520 | 0.003275 | 5/5 pass |
| BF16 | candidate fixed | 0.002999 | 0.003275 | 0.002856 | 0.003542 | 0.003175 | 0.003542 | 5/5 pass |

All outputs and gradients finite; zero nonfinite elements; max abs error <= 0.25 (BF16) and <= 0.034 (FP16). Both routes clear their thresholds by more than 5x.

## Cold start (empty cache, one call, 600 s cap, all exit 0)

| dtype | route | forward s | backward s | call s | compiles | compile s | autotune configs | cache after |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 | baseline | 67.887 | 311.911 | 379.798 | 30 | 358.788 | 30 | 240 files / 71.1 MB |
| FP16 | candidate | 68.523 | 11.897 | 80.420 | 13 | 75.625 | 9 | 104 files / 23.0 MB |
| BF16 | baseline | 68.226 | 313.011 | 381.237 | 30 | 359.609 | 30 | 240 files / 71.2 MB |
| BF16 | candidate | 68.670 | 11.849 | 80.519 | 13 | 75.462 | 9 | 104 files / 23.0 MB |

This is the first **completed** exact-old-code cold call for this shape: PARALLAX-004's three attempts were killed at 175-180 s inside the DKV sweep. Here the backward alone needed 312 s of wall time, so those timeouts were compile/autotune limits, not evidence about runtime. The candidate removes 17 backward compiles and 21 autotune configs; the generic forward sweep (~68 s, 9 configs) is unchanged and now dominates its cold start. Cold numbers describe compilation, not steady performance, and are not converted into a speedup.

## Steady A/B (5 samples/round, medians; full matrix in evidence JSONL)

Round-median of forward / backward / full, in ms:

| dtype | round | baseline fwd | candidate fwd | baseline bwd | candidate bwd | baseline full | candidate full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | 0 | 0.421 | 0.433 | 0.826 | 1.139 | 0.995 | 1.337 |
| FP16 | 1 | 0.430 | 0.435 | 0.856 | 1.149 | 1.054 | 1.348 |
| FP16 | 2 | 0.470 | 0.447 | 0.877 | 1.159 | 1.107 | 1.341 |
| BF16 | 0 | 0.433 | 0.455 | 0.870 | 1.154 | 1.049 | 1.351 |
| BF16 | 1 | 0.426 | 0.433 | 0.890 | 1.140 | 1.103 | 1.334 |
| BF16 | 2 | 0.436 | 0.442 | 0.897 | 1.148 | 1.104 | 1.345 |

Median of the three round medians and observed round-median range:

| dtype | metric | baseline | candidate | candidate/baseline |
| --- | --- | ---: | ---: | ---: |
| FP16 | forward | 0.430 (0.421-0.470) | 0.435 (0.433-0.447) | 1.01x |
| FP16 | backward | 0.856 (0.826-0.877) | 1.149 (1.139-1.159) | **1.34x slower** |
| FP16 | full | 1.054 (0.995-1.107) | 1.341 (1.337-1.348) | **1.27x slower** |
| BF16 | forward | 0.433 (0.426-0.436) | 0.442 (0.433-0.455) | 1.02x |
| BF16 | backward | 0.890 (0.870-0.897) | 1.148 (1.140-1.154) | **1.29x slower** |
| BF16 | full | 1.103 (1.049-1.104) | 1.345 (1.334-1.351) | **1.22x slower** |

Within-round spread is 0.016-0.107 ms (5 samples), and round medians overlap for forward, so forward is a tie within noise. The backward regression does not overlap: every baseline round is below every candidate round in both dtypes (largest baseline bwd 0.897 ms, smallest candidate bwd 1.139 ms). Sample variance is reported per round in the JSONL; candidate backward variance is consistently small (stdev 0.008-0.020 ms), so the gap is not sampling noise.

Peak GCU memory (`torch.gcu.max_memory_allocated` after reset, every steady round): baseline 4.08 MB, candidate 5.08 MB (+1.00 MB, +24.5 %); reserved peak 262 MB for both. The candidate allocates separate dK/dV buffers because the fixed dense path bypasses the grouped-KV sharing.

## Interpretation and limits

- The fixed dense DQR/DKV path is **correct but slower in steady state** at this shape: about 0.29 ms more backward per call for both dtypes, while saving ~300 s of one-time compilation. The extra launches (`preprocess` and `reduce_gqa`) plus the split dK/dV buffers outweigh the removal of two autotune dispatches when kernels are warm.
- No speedup is claimed. The only defensible benefit is cold-start/compile time, which is a compile metric, not inference latency.
- Cache and process scope: baseline and candidate ran in different processes from different worktrees with separate cache directories (`/tmp/p005-cache-<route>-<dtype>-cold|steady`). Binary caches are not interchangeable because the cache key covers the modified source. Autotune results are not persisted by this FlagTree build, so each process re-benchmarks during the untimed path probe; no autotune wall time enters a sample.
- The switch-off candidate was not re-timed: its Triton cache key misses the baseline binaries, and a 200 s wrapper expired during recompilation (`p005-candoff-fp16.log`, exit 124). Default-off equivalence is therefore supported by the static gate check and the source diff (the environment read is the only added condition), not by a new timing run — and no number is inferred from that timeout.
- One BF16 baseline cold log (`p005-base-bf16-cold.log`) contains a second, warm run (23.296 s) because the runner script was edited while that shell was executing; the runner re-read the changed file. The first run is the valid empty-cache cold measurement and is the one tabulated. Scripts were frozen during all later runs. This is a harness/process artifact, not a GCU error.
- Model/SSH vs GCU: the SSH route and all GCU calls succeeded (exit 0). No model-gateway calls were needed for measurement. The only non-zero exit is the deliberate 200 s switch-off probe above (124), caused by recompilation, not a kernel failure.
- Not measured: nearby shapes (other T/B/G/D), T63 regression, multi-host or GCU-0 behavior, device-event timing, and profiler-level attribution of the +0.29 ms.

## Decision and next recommendation

Keep `PARALLAX_FIXED_D64` experimental and **off by default**; do not propose a production dispatch change on this evidence. The fixed dense backward is validated for correctness but regresses steady latency ~22-34 % at the target shape and costs 1 MB more peak memory. Recommended next step, only if cold-start matters operationally: reduce the generic forward autotune sweep (9 configs, ~68 s) or persist autotune results, since that is now the dominant remaining cold cost for both routes. If steady latency is the goal, abandon this dispatch for D64/T256/G4 and profile the two extra launches before any further variant.

## Reproduction and artifacts

- `experiments/parallax_steady_ab.py` — cold/steady harness (route/dtype/mode), import and path probes, reference metrics, per-sample timings, peak memory.
- `experiments/run_p005.sh` — single-command runner: worktree + `PYTHONPATH` + cache selection, `taskset -c 160-191`, `timeout 600`, raw log and exit-code file.
- `experiments/ab_driver.sh` — 3-round alternating baseline/candidate driver.
- `experiments/p005_evidence.py` — rebuilds `experiments/results/PARALLAX-005.jsonl` from raw logs (per-log SHA-256, exit codes, import/path probes, cold calls, validations, per-sample steady data and summaries).
- `experiments/logs/p005-*.log` — raw stdout/stderr for all 17 runs; `experiments/results/p005-*.exit` — exit codes.
