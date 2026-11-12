# PARALLAX-003: where FP16 target backward spends its ~204 s

## Scope and method

- Baseline 16d168d20801d3e91aab6797fade72bd7ec41401, branch `agent/enflame-parallax-fp16-compile`.
- Shape B=2, T=256, H=2, HQ=8, D=64 (G=4), dense causal, seed 42, output/gradient comparison against `tests/flag_attn/test_parallax.py::_reference_parallax`.
- Container GCU 0 only (`torch.gcu.set_device(0)`, asserted at runtime; device properties: 2 SMs). `experiments/parallax_target.py` hardcodes `gcu:1` and was not executed. Its copy `experiments/parallax_fp16_stages.py` pins GCU 0, keeps the same reference/error metrics, adds per-stage probes and explicit `torch.gcu.synchronize()`.
- Probes wrap `triton.runtime.autotuner.Autotuner._bench` / `.run` and `JITFunction._do_compile` to timestamp each autotune config, each compile and each launch. Logs keep raw stdout/stderr and the process exit code.
- Every device command had a timeout. Ordinary runs: 180 s. Exactly one FP16 diagnostic used 240 s. Session stayed within the 35 min timebox.

## Static routing divergence (measured, experiments/logs/static-divergence.log)

`parallel_parallax_bwd` at this shape for both dtypes:

| decision | FP16 | BF16 |
|---|---|---|
| `use_dense_aligned` | False | True |
| `use_split_dqr` (`K >= 128`) | False | False |
| DQR kernel launched | `..._dqr_fused` | `..._dqr_fused` |
| DKV kernel launched | `..._dkv_grouped` | `..._dkv_grouped` |
| DQR tuner configs (BK=64) | 12 | 12 |
| DKV tuner configs (BK=64) | 9 | 9 |

Two consequences:

1. The FP16/BF16 difference is confined to the `use_dense_aligned` boolean itself. Both of its consumers are gated on `use_split_dqr` (parallel.py:2422) or on `not use_grouped_dkv` (parallel.py:2500). With K=64, `use_split_dqr` is False; with T=256, G=4, NQ=4 > 2, `use_grouped_dkv` is True. Therefore commit 16d168d changes no kernel at D64/T256/G4. The `dkv_dense_aligned` launch is shadowed by the grouped branch and the `dqr_dense_aligned` branch is unreachable for D64. This is observed dispatch logic, not a hypothesis.
2. Consequently the FP16 cold cost is not an FP16-only path; it is the cost of the generic autotuned backward that BF16 also takes.

## Stage timings

### FP16, truly cold (empty TRITON_CACHE_DIR), 240 s diagnostic — experiments/logs/fp16-cold-240s.log, EXIT=124

| stage | wall | detail |
|---|---|---|
| forward | 69.05 s | 9 JIT compiles (~6.1-9.4 s each) + 9 autotune configs; best BK64 BS128 w1 s2 |
| backward DQR (`dqr_fused`) | 117.61 s | 12 autotune configs, 12 compiles; BS=16 configs alone cost 7.9-11.9 s each and bench at ~260 ms/call |
| backward DKV (`dkv_grouped`) | 36.5 s then timeout | 2 of 9 configs at 18.3 s and 18.2 s each |
| reference | not reached | process killed at 240 s |

Projection from measured per-config costs: a fully cold FP16 forward+backward needs about 69 + 118 + 9*20 = ~370 s, so the known 204 s figure cannot be a fully cold run.

### BF16, cold (empty cache), 180 s — experiments/logs/bf16-cold-180s.log, EXIT=124

Forward 69.21 s (9 compiles, same best config), then the `dqr_fused` sweep ran 11 of 12 configs in ~117 s and was killed. Kernel set and per-stage costs match FP16 within noise.

### FP16 reusing the binary cache from the 240 s run — experiments/logs/fp16-warm2-180s.log, EXIT=124

DQR compiles drop to 0.002-0.013 s (cache hits) and the DQR sweep is 13.0 s instead of 117.6 s, though it selects a different best config (BS128 s2 vs s1): the tuner choice is noise-sensitive. Forward still re-benchmarks: 8.2 s. DKV grouped had only 2 of 9 configs cached, so the other 7 recompiled at 18-25 s each and the run hit 180 s. This residual cost is exactly what produces a ~204 s "second attempt" after a partially successful earlier attempt.

### FP16 with the shared host cache and TRITON_CACHE_AUTOTUNING=1 — experiments/logs/fp16-warmhost-180s.log, EXIT=124

Forward 69.2 s and DQR 117 s again: binaries from the earlier GCU-1 session do not satisfy these cache keys, and this FlagTree build does not persist autotune results for these kernels by default (`Autotuner(cache_results=False)` at triton/runtime/autotuner.py:46, enabled only by TRITON_CACHE_AUTOTUNING). A new process therefore re-sweeps every config.

## Findings

1. The 204 s is autotune/compile time in the backward, not kernel execution and not a first-run kernel launch. Warm steady state for the same call is sub-millisecond per launch (earlier 0.97-1.00 ms backward in experiments/steady.log); the cold cost is dominated by 12 DQR + 9 DKV config compiles at ~6-20 s each on this toolchain.
2. The DQR sweep is the single largest backward stage (~118 s cold) and includes clearly unprofitable BS=16 configs (~260 ms per bench call, ~8-12 s compile each), 60-190x slower than the selected tile.
3. The DKV grouped sweep is the longest tail (~20 s per config, 9 configs ~180 s) and is the reason a partially cached rerun still finishes at ~204 s: forward and DQR are cache hits, DKV is not.
4. Commit 16d168d does not affect D64/T256/G4 at all, for the reason above. The PARALLAX-002 statement that BF16 is "routed through the validated dense kernel" is inaccurate for this shape: the dense DKV launch is shadowed by `use_grouped_dkv`, and dense DQR requires K >= 128. Both dtypes take the identical autotuned path, which is why their cold stage timings match.

## Inferences (not directly proven)

- The reported 203.97 s backward in experiments/dense64-fp16-rerun.log is best explained by a partially warm binary cache (forward + DQR hits, DKV misses), not by a genuinely cold run.
- The ~20 s per-config DKV compile time, versus ~6-9 s for the forward and DQR kernels, suggests larger generated code / longer GCU lowering for the grouped kernel; not profiled.

## Why no production routing change was committed

A minimal candidate is supported by the evidence: for D64/T256/G4, call `parallel_parallax_bwd_kernel_dqr_dense_aligned` (BS=32) and `parallel_parallax_bwd_kernel_dkv_dense_aligned` (BS=64), the fixed kernels the BF16 commit intended to use. Both are non-autotuned single-compile kernels, and the shape satisfies their static asserts (K == BK == 64, BT % BS == 0, BT == BS == 64, T % BT == 0), so expected cold cost drops from ~370 s to roughly the forward sweep (~69 s) plus two compiles.

It was not committed because the remaining budget did not allow the required verification: FP16 target output plus q/r/k/v gradients against the reference, BF16 regression, small-shape regression (`tests/flag_attn/test_parallax.py`, previously 2 passed / 30 deselected), and separate cold/warm timings. Per the task rules an unverified routing change is not shipped. Also note that merely admitting FP16 into `use_dense_aligned` would be a no-op unless `use_split_dqr` and the `use_grouped_dkv` shadowing are addressed, which is a larger change than it appears.

## Next minimal experiment (recommended, ~10 min device time with a warm forward)

On a clean binary cache, patch only the D64/T256/G4 dispatch so both dtypes take the two fixed dense kernels, and validate in one process:

1. FP16 target vs `_reference_parallax`: output and q/r/k/v finite, max_abs, relative_rms (expected FP16 rel-RMS about 4e-4 from the historical rerun).
2. BF16 target: same metrics (expected about 3e-3).
3. Small-shape regression: the dense FP16/BF16 cases in `tests/flag_attn/test_parallax.py` (previously 2 passed, 30 deselected).
4. Cold vs warm timings reported separately, with the cold number labelled compile-dominated.

## Artifacts

- experiments/parallax_fp16_stages.py — GCU-0 stage/compile/autotune probe (seed 42, reference metrics, `--mode cold|warm|compare`, `--phase forward|backward|both`).
- experiments/parallax_fp16_divergence.py — static dispatch divergence probe.
- experiments/logs/fp16-cold-240s.log, bf16-cold-180s.log, fp16-warm2-180s.log, fp16-warmhost-180s.log, static-divergence.log — raw stdout/stderr with exit codes.

## Limits

- Only container GCU 0 (2 SMs) was used; timings are not transferable to GCU 1 or other hosts.
- No device-event timing was taken; all numbers are synchronized Python wall times and include dispatch overhead.
- No profiler trace of the compiler itself; why one DKV config needs ~20 s remains open.
- Autotune best-config selection differed between cold and warm-binary runs for the forward kernel, so config-level conclusions are noise-sensitive.
- No FP16 improvement was claimed: no routing/kernel change is included in this branch.
