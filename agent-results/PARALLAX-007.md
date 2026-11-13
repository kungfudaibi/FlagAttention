# PARALLAX-007: D64 static-tuning production-candidate decision

**Verdict: STOP — do not promote the static D64 table to a default. Keep
`PARALLAX_STATIC_D64` experimental and default-off.** Cold-start and numerical
evidence are strong, and steady state shows no regression, but the *config
fragility* axis fails the bar: the per-kernel winners are not reproducible
across rounds, and the shipped BF16 DQR tuple is a systematic 5–8 % behind the
best neighbour (up to 10.5 % in one round). No production behaviour change is
committed; this report plus the pytest regression are the deliverables.

- Task: `tasks/PARALLAX-007.md`
- Working clone (the only tree touched):
  `/workspace/agent-workspaces/parallax-static-tune-release`
- Branch `agent/enflame-parallax-static-tune-release`, base
  `dcf487e19e333b796ac7fbb0e535c60aaf7528b2`
- Candidate HEAD at measurement time: `ab9c807b2466fea69bff06e38b0e14847beb601b`
  (`ab9c807`), parent `dcf487e`. Working tree additionally carried the
  untracked probe/driver/test/report files described below.
- Baseline: detached read-only worktree `.baseline-70bdc5` at the exact old
  commit **`70bdc5e3ed75e877445b9888250b9155523ef852`**.
  `parallel.py` sha256 `51cf806459d5d043188b034fe093a76e80b5b6d2aaa161636f7d9e8919ab31a2`.
  Candidate `parallel.py` sha256
  `5f6fc383ec213bcc5b432aec1a19b90857cea2b629907359ef558da3318ea513`.
  `dcf487e` already contains the static patch (same file sha as HEAD), so it
  cannot serve as the "old" baseline; that is why `70bdc5e` was used.
- Device: container **GCU 1 only**, `torch.gcu.set_device(1)`, PCI
  `0000:33:00.0`, SN `A0D4P60511214`, UUID `TS9N12191103`; CPUs pinned to
  **32–63** with `taskset` and re-asserted in-process via
  `os.sched_setaffinity`. One GCU process at a time. GCU 6/7 never touched.
- Stack: `/flagos/bin/python` (CPython 3.12.13), torch `2.10.0+cpu` +
  torch_gcu, Triton `3.6.0`, FlagTree at `/opt/flagtree`,
  `TRITON_F32_DEFAULT=ieee`. Host `vm-dhrj-gd-zone7-d-s60-48g-1-17`,
  kernel 6.8.0-117. Full capture: `agent-results/raw/PARALLAX-007/environment.txt`.
- Target shape: B=2, T=256, H=2, HQ=8, D=K=64, G=HQ/H=4, dense causal,
  window `< 0`, `cu_seqlens is None`, seed 42, scale `0.125`.
  Tolerances: fp16 `0.005`, bf16 `0.02` (relative RMS).

Shipped table under review (`_STATIC_D64_CONFIGS`, `(BS, num_warps, num_stages)`):

| dtype | fwd | dqr | dkv |
|---|---|---|---|
| fp16 | (128, 1, 2) | (128, 1, 2) | (64, 1, 1) |
| bf16 | (128, 1, 2) | (64, 1, 1) | (64, 1, 2) |

## 1. Method

`experiments/parallax007_probe.py` runs exactly one measured process. It pins
the CPU mask, selects device 1, patches `JITFunction.run`,
`JITFunction._do_compile`, `Autotuner._bench` and `Autotuner.check_disk_cache`,
and emits one JSON object per event: `import_probe` (tree, HEAD, file hash,
affinity, device, env, cache knobs, autotuner table), `device_memory_before/after`,
`device_call`, `jit_compile`, `disk_cache`, `launch` (BS/warps/stages/grid/device),
`autotune_config` (per-config bench time), `validation` (o, dq, dr, dk, dv),
`sample`, `steady_summary`, `peak_memory`, `path_probe`.

`agent-results/scripts/parallax007_ab.py` is the formal driver. It re-validates
every raw record and **exits nonzero (5)** on: missing/duplicated records,
nonzero probe exit, device ≠ 1, affinity ≠ 32–63, wrong tree or file hash,
numerical failure, unexpected kernel/grid/static tuple, a candidate target that
autotunes, or a baseline target that does not. `--self-check` replays the
validator on eight deliberately corrupted copies of the real output and refuses
to pass if any corruption is accepted; all eight were rejected (missing sample,
launch off device 1, numeric `all_pass=False`, wrong static tuple, wrong kernel
family, baseline with no autotune record, affinity 32–59).

Phases: `probe` (numerics/paths: target, nearby, T63 × fp16/bf16),
`cold` (empty-cache target, baseline then candidate per dtype),
`steady` (3 alternating A/B rounds per dtype, 5 samples each, shared warm cache),
`cache` (`TRITON_CACHE_AUTOTUNING=1` first process → new-process reuse →
invalidation → static control → warm static),
`config` (one neighbouring tuple per kernel, to probe fragility).

## 2. Cold start, empty cache (one process, first call)

| dtype | route | wall s | compiles | compile s | autotune cfgs | peak alloc B | peak reserved B |
|---|---|---|---|---|---|---|---|
| fp16 | baseline `70bdc5e` | 389.56 | 30 | 364.09 | 30 | 272,711,680 | 274,726,912 |
| fp16 | candidate static | **36.77** | 3 | 36.75 | **0** | **4,276,224** | **6,291,456** |
| bf16 | baseline `70bdc5e` | 386.02 | 30 | 360.37 | 30 | 272,711,680 | 274,726,912 |
| bf16 | candidate static | **36.44** | 3 | 36.42 | **0** | **4,276,224** | **6,291,456** |

Speed-up **10.59×** (fp16) and **10.59×** (bf16) on cold time-to-first-call;
peak reserved memory drops **43.7×** (274.7 MB → 6.3 MB) because the autotune
sweep materialises every config. The 30 tunings are 9 fwd + 12 dqr + 9 dkv;
`jit_compile` timings show the cost is concentrated in DKV (18–28 s per config,
9 configs).

`efsmi -q -d MEMORY` before/after agrees in direction (81.0 → 379.6 for
baseline, 104.9 for the bf16 candidate) but the fp16 candidate reading
(186.0) is inconsistent with the bf16 one, so the efsmi numbers are treated as
auxiliary only and the verdict uses the in-process `torch.gcu` peaks.

## 3. Numerics and paths (all PASS, both dtypes)

Output plus **all four gradients** were compared against
`tests/flag_attn/test_parallax.py::_reference_parallax` on device 1:

| case | dtype | o | dq | dr | dk | dv | verdict |
|---|---|---|---|---|---|---|---|
| target | fp16 | 0.0156 / 3.69e-4 | 0.0335 / 4.32e-4 | 0.0156 / 3.55e-4 | 0.0313 / 3.68e-4 | 0.0313 / 3.23e-4 | PASS |
| target | bf16 | 0.125 / 3.00e-3 | 0.211 / 3.28e-3 | 0.125 / 2.86e-3 | 0.250 / 2.95e-3 | 0.250 / 2.52e-3 | PASS |
| nearby B=1 | fp16 | 0.0117 / 3.60e-4 | 0.0156 / 3.79e-4 | 0.0156 / 3.50e-4 | 0.0313 / 3.41e-4 | 0.0156 / 3.25e-4 | PASS |
| nearby B=1 | bf16 | 0.0938 / 3.00e-3 | 0.164 / 3.30e-3 | 0.0625 / 2.83e-3 | 0.250 / 2.96e-3 | 0.125 / 2.55e-3 | PASS |
| T63 | fp16 | 0.0078 / 3.61e-4 | 0.0044 / 4.34e-4 | 0.0043 / 3.85e-4 | 0.0078 / 3.96e-4 | 0.0156 / 3.39e-4 | PASS |
| T63 | bf16 | 0.0625 / 3.16e-3 | 0.0313 / 2.93e-3 | 0.0625 / 2.99e-3 | 0.125 / 3.09e-3 | 0.0625 / 2.72e-3 | PASS |

(cells are `max_abs / relative_rms`.)

Paths, recorded from the actual launches on device 1:

- target candidate: `parallel_parallax_fwd_kernel` grid `[4,16]`,
  `parallel_parallax_bwd_kernel_dqr_fused` grid `[4,16]`,
  `parallel_parallax_bwd_kernel_dkv_grouped` grid `[4,4]`, with exactly the
  table tuples and **0** `Autotuner._bench` calls.
- target baseline: same three kernels, 30 benchmarked configs, dkv launched
  through a callable grid (recorded as `grid: null`).
- nearby B=1/T=256 (gate must fall through): 30 configs, 30 tunings,
  `[4,8]` grids, dkv BS=32 — the static path is not taken.
- T63 (gate must fall through): `parallel_parallax_fwd_kernel_short` and
  `parallel_parallax_bwd_kernel_short`, grid `[1,1]`, warps 4, stages 1,
  2 compiles, 0 tunings.

## 4. Steady state: 3 alternating A/B rounds (full call, ms)

| dtype | round | baseline med (min–max) | candidate med (min–max) | Δ |
|---|---|---|---|---|
| fp16 | r0 | 0.999 (0.971–1.088) | 1.037 (1.022–1.094) | +3.8 % |
| fp16 | r1 | 1.061 (1.040–1.137) | 1.029 (1.005–1.200) | −3.0 % |
| fp16 | r2 | 1.061 (1.037–1.185) | 1.026 (1.004–1.056) | −3.3 % |
| bf16 | r0 | 1.065 (1.031–1.104) | 1.073 (1.038–1.122) | +0.8 % |
| bf16 | r1 | 1.083 (1.043–1.178) | 0.997 (0.979–1.046) | −7.9 % |
| bf16 | r2 | 1.089 (1.057–1.140) | 1.062 (1.044–1.131) | −2.5 % |

Pooled over the three rounds (15 samples each side): fp16 baseline 1.054 vs
candidate 1.036 (ratio 0.983, Mann–Whitney p = 0.281); bf16 1.078 vs 1.050
(ratio 0.974, p = 0.071). **No steady-state regression**: the sign flips
between rounds, the magnitude is inside the within-round spread, and the
pooled point estimate favours the static path. Forward-only is consistently a
little faster on the static path (fp16 0.441 → 0.412, bf16 0.436 → 0.406);
backward is a wash (fp16 0.863 → 0.848, bf16 0.863 → 0.887).

## 5. Configuration fragility — the reason for the stop verdict

Per-kernel pooled median over the three baseline rounds (same machine, same
process family), with the shipped tuple marked:

| dtype | kernel | shipped | shipped med | pooled best | best med | shipped penalty | rank |
|---|---|---|---|---|---|---|---|
| fp16 | fwd | (128,1,2) | 0.4268 | (128,1,1) | 0.4212 | +1.3 % | 2/9 |
| fp16 | dqr | (128,1,2) | 1.0875 | (128,1,1) | 1.0719 | +1.5 % | 2/12 |
| fp16 | dkv | (64,1,1) | 0.6212 | (64,1,2) | 0.6148 | +1.0 % | 2/9 |
| bf16 | fwd | (128,1,2) | 0.4296 | (128,1,1) | 0.4135 | +3.9 % | 2/9 |
| bf16 | dqr | (64,1,1) | **1.1594** | (128,1,2) | **1.0695** | **+8.4 %** | **5/12** |
| bf16 | dkv | (64,1,2) | 0.6314 | (64,1,2) | 0.6314 | 0.0 % | 1/9 |

Per-round winners are **not reproducible**:

- fp16 fwd: `(128,1,1)` → `(128,1,2)` → `(128,1,1)`
- fp16 dqr: `(128,1,2)` → `(128,1,1)` → `(128,1,2)`
- fp16 dkv: `(64,1,1)` → `(64,1,2)` → `(64,1,2)`
- bf16 fwd: `(128,1,1)` in all three rounds
- bf16 dqr: `(128,1,2)` → `(128,1,1)` → `(128,1,2)`; shipped `(64,1,1)` ranks 3, 3, 5 of 12
- bf16 dkv: `(64,1,1)` → `(64,1,2)` → `(64,1,2)`

Neighbour-substitution runs (steady full call, static table with exactly one
tuple swapped, 5 samples):

| dtype | swap | full med (ms) | vs shipped |
|---|---|---|---|
| fp16 | fwd (128,1,1) | 1.031 | −0.6 % |
| fp16 | dqr (128,1,1) | 0.987 | −4.8 % |
| fp16 | dkv (64,1,2) | 1.063 | +2.5 % |
| bf16 | fwd (128,1,1) | 1.113 | +3.7 % |
| bf16 | dqr (128,1,2) | 1.048 | −2.3 % |
| bf16 | dkv (64,1,1) | 1.029 | −4.1 % |

So a one-step change in `num_stages` moves the *same* kernel by up to ±5 %,
i.e. the same order as the round-to-round drift; the top-3 ordering is below
the measurement floor. This is not dangerous in magnitude — the table sits in
the fast basin and the danger is the slow tail, not the ordering: across the
full menu the worst config is `(32,2,1)` at 0.956 ms for fwd, `(16,1,1)` at
**257 ms** for dqr (≈245× the best), `(32,2,1)` at 1.276 ms for dkv. The
shipped table is in the top 3 of 9–12 everywhere, so it is safe; it just is not
*provably optimal*, and BF16 DQR is measurably off.

Derived (arithmetic from each round's own 30-config bench table, **not** an
end-to-end measurement — labelled as such):

| dtype | round | baseline-as-deployed | shipped static | oracle-best | static vs deployed | static vs oracle |
|---|---|---|---|---|---|---|
| fp16 | r0 | 2.229 | 2.128 | 2.128 | −4.5 % | 0.0 % |
| fp16 | r1 | 2.450 | 2.136 | 2.088 | −12.8 % | +2.3 % |
| fp16 | r2 | 2.183 | 2.114 | 2.089 | −3.1 % | +1.2 % |
| bf16 | r0 | 2.175 | 2.127 | 2.052 | −2.2 % | +3.7 % |
| bf16 | r1 | 2.263 | 2.215 | 2.144 | −2.1 % | +3.3 % |
| bf16 | r2 | 2.181 | 2.236 | 2.116 | +2.5 % | +5.7 % |

("baseline-as-deployed" = the tuples that the empty-cache autotune actually
selected, priced with that round's bench table.) This cuts both ways and is
part of why the verdict is *stop, not reject*: the frozen table is usually
**better than what the tuner actually installs**, because the tuner picks on a
cold, noisy first process — fp16 dqr picks `(32,1,1)`, 9–32 % behind the static
tuple. The problem is that neither the table nor the tuner is reproducible at
the few-percent level, so freezing one snapshot as the default bakes in a
non-reproducible choice. Worst full-call deviation from the oracle is +5.7 %.

## 6. Cross-process disk autotune cache (`TRITON_CACHE_AUTOTUNING=1`)

| run | cache state | wall s | compiles | compile s | tunings | JSON hits |
|---|---|---|---|---|---|---|
| baseline fp16, empty cache | new dir | 389.56 | 30 | 364.09 | 30 | 0 (writes 3) |
| baseline fp16, new process | 3 JSON + 240 binaries | **2.76** | 3 | 0.03 | **0** | **3/3** |
| baseline fp16, JSON deleted | binaries only | 23.75 | 30 | 0.09 | 30 | 0 (rewrites 3) |
| static fp16, empty cache | new dir | 36.77 | 3 | 36.75 | 0 | 0 |
| static fp16, warm binaries | binaries present | 2.69 | 3 | 2.67 | 0 | 0 |
| baseline bf16, new process | 3 JSON | 2.73 | 3 | 0.03 | 0 | 3/3 |
| static bf16, empty cache | new dir | 37.43 | 3 | 37.41 | 0 | 0 |
| static bf16, warm binaries | binaries present | 2.71 | 3 | 2.69 | 0 | 0 |

Findings:

1. The FlagTree Triton autotuner **does** honour `TRITON_CACHE_AUTOTUNING=1`
   here: 3 `*.autotune.json` files under `TRITON_CACHE_DIR`, 3 `disk_cache`
   hits in a fresh process, 0 `_bench` calls, per-kernel hit time 2.69 s
   (fwd, includes first-call setup) / 0.01 s / 0.01 s.
2. **The project wrapper does not enable it on its own.** `compat.libtuner()`
   builds `triton.autotune(configs=..., key=..., **kwargs)` and drops
   `autotune_cache_kwargs`, so `parallel_uses_compat_cache_kwargs` is `False`
   and plain runs show `cache_results=False` / `knobs_cache=False`; nothing is
   written. Only the env var flips it (`cache_results=True`, `knobs_cache=True`).
   A deployment that does not export the env var therefore gets **no**
   cross-process autotune reuse.
3. Deleting the 3 JSONs but keeping the binaries restores a full 30-config
   re-tune (0 hits, 23.75 s) — the cache is keyed on content and is not a
   substitute for the static table. With warm binaries the re-tune costs only
   23.75 s instead of 389 s, so the split is: ~365 s of compilation (unavoidable
   for the 30-config menu, paid once per cache directory) + ~24 s of
   benchmarking per new process.
4. The static route writes **no** autotune JSON (0 files) and is unaffected by
   the variable, so it cannot pollute the tuner cache.
5. **The repeat-process gain is not first-empty-cache gain.** The 389.56 s →
   2.76 s number is a *second* process reusing another process's work. The
   static table's 389.56 s → 36.77 s is achieved on the very first, empty-cache
   process. Reported separately on purpose.

## 7. Regression test

`tests/flag_attn/test_parallax_static_d64.py` (17 tests, marked
`parallel_parallax`):

- gate accepts the exact target shape for fp16/bf16 and returns exactly the
  documented tuples, including a check that the source table itself still holds
  them;
- gate rejects B∈{1,3}, T∈{255,512}, H=1, HQ∈{4,16}, K∈{32,128}, any
  `window_size >= 0`, `cu_seqlens`, float32, and every "off" value of the env
  var (`""`, `0`, `false`, `no`), while `1`/`true`/`yes` enable it;
- target numerics for output + dq/dr/dk/dv against the reference, with exactly
  the static tuples, the expected `[4,16]/[4,16]/[4,4]` grids and **0**
  autotune configs;
- T63 must still take `*_kernel_short` at grid `[1,1]`, warps 4, stages 1 with
  matching numerics;
- float32 input still raises `TypeError("requires bf16 or fp16")`.

Results on GCU 1, CPUs 32–63, fresh `TRITON_CACHE_DIR`:
`17 passed` in 85.49 s cold; `17 passed` in 3.91 s on a warm cache.
Command:

```
cd /workspace/agent-workspaces/parallax-static-tune-release
PYTHONPATH=$PWD/src:$PWD:/opt/flagtree TRITON_CACHE_DIR=/tmp/p007-pytest-cold \
PARALLAX_STATIC_D64=1 TRITON_F32_DEFAULT=ieee \
taskset -c 32-63 /flagos/bin/python -m pytest \
  tests/flag_attn/test_parallax_static_d64.py -q -p experiments.gcu1_plugin
```

The real autotuned nearby path costs ~6 min of cold compile per dtype, so it is
covered by the driver's `probe` phase rather than by the default pytest run.

## 8. Decision and preconditions to revisit

**Keep `PARALLAX_STATIC_D64` default-off.** The stop bar in the task is
"steady-state regression, numerical anomaly, or unstable config dependence".
Two of three are clearly met (no regression; numerics pass), and the third is
not: the per-kernel optimum is not reproducible at the few-percent level, the
shipped BF16 DQR tuple is 8.4 % behind the pooled best (rank 5/12, 10.5 % in
the worst round), and all evidence comes from a single 2-SM GCU 1 slice.

Promotion is defensible later, and the case for it is real (10.6× cold start,
43.7× lower peak reservation, deterministic launches, no steady regression,
and a table that usually beats what the tuner installs). Before flipping the
default, require all of:

1. **Re-fit BF16 DQR.** `(128,1,2)` beat `(64,1,1)` in 2 of 3 rounds (pooled
   1.0695 vs 1.1594) and the direct swap run measured 1.048 vs 1.073 ms
   (−2.3 %). Re-fit from ≥5 rounds per candidate tuple, not 3.
2. **Stability study.** ≥5 alternating rounds × ≥20 samples per config for the
   top-4 tuples per kernel, with an explicit "within noise" decision rule
   (e.g. accept only gaps > 2× the pooled per-config standard error).
3. **Device-scope gate.** The table was fitted on `get_num_sms(1) == 2`. Either
   add an SM-count/device-model condition to `_static_d64_config`, or measure a
   second partition size and gate on it, so the frozen tuples cannot be applied
   blind to a larger GCU slice.
4. **One real-workload check.** End-to-end step time on a model that actually
   hits B2/T256/H2/HQ8/D64/G4, since all numbers here are microbenchmarks.

Interim guidance: keep the switch opt-in for the exact shape, and if it is
turned on for latency-sensitive serving, also export
`TRITON_CACHE_AUTOTUNING=1` for the *non*-target shapes, since the wrapper does
not do it (section 6).

## 9. Not tested (explicit)

- Any GCU partition other than the 2-SM GCU 1 slice; no other host, driver,
  firmware or efsmi version; no GCU 6/7 (forbidden).
- Shapes other than B2/T256 (target), B1/T256 (nearby) and B1/T63 (fallback):
  no T>512, no non-multiple-of-64 T, no B>2, no G≠4, no D≠64, no HQ/H other
  than 4, no `window_size >= 0`, no varlen/`cu_seqlens`, no padding masks.
- Decode / KV-cache path (`parallax_attn_with_kvcache`, `parallax_decode`) —
  untouched by the switch, not exercised here.
- Interaction with `PARALLAX_FIXED_D64` (dense-aligned backward experiment),
  and with `parallel_parallax_bwd` called directly.
- fp8 or any dtype other than fp16/bf16 (float32 is only covered at the gate
  and by the expected `TypeError`).
- Backward with non-contiguous inputs, mutated inputs, double-backward,
  `torch.compile`, CUDA/HCU graph capture, multi-stream or concurrent callers,
  distributed/ECCL.
- Long-running soak, thermal/clock drift, and cross-tenant interference on the
  shared host; steady state is 5 samples per round, so no tail-latency claim.
- Disk-cache behaviour beyond one shape key per dtype in a dedicated directory:
  no multi-key eviction, no shared cache dir across dtypes, no corrupted or
  partially written JSON, no cache relocation between working directories.
- "baseline-as-deployed" in section 5 is derived from bench tables, not an
  end-to-end run pinned to those tuples.
- No end-to-end model or training-step measurement, no throughput under
  batching, no accuracy study beyond the single-seed (42) reference comparison.

## 10. Artifacts and reproduction

Raw evidence (`agent-results/raw/PARALLAX-007/`): `runs.jsonl` (one summary
line per measured process), 35 `<tag>.jsonl` probe streams with their `.log`
(command, tree, cache dir, wall, rc, stdout/stderr), `environment.txt`
(`efsmi -L`, SHAs, versions), `logs/phase-{probe,cold,steady,cache,config}.log`.
The per-run Triton cache directories under `cache/` are 353 MB of build
artifacts and are intentionally git-ignored; they are reproducible from the
commands below.

```
cd /workspace/agent-workspaces/parallax-static-tune-release
git rev-parse HEAD                                  # ab9c807...
git -C .baseline-70bdc5 rev-parse HEAD              # 70bdc5e...
sha256sum src/flag_attn/runtime/backend/_enflame/FLA/parallax/parallel.py
sha256sum .baseline-70bdc5/src/flag_attn/runtime/backend/_enflame/FLA/parallax/parallel.py
efsmi -L

# one phase at a time; the driver sets PYTHONPATH/TRITON_CACHE_DIR per tree,
# PARALLAX_STATIC_D64 per route, and pins taskset + device 1 in-process
taskset -c 32-63 /flagos/bin/python agent-results/scripts/parallax007_ab.py --phase probe
taskset -c 32-63 /flagos/bin/python agent-results/scripts/parallax007_ab.py --phase cold
taskset -c 32-63 /flagos/bin/python agent-results/scripts/parallax007_ab.py --phase steady
taskset -c 32-63 /flagos/bin/python agent-results/scripts/parallax007_ab.py --phase cache
taskset -c 32-63 /flagos/bin/python agent-results/scripts/parallax007_ab.py --phase config

# fail-closed proof of the validator (8 corrupted replays, exits 5 on a miss)
/flagos/bin/python agent-results/scripts/parallax007_ab.py --self-check

# table builders for this report
/flagos/bin/python agent-results/scripts/parallax007_analyze.py
```

Measured phase wall time on GCU 1: `probe` 06:36:10→06:52:05 (15.9 min),
`cold` 06:52:11→07:07:33 (15.4 min), `steady` 07:07:38→07:12:43 (5.1 min),
`config` 07:13:03→07:18:05 (5.0 min), `cache` ≈ 11 min including the two
restarts, pytest 85 s cold / 4 s warm. Phases are resumable:
a tag already present in `runs.jsonl` is revalidated, not re-run.

## 11. What was committed

Only the minimal production-candidate change plus the regression, on
`agent/enflame-parallax-static-tune-release`; **no** push, PR or merge, and no
change outside this clone. Because the verdict is "stop", the production side
is unchanged: the static table stays behind the default-off
`PARALLAX_STATIC_D64` switch (i.e. zero behaviour change). Committed:
`experiments/parallax007_probe.py`,
`agent-results/scripts/parallax007_ab.py`,
`agent-results/scripts/parallax007_analyze.py`,
`tests/flag_attn/test_parallax_static_d64.py`, this report, and the raw
JSONL/log/environment evidence.
