# PARALLAX-006: self-contained kernel static config for Parallax steady state

## Identity and scope

| item | value |
| --- | --- |
| task clone | `/workspace/agent-workspaces/parallax-static-tune` |
| branch | `agent/enflame-parallax-static-tune` |
| initial SHA | `00ebd31efe439aee153e4c68e920dca2b4402f6d` |
| exact old-code baseline | `.baseline-70bdc5`, detached at `70bdc5e3ed75e877445b9888250b9155523ef852`; never edited |
| device | container GCU 1 only (host GCU 5); GCU 0 untouched |
| CPU affinity | every device command under `taskset -c 160-191` (32 CPUs, probe recorded 160-191) |
| shape | B=2, T=256, H=2, HQ=8, D=64, G=4, dense causal, no window, no varlen |
| dtypes | FP16 and BF16, separate processes |
| seed / scale | `torch.manual_seed(42)`; `scale = 64**-0.5` |
| reference | `tests/flag_attn/test_parallax.py::_reference_parallax`, FP32 |
| thresholds | relative RMS < 0.005 (FP16), < 0.02 (BF16), output and dq/dr/dk/dv |
| runtime | `/flagos/bin/python`, `PYTHONPATH=<tree>/src:<tree>:/opt/flagtree` |
| time budget | start 06:29Z, hard stop 07:14Z (45 min) |

## Motivation

PARALLAX-005 measured the old route at this shape: 30 autotune configs
(9 forward + 12 fused DQR + 9 grouped DKV), 380 s cold call, steady full
1.054 ms FP16 / 1.103 ms BF16 on GCU 1. PARALLAX-004's fixed dense path removed
most compiles but was 22-34 % slower in backward. This task keeps the same fast
kernels as the old route (`parallel_parallax_fwd_kernel`,
`parallel_parallax_bwd_kernel_dqr_fused`, `parallel_parallax_bwd_kernel_dkv_grouped`)
and only removes their autotune sweep at the measured target shape.

## Configuration source (measured, per dtype)

Source: raw autotune events in
`/workspace/agent-workspaces/parallax-steady-ab/experiments/logs/p005-base-{fp16,bf16}-cold.log`
plus the three baseline steady rounds in `p005-ab-{fp16,bf16}-r{0,1,2}-baseline.log`.
Each BK=64 candidate's `median_ms` was aggregated across those runs (cold run +
3 steady rounds, 15 values per config). Selected config = minimum aggregate median.

| dtype | kernel | selected (BS, warps, stages) | aggregate median ms | runner-up |
| --- | --- | --- | ---: | --- |
| FP16 | forward | 128, 1, 2 | 0.3786 | 128,1,1 = 0.3834 |
| FP16 | fused DQR | 128, 1, 2 | 1.0991 | 128,1,1 = 1.1098 |
| FP16 | grouped DKV | 64, 1, 1 | 0.5805 | 64,1,2 = 0.5889 |
| BF16 | forward | 128, 1, 2 | 0.3888 | 128,1,1 = 0.3970 |
| BF16 | fused DQR | 64, 1, 1 | 1.0793 | 128,1,2 = 1.0829 |
| BF16 | grouped DKV | 64, 1, 2 | 0.5821 | 64,1,1 = 0.5877 |

Constants: `BK = next_power_of_2(D) = 64`, `BT = 64`, `query_grid = (T/BT, B*HQ) = (4,16)`,
`grouped DKV grid = (T/BS, B*H) = (4,4)` for BS=64. All are the same constexprs the
autotuner would pass; `num_ctas`, `maxnreg` and `ir_override` are None in the old
route and are not passed.

## Change (minimal, opt-in, default off)

`PARALLAX_STATIC_D64=1` (only `1`/`true`/`yes`) enables
`_static_d64_config()`, which returns a per-dtype tile map only when
`device.type == "gcu"`, `cu_seqlens is None`, `window_size_left < 0`,
`B=2, T=256, H=2, HQ=8, K=64, HQ//H=4`. Otherwise it returns `None` and the
original autotuned launch is used. Default (unset) is unchanged.

At the target shape the forward, fused-DQR and grouped-DKV `.fn` entries are
launched directly with the measured `BK/BS/num_warps/num_stages`. This is the
same pattern the repository already uses for varlen forward
(`parallel_parallax_fwd_kernel.fn[...]`). The decorated kernel object still
applies its `triton.heuristics` wrapper, so `IS_VARLEN` is computed exactly as
before; the actual JIT function, grid, output buffers and stores are unchanged.

## Validation

Seed 42, one process per case, `timeout 600`, `taskset -c 160-191`, GCU 1.
Launches recorded by patched `JITFunction.run`; autotune configs counted by
patched `Autotuner._bench`.

| case | switch | dtype | route observed | autotune configs | exit | result |
| --- | --- | --- | --- | ---: | ---: | --- |
| target B2/T256/HQ8 | 1 | FP16 | fwd + fused DQR + grouped DKV | 0 | 0 | 5/5 pass |
| target B2/T256/HQ8 | 1 | BF16 | fwd + fused DQR + grouped DKV | 0 | 0 | 5/5 pass |
| T63 B1/H1/HQ1 | 1 | FP16 | `*_kernel_short` (original short route) | 0 | 0 | 5/5 pass |
| nearby B1/T256/HQ8 | 1 | FP16 | autotuned fwd + fused DQR + grouped DKV | 30 | 0 | 5/5 pass |

The nearby case is the strongest gate check: it differs only in batch size, the
switch is still on, and it still runs the full 30-config old route.

| case/dtype | o rel-RMS | dq | dr | dk | dv | max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 target | 0.000369 | 0.000432 | 0.000355 | 0.000368 | 0.000323 | 0.0335 |
| BF16 target | 0.002999 | 0.003275 | 0.002856 | 0.002949 | 0.002520 | 0.250 |
| FP16 T63 | 0.000360 | 0.000431 | 0.000355 | 0.000368 | 0.000322 | see log |
| FP16 nearby | 0.000360 | 0.000431 | 0.000356 | 0.000368 | 0.000323 | see log |

All outputs and gradients finite, zero nonfinite elements, all metrics under
threshold. Metrics are the same as PARALLAX-005's old-route numbers.

## Cold start (fresh empty cache, 600 s cap)

| dtype | route | call s | compiles | compile s | autotune configs | cache files |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | candidate static | 36.577 | 3 | 36.534 | 0 | 24 / 7.5 MB |
| BF16 | candidate static | 35.842 | 3 | 35.798 | 0 | 24 / 7.5 MB |
| FP16 | old, same session | 394.087 | 30 | 361.856 | 30 | 241 / 71.2 MB |
| BF16 | old, PARALLAX-005 historical | 381.237 | 30 | 359.609 | 30 | 240 / 71.2 MB |

The FP16 old-code cold call was re-run in this session from
`.baseline-70bdc5` with a fresh `/tmp/p006-cache-baseline-fp16-target-cold`
(0 files before, 30 autotune configs, exit 0, validation 5/5 pass) and took
394.087 s, consistent with PARALLAX-005's 379.798 s. The BF16 old-code number is
PARALLAX-005's historical baseline cold run: the BF16 re-run started at 07:03Z
but cannot finish inside the 45-minute box (380 s plus validation), so it is
reported as historical context, not same-session A/B. Differences vs the 005
runs: different Triton cache directory (`/tmp/p006-*` vs `/tmp/p005-*`),
different process, different `parallel.py` for the candidate (static gate
only). This is a compile/cold metric, never a steady speedup.

## Steady A/B

Protocol: 3 alternating rounds per dtype, each round exact old worktree
(`.baseline-70bdc5`, switch 0) then candidate (static, switch 1), 5 synchronized
samples per metric per round, 2 untimed warmups, path probe before samples.
Each route has its own process and its own Triton cache directory copied from
its own cold cache, so compilation and first-call autotune never enter a sample.

Results are in `agent-results/PARALLAX-006-steady.md` and raw in
`experiments/results/PARALLAX-006.jsonl`.

## Limits

- FP16 cold was re-measured same-session against the exact old worktree; BF16
  old-code cold is PARALLAX-005 historical because another ~380 s sweep plus
  validation does not fit the box.
- The static config is only valid for the measured shape/dtype/device class.
  It is a fixed heuristic, not a persisted autotune cache; other shapes keep
  the autotuner.
- One target shape, one device, one seed, three rounds. No profiler attribution.
- Autotune results are not persisted by this FlagTree build, so the default path
  still re-benchmarks per process; this task only removes that for the explicit
  opt-in target shape.

## Reproduction

```bash
cd /workspace/agent-workspaces/parallax-static-tune
# cold target, candidate static, fresh cache
rm -rf /tmp/p006-cache-candidate-fp16-target-cold
experiments/run_p006.sh candidate fp16 cold 0 target cold-cand-fp16-target
# steady alternating A/B (3 rounds)
experiments/ab_driver.sh fp16
experiments/ab_driver.sh bf16
# rebuild evidence JSONL
/flagos/bin/python experiments/p006_evidence.py
```

Artifacts: `experiments/parallax_static_tune.py` (harness),
`experiments/run_p006.sh` (single-command runner), `experiments/ab_driver.sh`
(alternating driver), `experiments/p006_evidence.py` (JSONL builder),
`experiments/static.patch` (the exact code change), `experiments/logs/p006-*.log`
and `experiments/results/p006-*.exit`.
