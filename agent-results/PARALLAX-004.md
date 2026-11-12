# PARALLAX-004: fixed D64 Parallax backward experiment

## Scope

Branch `agent/enflame-parallax-fixed-d64` is based on diagnostic commit `70bdc5e3ed75e877445b9888250b9155523ef852`. The only code change is an opt-in `PARALLAX_FIXED_D64=1` branch for GCU, FP16/BF16, dense causal B/H-independent `T=256`, `D=64`, `G=4` backward. It launches fixed dense-aligned DQR and DKV kernels instead of the existing fused DQR and grouped DKV autotune route. The default environment value is off. No production dispatch or existing PR was changed.

The task ran on container GCU 1 (host GCU 5), seed 42, `/flagos/bin/python`, with host NUMA 1 logical CPUs 160–191. GLA-005 used GCU 0 and CPUs 128–159. The DeepSeek Harness continuation failed on HTTP 503 from the company model gateway; Codex continued the checks from the checkpoint. This was a model-service failure, not a GCU kernel error.

## Actual path and correctness

The runtime probe at B2/T256/H2/HQ8/D64 reported only device 1 and the following real backward launches for both FP16 and BF16: `parallel_parallax_bwd_kernel_preprocess`, `parallel_parallax_bwd_kernel_dqr_dense_aligned`, `parallel_parallax_bwd_kernel_dkv_dense_aligned`, and `parallel_parallax_bwd_kernel_reduce_gqa`. The generic forward kernel still autotunes; the experiment removes the two backward autotune selections only.

Against `tests/flag_attn/test_parallax.py::_reference_parallax` (same input/seed, FP32 reference), all finite outputs and q/r/k/v gradients passed the relative RMS thresholds of 0.005 for FP16 and 0.02 for BF16:

| Case | Maximum relative RMS across output and gradients | Result | Path |
| --- | ---: | --- | --- |
| FP16, B2/T256/H2/HQ8/D64 | 0.000440 | 5/5 pass | fixed dense DQR/DKV |
| BF16, B2/T256/H2/HQ8/D64 | 0.003542 | 5/5 pass | fixed dense DQR/DKV |
| FP16, B2/T63/H2/HQ8/D64 | 0.000449 | 5/5 pass | original short backward |
| BF16, B2/T63/H2/HQ8/D64 | 0.003455 | 5/5 pass | original short backward |

The T63 cases also verify a shape outside the new dispatch retains the old short path. Logs record kernel names, grids, device, nonfinite counts, max absolute errors, relative RMS, and exit 0. `experiments/parallax_fixed_d64.py` is the device-1 reference/probe harness. `experiments/gcu1_plugin.py` supports running repository pytest cases on device 1.

## Compilation and timing

The FP16 candidate reached the device call and reference validation in 10.226 s in a process with already populated compiler cache; the BF16 candidate device call took 79.960 s, including 75.265 s of JIT compilation reported by the probe. These are cold/process wall times under different cache states, so they cannot be compared as performance numbers.

The original FP16 path timed out twice at 180 and 175 s while compiling/autotuning the grouped DKV configs, before producing a reference result. Logs `p004-base-fp16-valid.log` and `p004-base-fp16-valid2.log` preserve both exit-124 cases. The original BF16 path also reached the 175 s timeout before producing a device-call result (`p004-base-bf16-valid.log`, exit 124). These are cold-start limits, not steady-state latency measurements. A three-round alternating warm A/B could not be completed within the per-command timebox. Therefore no runtime speedup or production recommendation is claimed.

## Decision

The fixed backward branch is a validated experiment for one target shape and two precisions. Keep the opt-in default off. A follow-up should obtain a completed baseline warm run, alternate baseline/candidate steady samples with GLA idle, and test additional B/H and nearby T/G/D shapes before any production routing change. The compile-time improvement is plausible because fixed backward launches avoid their autotune sweeps, but the generic forward still compiles and the baseline timeout prevents a quantified comparison.

## Reproduction

Set `PYTHONPATH=<task clone>/src:<task clone>:/opt/flagtree`. Run `PARALLAX_FIXED_D64=1 /flagos/bin/python experiments/parallax_fixed_d64.py --dtype fp16 --probe` and repeat with `--dtype bf16`; use `--T 63` for each outside-shape regression. Set `PARALLAX_FIXED_D64=0` for the original path. Each device command was wrapped in a timeout no greater than 180 s, except the BF16 candidate run where the wrapper was 220 s; its measured call completed in 79.960 s. `experiments/results/PARALLAX-004.jsonl` contains the parsed JSON events and each full console log's SHA-256 digest and exit code. The full console logs remain in `experiments/logs/` in the task clone.
