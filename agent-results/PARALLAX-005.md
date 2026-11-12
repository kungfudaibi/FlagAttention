# PARALLAX-005: fixed-D64 steady-state A/B (checkpoint)

## Identity and scope

| item | value |
| --- | --- |
| candidate SHA | `857cab96d64cf5d0e048c1b53096cd8580e2af8e` |
| exact old-code baseline SHA | `70bdc5e3ed75e877445b9888250b9155523ef852` |
| branch / clone | `agent/enflame-parallax-steady-ab` in `/workspace/agent-workspaces/parallax-steady-ab` |
| baseline worktree | `.baseline-70bdc5` detached at the baseline SHA (read-only; never edited) |
| container device | GCU 1 only (host GCU 5); GCU 0 is never touched |
| CPU affinity | logical CPUs 160-191 (`taskset -c 160-191`) on every GCU command |
| shape | B=2, T=256, H=2, HQ=8, D=64, G=4, dense causal, no window, no varlen |
| dtypes | FP16 and BF16, measured separately |
| seed / scale | `torch.manual_seed(42)`; `scale = 64**-0.5`; identical input and grad-output tensors per route |
| reference | `tests/flag_attn/test_parallax.py::_reference_parallax` in FP32 |
| pass thresholds | relative RMS < 0.005 (FP16), < 0.02 (BF16) for output and dq/dr/dk/dv |

## Measurement definitions

- **Cold**: a fresh process with an empty dedicated `TRITON_CACHE_DIR`, one device call (forward + backward), maximum 600 s wall via `timeout 600`. Record every JIT compile and every autotune config with wall time, the stage reached, cache file count/size before and after, and the real process exit code. A timeout (124) is never converted into a speedup.
- **Steady**: a fresh process per round; after at least two untimed warmup calls on the same tensors, five samples with `torch.gcu.synchronize()` on both sides of each interval. Forward = one `parallel_parallax` call; backward = `torch.autograd.grad` on that forward graph; full = a fresh forward plus its backward. Compilation/autotune during warmup is excluded. Report every sample, median, max-min spread, variance, and `torch.gcu.max_memory_allocated` after `reset_peak_memory_stats`.
- **A/B**: route processes alternate baseline (exact `70bdc5e` code) and candidate (`857cab9`, opt-in `PARALLAX_FIXED_D64=1`) for at least three rounds per dtype. Each route uses its own binary cache; process and cache differences are reported explicitly. The in-process env toggle provides only a preliminary comparison, never the final claim.

## Guardrails and time budget

- `/flagos/bin/python`, `PYTHONPATH` starts with the route's `src` and includes `/opt/flagtree`; the probe asserts the imported `flag_attn.__file__` resolves to the intended worktree.
- Experimental dispatch stays default-off; no production dispatch, PR, remote, SSH/network, credential, or baseline-worktree change.
- Timebox: started 2026-11-12T05:41Z, hard stop 2026-11-12T06:26Z (45 min). Checkpoint committed within the first 5 minutes; results are appended as each matrix cell completes.
- At checkpoint time `efsmi` reported device 1 `DUsed 0%` and no other Python/GCU performance process was running.

## Status

Measurement in progress. This checkpoint fixes SHAs, device, shape, dtypes, seed, thresholds, cold/steady definitions, and budget. Completed cells and raw logs are added below and committed with the scripts.

