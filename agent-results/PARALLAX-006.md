# PARALLAX-006 checkpoint: static target-shape configuration for Parallax

## Identity

| item | value |
| --- | --- |
| task clone | `/workspace/agent-workspaces/parallax-static-tune` |
| branch | `agent/enflame-parallax-static-tune` |
| initial / checkpoint SHA | `00ebd31efe439aee153e4c68e920dca2b4402f6d` |
| exact old-code worktree | `.baseline-70bdc5`, detached at `70bdc5e3ed75e877445b9888250b9155523ef852`; never edited |
| device | container GCU 1 only (host GCU 5); GCU 0 untouched |
| CPU affinity | every device command under `taskset -c 160-191` |
| shape | B=2, T=256, H=2, HQ=8, D=64, G=4, dense causal, no window, no varlen |
| dtypes | FP16 and BF16, separate processes |
| seed / scale | `torch.manual_seed(42)`; `scale = 64**-0.5` |
| runtime | `/flagos/bin/python`; `PYTHONPATH=<tree>/src:<tree>:/opt/flagtree` |

## Baseline evidence (PARALLAX-005, same device)

- Old path launches `parallel_parallax_fwd_kernel`, `parallel_parallax_bwd_kernel_dqr_fused`,
  `parallel_parallax_bwd_kernel_dkv_grouped`; 30 autotune configs total (9 fwd + 12 DQR + 9 DKV).
- Exact old-code cold call: FP16 379.798 s, BF16 381.237 s (empty cache, 600 s cap, exit 0).
- Steady median (3 rounds x 5 samples): FP16 full 1.054 ms, BF16 full 1.103 ms.
- Raw per-config autotune medians are in
  `/workspace/agent-workspaces/parallax-steady-ab/experiments/logs/p005-base-{fp16,bf16}-cold.log`.

## Config source and static plan

Measured fast configs from the PARALLAX-005 raw logs (per dtype, lowest
`median_ms` among the pruned BK=64 candidates):

| dtype | forward | fused DQR | grouped DKV |
| --- | --- | --- | --- |
| FP16 | BS=128, w1, s1 (0.3765 ms) | BS=128, w1, s2 (1.0692 ms) | BS=64, w1, s2 (0.5734 ms) |
| BF16 | BS=128, w1, s2 (0.3858 ms) | BS=128, w1, s1 (1.0838 ms) | BS=64, w1, s2 (0.5827 ms) |

Plan: opt-in env switch (default off) that, only for this exact dense target
shape, launches the same old forward / fused-DQR / grouped-DKV JIT kernels via
`.fn` with the measured constexprs, bypassing only the autotune sweep. Non-target
shapes and switch-off keep the original route. Candidate must be checked with
real launch probes, FP32-reference gradients, and 3-round alternating A/B.

## Time budget

Start 06:29Z, hard stop 07:14Z (45 min). Checkpoint commit now; code/validation
before ~07:00Z; final A/B and report before 07:14Z. Candidate cold call capped at
600 s and never converted into a speedup.
