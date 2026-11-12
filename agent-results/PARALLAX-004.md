# PARALLAX-004: D64/T256/G4 fixed dense backward experiment (checkpoint)

## Fixed context

- Branch: `agent/enflame-parallax-fixed-d64`, isolated clone `/workspace/agent-workspaces/parallax-fixed-d64`.
- Base/diagnostic commit: `70bdc5e3ed75e877445b9888250b9155523ef852` (also the branch HEAD at start).
- Device: container GCU 1 (host GCU 5) only. GCU 0 is reserved for concurrent GLA-005; no GCU-0 process is started by this task.
- Target shape: B=2, T=256, H=2, HQ=8, D=64 (G=4), dense causal, seed 42.
- Precision: FP16 and BF16, reported separately.
- Reference: `tests/flag_attn/test_parallax.py::_reference_parallax` (FP32, same process/inputs).
- Python: `/flagos/bin/python`; PYTHONPATH=`<clone>/src:<clone>:/opt/flagtree`.
- Per-device-command timeout: 180 s (`timeout 180`), exit codes preserved verbatim.
- Overall timebox: 45 min from 2026-11-12T03:28Z.

## Baseline dispatch (commit 70bdc5e, observed by PARALLAX-003)

For D64/T256/G4 both FP16 and BF16 launch `parallel_parallax_bwd_kernel_dqr_fused` (BT/BS=64 autotune) and `parallel_parallax_bwd_kernel_dkv_grouped` (9 autotune configs). `use_dense_aligned` is True only for BF16, and both of its consumers are unreachable at this shape (dense DQR needs `K >= 128`; dense DKV is shadowed by `use_grouped_dkv`). The experiment must change actual launches, not the boolean.

## Planned minimal change (experimental, shape-gated)

Add `use_fixed_d64` = GCU + (fp16|bf16) + `K == 64` + `T == 256` + `G == 4` + `cu_seqlens is None` + `window_size_left < 0` + `T % BT == 0`, and at that shape only: use the preprocess + `parallel_parallax_bwd_kernel_dqr_dense_aligned` (BS=32) and `parallel_parallax_bwd_kernel_dkv_dense_aligned` (BS=64) with per-query-head dK/dV buffers and the existing GQA reduction. All other shapes keep the existing dispatch byte-for-byte.

## Status

- [x] Task file read, branch/SHA verified, GCU 1 confirmed present (2 SMs), GCU 0 left to GLA-005.
- [ ] Runtime launch probes (kernel names/grids) for FP16 and BF16.
- [ ] FP16/BF16 output + q/r/k/v gradient vs reference.
- [ ] T=63 FP16/BF16 regression.
- [ ] Out-of-dispatch fallback (still original path).
- [ ] Cold/warm separated timing; 3-round steady A/B only if GLA-005 is idle.

Exit codes, raw logs and nonfinite counts are appended below as runs complete. No production routing change, PR or push is made by this task.
