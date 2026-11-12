# PARALLAX-003: FP16 target backward cold-stage diagnosis (checkpoint)

- Baseline: 16d168d20801d3e91aab6797fade72bd7ec41401, branch agent/enflame-parallax-fp16-compile.
- Shape: B=2, T=256, H=2, HQ=8, D=64 (G=4), dense causal, seed 42, FP16 primary; BF16 regression.
- Device: container GCU 0 only (host GCU 4), explicitly `torch.gcu.set_device(0)`. Existing `experiments/parallax_target.py` hardcodes `gcu:1` and will not be executed unchanged.
- Time budget: 35 minutes total; checkpoint by T+5 min; ordinary commands <=180 s; at most one FP16 cold diagnostic at 240 s.
- Stage probes planned: Python import/setup; forward cold (first launch, explicit `torch.gcu.synchronize()`); backward total split into preprocess/DQR/DKV dispatch, first-run compile, autotune config sweep, and warm repeats in the same process; cache state before/after; exit codes and raw stdout/stderr retained under `experiments/`.
- Static hypothesis from baseline diff: `use_dense_aligned` only admits FP16 when `K == 128`; for FP16 D64 T256 G4 it is false, while BF16 D64 T256 G4 is admitted. Also `use_split_dqr = K >= 128`, so even an admitted FP16/ BF16 D64 path uses the autotuned fused DQR rather than fixed dense DQR. To be confirmed by instrumentation, not assumed.
- Checkpoint status: repository clean at baseline; no device run started yet.
