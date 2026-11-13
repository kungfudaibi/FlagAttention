# SAGE-002: Enflame SageAttention bottleneck decision (checkpoint)

Status: in progress (phase 0 checkpoint committed within the first 10 minutes).

## Exact revision and device

- Base/working commit: `ad29b18c9171ad700129e38371bc4dde745d7357`
  (`docs(enflame): record SageAttention versus SDPA baseline`), branch
  `agent/enflame-sage-bottleneck`, clone `/workspace/agent-workspaces/sage-bottleneck`.
- Container GCU 1, `efsmi -L` DEV 1 = PCI **`0000:83:00.0`**, SN
  `A0D5460520864`. GCU 0 (DEV 0, PCI `63:00.0`) is not used.
- Every device command is pinned `taskset -c 160-191`; preflight is
  `agent-results/scripts/preflight_gcu1.sh`.
- Runtime prefix: `PYTHONPATH=/opt/flagtree:src:. FLAG_ATTN_TEST_DEVICE=gcu:1
  taskset -c 160-191 /flagos/bin/python ...`.

## Selected shapes and dtype (provisional)

The repository has no recorded production workload target, so the SAGE-001
long-sequence shapes are used **provisionally**:

- Primary: `[B,H,T,D] = [1,2,T,64]`, `T in {512, 1024, 2048, 4096}`, FP16
  Q/K/V, FP16 Sage output and FP16 SDPA output (same dtype).
- Secondary (repository benchmark default family, if time permits):
  `[B,H,T,D] = [4,32,T,128]`, `T in {1024, 2048, 4096}`.
- Baseline: `torch.nn.functional.scaled_dot_product_attention` on the same
  FP16 Q/K/V, `dropout_p=0.0`, `is_causal=False`.
- Accuracy gate (repository Sage benchmark thresholds, same input compared to
  SDPA): cosine similarity `>= 0.99`, relative L2 `<= 0.05`, finite outputs.
  Raw max-abs error is also recorded for information.

## Scope

Q/K quantization (`per_block_int8`), attention kernel on pre-quantized inputs,
and full call (quantization + attention) are timed separately with
device-event timing, synchronized every sample, 3 rounds, raw samples retained.
Agent-time numbers taken while GDN2-002 occupies GCU 0 are exploratory only;
the orchestrator will run the formal full-path A/B on a quiet host.

## Exploratory baseline reference (SAGE-001, not a new measurement)

SAGE-001 at FP16 `[1,2,T,64]` reported SDPA vs Sage-total medians of
0.048/0.211 ms (T512), 0.054/0.339 ms (T1024), 0.111/0.597 ms (T2048) and
0.263/1.601 ms (T4096), with cosine ~0.99992 and relative L2 ~0.013.
This checkpoint does not claim those numbers as final; fresh phase-1/phase-2
measurements and the full-path A/B follow below.

## Deliverables

- `tests/flag_attn/test_sage_bottleneck.py`: correctness on same FP16 inputs and
  output dtype, plus stage invariants.
- `agent-results/scripts/sage002_full_path_ab.py`: exact-baseline vs candidate
  full-call alternating A/B, per-round raw samples, memory, numerical error,
  nonzero exit on validity failure.
- `agent-results/raw/`: raw logs with exit codes.
