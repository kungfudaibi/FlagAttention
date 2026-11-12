#!/bin/bash
# Phase 2: BF16 controls through the existing parallel route, then FP32 opt-in smoke.
set -u
cd /workspace/agent-workspaces/gla-general-local-state
run() { # B T H K V dtype gln tag exp16 exp32 out
  echo "=== $(date -u +%FT%TZ) $8 ==="
  bash agent-results/scripts/preflight.sh || { echo "PREFLIGHT_ABORT $8"; return 1; }
  EXP16=${9:-0} EXP32=${10:-0} OUT=$PWD/agent-results/raw/${11:-phase2.jsonl} \
    taskset -c 128-159 /flagos/bin/python agent-results/scripts/run_case.py \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8"
}
run 2 256 2 32 64 bf16 1.0 bf16_B2T256H2_32x64 0 0 phase2_bf16.jsonl
run 2 256 2 64 32 bf16 1.0 bf16_B2T256H2_64x32 0 0 phase2_bf16.jsonl
run 2 256 2 256 64 bf16 1.0 bf16_B2T256H2_256x64 0 0 phase2_bf16.jsonl
run 2 256 2 64 64 bf16 1.0 bf16_B2T256H2_64x64 0 0 phase2_bf16.jsonl
run 2 256 2 128 128 bf16 1.0 bf16_B2T256H2_128x128 0 0 phase2_bf16.jsonl
run 1 128 1 64 64 fp32 1.0 fp32_B1T128H1_64x64 0 1 phase2_fp32.jsonl
run 1 128 1 32 64 fp32 1.0 fp32_B1T128H1_32x64 0 1 phase2_fp32.jsonl
run 1 128 1 64 32 fp32 1.0 fp32_B1T128H1_64x32 0 1 phase2_fp32.jsonl
run 1 128 1 32 32 fp32 1.0 fp32_B1T128H1_32x32 0 1 phase2_fp32.jsonl
echo "PHASE2_BATCH_DONE $(date -u +%FT%TZ)"
