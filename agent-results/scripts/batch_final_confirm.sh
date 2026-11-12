#!/bin/bash
# Final-code re-confirmation of representative FP16 shapes (post code cleanup).
set -u
cd /workspace/agent-workspaces/gla-general-local-state
run() { # B T H K V dtype gln tag
  echo "=== $(date -u +%FT%TZ) $8 ==="
  bash agent-results/scripts/preflight.sh || { echo "PREFLIGHT_ABORT $8"; return 1; }
  EXP16=1 EXP32=0 OUT=$PWD/agent-results/raw/final_confirm.jsonl \
    taskset -c 128-159 /flagos/bin/python agent-results/scripts/run_case.py "$@"
}
run 2 256 2 64 32 fp16 1.0 final_fp16_64x32
run 2 256 2 256 64 fp16 10 final_fp16_256x64_g10
run 2 256 2 64 256 fp16 1.0 final_fp16_64x256
run 2 300 2 96 128 fp16 1.0 final_fp16_T300_96x128
echo "FINAL_CONFIRM_DONE $(date -u +%FT%TZ)"
