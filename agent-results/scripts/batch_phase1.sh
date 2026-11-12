#!/bin/bash
# Serial phase-1 batch: one baseline+candidate pair per line, preflight before each.
set -u
cd /workspace/agent-workspaces/gla-general-local-state
OUT=$PWD/agent-results/raw/phase1_fp16.jsonl
run() { # B T H K V dtype gln tag [mem]
  echo "=== $(date -u +%FT%TZ) $8 ==="
  bash agent-results/scripts/preflight.sh || { echo "PREFLIGHT_ABORT $8"; return 1; }
  MEM=${9:-0} EXP16=1 EXP32=0 OUT="$OUT" MEM=$MEM \
    taskset -c 128-159 /flagos/bin/python agent-results/scripts/run_case.py "$@"
}
run 2 256 2 32 64 fp16 0.1 gate01_B2T256H2_32x64
run 2 256 2 32 64 fp16 10 gate10_B2T256H2_32x64
run 2 256 2 256 64 fp16 10 gate10_B2T256H2_256x64
run 1 512 8 32 64 fp16 1.0 b1t512h8_32x64
run 1 512 8 64 32 fp16 1.0 b1t512h8_64x32
run 1 512 8 256 64 fp16 1.0 b1t512h8_256x64 1
run 1 512 8 48 80 fp16 1.0 b1t512h8_48x80
run 1 512 8 96 128 fp16 1.0 b1t512h8_96x128 1
run 1 512 8 64 256 fp16 1.0 b1t512h8_64x256
echo "PHASE1_BATCH_DONE $(date -u +%FT%TZ)"
