#!/bin/bash
# Peak-memory probes, serial, one baseline/candidate pair per shape.
set -u
cd /workspace/agent-workspaces/gla-general-local-state
for spec in "2 32 64" "64 32" "256 64" "64 64"; do
  set -- $spec
  K=$1; V=$2
  echo "=== $(date -u +%FT%TZ) mem ${K}x${V} ==="
  bash agent-results/scripts/preflight.sh || { echo "PREFLIGHT_ABORT ${K}x${V}"; exit 1; }
  MEM=1 EXP16=1 EXP32=0 OUT=$PWD/agent-results/raw/phase3_mem.jsonl \
    taskset -c 128-159 /flagos/bin/python agent-results/scripts/run_case.py \
    2 256 2 "$K" "$V" fp16 1.0 "mem_B2T256H2_${K}x${V}"
done
echo "MEM_BATCH_DONE $(date -u +%FT%TZ)"
