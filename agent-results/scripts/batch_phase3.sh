#!/bin/bash
# Phase 3: serial old-vs-opt-in A/B, 3 rounds x 5 synchronized samples.
set -u
cd /workspace/agent-workspaces/gla-general-local-state
for spec in "32x64" "64x32" "256x64" "64x64" "48x80" "96x128" "64x256"; do
  echo "=== $(date -u +%FT%TZ) AB $spec ==="
  bash agent-results/scripts/preflight.sh || { echo "PREFLIGHT_ABORT $spec"; exit 1; }
  ROUNDS=3 SAMPLES=5 B=2 T=256 H=2 DTYPE=fp16 OUT=phase3_ab_fp16.jsonl \
    taskset -c 128-159 /flagos/bin/python agent-results/scripts/drive_ab6.py "$spec" || echo "AB_FAIL $spec"
done
echo "PHASE3_BATCH_DONE $(date -u +%FT%TZ)"
