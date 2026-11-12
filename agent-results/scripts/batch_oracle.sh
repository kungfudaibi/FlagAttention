#!/bin/bash
# Independent FP64 CPU recurrence oracle for small FP16/FP32 shapes.
set -u
cd /workspace/agent-workspaces/gla-general-local-state
run() { # tag shapes dtype T B H exp16 exp32 extra...
  tag=$1; shapes=$2; dtype=$3; T=$4; B=$5; H=$6; e16=$7; e32=$8; shift 8
  echo "=== $(date -u +%FT%TZ) oracle $tag ==="
  bash agent-results/scripts/preflight.sh || { echo "PREFLIGHT_ABORT $tag"; return 1; }
  PYTHONPATH=$PWD/src:$PWD:/opt/flagtree \
  FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP16_STATE=$e16 \
  FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP32_STATE=$e32 \
    taskset -c 128-159 /flagos/bin/python agent-results/scripts/gla006_oracle.py \
    --shapes "$shapes" --dtype $dtype --T $T --B $B --H $H \
    --exp-fp16 $e16 --exp-fp32 $e32 "$@" 2>&1 | tee -a agent-results/raw/oracle.log
}
run fp16_parallel_32x64_64x32_64x64 "32x64,64x32,64x64" fp16 64 1 1 1 0
run fp32_parallel_64x64_32x64_64x32_32x32 "64x64,32x64,64x32,32x32" fp32 64 1 1 0 1
echo "ORACLE_BATCH_DONE $(date -u +%FT%TZ)"
