#!/bin/bash
# Extra GLA-006 checks: default-off regression, T not a multiple of 64,
# BF16 FP64 oracle, FP32 tf32/ieee dot path.
set -u
cd /workspace/agent-workspaces/gla-general-local-state
RUN="taskset -c 128-159 /flagos/bin/python agent-results/scripts/run_case.py"
casepair() { # tag B T H K V dtype gln exp16 exp32 out
  echo "=== $(date -u +%FT%TZ) pair $1 ==="
  bash agent-results/scripts/preflight.sh || { echo "PREFLIGHT_ABORT $1"; return 1; }
  EXP16=$9 EXP32=${10} OUT=$PWD/agent-results/raw/${11} \
    $RUN "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$1"
}
# Default production dispatch must be unchanged: flags 0 on the candidate tree
# keeps FP16 32x64 and 256x64 on the old fallback route.
casepair defaultoff_32x64 2 256 2 32 64 fp16 1.0 0 0 phase_extra.jsonl
casepair defaultoff_256x64 2 256 2 256 64 fp16 1.0 0 0 phase_extra.jsonl
# Sequence length not a multiple of the 64-token chunk.
casepair fp16_B2T300H2_32x64 2 300 2 32 64 fp16 1.0 1 0 phase_extra.jsonl
casepair fp16_B2T300H2_64x256 2 300 2 64 256 fp16 1.0 1 0 phase_extra.jsonl

oracle() { # tag shapes dtype T B H e16 e32 extra
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
oracle bf16_parallel "32x64,64x32,256x64,64x64,128x128" bf16 64 1 1 0 0
oracle fp32_tf32 "64x64,32x64" fp32 64 1 1 0 1 --f32-default tf32
oracle fp32_unset "64x64" fp32 64 1 1 0 1 --f32-default unset
echo "EXTRA_BATCH_DONE $(date -u +%FT%TZ)"
