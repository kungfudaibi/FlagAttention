#!/bin/bash
# PARALLAX-008 hot driver: per-stage device time without per-launch sync,
# alternating full-call A/B, peak memory and numerics, default (warm) cache.
set -u
TREE=/workspace/agent-workspaces/parallax-kernel-profile-1
RAW=$TREE/agent-results/raw/PARALLAX-008
OUT=$RAW/hot-final.jsonl
SCRIPT=$TREE/agent-results/scripts/parallax008_final.py
: > "$OUT"
export PYTHONPATH=$TREE/src:$TREE:/opt/flagtree
export PARALLAX_SRC=$TREE/src
export PARALLAX_TESTS=$TREE/tests/flag_attn
taskset -c 32-63 /flagos/bin/python "$SCRIPT" --mode hot \
  --dtypes fp16 bf16 --iters 30 --rounds 7 \
  > >(grep '^{' >> "$OUT") 2> "$RAW/hot-final.err"
echo "HOT_DONE"
