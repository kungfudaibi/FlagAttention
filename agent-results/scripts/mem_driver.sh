#!/bin/bash
# PARALLAX-008 peak-memory driver: one process per (dtype, route) so the peak
# belongs to a single fresh call and nothing from the other route leaks in.
set -u
TREE=/workspace/agent-workspaces/parallax-kernel-profile-1
RAW=$TREE/agent-results/raw/PARALLAX-008
OUT=$RAW/mem.jsonl
: > "$OUT"
export PYTHONPATH=$TREE/src:$TREE:/opt/flagtree
export PARALLAX_SRC=$TREE/src
for dtype in fp16 bf16; do
  for route in shipped candidate; do
    taskset -c 32-63 /flagos/bin/python "$TREE/agent-results/scripts/parallax008_final.py" \
      --mode mem --dtype "$dtype" --route "$route" 2>/dev/null | grep '^{' >> "$OUT"
  done
done
echo "MEM_DONE"
