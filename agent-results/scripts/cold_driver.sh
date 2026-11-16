#!/bin/bash
# PARALLAX-008 cold-start driver: one fresh Triton cache per (dtype, route),
# one process per measurement, so the recorded wall is a genuine empty-cache
# first call.  Compile attribution comes from Triton's real compile hook.
set -u
TREE=/workspace/agent-workspaces/parallax-kernel-profile-1
OUT=$TREE/agent-results/raw/PARALLAX-008/cold-attrib.jsonl
SCRIPT=$TREE/agent-results/scripts/parallax008_final.py
: > "$OUT"
export PYTHONPATH=$TREE/src:$TREE:/opt/flagtree
export PARALLAX_SRC=$TREE/src
export PARALLAX_TESTS=$TREE/tests/flag_attn
for dtype in fp16 bf16; do
  for route in shipped candidate; do
    cache=/tmp/p008_cache_${dtype}_${route}
    rm -rf "$cache"
    echo "# cold $dtype $route cache=$cache" >&2
    TRITON_CACHE_DIR=$cache taskset -c 32-63 /flagos/bin/python "$SCRIPT" \
      --mode cold --dtype "$dtype" --route "$route" 2>/dev/null | grep '^{' >> "$OUT"
    du -sh "$cache" >&2
  done
done
echo "COLD_DONE"
