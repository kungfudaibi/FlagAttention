#!/bin/bash
# PARALLAX-006 single-command runner: one GCU-1 process at a time.
# usage: run_p006.sh <baseline|candidate> <fp16|bf16> <cold|steady> <round> <case> <tag>
set -uo pipefail
route=$1; dtype=$2; mode=$3; round=${4:-0}; case=${5:-target}; tag=${6:-$1-$2-$mode-$case}
root=/workspace/agent-workspaces/parallax-static-tune
case "$route" in
  baseline) tree=$root/.baseline-70bdc5; export PARALLAX_STATIC_D64=0 ;;
  candidate)
    # Switch on for every case: the gate itself must reject non-target shapes.
    tree=$root; export PARALLAX_STATIC_D64=1 ;;
  *) echo "bad route" >&2; exit 2 ;;
esac
cache=/tmp/p006-cache-$route-$dtype-$case-$mode
if [ "$mode" = steady ] && [ -d "$cache" ]; then :; fi
mkdir -p "$cache" "$root/experiments/logs" "$root/experiments/results"
log=$root/experiments/logs/p006-$tag.log
: > "$log"
{
  echo "CMD route=$route dtype=$dtype mode=$mode round=$round case=$case tag=$tag"
  echo "TREE=$tree HEAD=$(git -C "$tree" rev-parse HEAD)"
  echo "PARALLAX_STATIC_D64=${PARALLAX_STATIC_D64:-unset} TRITON_CACHE_DIR=$cache CACHE_FILES_BEFORE=$(find "$cache" -type f | wc -l)"
  echo "DATE_START=$(date -u +%FT%TZ)"
  echo "DEVICE_BEFORE:"; taskset -c 160-191 efsmi 2>/dev/null | grep -E "^\| [01] |DUsed|MiB" | head -8
  echo "OTHER_PROCS:"; ps -eo pid,etime,args | grep -E "flagos/bin/python" | grep -v grep | head -5
} | tee -a "$log"
export PYTHONPATH="$tree/src:$tree:/opt/flagtree"
export TRITON_CACHE_DIR="$cache"
export TRITON_F32_DEFAULT=ieee
cd "$root" || exit 2
timeout 600 taskset -c 160-191 /flagos/bin/python "$root/experiments/parallax_static_tune.py" \
  --route "$route" --dtype "$dtype" --mode "$mode" --case "$case" --round "$round" \
  --samples "${SAMPLES:-5}" --warmups 2 >> "$log" 2>&1
code=$?
echo "DATE_END=$(date -u +%FT%TZ)" | tee -a "$log"
echo "EXIT=$code" | tee -a "$log"
echo "$code" > "$root/experiments/results/p006-$tag.exit"
exit "$code"
