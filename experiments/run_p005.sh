#!/bin/bash
# PARALLAX-005 single-command runner: one GCU-1 process at a time.
# usage: run_p005.sh <baseline|candidate> <fp16|bf16> <cold|steady> <round> <tag>
set -uo pipefail
route=$1; dtype=$2; mode=$3; round=${4:-0}; tag=${5:-$1-$2-$3}
root=/workspace/agent-workspaces/parallax-steady-ab
case "$route" in
  baseline) tree=$root/.baseline-70bdc5; export PARALLAX_FIXED_D64=0 ;;
  candidate) tree=$root; export PARALLAX_FIXED_D64=1 ;;
  *) echo "bad route" >&2; exit 2 ;;
esac
# Steady reuses binaries from the route's own cold run. Autotune choices are
# re-benchmarked in the untimed path probe; no compile enters a sample.
if [ "$mode" = steady ]; then
  cache=/tmp/p005-cache-$route-$dtype-steady
  src=/tmp/p005-cache-$route-$dtype-cold
  if [ -d "$src" ] && [ ! -d "$cache" ]; then cp -a "$src" "$cache"; fi
else
  cache=/tmp/p005-cache-$route-$dtype-cold
fi
mkdir -p "$cache" "$root/experiments/logs" "$root/experiments/results"
log=$root/experiments/logs/p005-$tag.log
: > "$log"
{
  echo "CMD route=$route dtype=$dtype mode=$mode round=$round tag=$tag"
  echo "TREE=$tree HEAD=$(git -C "$tree" rev-parse HEAD)"
  echo "PARALLAX_FIXED_D64=${PARALLAX_FIXED_D64:-unset} TRITON_CACHE_DIR=$cache CACHE_FILES_BEFORE=$(find "$cache" -type f | wc -l)"
  echo "DATE_START=$(date -u +%FT%TZ)"
} | tee -a "$log"
export PYTHONPATH="$tree/src:$tree:/opt/flagtree"
export TRITON_CACHE_DIR="$cache"
export TRITON_F32_DEFAULT=ieee
cd "$root" || exit 2
timeout 600 taskset -c 160-191 /flagos/bin/python "$root/experiments/parallax_steady_ab.py" \
  --route "$route" --dtype "$dtype" --mode "$mode" --round "$round" --samples 5 --warmups 2 \
  >> "$log" 2>&1
code=$?
echo "DATE_END=$(date -u +%FT%TZ)" | tee -a "$log"
echo "EXIT=$code" | tee -a "$log"
echo "$code" > "$root/experiments/results/p005-$tag.exit"
exit "$code"
