#!/usr/bin/env bash
# MSA-005 full-call alternating A/B on GCU 2.
# usage: run_ab.sh [output-jsonl] [log]
set -u
cd /workspace/agent-workspaces/msa-group16-formal-5 || exit 90
export PYTHONPATH="$PWD/src:$PWD:/opt/flagtree"
OUT=${1:-$PWD/agent-results/raw/MSA-005/ab.jsonl}
LOG=${2:-$PWD/agent-results/raw/MSA-005/ab_run.log}
export MSA_AB_OUT="$OUT"
mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"
START=$(date +%s)
timeout --signal=TERM --kill-after=10s 1800s \
    taskset -c 64-95 /flagos/bin/python agent-results/scripts/msa003_ab.py \
    > "$LOG" 2>&1
AB_EXIT=$?
END=$(date +%s)
echo "AB_EXIT=$AB_EXIT AB_SECONDS=$((END-START)) OUT=$OUT"
grep -E "^(WROTE|PASSED|FAILED)" "$LOG" | tail -3
exit "$AB_EXIT"
