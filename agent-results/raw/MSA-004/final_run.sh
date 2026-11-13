set -u
cd /workspace/agent-workspaces/msa-group16-validate-2 || exit 90
export PYTHONPATH="/opt/flagtree:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MSA004_REQUIRE_DEVICE=1
mkdir -p agent-results/raw/MSA-004
LOG=agent-results/raw/MSA-004/final_run.log
: > "$LOG"
say() { echo "$@" | tee -a "$LOG"; }

say "== git =="
say "HEAD=$(git rev-parse HEAD)"
say "STATUS=$(git status --porcelain | tr '\n' '|')"
say "== env =="
say "PYTHON=$(/flagos/bin/python -V 2>&1)"
say "LOADAVG_BEFORE=$(cat /proc/loadavg)"
say "AFFINITY=$(/flagos/bin/python -c 'import os;print(sorted(os.sched_getaffinity(0)))')"
say "== efsmi before =="
efsmi -L 2>/dev/null | sed -n '7,10p' | tee -a "$LOG"
efsmi -i 2 -q -d MEMORY 2>/dev/null | tee -a "$LOG"

say "== 1. harness correctness: tests/flag_attn/test_minimax_sparse_attention.py (default path) =="
timeout 2400 taskset -c 64-95 /flagos/bin/python -m pytest tests/flag_attn/test_minimax_sparse_attention.py -q \
    > /tmp/msa004_target.log 2>&1
TARGET_EXIT=$?
say "TARGET_EXIT=$TARGET_EXIT"
tail -3 /tmp/msa004_target.log | tee -a "$LOG"

say "== 2. new routing tests: tests/flag_attn/test_msa_group16.py =="
timeout 2400 taskset -c 64-95 /flagos/bin/python -m pytest tests/flag_attn/test_msa_group16.py -q \
    > /tmp/msa004_newtests.log 2>&1
NEWTESTS_EXIT=$?
say "NEWTESTS_EXIT=$NEWTESTS_EXIT"
tail -3 /tmp/msa004_newtests.log | tee -a "$LOG"

say "== 3. A/B three alternating rounds: agent-results/scripts/msa003_ab.py =="
timeout 2400 taskset -c 64-95 /flagos/bin/python agent-results/scripts/msa003_ab.py \
    > /tmp/msa004_ab.log 2>&1
AB_EXIT=$?
say "AB_EXIT=$AB_EXIT"
grep -E "^(PASSED|FAILED|WROTE)" /tmp/msa004_ab.log | tee -a "$LOG"
tail -2 /tmp/msa004_ab.log | tee -a "$LOG"

say "== efsmi after =="
efsmi -i 2 -q -d MEMORY 2>/dev/null | tee -a "$LOG"
say "LOADAVG_AFTER=$(cat /proc/loadavg)"

cp /tmp/msa004_target.log agent-results/raw/MSA-004/final_target_run2.log
cp /tmp/msa004_newtests.log agent-results/raw/MSA-004/final_newtests_run2.log
cp /tmp/msa004_ab.log agent-results/raw/MSA-004/final_ab_run2.log
say "SUMMARY TARGET=$TARGET_EXIT NEWTESTS=$NEWTESTS_EXIT AB=$AB_EXIT"
if [ "$TARGET_EXIT" -ne 0 ] || [ "$NEWTESTS_EXIT" -ne 0 ] || [ "$AB_EXIT" -ne 0 ]; then
    say "OVERALL=FAIL"
    exit 1
fi
say "OVERALL=PASS"
exit 0
