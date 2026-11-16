#!/usr/bin/env bash
# MSA-005 formal-equivalent correctness run, exactly the declared argv.
set -u
cd /workspace/agent-workspaces/msa-group16-formal-5 || exit 90
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
LOG=agent-results/raw/MSA-005/target_formal.log
: > "$LOG"
{
  echo "== target formal-equivalent run =="
  date -u +%Y-%m-%dT%H:%M:%SZ
  echo "HEAD=$(git rev-parse HEAD)"
  echo "BRANCH=$(git rev-parse --abbrev-ref HEAD)"
  echo "STATUS=$(git status --porcelain=v1 | tr "\n" "|")"
  echo "PYTHONPATH=$PYTHONPATH"
  echo "AFFINITY=$(/flagos/bin/python -c "import os; print(sorted(os.sched_getaffinity(0)))")"
  echo "LOADAVG_BEFORE=$(cat /proc/loadavg)"
  echo "EFSMI_DEV2_BEFORE=$(efsmi -i 2 -q -d MEMORY 2>/dev/null | grep -E "Used Size" | head -1)"
  echo "== command =="
  echo "timeout --signal=TERM --kill-after=10s 900s taskset -c 64-95 /flagos/bin/python -c <declared target>"
  START=$(date +%s)
  timeout --signal=TERM --kill-after=10s 900s taskset -c 64-95 /flagos/bin/python -c "import torch, torch_gcu, pytest; torch.gcu.set_device(2); raise SystemExit(pytest.main([\"tests/flag_attn/test_minimax_sparse_attention.py\",\"-q\"]))"
  EXIT=$?
  END=$(date +%s)
  echo "TARGET_EXIT=$EXIT"
  echo "TARGET_SECONDS=$((END-START))"
  echo "LOADAVG_AFTER=$(cat /proc/loadavg)"
  echo "EFSMI_DEV2_AFTER=$(efsmi -i 2 -q -d MEMORY 2>/dev/null | grep -E "Used Size" | head -1)"
  echo "STATUS_AFTER=$(git status --porcelain=v1 | tr "\n" "|")"
  exit "$EXIT"
} 2>&1 | tee "$LOG"
