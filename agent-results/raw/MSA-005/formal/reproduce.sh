#!/usr/bin/env bash
# Re-derive the noise/bimodality verdicts from the committed formal A/B
# records.  Read-only: no GCU, no torch, no pytest needed.  Run from the
# repository root.
set -u
root="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$root"
mkdir -p /tmp/msa006-repro
python3 agent-results/scripts/msa_ab_reanalyze.py \
  --run MSA-004=agent-results/raw/MSA-004/ab.jsonl \
  --run MSA-005-run1=agent-results/raw/MSA-005/run1/ab.jsonl \
  --run MSA-005-run2=agent-results/raw/MSA-005/ab.jsonl \
  --run MSA-005-formal=agent-results/raw/MSA-005/formal/20261116T044252Z-99672caa/ab.jsonl \
  --json-out /tmp/msa006-repro/reanalysis.json
rc=$?
echo "REANALYZE_EXIT=$rc"
cmp -s /tmp/msa006-repro/reanalysis.json agent-results/raw/MSA-006/reanalysis.json
echo "MATCHES_COMMITTED=$?"
python3 - <<'PY'
import json
d = json.load(open("/tmp/msa006-repro/reanalysis.json"))
for name, run in d["runs"].items():
    print(name, "stable_gain", len(run["stable_gain_cases"]),
          "unstable_controls", len(run["unstable_controls"]),
          "noise_floor", run["noise_floor"])
PY
exit "$rc"
