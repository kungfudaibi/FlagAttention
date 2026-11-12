set -u
CAND=/workspace/agent-workspaces/gla-shape-sweep
BASE=$CAND/.baseline-4f437de
CASE=$CAND/agent-results/scripts/gla004_case.py
OUT=$CAND/agent-results/raw/gate_stress_T256.jsonl
for GLN in 0.1 10.0; do
  for SPEC in 32x32 128x128; do
    K=${SPEC%x*}; V=${SPEC#*x}
    TAG="${SPEC}_T256_gln${GLN}"
    BP=/tmp/gla004_gate_base_${TAG}.pt
    PYTHONPATH=$BASE/src:$BASE:/opt/flagtree timeout 175 /flagos/bin/python $CASE --T 256 --K $K --V $V --gln $GLN --tag base_$TAG --no-timing --save $BP 2>/dev/null | grep '^GLA004_JSON' | sed 's/^GLA004_JSON //' >> $OUT
    PYTHONPATH=$CAND/src:$CAND:/opt/flagtree timeout 175 /flagos/bin/python $CASE --T 256 --K $K --V $V --gln $GLN --tag cand_$TAG --no-timing --compare $BP 2>/dev/null | grep '^GLA004_JSON' | sed 's/^GLA004_JSON //' >> $OUT
    echo done $TAG
  done
done
