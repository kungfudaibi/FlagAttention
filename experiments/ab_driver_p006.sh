#!/bin/bash
# PARALLAX-006 serial alternating A/B: 3 rounds x (baseline then candidate),
# 5 synchronized samples per round per route, per dtype.
set -uo pipefail
cd /workspace/agent-workspaces/parallax-static-tune || exit 2
order=${1:-fp16}
for rnd in 0 1 2; do
  for route in baseline candidate; do
    tag="ab-$order-r$rnd-$route"
    experiments/run_p006.sh "$route" "$order" steady "$rnd" target "$tag" > /tmp/$tag.out 2>&1
    echo "$tag exit=$? $(date -u +%FT%TZ)" >> experiments/results/p006-ab-driver-$order.log
  done
done
