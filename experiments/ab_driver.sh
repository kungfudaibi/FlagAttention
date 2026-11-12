#!/bin/bash
# Serial alternating A/B: baseline then candidate, 3 rounds per dtype.
set -uo pipefail
cd /workspace/agent-workspaces/parallax-steady-ab || exit 2
order=${1:-fp16}
for rnd in 0 1 2; do
  for route in baseline candidate; do
    tag="ab-$order-r$rnd-$route"
    experiments/run_p005.sh "$route" "$order" steady "$rnd" "$tag" > /tmp/$tag.out 2>&1
    echo "$tag exit=$?" >> experiments/results/p005-ab-driver-$order.log
  done
done
