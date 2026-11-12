#!/bin/bash
# GCU 0 preflight: refuse to launch if another process holds device memory or a
# competing python device job is running.  Read-only inspection.
set -u
echo "== preflight $(date -u +%FT%TZ) =="
/flagos/bin/python - <<'PY' 2>&1 | grep -v "GCU Allocator"
import torch
free, total = torch.gcu.mem_get_info(0)
print(f"gcu0_free={free} gcu0_total={total} gcu0_allocated={torch.gcu.memory_allocated(0)}")
print("gcu0_reserved", torch.gcu.memory_reserved(0))
assert free > 4 * 1024**3, "gcu0 free memory below 4 GiB"
PY
rc=$?
if [ $rc -ne 0 ]; then echo "PREFLIGHT_FAIL"; exit 1; fi
comp=$(ps -eo pid,args | grep -E "gla00[0-9]_case|gla005_case|chunk_gla|torch.*gcu" | grep -v grep | grep -v preflight || true)
if [ -n "$comp" ]; then echo "COMPETING_PROCESS:"; echo "$comp"; exit 1; fi
echo "PREFLIGHT_OK"
