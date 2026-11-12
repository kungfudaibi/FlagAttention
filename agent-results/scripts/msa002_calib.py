"""MSA-002 timing calibration: event overhead and per-launch fixed cost."""

import sys
import time

import torch
import torch_gcu  # noqa: F401

sys.path.insert(0, "/workspace/agent-workspaces/msa-stage-profile")
sys.path.insert(0, "/workspace/agent-workspaces/msa-stage-profile/src")
torch.gcu.set_device(1)

print("device", torch.gcu.current_device(), torch.gcu.get_device_name(1), flush=True)

x = torch.randn(1024, 1024, device="gcu:1", dtype=torch.bfloat16)
y = torch.empty_like(x)

# warm compile
for _ in range(10):
    torch.mul(x, 2, out=y)
torch.gcu.synchronize()

# 1) per-iteration event pairs
vals = []
for _ in range(30):
    a = torch.gcu.Event(enable_timing=True)
    b = torch.gcu.Event(enable_timing=True)
    a.record()
    torch.mul(x, 2, out=y)
    b.record()
    torch.gcu.synchronize()
    vals.append(a.elapsed_time(b))
vals.sort()
print("per-iter event ms: min %.4f med %.4f max %.4f" % (vals[0], vals[len(vals) // 2], vals[-1]), flush=True)

# 2) N launches inside one event pair
for n in (1, 10, 50, 100):
    a = torch.gcu.Event(enable_timing=True)
    b = torch.gcu.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        torch.mul(x, 2, out=y)
    b.record()
    torch.gcu.synchronize()
    total = a.elapsed_time(b)
    print("n=%d total_ms %.4f per_launch_ms %.4f" % (n, total, total / n), flush=True)

# 3) host wall time of launch-only (no sync) and with sync
for n in (100,):
    torch.gcu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        torch.mul(x, 2, out=y)
    t1 = time.perf_counter()
    torch.gcu.synchronize()
    t2 = time.perf_counter()
    print("host enqueue per launch ms %.4f; sync wait ms %.4f" % ((t1 - t0) * 1000 / n, (t2 - t1) * 1000), flush=True)

# 4) empty event pair after warmup
vals = []
for _ in range(30):
    a = torch.gcu.Event(enable_timing=True)
    b = torch.gcu.Event(enable_timing=True)
    a.record()
    b.record()
    torch.gcu.synchronize()
    vals.append(a.elapsed_time(b))
vals.sort()
print("empty event ms: min %.4f med %.4f max %.4f" % (vals[0], vals[len(vals) // 2], vals[-1]), flush=True)

# 5) same events but elapsed measured after many records (queue depth)
a = torch.gcu.Event(enable_timing=True)
b = torch.gcu.Event(enable_timing=True)
a.record()
for _ in range(50):
    torch.mul(x, 2, out=y)
b.record()
torch.gcu.synchronize()
print("50 launches one pair ms %.4f" % a.elapsed_time(b), flush=True)
