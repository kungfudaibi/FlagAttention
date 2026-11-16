"""PARALLAX-008 memory-traffic calibration on GCU 1.

Gives the copy bandwidth the DKV/FWD tile comparison is judged against, so a
launch-shape win cannot be mistaken for a bandwidth wall.  Copy moves
2 x bytes (read + write).

The fill/add cases are reported for completeness but are NOT usable as a
bandwidth figure on this stack: torch.gcu fill_ and add land at 1.4-2.7 GB/s
against a 428-434 GB/s copy, i.e. ~200x off, which is an unfused eager elementwise
path and not a memory limit.  Only the copy rows should be quoted.
"""
import json
import os

os.sched_setaffinity(0, set(range(32, 64)))

import torch
import torch_gcu  # noqa: F401

torch.gcu.set_device(1)
assert torch.gcu.current_device() == 1


def timed(fn, iters=20, warm=10):
    for _ in range(warm):
        fn()
    torch.gcu.synchronize()
    e0 = torch.gcu.Event(enable_timing=True)
    e1 = torch.gcu.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.gcu.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3


out = {"mode": "bandwidth", "device": torch.gcu.current_device()}
for mb in (8, 64, 256):
    n = mb * 1024 * 1024 // 2
    src = torch.empty(n, dtype=torch.float16, device="gcu:1")
    dst = torch.empty_like(src)
    us = timed(lambda: dst.copy_(src))
    out[f"copy_{mb}MB"] = {"us": round(us, 2),
                           "GBps": round(2 * n * 2 / (us * 1e-6) / 1e9, 1)}
    us = timed(lambda: dst.fill_(1.0))
    out[f"fill_{mb}MB"] = {"us": round(us, 2),
                           "GBps": round(n * 2 / (us * 1e-6) / 1e9, 1)}
print(json.dumps(out), flush=True)
