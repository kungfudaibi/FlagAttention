"""MSA-002 roofline reference: peak BF16 matmul throughput and HBM bandwidth."""

import json
import statistics
import sys
import time

import torch
import torch_gcu  # noqa: F401

sys.path.insert(0, "/workspace/agent-workspaces/msa-stage-profile")
sys.path.insert(0, "/workspace/agent-workspaces/msa-stage-profile/src")
torch.gcu.set_device(1)

print("device", torch.gcu.current_device(), torch.gcu.get_device_name(1), flush=True)
props = torch.gcu.get_device_properties(1)
print("props", props, flush=True)


def bench(fn, n=10, samples=10):
    fn()
    torch.gcu.synchronize()
    vals = []
    for _ in range(samples):
        a = torch.gcu.Event(enable_timing=True)
        b = torch.gcu.Event(enable_timing=True)
        torch.gcu.synchronize()
        a.record()
        for _ in range(n):
            fn()
        b.record()
        torch.gcu.synchronize()
        vals.append(a.elapsed_time(b) / n)
    vals.sort()
    return statistics.median(vals)


for m, k, n in ((4096, 4096, 4096), (8192, 8192, 8192), (16384, 4096, 4096)):
    x = torch.randn(m, k, device="gcu:1", dtype=torch.bfloat16)
    w = torch.randn(k, n, device="gcu:1", dtype=torch.bfloat16)
    ms = bench(lambda: torch.matmul(x, w), n=5, samples=8)
    tflops = 2 * m * k * n / (ms * 1e-3) / 1e12
    print(json.dumps({"probe": "matmul", "m": m, "k": k, "n": n, "ms": ms, "tflops": tflops}), flush=True)
    del x, w
    torch.gcu.empty_cache()

# memory bandwidth: large copy
for numel in (1 << 24, 1 << 26, 1 << 28):
    src = torch.randn(numel, device="gcu:1", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    ms = bench(lambda: dst.copy_(src), n=5, samples=8)
    gbytes = numel * 2 * 2 / (ms * 1e-3) / 1e9
    print(json.dumps({"probe": "copy", "numel": numel, "ms": ms, "gbyte_per_s": gbytes}), flush=True)
    del src, dst
    torch.gcu.empty_cache()

# reduction and elementwise ceilings
x = torch.randn(1 << 26, device="gcu:1", dtype=torch.bfloat16)
ms = bench(lambda: torch.sum(x), n=5, samples=8)
print(json.dumps({"probe": "sum", "numel": 1 << 26, "ms": ms,
                  "gbyte_per_s": (1 << 26) * 2 / (ms * 1e-3) / 1e9}), flush=True)
print("DONE", flush=True)
