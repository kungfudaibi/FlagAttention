#!/usr/bin/env python3
"""Show which implementation the repository SDPA baseline actually executes.

The repository baseline is ``F.scaled_dot_product_attention``.  On this
Enflame image that dispatcher entry can fall back to the CPU math kernel and
copy tensors over PCIe; the explicit vendor fused ops stay on the GCU.  This
script records the profiler evidence and the CPU/device cost of both paths.
"""

import json
import os
import statistics
import time

import torch
import torch.nn.functional as F
import torch_gcu


torch.gcu.set_device(1)


def cpu_seconds():
    fields = open("/proc/self/stat").read().split()
    ticks = os.sysconf("SC_CLK_TCK")
    return (int(fields[13]) + int(fields[14])) / ticks


def timed(fn, samples=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.gcu.synchronize()
    events, walls, cpus = [], [], []
    for _ in range(samples):
        c0 = cpu_seconds()
        start = torch.gcu.Event(enable_timing=True)
        end = torch.gcu.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        out = fn()
        end.record()
        torch.gcu.synchronize()
        walls.append((time.perf_counter() - t0) * 1000)
        cpus.append((cpu_seconds() - c0) * 1000)
        events.append(start.elapsed_time(end))
    return {
        "event_median_ms": statistics.median(events),
        "event_min_ms": min(events),
        "wall_median_ms": statistics.median(walls),
        "cpu_median_ms": statistics.median(cpus),
        "samples_ms": events,
        "out": out,
    }


def sdpa_exact(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


def sdpa_vendor(q, k, v):
    return torch.ops.aten._scaled_dot_product_flash_attention(
        q, k, v, 0.0, False, False
    )[0]


def profile_names(fn):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        fn()
        torch.gcu.synchronize()
    names = []
    for event in prof.key_averages():
        name = event.key
        if any(token in name for token in (
            "scaled_dot_product", "_to_cpu", "_to_copy", "copy_", "matmul", "bmm", "softmax"
        )):
            names.append({
                "name": name,
                "self_cpu_ms": event.self_cpu_time_total / 1000.0,
                "calls": event.count,
            })
    return names


def main() -> int:
    print(json.dumps({
        "event": "meta",
        "device": "gcu:1",
        "pci": "0000:83:00.0",
        "torch": torch.__version__,
        "torch_gcu": torch_gcu.__version__,
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }), flush=True)
    for T in (512, 1024, 2048, 4096):
        torch.manual_seed(42)
        q = torch.randn(1, 2, T, 64, device="gcu:1", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        exact = timed(lambda: sdpa_exact(q, k, v))
        vendor = timed(lambda: sdpa_vendor(q, k, v))
        a = exact.pop("out").float().flatten()
        b = vendor.pop("out").float().flatten()
        cosine = F.cosine_similarity(a, b, dim=0).item()
        rel_l2 = ((a - b).norm() / b.norm().clamp_min(1e-8)).item()
        exact_names = profile_names(lambda: sdpa_exact(q, k, v))
        vendor_names = profile_names(lambda: sdpa_vendor(q, k, v))
        print(json.dumps({
            "event": "shape",
            "shape": [1, 2, T, 64],
            "exact": exact,
            "vendor": vendor,
            "exact_vs_vendor": {"cosine": cosine, "relative_l2": rel_l2},
            "exact_profile": exact_names,
            "vendor_profile": vendor_names,
        }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
