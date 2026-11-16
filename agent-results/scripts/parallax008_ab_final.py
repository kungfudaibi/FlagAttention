"""PARALLAX-008 final full-call A/B on the shipped code path.

Uses only PARALLAX_ROW_TILE_D64 (no monkeypatched configs), alternating order,
9 rounds per dtype, plus a launch-floor calibration and peak-memory reset.
"""
import argparse
import json
import os
import statistics
import sys

os.sched_setaffinity(0, set(range(32, 64)))

import torch
import torch_gcu  # noqa: F401
import triton
import triton.language as tl

torch.gcu.set_device(1)
assert torch.gcu.current_device() == 1

sys.path.insert(0, os.environ["PARALLAX_SRC"])
from flag_attn import parallel_parallax  # noqa: E402

B, T, H, HQ, D = 2, 256, 2, 8, 64
ROUTES = {"shipped": "0", "candidate": "1"}
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def tensors(dtype):
    gen = torch.Generator(device="cpu").manual_seed(42)

    def mk(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(
            device="gcu:1", dtype=dtype)
    return [mk(B, T, HQ, D), mk(B, T, HQ, D), mk(B, T, H, D), mk(B, T, H, D)]


def grad_out(dtype):
    gen = torch.Generator(device="cpu").manual_seed(7)
    return torch.randn(B, T, HQ, D, generator=gen, dtype=torch.float32).to(
        device="gcu:1", dtype=dtype)


def call(base, go):
    q, r, k, v = (x.detach().clone().requires_grad_(True) for x in base)
    o = parallel_parallax(q, r, k, v, scale=D ** -0.5)
    return o, torch.autograd.grad(o, (q, r, k, v), go)


def full_us(base, go, n):
    for _ in range(10):
        call(base, go)
    torch.gcu.synchronize()
    e0 = torch.gcu.Event(enable_timing=True)
    e1 = torch.gcu.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        call(base, go)
    e1.record()
    torch.gcu.synchronize()
    return e0.elapsed_time(e1) / n * 1e3


@triton.jit
def noop(x):
    pass


def launch_floor():
    x = torch.zeros(4, dtype=torch.float32, device="gcu:1")
    noop[(1,)](x, num_warps=1)
    torch.gcu.synchronize()
    e0 = torch.gcu.Event(enable_timing=True)
    e1 = torch.gcu.Event(enable_timing=True)
    e0.record()
    for _ in range(200):
        noop[(1,)](x, num_warps=1)
    e1.record()
    torch.gcu.synchronize()
    return round(e0.elapsed_time(e1) / 200 * 1e3, 3)


ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=9)
ap.add_argument("--iters", type=int, default=30)
a = ap.parse_args()

print(json.dumps({"mode": "floor", "launch_floor_us": launch_floor()}), flush=True)

for dtype_name, dt in DTYPES.items():
    base, go = tensors(dt), grad_out(dt)
    for route, flag in ROUTES.items():
        os.environ["PARALLAX_ROW_TILE_D64"] = flag
        call(base, go)
    torch.gcu.synchronize()

    rows = {r: [] for r in ROUTES}
    for rnd in range(a.rounds):
        order = list(ROUTES) if rnd % 2 == 0 else list(ROUTES)[::-1]
        for route in order:
            os.environ["PARALLAX_ROW_TILE_D64"] = ROUTES[route]
            rows[route].append(round(full_us(base, go, a.iters), 2))
    s, c = rows["shipped"], rows["candidate"]
    print(json.dumps({
        "mode": "final-ab", "dtype": dtype_name, "rounds": a.rounds,
        "iters": a.iters,
        "shipped_med_us": round(statistics.median(s), 2),
        "candidate_med_us": round(statistics.median(c), 2),
        "ratio": round(statistics.median(c) / statistics.median(s), 4),
        "delta_pct": round((statistics.median(c) / statistics.median(s) - 1) * 100, 2),
        "per_round_delta_pct": [round((x / y - 1) * 100, 2) for x, y in zip(c, s)],
        "shipped_all": s, "candidate_all": c,
        "all_rounds_improve": all(x < y for x, y in zip(c, s)),
    }), flush=True)

    # peak memory on a warm allocator, both routes, same process reset
    for route, flag in ROUTES.items():
        os.environ["PARALLAX_ROW_TILE_D64"] = flag
        call(base, go)
        torch.gcu.synchronize()
        torch.gcu.reset_peak_memory_stats()
        call(base, go)
        torch.gcu.synchronize()
        print(json.dumps({"mode": "mem-warm", "dtype": dtype_name, "route": route,
                          "peak_alloc_MB": round(torch.gcu.max_memory_allocated() / 2**20, 2),
                          "peak_reserved_MB": round(torch.gcu.max_memory_reserved() / 2**20, 2)}),
              flush=True)
