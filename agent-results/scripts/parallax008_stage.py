"""PARALLAX-008 per-stage attribution with minimal instrumentation.

Wrapping all three kernels in one pass inflates and even reorders the middle
stage's measured time, so each stage is measured in its own pass with only that
kernel instrumented.  Routes alternate every round; the full call is timed in
the same round so stage numbers can be sanity-checked against it.
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

torch.gcu.set_device(1)
assert torch.gcu.current_device() == 1

sys.path.insert(0, os.environ["PARALLAX_SRC"])
from flag_attn import parallel_parallax  # noqa: E402

B, T, H, HQ, D = 2, 256, 2, 8, 64
NAMES = {
    "fwd": "parallel_parallax_fwd_kernel",
    "dqr": "parallel_parallax_bwd_kernel_dqr_fused",
    "dkv": "parallel_parallax_bwd_kernel_dkv_grouped",
}
ROUTES = {"shipped": "0", "candidate": "1"}
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

_orig_run = triton.runtime.jit.JITFunction.run
REC = {"events": [], "target": None, "active": False}


def _run(self, *args, grid, warmup, **kw):
    name = getattr(self, "__name__", "")
    if not warmup and REC["active"] and name == REC["target"]:
        e0 = torch.gcu.Event(enable_timing=True)
        e1 = torch.gcu.Event(enable_timing=True)
        e0.record()
        out = _orig_run(self, *args, grid=grid, warmup=warmup, **kw)
        e1.record()
        REC["events"].append((e0, e1))
        return out
    return _orig_run(self, *args, grid=grid, warmup=warmup, **kw)


triton.runtime.jit.JITFunction.run = _run


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


def stage_pass(base, go, stage, n):
    """One timed pass of n calls with only `stage` instrumented."""
    REC["target"] = NAMES[stage]
    REC["events"] = []
    for _ in range(5):
        call(base, go)
    torch.gcu.synchronize()
    REC["active"] = True
    for _ in range(n):
        call(base, go)
    REC["active"] = False
    torch.gcu.synchronize()
    vals = [e0.elapsed_time(e1) * 1e3 for (e0, e1) in REC["events"]]
    assert len(vals) == n, (stage, len(vals), n)
    return round(statistics.median(vals), 2), round(min(vals), 2), round(max(vals), 2)


def full_pass(base, go, n):
    for _ in range(5):
        call(base, go)
    torch.gcu.synchronize()
    e0 = torch.gcu.Event(enable_timing=True)
    e1 = torch.gcu.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        call(base, go)
    e1.record()
    torch.gcu.synchronize()
    return round(e0.elapsed_time(e1) / n * 1e3, 2)


ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--iters", type=int, default=30)
a = ap.parse_args()

for dtype_name, dt in DTYPES.items():
    base, go = tensors(dt), grad_out(dt)
    for route, flag in ROUTES.items():
        os.environ["PARALLAX_ROW_TILE_D64"] = flag
        call(base, go)
    torch.gcu.synchronize()

    rows = {s: {r: [] for r in ROUTES} for s in NAMES}
    fulls = {r: [] for r in ROUTES}
    for rnd in range(a.rounds):
        order = list(ROUTES) if rnd % 2 == 0 else list(ROUTES)[::-1]
        for stage in ("fwd", "dqr", "dkv"):
            for route in order:
                os.environ["PARALLAX_ROW_TILE_D64"] = ROUTES[route]
                med, lo, hi = stage_pass(base, go, stage, a.iters)
                rows[stage][route].append({"med": med, "min": lo, "max": hi})
        for route in order:
            os.environ["PARALLAX_ROW_TILE_D64"] = ROUTES[route]
            fulls[route].append(full_pass(base, go, a.iters))

    out = {"mode": "stage-clean", "dtype": dtype_name, "rounds": a.rounds,
           "iters": a.iters, "stages": {}, "full_call_med_us": {}}
    for stage, per_route in rows.items():
        out["stages"][stage] = {
            route: {"med_us": round(statistics.median(x["med"] for x in vals), 2),
                    "rounds": vals}
            for route, vals in per_route.items()}
        s = out["stages"][stage]["shipped"]["med_us"]
        c = out["stages"][stage]["candidate"]["med_us"]
        out["stages"][stage]["delta_pct"] = round((c / s - 1) * 100, 2)
    for route, vals in fulls.items():
        out["full_call_med_us"][route] = round(statistics.median(vals), 2)
        out["full_call_med_us"][route + "_rounds"] = vals
    s = out["full_call_med_us"]["shipped"]
    c = out["full_call_med_us"]["candidate"]
    out["full_call_ratio"] = round(c / s, 4)
    out["full_call_delta_pct"] = round((c / s - 1) * 100, 2)
    out["stage_sum_us"] = {
        route: round(sum(out["stages"][st][route]["med_us"] for st in NAMES), 2)
        for route in ROUTES}
    print(json.dumps(out), flush=True)
