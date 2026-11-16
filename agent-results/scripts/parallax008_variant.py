"""PARALLAX-008 row-tile variant selection.

The first full-call measurement showed the isolated DQR microbenchmark's
preference (BS=128, BT=128) does NOT transfer to the real call: DQR regressed
from ~208 us (BT=64) to ~238 us (BT=128) at fp16.  This probe re-decides the
DQR tuple inside the real call, keeping the row-tile experiment itself fixed.

Same inputs, alternating order, per-stage device events without per-launch
synchronisation, plus full-call medians.
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
from flag_attn.runtime.backend._enflame.FLA.parallax import parallel as P  # noqa: E402

B, T, H, HQ, D = 2, 256, 2, 8, 64
FWD = "parallel_parallax_fwd_kernel"
DQR = "parallel_parallax_bwd_kernel_dqr_fused"
DKV = "parallel_parallax_bwd_kernel_dkv_grouped"
STAGES = (FWD, DQR, DKV)
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

# name -> (gate, fwd, dqr, dkv)
VARIANTS = {
    "shipped": (0, None, None, None),
    "rowtile-bt128dqr": (1, (128, 256, 1, 1), (128, 128, 1, 1), (64, 256, 1, 1)),
    "rowtile-bt64dqr": (1, (128, 256, 1, 1), (128, 64, 1, 1), (64, 256, 1, 1)),
    "rowtile-dqr64x128": (1, (128, 256, 1, 1), (64, 128, 1, 1), (64, 256, 1, 1)),
    "rowtile-bt64dqr-nofwd": (1, (128, 64, 1, 1), (128, 64, 1, 1), (64, 256, 1, 1)),
}

_orig_run = triton.runtime.jit.JITFunction.run
REC = {"events": [], "active": False}


def _run(self, *args, grid, warmup, **kw):
    name = getattr(self, "__name__", "")
    if not warmup and REC["active"] and name in STAGES:
        e0 = torch.gcu.Event(enable_timing=True)
        e1 = torch.gcu.Event(enable_timing=True)
        e0.record()
        out = _orig_run(self, *args, grid=grid, warmup=warmup, **kw)
        e1.record()
        REC["events"].append((name, e0, e1))
        return out
    return _orig_run(self, *args, grid=grid, warmup=warmup, **kw)


triton.runtime.jit.JITFunction.run = _run


def tensors(dtype, seed=42):
    gen = torch.Generator(device="cpu").manual_seed(seed)

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
    g = torch.autograd.grad(o, (q, r, k, v), go)
    return o, g


def apply(name, dt):
    gate, fwd, dqr, dkv = VARIANTS[name]
    os.environ["PARALLAX_ROW_TILE_D64"] = str(gate)
    if gate:
        P._ROW_TILE_D64_CONFIGS[dt] = {"fwd": fwd, "dqr": dqr, "dkv": dkv}


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


def stage_us(base, go, n):
    REC["events"] = []
    REC["active"] = True
    for _ in range(n):
        call(base, go)
    REC["active"] = False
    torch.gcu.synchronize()
    per = {}
    for s in STAGES:
        vals = [e0.elapsed_time(e1) * 1e3 for (nm, e0, e1) in REC["events"] if nm == s]
        per[s] = round(statistics.median(vals), 2) if vals else None
    return per


ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--iters", type=int, default=30)
a = ap.parse_args()

names = list(VARIANTS)
for dtype_name, dt in DTYPES.items():
    base, go = tensors(dt), grad_out(dt)
    for name in names:                      # warm every variant first
        apply(name, dt)
        call(base, go)
    torch.gcu.synchronize()

    for name in names:
        apply(name, dt)
        per = stage_us(base, go, a.iters)
        full = full_us(base, go, a.iters)
        print(json.dumps({"mode": "variant-stage", "dtype": dtype_name, "variant": name,
                          "stage_med_us": per,
                          "sum_stage_med_us": round(sum(v for v in per.values() if v), 2),
                          "full_call_med_us": round(full, 2)}), flush=True)

    rows = {n: [] for n in names}
    for rnd in range(a.rounds):
        order = names[rnd % len(names):] + names[:rnd % len(names)]
        for name in order:
            apply(name, dt)
            rows[name].append(round(full_us(base, go, a.iters), 2))
    base_med = statistics.median(rows["shipped"])
    summary = {"mode": "variant-ab", "dtype": dtype_name, "rounds": a.rounds,
               "iters": a.iters, "shipped_med_us": round(base_med, 2), "variants": {}}
    for name in names[1:]:
        med = statistics.median(rows[name])
        summary["variants"][name] = {
            "med_us": round(med, 2),
            "ratio_vs_shipped": round(med / base_med, 4),
            "all": rows[name],
            "per_round_delta_pct": [round((x / y - 1) * 100, 2)
                                    for x, y in zip(rows[name], rows["shipped"])],
        }
    summary["shipped_all"] = rows["shipped"]
    print(json.dumps(summary), flush=True)
