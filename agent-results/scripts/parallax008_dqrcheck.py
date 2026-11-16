"""PARALLAX-008 DQR row-tile cross-check inside the real call.

The variant probe produced one inconsistency: at fp16 the per-launch event
timing ranked DQR BT=64 (228 us) ahead of BT=128 (239 us), while the batched
full-call A/B ranked the BT=128 variant ~5 % ahead overall.  This probe
re-measures only that axis, interleaved, several rounds, to say whether the
per-stage inversion is reproducible or an artefact of per-launch events.
"""
import json
import os
import statistics

os.sched_setaffinity(0, set(range(32, 64)))

import torch
import torch_gcu  # noqa: F401
import triton

torch.gcu.set_device(1)

import sys
sys.path.insert(0, os.environ["PARALLAX_SRC"])
from flag_attn import parallel_parallax  # noqa: E402
from flag_attn.runtime.backend._enflame.FLA.parallax import parallel as P  # noqa: E402

B, T, H, HQ, D = 2, 256, 2, 8, 64
DQR = "parallel_parallax_bwd_kernel_dqr_fused"
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
VARIANTS = {
    "bt128": (128, 256, 1, 1, 128, 128, 1, 1, 64, 256, 1, 1),
    "bt64": (128, 256, 1, 1, 128, 64, 1, 1, 64, 256, 1, 1),
}

_orig_run = triton.runtime.jit.JITFunction.run
REC = {"events": [], "active": False}


def _run(self, *args, grid, warmup, **kw):
    name = getattr(self, "__name__", "")
    if not warmup and REC["active"] and name == DQR:
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


def apply(name, dt):
    v = VARIANTS[name]
    os.environ["PARALLAX_ROW_TILE_D64"] = "1"
    P._ROW_TILE_D64_CONFIGS[dt] = {
        "fwd": v[0:4], "dqr": v[4:8], "dkv": v[8:12]}


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


def dqr_us(base, go, n):
    REC["events"] = []
    REC["active"] = True
    for _ in range(n):
        call(base, go)
    REC["active"] = False
    torch.gcu.synchronize()
    vals = [e0.elapsed_time(e1) * 1e3 for (e0, e1) in REC["events"]]
    return statistics.median(vals), len(vals)


for dtype_name, dt in DTYPES.items():
    base, go = tensors(dt), grad_out(dt)
    for name in VARIANTS:
        apply(name, dt)
        call(base, go)
    torch.gcu.synchronize()
    dq = {n: [] for n in VARIANTS}
    fu = {n: [] for n in VARIANTS}
    for rnd in range(3):
        order = list(VARIANTS) if rnd % 2 == 0 else list(VARIANTS)[::-1]
        for name in order:
            apply(name, dt)
            med, cnt = dqr_us(base, go, 30)
            dq[name].append(round(med, 2))
            fu[name].append(round(full_us(base, go, 30), 2))
            assert cnt == 30, (name, cnt)
    print(json.dumps({
        "mode": "dqr-crosscheck", "dtype": dtype_name,
        "dqr_stage_med_us": {k: v for k, v in dq.items()},
        "full_call_med_us": {k: v for k, v in fu.items()},
        "dqr_medians": {k: round(statistics.median(v), 2) for k, v in dq.items()},
        "full_medians": {k: round(statistics.median(v), 2) for k, v in fu.items()},
    }), flush=True)
