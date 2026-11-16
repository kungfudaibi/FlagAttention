"""PARALLAX-008 nearby/fallback-shape numerics with the opt-in flag on and off.

The row-tile gate only accepts B2/T256/H2/HQ8/D64.  Every other shape must take
the untouched autotuned route; this re-checks that the outputs and all four
gradients are bit-identical between flag=0 and flag=1 for a nearby shape, and
that they still match the FP32 reference.
"""
import importlib.util
import json
import os
import sys

os.sched_setaffinity(0, set(range(32, 64)))

import torch
import torch_gcu  # noqa: F401

torch.gcu.set_device(1)
sys.path.insert(0, os.environ["PARALLAX_SRC"])
from flag_attn import parallel_parallax  # noqa: E402

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
SHAPES = ((1, 256, 2, 8), (2, 128, 2, 8), (1, 63, 1, 1))
D = 64

spec = importlib.util.spec_from_file_location(
    "_p008_ref", os.path.join(os.environ["PARALLAX_TESTS"], "test_parallax.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["_p008_ref"] = mod
spec.loader.exec_module(mod)


def mk(dtype, b, t, h, hq, seed=42):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def one(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(
            device="gcu:1", dtype=dtype)
    return ([one(b, t, hq, D), one(b, t, hq, D), one(b, t, h, D), one(b, t, h, D)],
            one(b, t, hq, D))


def run(base, go):
    q, r, k, v = (x.detach().clone().requires_grad_(True) for x in base)
    o = parallel_parallax(q, r, k, v, scale=D ** -0.5)
    return o, torch.autograd.grad(o, (q, r, k, v), go)


def rel(x, y):
    d = x.float() - y.float()
    return (d.square().mean().sqrt()
            / y.float().square().mean().sqrt().clamp_min(1e-9)).item()


for name, dt in DTYPES.items():
    for (b, t, h, hq) in SHAPES:
        base, go = mk(dt, b, t, h, hq)
        got = {}
        for flag in ("0", "1"):
            os.environ["PARALLAX_ROW_TILE_D64"] = flag
            got[flag] = run(base, go)
            torch.gcu.synchronize()
        ins = tuple(x.detach().float().requires_grad_(True) for x in base)
        ref = mod._reference_parallax(*ins, scale=D ** -0.5, window_size=None)
        rg = torch.autograd.grad(ref, ins, go.float())
        out = {"mode": "nearby", "dtype": name, "shape": [b, t, h, hq, D],
               "identical_flag0_vs_flag1": True}
        o, g = got["1"]
        out["rel_vs_fp32_ref"] = {"o": rel(o, ref)}
        for nm, gg, rr in zip(("dq", "dr", "dk", "dv"), g, rg):
            out["rel_vs_fp32_ref"][nm] = rel(gg, rr)
        for i, nm in enumerate(("o", "dq", "dr", "dk", "dv")):
            a = got["0"][0] if i == 0 else got["0"][1][i - 1]
            c = got["1"][0] if i == 0 else got["1"][1][i - 1]
            if not torch.equal(a, c):
                out["identical_flag0_vs_flag1"] = False
            out[f"max_abs_diff_{nm}"] = (a.float() - c.float()).abs().max().item()
        out["finite"] = bool(torch.isfinite(o).all()
                             and all(torch.isfinite(x).all() for x in g))
        print(json.dumps(out), flush=True)
