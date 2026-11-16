"""PARALLAX-008 final evidence probe.

Modes
  cold    : one fresh-cache cold call, counting real compiles and autotune
            bench calls/time.  Run in its own process per (dtype, route).
  hot     : per-stage device timing with NO per-launch synchronisation, plus
            same-input alternating full-call A/B, launch counts, peak memory
            and numeric agreement against the FP32 reference.

Everything is pinned to GCU 1 and CPUs 32-63.
"""
import argparse
import json
import os
import statistics
import sys
import time

os.sched_setaffinity(0, set(range(32, 64)))

import torch
import torch_gcu  # noqa: F401
import triton

torch.gcu.set_device(1)
assert torch.gcu.current_device() == 1

sys.path.insert(0, os.environ["PARALLAX_SRC"])
from flag_attn import parallel_parallax  # noqa: E402

B, T, H, HQ, D, G = 2, 256, 2, 8, 64, 4
FWD = "parallel_parallax_fwd_kernel"
DQR = "parallel_parallax_bwd_kernel_dqr_fused"
DKV = "parallel_parallax_bwd_kernel_dkv_grouped"
STAGES = (FWD, DQR, DKV)
ROUTES = {"shipped": "0", "candidate": "1"}
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

_orig_run = triton.runtime.jit.JITFunction.run
_orig_bench = triton.runtime.autotuner.Autotuner._bench


class Recorder:
    """Counts compiles / autotune benches and records non-blocking events."""

    def __init__(self):
        self.compiles = 0
        self.bench_calls = 0
        self.bench_s = 0.0
        self.launches = {s: 0 for s in STAGES}
        self.events = []          # (stage, e0, e1)
        self.cfgs = {}
        self.trace = []           # ordered stage names, one list per call
        self.call = None
        self.active = False
        self.bench_depth = 0

    def reset_events(self):
        self.events = []
        self.launches = {s: 0 for s in STAGES}
        self.trace = []
        self.cfgs = {}

    def reset_all(self):
        self.reset_events()
        self.compiles = 0
        self.bench_calls = 0
        self.bench_s = 0.0
        self.cfgs = {}


REC = Recorder()


def _post_compile_hook(*args, **kwargs):
    REC.compiles += 1


def _bench(self, *args, config=None, **meta):
    # Autotune benchmarks replay the kernel many times with candidate configs;
    # none of those launches may be counted or timed as a real call.
    REC.bench_depth += 1
    t0 = time.time()
    try:
        return _orig_bench(self, *args, config=config, **meta)
    finally:
        REC.bench_depth -= 1
        REC.bench_calls += 1
        REC.bench_s += time.time() - t0


def _run(self, *args, grid, warmup, **kw):
    name = getattr(self, "__name__", "")
    if not warmup and name in REC.launches and REC.bench_depth == 0:
        if REC.active:
            e0 = torch.gcu.Event(enable_timing=True)
            e1 = torch.gcu.Event(enable_timing=True)
            e0.record()
            out = _orig_run(self, *args, grid=grid, warmup=warmup, **kw)
            e1.record()
            REC.events.append((name, e0, e1))
            REC.launches[name] += 1
            cfg = (kw.get("BS"), kw.get("BT"), kw.get("num_warps"), kw.get("num_stages"))
            REC.cfgs.setdefault(name, [])
            if cfg not in REC.cfgs[name]:
                REC.cfgs[name].append(cfg)
            if REC.call is not None:
                REC.call.append(name)
            return out
        REC.launches[name] += 1
        cfg = (kw.get("BS"), kw.get("BT"), kw.get("num_warps"), kw.get("num_stages"))
        REC.cfgs.setdefault(name, [])
        if cfg not in REC.cfgs[name]:
            REC.cfgs[name].append(cfg)
        if REC.call is not None:
            REC.call.append(name)
    return _orig_run(self, *args, grid=grid, warmup=warmup, **kw)


triton.runtime.jit.JITFunction.run = _run
triton.runtime.autotuner.Autotuner._bench = _bench
triton.knobs.runtime.jit_post_compile_hook = _post_compile_hook


def tensors(dtype, seed=42, b=B, t=T, h=H, hq=HQ, d=D):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    def mk(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(
            device="gcu:1", dtype=dtype)
    return [mk(b, t, hq, d), mk(b, t, hq, d), mk(b, t, h, d), mk(b, t, h, d)]


def grad_out(dtype, b=B, t=T, hq=HQ, d=D):
    gen = torch.Generator(device="cpu").manual_seed(7)
    return torch.randn(b, t, hq, d, generator=gen, dtype=torch.float32).to(
        device="gcu:1", dtype=dtype)


def call(base, go):
    q, r, k, v = (x.detach().clone().requires_grad_(True) for x in base)
    o = parallel_parallax(q, r, k, v, scale=D ** -0.5)
    g = torch.autograd.grad(o, (q, r, k, v), go)
    return o, g


def rel(x, y):
    diff = x.float() - y.float()
    return (diff.square().mean().sqrt()
            / y.float().square().mean().sqrt().clamp_min(1e-9)).item()


# --------------------------------------------------------------------------- cold
def mode_cold(a):
    os.environ["PARALLAX_ROW_TILE_D64"] = ROUTES[a.route]
    REC.reset_all()
    dt = DTYPES[a.dtype]
    base, go = tensors(dt), grad_out(dt)
    t0 = time.time()
    o, g = call(base, go)
    torch.gcu.synchronize()
    wall = time.time() - t0
    print(json.dumps({
        "mode": "cold", "dtype": a.dtype, "route": a.route,
        "wall_s": round(wall, 2),
        "compiles": REC.compiles, "autotune_bench_calls": REC.bench_calls,
        "autotune_bench_s": round(REC.bench_s, 2),
        "launch_cfgs": REC.cfgs,
        "launches": REC.launches,
    }), flush=True)


# --------------------------------------------------------------------------- hot
def stage_timing(base, go, iters):
    REC.reset_events()
    REC.active = True
    for _ in range(iters):
        REC.call = []
        call(base, go)
        REC.trace.append(REC.call)
    REC.active = False
    torch.gcu.synchronize()
    per = {}
    for name in STAGES:
        vals = [e0.elapsed_time(e1) * 1e3 for (n, e0, e1) in REC.events if n == name]
        per[name] = {"med_us": round(statistics.median(vals), 2),
                     "min_us": round(min(vals), 2),
                     "max_us": round(max(vals), 2),
                     "n": len(vals)}
    # device-busy span: first event start to last event end of each call is not
    # recoverable from events alone, so report the sum of stage medians plus the
    # full-call wall measured separately.
    shapes = {tuple(sorted(set(c))) for c in REC.trace}
    return per, shapes


def full_call_timing(base, go, iters):
    for _ in range(10):
        call(base, go)
    torch.gcu.synchronize()
    e0 = torch.gcu.Event(enable_timing=True)
    e1 = torch.gcu.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        call(base, go)
    e1.record()
    torch.gcu.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3


def mode_hot(a):
    out = []
    for dtype_name in a.dtypes:
        dt = DTYPES[dtype_name]
        base, go = tensors(dt), grad_out(dt)

        # ---- correctness against the FP32 reference and cross-route agreement
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_p008_ref", os.path.join(os.environ["PARALLAX_TESTS"], "test_parallax.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_p008_ref"] = mod
        spec.loader.exec_module(mod)

        ins32 = tuple(x.detach().float().requires_grad_(True) for x in base)
        o32 = mod._reference_parallax(*ins32, scale=D ** -0.5, window_size=None)
        g32 = torch.autograd.grad(o32, ins32, go.float())
        torch.gcu.synchronize()

        res = {"mode": "hot", "dtype": dtype_name, "route": {}}
        outs = {}
        # Warm (and, on a cold cache, autotune) both routes first so no
        # warm-up launch can be counted or attributed to the other route.
        for route, flag in ROUTES.items():
            os.environ["PARALLAX_ROW_TILE_D64"] = flag
            call(base, go)
        torch.gcu.synchronize()
        for route, flag in ROUTES.items():
            os.environ["PARALLAX_ROW_TILE_D64"] = flag
            REC.reset_events()
            o, g = call(base, go)
            torch.gcu.synchronize()
            outs[route] = (o, g)
            r = {"o": rel(o, o32.detach())}
            for nm, gg, rr in zip(("dq", "dr", "dk", "dv"), g, g32):
                r[nm] = rel(gg, rr)
            res["route"][route] = {"rel_vs_fp32": r,
                                   "launch_cfgs": {k: list(v) for k, v in REC.cfgs.items()},
                                   "launches_per_call": dict(REC.launches)}
        res["rel_candidate_vs_shipped"] = {"o": rel(outs["candidate"][0], outs["shipped"][0])}
        for nm, cc, ss in zip(("dq", "dr", "dk", "dv"),
                              outs["candidate"][1], outs["shipped"][1]):
            res["rel_candidate_vs_shipped"][nm] = rel(cc, ss)
        out.append(res)
        print(json.dumps(res), flush=True)


        # ---- per-stage device time (no per-launch sync)
        for route, flag in ROUTES.items():
            os.environ["PARALLAX_ROW_TILE_D64"] = flag
            for _ in range(10):
                call(base, go)
            torch.gcu.synchronize()
            per, shapes = stage_timing(base, go, a.iters)
            full = full_call_timing(base, go, a.iters)
            ssum = round(sum(v["med_us"] for v in per.values()), 2)
            print(json.dumps({"mode": "stage", "dtype": dtype_name, "route": route,
                              "stage": per, "sum_stage_med_us": ssum,
                              "full_call_med_us": round(full, 2),
                              "stage_sum_over_full": round(ssum / full, 4),
                              "launch_sequences": sorted(list(s) for s in shapes),
                              "cfgs": REC.cfgs}), flush=True)

        # ---- alternating full-call A/B
        rows = []
        for rnd in range(a.rounds):
            order = ["shipped", "candidate"] if rnd % 2 == 0 else ["candidate", "shipped"]
            for name in order:
                os.environ["PARALLAX_ROW_TILE_D64"] = ROUTES[name]
                rows.append({"round": rnd, "route": name,
                             "us": round(full_call_timing(base, go, a.iters), 2)})
        s = [r["us"] for r in rows if r["route"] == "shipped"]
        c = [r["us"] for r in rows if r["route"] == "candidate"]
        print(json.dumps({
            "mode": "ab", "dtype": dtype_name, "rounds": a.rounds, "iters": a.iters,
            "shipped_med_us": round(statistics.median(s), 2),
            "candidate_med_us": round(statistics.median(c), 2),
            "ratio": round(statistics.median(c) / statistics.median(s), 4),
            "per_round_delta_pct": [round((cc / ss - 1) * 100, 2)
                                    for ss, cc in zip(s, c)],
            "shipped_all": s, "candidate_all": c}), flush=True)

def mode_mem(a):
    """Peak memory for a single fresh call (one route, one process)."""
    os.environ["PARALLAX_ROW_TILE_D64"] = ROUTES[a.route]
    dt = DTYPES[a.dtype]
    base, go = tensors(dt), grad_out(dt)
    torch.gcu.synchronize()
    torch.gcu.empty_cache()
    torch.gcu.reset_peak_memory_stats()
    t0 = time.time()
    call(base, go)
    torch.gcu.synchronize()
    print(json.dumps({"mode": "mem", "dtype": a.dtype, "route": a.route,
                      "peak_alloc_MB": round(torch.gcu.max_memory_allocated() / 2**20, 2),
                      "peak_reserved_MB": round(torch.gcu.max_memory_reserved() / 2**20, 2),
                      "wall_s": round(time.time() - t0, 2)}), flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["cold", "hot", "mem"])
ap.add_argument("--dtype", default="fp16")
ap.add_argument("--dtypes", nargs="+", default=["fp16", "bf16"])
ap.add_argument("--route", default="shipped")
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--rounds", type=int, default=7)
a = ap.parse_args()

if a.mode == "cold":
    mode_cold(a)
elif a.mode == "mem":
    mode_mem(a)
else:
    mode_hot(a)
