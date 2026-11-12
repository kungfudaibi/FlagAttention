"""PARALLAX-003: GCU-0-only stage/compile probe for target Parallax backward.

Adapted from experiments/parallax_target.py (which hardcodes gcu:1). This copy
pins GCU 0, uses seed 42, and instruments Triton autotune sweeps and JIT
compiles. It never touches GCU 1.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch_gcu  # noqa: F401
import triton
import triton.runtime.autotuner as autotuner_mod
from triton import knobs

from flag_attn import parallel_parallax

DEVICE = 0
STATE = {"phase": "setup", "compiles": 0, "compile_s": 0.0}


def emit(**kw):
    kw.setdefault("phase", STATE["phase"])
    kw["t"] = round(time.time(), 3)
    print(json.dumps(kw), flush=True)


def _cache_stats():
    root = Path(knobs.cache.dir or "")
    files = list(root.rglob("*")) if root.exists() else []
    nfiles = sum(1 for p in files if p.is_file())
    nbytes = sum(p.stat().st_size for p in files if p.is_file())
    return {"cache_dir": str(root), "cache_files": nfiles, "cache_mb": round(nbytes / 2**20, 1)}


def install_probes():
    orig_bench = autotuner_mod.Autotuner._bench

    def bench(self, *args, config, **meta):
        before = STATE["compiles"]
        started = time.perf_counter()
        ret = orig_bench(self, *args, config=config, **meta)
        elapsed = time.perf_counter() - started
        median = ret[0] if isinstance(ret, (list, tuple)) and ret else None
        emit(event="autotune_config", kernel=self.base_fn.__name__,
             config=str(config), wall_s=round(elapsed, 3),
             compiles=STATE["compiles"] - before,
             median_ms=None if median is None else round(float(median), 4))
        return ret

    autotuner_mod.Autotuner._bench = bench

    orig_run = autotuner_mod.Autotuner.run

    def run(self, *args, **kwargs):
        before = STATE["compiles"]
        prev_bench = getattr(self, "bench_time", None)
        started = time.perf_counter()
        ret = orig_run(self, *args, **kwargs)
        elapsed = time.perf_counter() - started
        tuned = getattr(self, "bench_time", None) != prev_bench
        emit(event="kernel_launch", kernel=self.base_fn.__name__,
             wall_s=round(elapsed, 3), tuned_sweep=tuned,
             bench_time_s=round(getattr(self, "bench_time", 0.0), 3),
             best_config=str(getattr(self, "best_config", None)),
             compiles=STATE["compiles"] - before)
        return ret

    autotuner_mod.Autotuner.run = run

    from triton.runtime.jit import JITFunction
    orig_compile = JITFunction._do_compile

    def do_compile(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return orig_compile(self, *args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            STATE["compiles"] += 1
            STATE["compile_s"] += elapsed
            emit(event="jit_compile", kernel=getattr(self, "fn", self).__name__,
                 wall_s=round(elapsed, 3))

    JITFunction._do_compile = do_compile


def make_inputs(dtype):
    B, T, H, HQ, D = 2, 256, 2, 8, 64
    torch.manual_seed(42)
    q, r = [torch.randn(B, T, HQ, D, dtype=dtype).to(f"gcu:{DEVICE}").requires_grad_() for _ in range(2)]
    k, v = [torch.randn(B, T, H, D, dtype=dtype).to(f"gcu:{DEVICE}").requires_grad_() for _ in range(2)]
    grad_output = torch.randn(B, T, HQ, D, dtype=dtype).to(f"gcu:{DEVICE}")
    return (q, r, k, v), grad_output


def diff(actual, reference):
    a, r = actual.detach().float().cpu(), reference.detach().float().cpu()
    delta = a - r
    return {"relative_rms": (delta.square().mean().sqrt() / r.square().mean().sqrt().clamp_min(1e-8)).item(),
            "max_abs": delta.abs().max().item(),
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(r).all())}


parser = argparse.ArgumentParser()
parser.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
parser.add_argument("--mode", choices=("cold", "warm", "compare"), default="cold")
parser.add_argument("--phase", choices=("forward", "backward", "both"), default="both")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--save", default=None)
parser.add_argument("--load", default=None)
args = parser.parse_args()

torch.gcu.set_device(DEVICE)
assert torch.gcu.current_device() == DEVICE, torch.gcu.current_device()
dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
inputs, grad_output = make_inputs(dtype)

if args.mode != "compare":
    install_probes()
    emit(event="cache_before", dtype=args.dtype, mode=args.mode, **_cache_stats())
    STATE["phase"] = "forward"
    started = time.perf_counter()
    out = parallel_parallax(*inputs, scale=64**-0.5)
    torch.gcu.synchronize()
    forward_s = time.perf_counter() - started
    emit(event="stage_done", stage="forward", wall_s=round(forward_s, 3))
    backward_s = None
    grads = None
    if args.phase in ("backward", "both"):
        STATE["phase"] = "backward"
        started = time.perf_counter()
        grads = torch.autograd.grad(out, inputs, grad_output)
        torch.gcu.synchronize()
        backward_s = time.perf_counter() - started
        emit(event="stage_done", stage="backward", wall_s=round(backward_s, 3))
    STATE["phase"] = "warm"
    warm = []
    for repeat in range(args.repeats if args.mode == "warm" else 0):
        started = time.perf_counter()
        o2 = parallel_parallax(*inputs, scale=64**-0.5)
        torch.gcu.synchronize()
        f_s = time.perf_counter() - started
        started = time.perf_counter()
        g2 = torch.autograd.grad(o2, inputs, grad_output)
        torch.gcu.synchronize()
        b_s = time.perf_counter() - started
        warm.append({"repeat": repeat, "forward_s": round(f_s, 4), "backward_s": round(b_s, 4),
                     "finite": all(bool(torch.isfinite(g).all()) for g in g2)})
        emit(event="warm_repeat", **warm[-1])
    emit(event="cache_after", **_cache_stats())
    emit(event="summary", dtype=args.dtype, mode=args.mode,
         forward_s=round(forward_s, 3),
         backward_s=None if backward_s is None else round(backward_s, 3),
         compile_count=STATE["compiles"], compile_s=round(STATE["compile_s"], 3),
         warm=warm)
    if args.save and grads is not None:
        torch.save({"dtype": args.dtype, "out": out.detach().cpu(),
                    "grads": tuple(g.detach().cpu() for g in grads)}, args.save)
        emit(event="saved", path=args.save)
else:
    sys.path.insert(0, str(Path("tests/flag_attn").resolve()))
    from test_parallax import _reference_parallax
    saved = torch.load(args.load, map_location="cpu")
    reference = _reference_parallax(*inputs, scale=64**-0.5, window_size=None)
    ref_grads = torch.autograd.grad(reference, inputs, grad_output)
    torch.gcu.synchronize()
    out = saved["out"].to(f"gcu:{DEVICE}")
    grads = tuple(g.to(f"gcu:{DEVICE}") for g in saved["grads"])
    emit(event="accuracy", dtype=args.dtype,
         output=diff(out, reference),
         grads={name: diff(actual, expected) for name, actual, expected in
                zip(("q", "r", "k", "v"), grads, ref_grads)})
