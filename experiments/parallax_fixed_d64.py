"""PARALLAX-004: GCU-1 runtime path probe + reference validation.

Runs one Parallax forward/backward on container GCU 1 (host GCU 5), records
every real Triton kernel launch (name/grid/device) and compares output plus
q/r/k/v gradients with tests/flag_attn/test_parallax.py::_reference_parallax.

Never touches device 0.
"""

import argparse
import importlib.util
import json
import os
import time
import traceback

import torch
import torch_gcu  # noqa: F401
import triton.runtime.autotuner as autotuner_mod
import triton.runtime.jit as jit_mod
from triton.runtime import driver

from flag_attn import parallel_parallax

DEVICE = 1
TOL = {torch.float16: 0.005, torch.bfloat16: 0.02}
STATE = {"in_tune": 0, "launches": [], "tunes": [], "compiles": 0, "compile_s": 0.0}


def emit(**kw):
    print(json.dumps(kw, default=str), flush=True)


def _kname(fn):
    return getattr(getattr(fn, "fn", fn), "__name__", str(fn))


def install_probes():
    orig_run = jit_mod.JITFunction.run

    def run(self, *args, grid, warmup, **kwargs):
        if not warmup:
            g = None
            if not callable(grid):
                g = tuple(grid) if isinstance(grid, (tuple, list)) else grid
            if STATE["in_tune"]:
                tag = "autotune_bench"
            else:
                tag = "launch"
            STATE["launches"].append({
                "kernel": _kname(self),
                "grid": g,
                "grid_callable": callable(grid),
                "tag": tag,
                "device": driver.active.get_current_device(),
            })
        return orig_run(self, *args, grid=grid, warmup=warmup, **kwargs)

    jit_mod.JITFunction.run = run

    orig_bench = autotuner_mod.Autotuner._bench

    def bench(self, *args, config, **meta):
        STATE["in_tune"] += 1
        started = time.perf_counter()
        try:
            ret = orig_bench(self, *args, config=config, **meta)
        finally:
            STATE["in_tune"] -= 1
        elapsed = time.perf_counter() - started
        median = ret[0] if isinstance(ret, (list, tuple)) and ret else None
        emit(event="tune_live", kernel=_kname(self), wall_s=round(elapsed, 3),
             median_ms=None if median is None else round(float(median), 4))
        STATE["tunes"].append({
            "kernel": _kname(self),
            "config": str(config),
            "wall_s": round(elapsed, 3),
            "median_ms": None if median is None else round(float(median), 4),
        })
        return ret

    autotuner_mod.Autotuner._bench = bench

    orig_compile = jit_mod.JITFunction._do_compile

    def do_compile(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return orig_compile(self, *args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            STATE["compiles"] += 1
            STATE["compile_s"] += elapsed
            emit(event="jit_compile", kernel=_kname(self), wall_s=round(elapsed, 3))
    jit_mod.JITFunction._do_compile = do_compile


def load_reference():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "tests", "flag_attn", "test_parallax.py")
    spec = importlib.util.spec_from_file_location("p004_test_parallax", path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._reference_parallax


def make_inputs(B, T, H, HQ, D, dtype):
    torch.manual_seed(42)
    q, r = [torch.randn(B, T, HQ, D, dtype=dtype, device=f"gcu:{DEVICE}").requires_grad_() for _ in range(2)]
    k, v = [torch.randn(B, T, H, D, dtype=dtype, device=f"gcu:{DEVICE}").requires_grad_() for _ in range(2)]
    torch.manual_seed(42)
    grad_output = torch.randn(B, T, HQ, D, dtype=dtype, device=f"gcu:{DEVICE}")
    return (q, r, k, v), grad_output


def metrics(name, actual, expected, tol):
    a = actual.detach().float()
    e = expected.detach().float()
    delta = a - e
    finite_a = bool(torch.isfinite(a).all())
    finite_e = bool(torch.isfinite(e).all())
    ratio = (delta.square().mean().sqrt() / e.square().mean().sqrt().clamp_min(1e-8)).item()
    max_abs = delta.abs().max().item()
    nonfinite_a = int((~torch.isfinite(a)).sum())
    nonfinite_e = int((~torch.isfinite(e)).sum())
    passed = finite_a and finite_e and (max_abs <= 1e-6 or ratio < tol)
    return {
        "name": name,
        "shape": list(a.shape),
        "finite_actual": finite_a,
        "finite_reference": finite_e,
        "nonfinite_actual": nonfinite_a,
        "nonfinite_reference": nonfinite_e,
        "max_abs": max_abs,
        "relative_rms": ratio,
        "tolerance": tol,
        "pass": passed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--HQ", type=int, default=8)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--window", type=int, default=-1)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--warmup-only", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--skip-reference", action="store_true")
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    torch.gcu.set_device(DEVICE)
    assert torch.gcu.current_device() == DEVICE, torch.gcu.current_device()
    if args.probe:
        install_probes()

    B, T, H, HQ, D = args.B, args.T, args.H, args.HQ, args.D
    window_size = None if args.window < 0 else args.window
    inputs, grad_output = make_inputs(B, T, H, HQ, D, dtype)
    scale = args.scale
    emit(event="setup", device=torch.gcu.current_device(), dtype=args.dtype,
         shape=[B, T, H, HQ, D], window=window_size, fixed_d64_env=os.environ.get("PARALLAX_FIXED_D64"),
         git_head=os.popen("git rev-parse HEAD").read().strip())

    started = time.perf_counter()
    output = parallel_parallax(*inputs, scale=scale, window_size=window_size)
    grads = torch.autograd.grad(output, inputs, grad_output, allow_unused=True)
    torch.gcu.synchronize()
    wall_s = time.perf_counter() - started
    emit(event="device_call", wall_s=round(wall_s, 3),
         finite=bool(torch.isfinite(output).all()) and all(g is not None and bool(torch.isfinite(g).all()) for g in grads),
         compiles=STATE["compiles"], compile_s=round(STATE["compile_s"], 3))

    launches = STATE["launches"]
    summary = {}
    for item in launches:
        if item["tag"] != "launch":
            continue
        key = item["kernel"]
        summary[key] = summary.get(key, 0) + 1
    emit(event="launch_summary", launches=summary,
         devices=sorted({i["device"] for i in launches if i["tag"] == "launch"}),
         autotune_benches=len(STATE["tunes"]))

    # Print target-relevant launch records only (the forward sweep is generic).
    for item in launches:
        if item["tag"] == "launch" and ("parallel_parallax" in item["kernel"]):
            emit(event="launch", **item)

    if args.warmup_only or args.skip_reference:
        emit(event="done", reference="skipped", output_checksum=float(output.detach().float().abs().sum().item()))
        return

    reference = load_reference()
    ref_inputs = tuple(t.detach().clone().requires_grad_() for t in inputs)
    expected = reference(*ref_inputs, scale=scale, window_size=window_size)
    expected_grads = torch.autograd.grad(expected, ref_inputs, grad_output)
    torch.gcu.synchronize()

    results = [metrics("o", output, expected, TOL[dtype])]
    for name, got, want in zip(("dq", "dr", "dk", "dv"), grads, expected_grads):
        results.append(metrics(name, got, want, TOL[dtype]))
    passed = all(r["pass"] for r in results)
    emit(event="validation", dtype=args.dtype, shape=[B, T, H, HQ, D], window=window_size,
         fixed_d64_env=os.environ.get("PARALLAX_FIXED_D64"), results=results, all_pass=passed,
         device_call_wall_s=round(wall_s, 3))
    for r in results:
        emit(event="metric", **r)
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(4)
