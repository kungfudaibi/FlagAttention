"""PARALLAX-006 harness: opt-in static target-shape config on GCU 1.

One process = one route (exact old code or candidate), one dtype, one case.
The candidate route sets PARALLAX_STATIC_D64=1 for the static target case and
PARALLAX_STATIC_D64=0 otherwise.

cold:   empty dedicated Triton cache, one forward+backward, probes on,
        reference validation, compile/autotune events.
steady: path-probe call, probes off, warmups, then N synchronized samples for
        forward, backward and full call, with peak memory.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
import torch_gcu  # noqa: F401
import triton
import triton.runtime.autotuner as autotuner_mod
import triton.runtime.jit as jit_mod
from triton import knobs
from triton.runtime import driver

import flag_attn
from flag_attn import parallel_parallax

DEVICE = 1
TOL = {torch.float16: 0.005, torch.bfloat16: 0.02}
STATE = {"in_tune": 0, "launches": [], "tunes": [], "compiles": 0, "compile_s": 0.0}
ORIG = {}

CASES = {
    "target": dict(B=2, T=256, H=2, HQ=8, D=64, window=None),
    "t63": dict(B=1, T=63, H=1, HQ=1, D=64, window=None),
    "nearby": dict(B=1, T=256, H=2, HQ=8, D=64, window=None),
}


def emit(**kw):
    print(json.dumps(kw, default=str), flush=True)


def kname(fn):
    return getattr(getattr(fn, "fn", fn), "__name__", str(fn))


def cache_stats():
    root = Path(knobs.cache.dir or "")
    files = list(root.rglob("*")) if root.exists() else []
    return {"cache_dir": str(root), "cache_files": sum(1 for p in files if p.is_file()),
            "cache_mb": round(sum(p.stat().st_size for p in files if p.is_file()) / 2**20, 1)}


def install_probes():
    ORIG["run"] = jit_mod.JITFunction.run
    ORIG["bench"] = autotuner_mod.Autotuner._bench
    ORIG["compile"] = jit_mod.JITFunction._do_compile

    def run(self, *args, grid, warmup, **kwargs):
        if not warmup:
            g = None if callable(grid) else (tuple(grid) if isinstance(grid, (tuple, list)) else grid)
            rec = {"kernel": kname(self), "grid": g, "grid_callable": callable(grid),
                   "tag": "autotune_bench" if STATE["in_tune"] else "launch",
                   "device": driver.active.get_current_device()}
            for key in ("BT", "BS", "BK", "BP", "num_warps", "num_stages"):
                if key in kwargs:
                    rec[key] = kwargs[key]
            STATE["launches"].append(rec)
        return ORIG["run"](self, *args, grid=grid, warmup=warmup, **kwargs)

    def bench(self, *args, config, **meta):
        STATE["in_tune"] += 1
        started = time.perf_counter()
        try:
            ret = ORIG["bench"](self, *args, config=config, **meta)
        finally:
            STATE["in_tune"] -= 1
        elapsed = time.perf_counter() - started
        median = ret[0] if isinstance(ret, (list, tuple)) and ret else None
        rec = {"kernel": getattr(self, "base_fn", self).__name__, "config": str(config),
               "wall_s": round(elapsed, 3),
               "median_ms": None if median is None else round(float(median), 4)}
        STATE["tunes"].append(rec)
        emit(event="autotune_config", **rec)
        return ret

    def do_compile(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return ORIG["compile"](self, *args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            STATE["compiles"] += 1
            STATE["compile_s"] += elapsed
            emit(event="jit_compile", kernel=kname(self), wall_s=round(elapsed, 3))

    jit_mod.JITFunction.run = run
    autotuner_mod.Autotuner._bench = bench
    jit_mod.JITFunction._do_compile = do_compile


def restore_probes():
    jit_mod.JITFunction.run = ORIG["run"]
    autotuner_mod.Autotuner._bench = ORIG["bench"]
    jit_mod.JITFunction._do_compile = ORIG["compile"]


def launch_summary():
    summary = {}
    for item in STATE["launches"]:
        if item["tag"] == "launch":
            summary[item["kernel"]] = summary.get(item["kernel"], 0) + 1
    return summary


def load_reference():
    tree = Path(flag_attn.__file__).resolve().parents[2]
    path = tree / "tests" / "flag_attn" / "test_parallax.py"
    spec = importlib.util.spec_from_file_location("p006_test_parallax", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._reference_parallax


def make_inputs(dtype, case):
    shape = CASES[case]
    B, T, H, HQ, D = shape["B"], shape["T"], shape["H"], shape["HQ"], shape["D"]
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
    passed = finite_a and finite_e and (max_abs <= 1e-6 or ratio < tol)
    return {"name": name, "shape": list(a.shape), "finite_actual": finite_a,
            "finite_reference": finite_e, "nonfinite_actual": int((~torch.isfinite(a)).sum()),
            "nonfinite_reference": int((~torch.isfinite(e)).sum()), "max_abs": max_abs,
            "relative_rms": ratio, "tolerance": tol, "pass": passed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=("baseline", "candidate"), required=True)
    ap.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    ap.add_argument("--mode", choices=("cold", "steady"), required=True)
    ap.add_argument("--case", choices=tuple(CASES), default="target")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--skip-reference", action="store_true")
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    torch.gcu.set_device(DEVICE)
    assert torch.gcu.current_device() == DEVICE, torch.gcu.current_device()
    pkg = Path(flag_attn.__file__).resolve()
    tree = pkg.parents[2]
    head = os.popen(f"git -C {tree} rev-parse HEAD").read().strip()
    par = tree / "src/flag_attn/runtime/backend/_enflame/FLA/parallax/parallel.py"
    digest = hashlib.sha256(par.read_bytes()).hexdigest()[:16]
    emit(event="import_probe", route=args.route, dtype=args.dtype, mode=args.mode, case=args.case,
         flag_attn_file=str(pkg), tree=str(tree), git_head=head, parallel_py_sha256_16=digest,
         static_d64_env=os.environ.get("PARALLAX_STATIC_D64"), device=torch.gcu.current_device(),
         pid=os.getpid(), affinity=sorted(os.sched_getaffinity(0))[:3] + ["..."],
         affinity_count=len(os.sched_getaffinity(0)))

    inputs, grad_output = make_inputs(dtype, args.case)
    scale = 64 ** -0.5
    emit(event="setup", case=args.case, shape=list(inputs[0].shape), dtype=args.dtype, scale=scale,
         torch_version=torch.__version__, cache=cache_stats())

    if args.mode == "cold":
        install_probes()
        emit(event="cache_before", **cache_stats())
        t0 = time.perf_counter()
        out = parallel_parallax(*inputs, scale=scale)
        torch.gcu.synchronize()
        fwd = time.perf_counter() - t0
        emit(event="stage_done", stage="forward", wall_s=round(fwd, 3),
             compiles=STATE["compiles"], compile_s=round(STATE["compile_s"], 3))
        t0 = time.perf_counter()
        grads = torch.autograd.grad(out, inputs, grad_output)
        torch.gcu.synchronize()
        bwd = time.perf_counter() - t0
        emit(event="stage_done", stage="backward", wall_s=round(bwd, 3),
             compiles=STATE["compiles"], compile_s=round(STATE["compile_s"], 3))
        emit(event="device_call", wall_s=round(fwd + bwd, 3), compiles=STATE["compiles"],
             compile_s=round(STATE["compile_s"], 3), autotune_configs=len(STATE["tunes"]))
        emit(event="launch_summary", launches=launch_summary(),
             devices=sorted({i["device"] for i in STATE["launches"] if i["tag"] == "launch"}))
        for item in STATE["launches"]:
            if item["tag"] == "launch":
                emit(event="launch", **item)
        emit(event="cache_after", **cache_stats())
        if args.skip_reference:
            emit(event="done", reference="skipped")
            return
        reference = load_reference()
        ref_inputs = tuple(t.detach().clone().requires_grad_() for t in inputs)
        expected = reference(*ref_inputs, scale=scale, window_size=None)
        expected_grads = torch.autograd.grad(expected, ref_inputs, grad_output)
        torch.gcu.synchronize()
        results = [metrics("o", out, expected, TOL[dtype])]
        for name, got, want in zip(("dq", "dr", "dk", "dv"), grads, expected_grads):
            results.append(metrics(name, got, want, TOL[dtype]))
        passed = all(r["pass"] for r in results)
        emit(event="validation", dtype=args.dtype, case=args.case, results=results, all_pass=passed)
        for r in results:
            emit(event="metric", **r)
        raise SystemExit(0 if passed else 3)

    # steady
    install_probes()
    out = parallel_parallax(*inputs, scale=scale)
    grads = torch.autograd.grad(out, inputs, grad_output)
    torch.gcu.synchronize()
    emit(event="path_probe", launches=launch_summary(),
         devices=sorted({i["device"] for i in STATE["launches"] if i["tag"] == "launch"}),
         autotune_configs=len(STATE["tunes"]), static_d64_env=os.environ.get("PARALLAX_STATIC_D64"))
    for item in STATE["launches"]:
        if item["tag"] == "launch":
            emit(event="launch", **item)
    restore_probes()
    del out, grads

    for _ in range(args.warmups):
        out = parallel_parallax(*inputs, scale=scale)
        torch.autograd.grad(out, inputs, grad_output)
        torch.gcu.synchronize()
        del out
    torch.gcu.synchronize()
    torch.gcu.reset_peak_memory_stats()

    records = []
    for i in range(args.samples):
        torch.gcu.synchronize()
        t0 = time.perf_counter()
        out = parallel_parallax(*inputs, scale=scale)
        torch.gcu.synchronize()
        fwd = time.perf_counter() - t0

        torch.gcu.synchronize()
        t0 = time.perf_counter()
        grads = torch.autograd.grad(out, inputs, grad_output)
        torch.gcu.synchronize()
        bwd = time.perf_counter() - t0
        del out, grads

        torch.gcu.synchronize()
        t0 = time.perf_counter()
        out = parallel_parallax(*inputs, scale=scale)
        grads = torch.autograd.grad(out, inputs, grad_output)
        torch.gcu.synchronize()
        full = time.perf_counter() - t0
        del out, grads
        rec = {"route": args.route, "dtype": args.dtype, "round": args.round,
               "sample": i, "forward_s": round(fwd, 6), "backward_s": round(bwd, 6),
               "full_s": round(full, 6)}
        records.append(rec)
        emit(event="sample", **rec)

    peak_alloc = int(torch.gcu.max_memory_allocated())
    peak_reserved = int(torch.gcu.max_memory_reserved())
    summary = {}
    for key in ("forward_s", "backward_s", "full_s"):
        vals = [r[key] for r in records]
        summary[key] = {"median": statistics.median(vals), "mean": statistics.fmean(vals),
                        "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                        "variance": statistics.pvariance(vals), "min": min(vals), "max": max(vals),
                        "spread": max(vals) - min(vals), "samples": vals}
    emit(event="steady_summary", route=args.route, dtype=args.dtype, round=args.round,
         samples=len(records), peak_memory_allocated_bytes=peak_alloc,
         peak_memory_reserved_bytes=peak_reserved, summary=summary)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(4)
