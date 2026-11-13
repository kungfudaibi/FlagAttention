"""PARALLAX-007 probe: one route/dtype/case per process on container GCU 1.

Modes:
  cold    fresh cache, first process, probes on, numeric validation
  steady  path probe then synchronized forward/backward/full samples
  Any route behavior is selected by the tree on PYTHONPATH and
  PARALLAX_STATIC_D64 (candidate=1, baseline=0).

The harness deliberately records every JIT compile, every autotune config,
every launch (kernel/grid/constexprs), disk-autotune-cache decisions, cache
directory contents, device affinity and efsmi memory/usage samples.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

DEVICE = 1
CPU_START, CPU_END = 32, 63
CPU_LIST = list(range(CPU_START, CPU_END + 1))
TOL = {"fp16": 0.005, "bf16": 0.02}

CASES = {
    # B, T, H, HQ, D, window
    "target": dict(B=2, T=256, H=2, HQ=8, D=64, window=None),
    "nearby": dict(B=1, T=256, H=2, HQ=8, D=64, window=None),
    "t63": dict(B=1, T=63, H=1, HQ=1, D=64, window=None),
    "nearby_t128": dict(B=1, T=128, H=2, HQ=8, D=64, window=None),
    "window": dict(B=2, T=256, H=2, HQ=8, D=64, window=64),
}
TARGET_LAUNCHES = {
    "parallel_parallax_fwd_kernel",
    "parallel_parallax_bwd_kernel_dqr_fused",
    "parallel_parallax_bwd_kernel_dkv_grouped",
}

STATE = {
    "in_tune": 0,
    "launches": [],
    "tunes": [],
    "compiles": 0,
    "compile_s": 0.0,
    "disk_cache": [],
    "autotuners": [],
}
ORIG = {}


def emit(**kw):
    print(json.dumps(kw, default=str), flush=True)


def kname(fn):
    return getattr(getattr(fn, "fn", fn), "__name__", str(fn))


def efsmi_memory():
    try:
        out = subprocess.run(
            ["efsmi", "-q", "-d", "MEMORY", "--json-format"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        data = json.loads(out)["devices"][DEVICE]["Device Mem Info"]
        return {str(k): float(v) for k, v in data.items()}
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"error": repr(exc)}


def efsmi_usage():
    try:
        out = subprocess.run(
            ["efsmi", "-q", "-d", "USAGE", "--json-format"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        data = json.loads(out)["devices"][DEVICE]["Device Usage Info"]
        return {str(k): v for k, v in data.items()}
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"error": repr(exc)}


def cache_stats():
    root = Path(str(getattr(knobs.cache, "dir", "") or ""))
    files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
    autotune = [p for p in files if p.name.endswith(".autotune.json")]
    return {
        "cache_dir": str(root),
        "cache_files": len(files),
        "cache_mb": round(sum(p.stat().st_size for p in files) / 2**20, 3),
        "autotune_files": len(autotune),
        "autotune_paths": [str(p) for p in autotune[:20]],
        "autotune_bytes": sum(p.stat().st_size for p in autotune),
    }


# Import torch/triton first, then install the Autotuner.__init__ probe before
# importing flag_attn so every kernel decorator is observed at construction.
import torch  # noqa: E402
import torch_gcu  # noqa: F401,E402
import triton  # noqa: E402
import triton.runtime.autotuner as autotuner_mod  # noqa: E402
import triton.runtime.jit as jit_mod  # noqa: E402
from triton import knobs  # noqa: E402
from triton.runtime import driver  # noqa: E402


def _autotuner_init_probe(original):
    def wrapper(self, *args, **kwargs):
        original(self, *args, **kwargs)
        try:
            base = getattr(self, "base_fn", None)
            name = getattr(base, "__name__", str(base))
            STATE["autotuners"].append(
                {
                    "kernel": name,
                    "configs": len(self.configs),
                    "cache_results": bool(self.cache_results),
                    "keys": list(self.keys),
                    "knobs_cache": bool(knobs.autotuning.cache),
                }
            )
        except Exception as exc:  # pragma: no cover
            STATE["autotuners"].append({"error": repr(exc)})
    return wrapper


autotuner_mod.Autotuner.__init__ = _autotuner_init_probe(
    autotuner_mod.Autotuner.__init__
)

import flag_attn  # noqa: E402
from flag_attn import parallel_parallax  # noqa: E402
from flag_attn.runtime.backend._enflame.FLA import compat  # noqa: E402
from flag_attn.runtime.backend._enflame.FLA.parallax import parallel as parallax_parallel  # noqa: E402


def install_probes():
    ORIG["run"] = jit_mod.JITFunction.run
    ORIG["bench"] = autotuner_mod.Autotuner._bench
    ORIG["compile"] = jit_mod.JITFunction._do_compile
    ORIG["check_disk_cache"] = autotuner_mod.Autotuner.check_disk_cache

    def run(self, *args, grid, warmup, **kwargs):
        if not warmup:
            canonical = None if callable(grid) else list(grid)
            rec = {
                "kernel": kname(self),
                "grid": canonical,
                "grid_callable": callable(grid),
                "tag": "autotune_bench" if STATE["in_tune"] else "launch",
                "device": driver.active.get_current_device(),
            }
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
        rec = {
            "kernel": getattr(getattr(self, "base_fn", self), "__name__", str(self)),
            "config": str(config),
            "wall_s": round(elapsed, 6),
            "median_ms": None if median is None else round(float(median), 6),
        }
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
            emit(event="jit_compile", kernel=kname(self), wall_s=round(elapsed, 6))

    def check_disk_cache(self, tuning_key, configs, bench_fn):
        started = time.perf_counter()
        try:
            hit = ORIG["check_disk_cache"](self, tuning_key, configs, bench_fn)
        finally:
            elapsed = time.perf_counter() - started
            rec = {
                "event": "disk_cache",
                "kernel": getattr(getattr(self, "base_fn", self), "__name__", str(self)),
                "tuning_key": str(tuning_key),
                "configs": len(configs),
                "cache_results": bool(self.cache_results),
                "knobs_cache": bool(knobs.autotuning.cache),
                "wall_s": round(elapsed, 6),
            }
            if hit is not None:
                rec["hit"] = bool(hit)
            STATE["disk_cache"].append(rec)
            emit(**rec)
        return hit

    jit_mod.JITFunction.run = run
    autotuner_mod.Autotuner._bench = bench
    jit_mod.JITFunction._do_compile = do_compile
    autotuner_mod.Autotuner.check_disk_cache = check_disk_cache


def restore_probes():
    for key in ("run", "bench", "compile", "check_disk_cache"):
        if key not in ORIG:
            continue
    jit_mod.JITFunction.run = ORIG["run"]
    autotuner_mod.Autotuner._bench = ORIG["bench"]
    jit_mod.JITFunction._do_compile = ORIG["compile"]
    autotuner_mod.Autotuner.check_disk_cache = ORIG["check_disk_cache"]


def launch_summary():
    summary = {}
    for item in STATE["launches"]:
        if item["tag"] == "launch":
            summary[item["kernel"]] = summary.get(item["kernel"], 0) + 1
    return summary


def load_reference():
    tree = Path(flag_attn.__file__).resolve().parents[2]
    path = tree / "tests" / "flag_attn" / "test_parallax.py"
    spec = importlib.util.spec_from_file_location("p007_test_parallax", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._reference_parallax


def make_inputs(dtype, case):
    shape = CASES[case]
    B, T, H, HQ, D = shape["B"], shape["T"], shape["H"], shape["HQ"], shape["D"]
    torch.manual_seed(42)
    q, r = [
        torch.randn(B, T, HQ, D, dtype=dtype, device=f"gcu:{DEVICE}").requires_grad_()
        for _ in range(2)
    ]
    k, v = [
        torch.randn(B, T, H, D, dtype=dtype, device=f"gcu:{DEVICE}").requires_grad_()
        for _ in range(2)
    ]
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
    return {
        "name": name,
        "shape": list(a.shape),
        "finite_actual": finite_a,
        "finite_reference": finite_e,
        "nonfinite_actual": int((~torch.isfinite(a)).sum()),
        "nonfinite_reference": int((~torch.isfinite(e)).sum()),
        "max_abs": max_abs,
        "relative_rms": ratio,
        "tolerance": tol,
        "pass": passed,
    }


def apply_config_override(dtype, config_json):
    if not config_json:
        return
    payload = json.loads(config_json)
    key = torch.float16 if dtype == "fp16" else torch.bfloat16
    parallax_parallel._STATIC_D64_CONFIGS[key] = {
        name: tuple(int(x) for x in value) for name, value in payload.items()
    }
    emit(event="config_override", dtype=dtype, config=payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=("baseline", "candidate"), required=True)
    ap.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    ap.add_argument("--mode", choices=("cold", "steady"), required=True)
    ap.add_argument("--case", choices=tuple(CASES), default="target")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--samples", type=int, default=7)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--require-empty", action="store_true")
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--forward-only", action="store_true")
    ap.add_argument("--config-json", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    # Pin CPU before any device work and verify it.
    os.sched_setaffinity(0, CPU_LIST)
    affinity = sorted(os.sched_getaffinity(0))
    if affinity != CPU_LIST:
        emit(event="fatal", reason="cpu_affinity", expected=CPU_LIST, actual=affinity)
        raise SystemExit(2)

    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    torch.set_float32_matmul_precision("highest")
    if torch.gcu.device_count() < 2:
        emit(event="fatal", reason="device_count", count=torch.gcu.device_count())
        raise SystemExit(2)
    torch.gcu.set_device(DEVICE)
    if torch.gcu.current_device() != DEVICE:
        emit(event="fatal", reason="wrong_device", current=torch.gcu.current_device())
        raise SystemExit(2)

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    pkg = Path(flag_attn.__file__).resolve()
    tree = pkg.parents[2]
    head = subprocess.check_output(["git", "-C", str(tree), "rev-parse", "HEAD"], text=True).strip()
    par = tree / "src/flag_attn/runtime/backend/_enflame/FLA/parallax/parallel.py"
    digest = hashlib.sha256(par.read_bytes()).hexdigest()[:16]
    par_source = par.read_text()
    emit(
        event="import_probe",
        route=args.route,
        dtype=args.dtype,
        mode=args.mode,
        case=args.case,
        tag=args.tag,
        flag_attn_file=str(pkg),
        tree=str(tree),
        git_head=head,
        parallel_py_sha256_16=digest,
        static_d64_env=os.environ.get("PARALLAX_STATIC_D64"),
        device=torch.gcu.current_device(),
        device_count=torch.gcu.device_count(),
        pid=os.getpid(),
        affinity=affinity,
        affinity_count=len(affinity),
        triton_cache_dir=str(getattr(knobs.cache, "dir", "")),
        triton_cache_autotuning=os.environ.get("TRITON_CACHE_AUTOTUNING"),
        knobs_cache=bool(knobs.autotuning.cache),
        compat_supports_cache=bool(compat._SUPPORTS_AUTOTUNE_CACHE),
        compat_cache_kwargs=dict(compat.autotune_cache_kwargs),
        parallel_uses_compat_cache_kwargs=("autotune_cache_kwargs" in par_source),
        torch_version=torch.__version__,
        triton_version=triton.__version__,
        cwd=os.getcwd(),
        autotuners=STATE["autotuners"],
    )
    emit(event="device_memory_before", **efsmi_memory())
    emit(event="device_usage_before", **efsmi_usage())

    apply_config_override(args.dtype, args.config_json)

    inputs, grad_output = make_inputs(dtype, args.case)
    shape = CASES[args.case]
    scale = 64 ** -0.5
    emit(
        event="setup",
        case=args.case,
        shape=list(inputs[0].shape),
        dtype=args.dtype,
        scale=scale,
        window=shape["window"],
        torch_version=torch.__version__,
        cache=cache_stats(),
    )

    if args.mode == "cold":
        before = cache_stats()
        emit(event="cache_before", **before)
        if args.require_empty and before["cache_files"] != 0:
            emit(event="fatal", reason="cache_not_empty", cache=before)
            raise SystemExit(2)
        install_probes()
        t0 = time.perf_counter()
        out = parallel_parallax(*inputs, scale=scale, window_size=shape["window"])
        torch.gcu.synchronize()
        fwd = time.perf_counter() - t0
        emit(
            event="stage_done",
            stage="forward",
            wall_s=round(fwd, 6),
            compiles=STATE["compiles"],
            compile_s=round(STATE["compile_s"], 6),
        )
        t0 = time.perf_counter()
        grads = torch.autograd.grad(out, inputs, grad_output)
        torch.gcu.synchronize()
        bwd = time.perf_counter() - t0
        emit(
            event="stage_done",
            stage="backward",
            wall_s=round(bwd, 6),
            compiles=STATE["compiles"],
            compile_s=round(STATE["compile_s"], 6),
        )
        emit(
            event="device_call",
            wall_s=round(fwd + bwd, 6),
            compiles=STATE["compiles"],
            compile_s=round(STATE["compile_s"], 6),
            autotune_configs=len(STATE["tunes"]),
        )
        emit(
            event="launch_summary",
            launches=launch_summary(),
            devices=sorted({i["device"] for i in STATE["launches"] if i["tag"] == "launch"}),
        )
        for item in STATE["launches"]:
            if item["tag"] == "launch":
                emit(event="launch", **item)
        after = cache_stats()
        emit(event="cache_after", **after)
        emit(
            event="peak_memory",
            peak_memory_allocated_bytes=int(torch.gcu.max_memory_allocated()),
            peak_memory_reserved_bytes=int(torch.gcu.max_memory_reserved()),
        )
        if args.forward_only:
            emit(event="done", reference="skipped", forward_only=True)
            return
        emit(event="autotuners_seen", autotuners=STATE["autotuners"])
        restore_probes()
        emit(event="device_memory_after", **efsmi_memory())
        emit(event="device_usage_after", **efsmi_usage())
        if args.skip_reference:
            emit(event="done", reference="skipped")
            return
        reference = load_reference()
        ref_inputs = tuple(t.detach().clone().requires_grad_() for t in inputs)
        expected = reference(
            *ref_inputs, scale=scale, window_size=shape["window"]
        )
        expected_grads = torch.autograd.grad(expected, ref_inputs, grad_output)
        torch.gcu.synchronize()
        results = [metrics("o", out, expected, TOL[args.dtype])]
        for name, got, want in zip(("dq", "dr", "dk", "dv"), grads, expected_grads):
            results.append(metrics(name, got, want, TOL[args.dtype]))
        passed = all(r["pass"] for r in results)
        emit(
            event="validation",
            dtype=args.dtype,
            case=args.case,
            results=results,
            all_pass=passed,
        )
        for r in results:
            emit(event="metric", **r)
        raise SystemExit(0 if passed else 3)

    # steady
    emit(event="cache_before_probe", **cache_stats())
    install_probes()
    out = parallel_parallax(*inputs, scale=scale, window_size=shape["window"])
    grads = torch.autograd.grad(out, inputs, grad_output)
    torch.gcu.synchronize()
    emit(
        event="path_probe",
        launches=launch_summary(),
        devices=sorted({i["device"] for i in STATE["launches"] if i["tag"] == "launch"}),
        autotune_configs=len(STATE["tunes"]),
        static_d64_env=os.environ.get("PARALLAX_STATIC_D64"),
    )
    for item in STATE["launches"]:
        if item["tag"] == "launch":
            emit(event="launch", **item)
    emit(event="autotuners_seen", autotuners=STATE["autotuners"])
    emit(event="cache_after_probe", **cache_stats())
    restore_probes()
    del out, grads

    for _ in range(args.warmups):
        out = parallel_parallax(*inputs, scale=scale, window_size=shape["window"])
        torch.autograd.grad(out, inputs, grad_output)
        torch.gcu.synchronize()
        del out
    torch.gcu.synchronize()
    torch.gcu.reset_peak_memory_stats()
    emit(event="device_memory_steady_start", **efsmi_memory())

    records = []
    for i in range(args.samples):
        torch.gcu.synchronize()
        t0 = time.perf_counter()
        out = parallel_parallax(*inputs, scale=scale, window_size=shape["window"])
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
        out = parallel_parallax(*inputs, scale=scale, window_size=shape["window"])
        grads = torch.autograd.grad(out, inputs, grad_output)
        torch.gcu.synchronize()
        full = time.perf_counter() - t0
        del out, grads
        rec = {
            "route": args.route,
            "dtype": args.dtype,
            "round": args.round,
            "sample": i,
            "forward_s": round(fwd, 6),
            "backward_s": round(bwd, 6),
            "full_s": round(full, 6),
        }
        records.append(rec)
        emit(event="sample", **rec)

    peak_alloc = int(torch.gcu.max_memory_allocated())
    peak_reserved = int(torch.gcu.max_memory_reserved())
    summary = {}
    for key in ("forward_s", "backward_s", "full_s"):
        vals = [r[key] for r in records]
        summary[key] = {
            "median": statistics.median(vals),
            "mean": statistics.fmean(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "variance": statistics.pvariance(vals),
            "min": min(vals),
            "max": max(vals),
            "spread": max(vals) - min(vals),
            "samples": vals,
        }
    emit(
        event="steady_summary",
        route=args.route,
        dtype=args.dtype,
        round=args.round,
        samples=len(records),
        peak_memory_allocated_bytes=peak_alloc,
        peak_memory_reserved_bytes=peak_reserved,
        summary=summary,
    )
    emit(event="device_memory_after", **efsmi_memory())
    emit(event="device_usage_after", **efsmi_usage())


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(4)
