"""GLA-006 single-case harness: real launch probe, correctness dump, timing, memory.

One process per (tree, case).  Tree chosen by PYTHONPATH.  Baseline and candidate
use identical seeds/inputs.  Experimental opt-in flags are explicit so the old
tree can never inherit them.
"""
import argparse
import json
import os
import sys
import time

import triton
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--B", type=int, default=2)
p.add_argument("--T", type=int, default=256)
p.add_argument("--H", type=int, default=2)
p.add_argument("--K", type=int, default=64)
p.add_argument("--V", type=int, default=64)
p.add_argument("--gln", type=float, default=1.0)
p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
p.add_argument("--seed", type=int, default=42)
p.add_argument("--exp-fp16", type=int, default=0)
p.add_argument("--exp-fp32", type=int, default=0)
p.add_argument("--tag", default="run")
p.add_argument("--save", default=None)
p.add_argument("--compare", default=None)
p.add_argument("--samples", type=int, default=5)
p.add_argument("--warmup", type=int, default=2)
p.add_argument("--no-timing", action="store_true")
p.add_argument("--mem", action="store_true")
p.add_argument("--force-fallback", action="store_true",
               help="disable the parallel state route in this process (numerical oracle)")
p.add_argument("--f32-default", default="ieee", choices=["ieee", "tf32", "unset"],
               help="TRITON_F32_DEFAULT used for this process")
a = p.parse_args()

os.environ["FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP16_STATE"] = str(a.exp_fp16)
os.environ["FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP32_STATE"] = str(a.exp_fp32)
if a.f32_default != "unset":
    os.environ["TRITON_F32_DEFAULT"] = a.f32_default
if a.force_fallback:
    os.environ["FLAGGEMS_ENFLAME_GLA_PARALLEL_STATE"] = "0"
    os.environ["FLAGGEMS_ENFLAME_GLA_PARALLEL_BWD_STATE"] = "0"

from flag_attn import chunk_gla  # noqa: E402
import importlib  # noqa: E402

C = importlib.import_module("flag_attn.runtime.backend._enflame.FLA.gla.chunk_gla")
Hmod = importlib.import_module("flag_attn.runtime.backend._enflame.FLA.chunk_h")

dev = torch.device("gcu:0")
B, T, H, K, V = a.B, a.T, a.H, a.K, a.V
DT = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]


def make_inputs():
    torch.manual_seed(a.seed)
    q = torch.rand(B, T, H, K, dtype=DT, device=dev).requires_grad_()
    k = torch.rand(B, T, H, K, dtype=DT, device=dev).requires_grad_()
    v = torch.rand(B, T, H, V, dtype=DT, device=dev).requires_grad_()
    g = (F.logsigmoid(torch.rand(B, T, H, K, dtype=DT, device=dev)) / a.gln).requires_grad_()
    h0 = torch.rand(B, H, K, V, dtype=torch.float32, device=dev).requires_grad_()
    do = torch.randn(B, T, H, V, dtype=DT, device=dev) if a.dtype != "fp32" else torch.randn(B, T, H, V, dtype=torch.float32, device=dev)
    dht = torch.randn(B, H, K, V, dtype=torch.float32, device=dev)
    return q, k, v, g, h0, do, dht


ATTEMPT, DONE = {}, {}
PROBE_NAMES = (
    "_chunk_fwd_local_state_kernel", "_chunk_bwd_local_state_kernel",
    "_chunk_fwd_state_scan_kernel", "_chunk_bwd_state_scan_kernel",
    "_chunk_fwd_state_scan_fp32_2d_group_kernel",
    "_chunk_bwd_state_scan_fp32_2d_group_kernel",
    "chunk_fwd_kernel_h", "chunk_bwd_kernel_dh",
)


class _Probe:
    def __init__(self, name, real):
        self.name = name
        self.real = real

    def __getitem__(self, grid):
        launcher = self.real[grid]

        def call(*args, **kwargs):
            ATTEMPT[self.name] = ATTEMPT.get(self.name, 0) + 1
            out = launcher(*args, **kwargs)
            DONE[self.name] = DONE.get(self.name, 0) + 1
            return out

        return call


for _n in PROBE_NAMES:
    if hasattr(Hmod, _n):
        setattr(Hmod, _n, _Probe(_n, getattr(Hmod, _n)))

result = {
    "tag": a.tag, "shape": [K, V], "B": B, "T": T, "H": H,
    "dtype": a.dtype, "gln": a.gln, "seed": a.seed,
    "exp_fp16": a.exp_fp16, "exp_fp32": a.exp_fp32,
    "pythonpath_head": os.environ.get("PYTHONPATH", "").split(":")[0],
    "chunk_h_file": Hmod.__file__,
    "git_head": os.environ.get("GLA_TREE_HEAD", ""),
    "device": "gcu:0",
    "triton_version": triton.__version__,
    "force_fallback": bool(a.force_fallback),
    "triton_f32_default": os.environ.get("TRITON_F32_DEFAULT", "<unset>"),
    "exit": 0,
}
t_start = time.perf_counter()

try:
    q, k, v, g, h0, do, dht = make_inputs()
    scale = K ** -0.5

    def full_fwd_bwd():
        for t in (q, k, v, g, h0):
            t.grad = None
        o, ht = chunk_gla(q=q, k=k, v=v, g=g, scale=scale, initial_state=h0,
                          output_final_state=True, cu_seqlens=None)
        ((o * do).sum() + (ht.to(o.dtype) * dht.to(o.dtype)).sum()).backward()
        return o, ht

    t0 = time.perf_counter()
    o, ht = full_fwd_bwd()
    torch.gcu.synchronize()
    result["cold_full_fwd_bwd_ms"] = (time.perf_counter() - t0) * 1000
    grads = dict(o=o, ht=ht, dq=q.grad, dk=k.grad, dv=v.grad, dg=g.grad, dh0=h0.grad)
    result["grad_finite"] = {n: bool(torch.isfinite(t).all()) for n, t in grads.items()}

    with torch.no_grad():
        qd, kd, vd = q.detach(), k.detach(), v.detach()
        gd, h0d = g.detach(), h0.detach()
        for t in (qd, kd, vd, gd, h0d):
            t.grad = torch.zeros_like(t)

        t0 = time.perf_counter()
        g_cumsum, A, h, ht_s, o_s = C.chunk_gla_fwd(
            q=qd, k=kd, v=vd, g=gd, g_cumsum=None, scale=scale,
            initial_state=h0d, output_final_state=True, cu_seqlens=None,
            chunk_size=64, chunk_indices=None)
        torch.gcu.synchronize()
        result["cold_fwd_ms"] = (time.perf_counter() - t0) * 1000

        def fwd_h():
            return C.chunk_fwd_h(k=kd, v=vd, g=None, gk=g_cumsum, gv=None, h0=h0d,
                                 output_final_state=True, cu_seqlens=None,
                                 chunk_size=64, states_in_fp32=False,
                                 state_v_first=False)

        def bwd_dh_fp32():
            return C.chunk_bwd_dh(q=qd, k=kd, v=vd, g=None, gk=g_cumsum, gv=None,
                                  do=do, h0=h0d, dht=dht, scale=scale,
                                  cu_seqlens=None, chunk_size=64,
                                  states_in_fp32=True, state_v_first=False)

        def whole_fwd():
            return C.chunk_gla_fwd(q=qd, k=kd, v=vd, g=gd, g_cumsum=None, scale=scale,
                                   initial_state=h0d, output_final_state=True,
                                   cu_seqlens=None, chunk_size=64,
                                   chunk_indices=None)

        def probe(fn):
            ATTEMPT.clear(); DONE.clear()
            out = fn()
            torch.gcu.synchronize()
            return out, {"attempt": dict(ATTEMPT), "done": dict(DONE)}

        (h_st, ht_st), fwd_path = probe(fwd_h)
        (dh_st, dh0_st), bwd_path = probe(bwd_dh_fp32)
        _, fwd_all_path = probe(whole_fwd)
        result["path_fwd_h"] = fwd_path
        result["path_bwd_dh"] = bwd_path
        result["path_whole_fwd"] = fwd_all_path
        result["parallel_fwd_h"] = fwd_path["done"].get("_chunk_fwd_local_state_kernel", 0) > 0
        result["parallel_bwd_dh"] = bwd_path["done"].get("_chunk_bwd_local_state_kernel", 0) > 0
        result["fallback_fwd_h"] = fwd_path["done"].get("chunk_fwd_kernel_h", 0)
        result["fallback_bwd_dh"] = bwd_path["done"].get("chunk_bwd_kernel_dh", 0)
        result["local_state_fp32_bytes_theory"] = {
            "fwd": B * ((T + 63) // 64) * H * K * V * 4,
            "bwd": B * ((T + 63) // 64) * H * K * V * 4,
        }

        stage_vals = dict(h=h_st, ht_state=ht_st, dh=dh_st, dh0_state=dh0_st)

        if a.mem:
            mem = {}

            def measure(fn):
                torch.gcu.synchronize()
                torch.gcu.reset_peak_memory_stats()
                before = torch.gcu.memory_allocated()
                out = fn()
                torch.gcu.synchronize()
                peak = torch.gcu.max_memory_allocated()
                after = torch.gcu.memory_allocated()
                del out
                return {"before": before, "peak": peak, "after": after,
                        "peak_delta": peak - before}

            mem["fwd_h"] = measure(fwd_h)
            mem["bwd_dh_fp32"] = measure(bwd_dh_fp32)
            mem["whole_fwd"] = measure(whole_fwd)
            result["memory"] = mem
            result["memory_reserved"] = torch.gcu.memory_reserved()

        if not a.no_timing:
            stages = {"fwd_h": fwd_h, "bwd_dh_fp32": bwd_dh_fp32,
                      "whole_fwd": whole_fwd}
            for _ in range(a.warmup):
                for f in stages.values():
                    f()
            torch.gcu.synchronize()
            timing = {}
            for name, f in stages.items():
                ts = []
                for _ in range(a.samples):
                    torch.gcu.synchronize()
                    t0 = time.perf_counter()
                    f()
                    torch.gcu.synchronize()
                    ts.append((time.perf_counter() - t0) * 1000)
                s = sorted(ts)
                timing[name] = {"ms": ts, "median": s[len(s) // 2], "min": s[0]}
            result["timing"] = timing

    if not a.no_timing:
        for _ in range(a.warmup):
            full_fwd_bwd()
        torch.gcu.synchronize()
        ts = []
        for _ in range(a.samples):
            torch.gcu.synchronize()
            t0 = time.perf_counter()
            full_fwd_bwd()
            torch.gcu.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        s = sorted(ts)
        result.setdefault("timing", {})["full_fwd_bwd"] = {
            "ms": ts, "median": s[len(s) // 2], "min": s[0]}

    # Probe the autograd full pass separately so backward launches are counted
    # from the production route, not only from the direct helper.
    if not a.no_timing:
        ATTEMPT.clear(); DONE.clear()
        o2, ht2 = full_fwd_bwd()
        torch.gcu.synchronize()
        result["path_full_autograd"] = {"attempt": dict(ATTEMPT), "done": dict(DONE)}
        del o2, ht2

    if a.save:
        payload = {n: t.detach().float().cpu() for n, t in grads.items()}
        payload.update({n: t.detach().float().cpu() for n, t in stage_vals.items()})
        torch.save(payload, a.save)
        result["saved"] = a.save

    if a.compare:
        cur = {n: t.detach().float().cpu() for n, t in grads.items()}
        cur.update({n: t.detach().float().cpu() for n, t in stage_vals.items()})
        ref = torch.load(a.compare, map_location="cpu")
        rows = []
        for n in ["o", "ht", "dq", "dk", "dv", "dg", "dh0", "h", "ht_state",
                  "dh", "dh0_state"]:
            x, y = cur[n], ref[n]
            diff = (x.float() - y.float()).abs()
            max_abs = diff.max().item()
            rel = ((x.float() - y.float()).square().mean().sqrt()
                   / x.float().square().mean().sqrt().clamp_min(1e-8)).item()
            bad = bool(not torch.isfinite(x).all() or not torch.isfinite(y).all())
            ok = (not bad) and (max_abs <= 1e-6 or rel < 4e-3)
            strict = (not bad) and (max_abs <= 1e-5 or rel < 1e-5)
            rows.append(dict(name=n, max_abs=max_abs, rel_rms=rel,
                             nan_inf=bad, ok=ok, strict_ok=strict,
                             ref_shape=list(y.shape), cur_shape=list(x.shape)))
        result["compare"] = rows
        result["ALL_PASS"] = all(r["ok"] for r in rows)
        result["ALL_STRICT"] = all(r["strict_ok"] for r in rows)
except Exception as exc:
    import traceback
    result["exit"] = 1
    result["error"] = f"{type(exc).__name__}: {exc}"
    result["traceback"] = traceback.format_exc()[-6000:]
    result["path_attempt_at_error"] = dict(ATTEMPT)
    result["path_done_at_error"] = dict(DONE)

result["wall_s"] = time.perf_counter() - t_start
print("GLA006_JSON " + json.dumps(result))
sys.exit(result["exit"])
