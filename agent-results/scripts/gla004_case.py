"""GLA-004 single-case harness: path probe, correctness dump, stage timing.

Run once per (worktree, shape, T, mode).  Worktree is chosen via PYTHONPATH.
Timing is wall time around explicit torch.gcu.synchronize(), warmup excluded;
cold compile is recorded separately and never used for speedups.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--B", type=int, default=1)
p.add_argument("--T", type=int, default=256)
p.add_argument("--H", type=int, default=4)
p.add_argument("--K", type=int, default=64)
p.add_argument("--V", type=int, default=64)
p.add_argument("--gln", type=float, default=1.0)
p.add_argument("--tag", default="run")
p.add_argument("--save", default=None, help="save o/ht/grads/stages here")
p.add_argument("--compare", default=None, help="baseline .pt to compare against")
p.add_argument("--samples", type=int, default=5)
p.add_argument("--warmup", type=int, default=2)
p.add_argument("--no-timing", action="store_true")
a = p.parse_args()

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
from flag_attn import chunk_gla
import importlib

C = importlib.import_module("flag_attn.runtime.backend._enflame.FLA.gla.chunk_gla")
Hmod = importlib.import_module("flag_attn.runtime.backend._enflame.FLA.chunk_h")

dev = torch.device("gcu:0")
torch.manual_seed(42)
B, T, H, K, V = a.B, a.T, a.H, a.K, a.V
dt = torch.float16


def make_inputs():
    torch.manual_seed(42)
    q = torch.rand(B, T, H, K, dtype=dt, device=dev).requires_grad_()
    k = torch.rand(B, T, H, K, dtype=dt, device=dev).requires_grad_()
    v = torch.rand(B, T, H, V, dtype=dt, device=dev).requires_grad_()
    g = (F.logsigmoid(torch.rand(B, T, H, K, dtype=dt, device=dev)) / a.gln).requires_grad_()
    h0 = torch.rand(B, H, K, V, dtype=torch.float32, device=dev).requires_grad_()
    do = torch.randn_like(v)
    dht = torch.randn(B, H, K, V, dtype=torch.float32, device=dev)
    return q, k, v, g, h0, do, dht


# ---- path probe: count launches of parallel vs fallback state kernels ----
COUNTS = {}


class _Probe:
    def __init__(self, name, real):
        self.name = name
        self.real = real

    def __getitem__(self, grid):
        launcher = self.real[grid]

        def call(*args, **kwargs):
            COUNTS[self.name] = COUNTS.get(self.name, 0) + 1
            return launcher(*args, **kwargs)

        return call


for _n in ("_chunk_fwd_local_state_kernel", "_chunk_bwd_local_state_kernel",
           "_chunk_fwd_state_scan_kernel", "_chunk_bwd_state_scan_kernel",
           "chunk_fwd_kernel_h", "chunk_bwd_kernel_dh",
           "_chunk_fwd_state_scan_fp32_2d_group_kernel",
           "_chunk_bwd_state_scan_fp32_2d_group_kernel"):
    if hasattr(Hmod, _n):
        setattr(Hmod, _n, _Probe(_n, getattr(Hmod, _n)))

result = {
    "tag": a.tag,
    "shape": [K, V],
    "B": B, "T": T, "H": H, "dtype": "fp16", "gln": a.gln,
    "pythonpath_head": os.environ.get("PYTHONPATH", "").split(":")[0],
    "chunk_h_file": Hmod.__file__,
    "exit": 0,
}
t_start = time.perf_counter()

try:
    q, k, v, g, h0, do, dht = make_inputs()
    scale = K ** -0.5

    def full_fwd_bwd():
        q.grad = None; k.grad = None; v.grad = None; g.grad = None; h0.grad = None
        o, ht = chunk_gla(q=q, k=k, v=v, g=g, scale=scale, initial_state=h0,
                          output_final_state=True, cu_seqlens=None)
        ((o * do).sum() + (ht * dht).sum().to(o.dtype)).backward()
        return o, ht

    t0 = time.perf_counter()
    o, ht = full_fwd_bwd()
    torch.gcu.synchronize()
    result["cold_full_fwd_bwd_ms"] = (time.perf_counter() - t0) * 1000
    grads = dict(o=o, ht=ht, dq=q.grad, dk=k.grad, dv=v.grad, dg=g.grad, dh0=h0.grad)

    with torch.no_grad():
        qd, kd, vd = q.detach(), k.detach(), v.detach()
        gd, h0d = g.detach(), h0.detach()
        qd.grad = torch.zeros_like(qd); kd.grad = torch.zeros_like(kd)
        vd.grad = torch.zeros_like(vd); gd.grad = torch.zeros_like(gd)
        h0d.grad = torch.zeros_like(h0d)

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

        COUNTS.clear()
        h_st, ht_st = fwd_h()
        torch.gcu.synchronize()
        fwd_path = dict(COUNTS)
        COUNTS.clear()
        dh_st, dh0_st = bwd_dh_fp32()
        torch.gcu.synchronize()
        bwd_path = dict(COUNTS)
        COUNTS.clear()
        whole_fwd()
        torch.gcu.synchronize()
        fwd_all_path = dict(COUNTS)
        result["path_fwd_h"] = fwd_path
        result["path_bwd_dh"] = bwd_path
        result["path_whole_fwd"] = fwd_all_path
        result["parallel_fwd_h"] = fwd_path.get("_chunk_fwd_local_state_kernel", 0) > 0
        result["parallel_bwd_dh"] = bwd_path.get("_chunk_bwd_local_state_kernel", 0) > 0
        result["fallback_fwd_h"] = fwd_path.get("chunk_fwd_kernel_h", 0)
        result["fallback_bwd_dh"] = bwd_path.get("chunk_bwd_kernel_dh", 0)

        stage_vals = dict(h=h_st, ht_state=ht_st, dh=dh_st, dh0_state=dh0_st)

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
                ts_sorted = sorted(ts)
                timing[name] = {"ms": ts, "median": ts_sorted[len(ts_sorted) // 2],
                                "min": ts_sorted[0]}
            result["timing"] = timing
            result["compile_counts"] = dict(COUNTS)

    if not a.no_timing:
        # end-to-end needs the autograd graph, so time it outside no_grad
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
        ts_sorted = sorted(ts)
        result.setdefault("timing", {})["full_fwd_bwd"] = {
            "ms": ts, "median": ts_sorted[len(ts_sorted) // 2], "min": ts_sorted[0]}

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
            diff = (x - y).abs()
            max_abs = diff.max().item()
            rel = ((x - y).square().mean().sqrt()
                   / x.square().mean().sqrt().clamp_min(1e-8)).item()
            bad = bool(torch.isnan(x).any() or torch.isinf(x).any()
                       or torch.isnan(y).any() or torch.isinf(y).any())
            ok = (not bad) and (max_abs <= 1e-6 or rel < 4e-3)
            rows.append(dict(name=n, max_abs=max_abs, rel_rms=rel,
                             nan_inf=bad, ok=ok))
        result["compare"] = rows
        result["ALL_PASS"] = all(r["ok"] for r in rows)
except Exception as exc:  # preserve failures rather than hiding them
    import traceback
    result["exit"] = 1
    result["error"] = f"{type(exc).__name__}: {exc}"
    result["traceback"] = traceback.format_exc()[-4000:]
    if a.save:
        try:
            torch.save({"error": result["error"]}, a.save)
        except Exception:
            pass

result["wall_s"] = time.perf_counter() - t_start
print("GLA004_JSON " + json.dumps(result))
sys.exit(result["exit"])
