"""GLA-006 independent FP64 CPU recurrence oracle for small shapes.

The GCU result comes from the public chunk_gla in whichever tree PYTHONPATH
selects.  The reference is a plain step-by-step FP64 recurrence on CPU, so it
is independent of the Enflame kernels and of the old fallback implementation.
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--B", type=int, default=1)
p.add_argument("--T", type=int, default=128)
p.add_argument("--H", type=int, default=1)
p.add_argument("--shapes", default="32x64,64x32,64x64")
p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
p.add_argument("--gln", type=float, default=1.0)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--exp-fp16", type=int, default=0)
p.add_argument("--exp-fp32", type=int, default=0)
p.add_argument("--force-fallback", action="store_true")
p.add_argument("--f32-default", default="ieee", choices=["ieee", "tf32", "unset"])
a = p.parse_args()

os.environ["FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP16_STATE"] = str(a.exp_fp16)
os.environ["FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP32_STATE"] = str(a.exp_fp32)
if a.f32_default != "unset":
    os.environ["TRITON_F32_DEFAULT"] = a.f32_default
if a.force_fallback:
    os.environ["FLAGGEMS_ENFLAME_GLA_PARALLEL_STATE"] = "0"
    os.environ["FLAGGEMS_ENFLAME_GLA_PARALLEL_BWD_STATE"] = "0"

import triton  # noqa: E402
from flag_attn import chunk_gla  # noqa: E402
import importlib  # noqa: E402

Hmod = importlib.import_module("flag_attn.runtime.backend._enflame.FLA.chunk_h")
dev = torch.device("gcu:0")
DT = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]

ATTEMPT, DONE = {}, {}
class _Probe:
    def __init__(self, name, real):
        self.name, self.real = name, real
    def __getitem__(self, grid):
        launcher = self.real[grid]
        def call(*args, **kwargs):
            ATTEMPT[self.name] = ATTEMPT.get(self.name, 0) + 1
            out = launcher(*args, **kwargs)
            DONE[self.name] = DONE.get(self.name, 0) + 1
            return out
        return call
for _n in ("_chunk_fwd_local_state_kernel", "_chunk_bwd_local_state_kernel",
           "_chunk_fwd_state_scan_kernel", "_chunk_bwd_state_scan_fp32_2d_group_kernel",
           "chunk_fwd_kernel_h", "chunk_bwd_kernel_dh"):
    if hasattr(Hmod, _n):
        setattr(Hmod, _n, _Probe(_n, getattr(Hmod, _n)))

rows = []
for spec in a.shapes.split(","):
    K, V = (int(x) for x in spec.split("x"))
    B, T, H = a.B, a.T, a.H
    torch.manual_seed(a.seed)
    q = torch.rand(B, T, H, K, dtype=DT, device=dev).requires_grad_()
    k = torch.rand(B, T, H, K, dtype=DT, device=dev).requires_grad_()
    v = torch.rand(B, T, H, V, dtype=DT, device=dev).requires_grad_()
    g = (F.logsigmoid(torch.rand(B, T, H, K, dtype=DT, device=dev)) / a.gln).requires_grad_()
    h0 = torch.rand(B, H, K, V, dtype=torch.float32, device=dev).requires_grad_()
    do = torch.randn(B, T, H, V, dtype=DT, device=dev)
    dht = torch.randn(B, H, K, V, dtype=torch.float32, device=dev)
    scale = K ** -0.5

    ATTEMPT.clear(); DONE.clear()
    o, ht = chunk_gla(q=q, k=k, v=v, g=g, scale=scale, initial_state=h0,
                      output_final_state=True, cu_seqlens=None)
    ((o * do).sum() + (ht.to(o.dtype) * dht.to(o.dtype)).sum()).backward()
    torch.gcu.synchronize()
    got = dict(o=o.detach().float().cpu(), ht=ht.detach().float().cpu(),
               dq=q.grad.float().cpu(), dk=k.grad.float().cpu(),
               dv=v.grad.float().cpu(), dg=g.grad.float().cpu(),
               dh0=h0.grad.float().cpu())

    qr = q.detach().double().cpu().requires_grad_()
    kr = k.detach().double().cpu().requires_grad_()
    vr = v.detach().double().cpu().requires_grad_()
    gr = g.detach().double().cpu().requires_grad_()
    hr = h0.detach().double().cpu().requires_grad_()
    state = hr.clone()
    outs = []
    for i in range(T):
        state = state * gr[:, i].exp()[..., None]
        state = state + kr[:, i].unsqueeze(-1) * vr[:, i].unsqueeze(-2)
        outs.append(torch.einsum("bhk,bhkv->bhv", qr[:, i] * scale, state))
    o_ref = torch.stack(outs, dim=1)
    ht_ref = state
    ((o_ref * do.double().cpu()).sum() + (ht_ref * dht.double().cpu()).sum()).backward()
    ref = dict(o=o_ref.detach(), ht=ht_ref.detach(), dq=qr.grad, dk=kr.grad,
               dv=vr.grad, dg=gr.grad, dh0=hr.grad)

    row = {"shape": [K, V], "dtype": a.dtype, "B": B, "T": T, "H": H,
           "gln": a.gln, "seed": a.seed, "f32_default": os.environ.get("TRITON_F32_DEFAULT", "<unset>"),
           "force_fallback": bool(a.force_fallback),
           "paths": {"attempt": dict(ATTEMPT), "done": dict(DONE)}, "tensors": {}}
    for n in ["o", "ht", "dq", "dk", "dv", "dg", "dh0"]:
        x = got[n]; y = ref[n].float()
        max_abs = (x - y).abs().max().item()
        rel = ((x - y).square().mean().sqrt()
               / x.square().mean().sqrt().clamp_min(1e-8)).item()
        bad = bool(not torch.isfinite(x).all())
        row["tensors"][n] = {"max_abs": max_abs, "rel_rms": rel, "nan_inf": bad,
                             "ok": (not bad) and (max_abs <= 1e-3 or rel < 5e-3)}
    row["ALL_PASS"] = all(x["ok"] for x in row["tensors"].values())
    rows.append(row)
    print(spec, json.dumps({n: [round(t["max_abs"], 8), round(t["rel_rms"], 8)]
                            for n, t in row["tensors"].items()}),
          "PASS", row["ALL_PASS"], row["paths"]["done"], flush=True)

print("GLA006_ORACLE " + json.dumps(rows))
