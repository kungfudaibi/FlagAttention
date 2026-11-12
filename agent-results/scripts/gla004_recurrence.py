"""Independent PyTorch recurrence cross-check for GLA-004 shapes (small T)."""
import argparse
import json
import os

import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--T", type=int, default=128)
p.add_argument("--H", type=int, default=2)
p.add_argument("--shapes", default="32x32,64x64,64x128,128x64,128x128")
p.add_argument("--gln", type=float, default=1.0)
a = p.parse_args()

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
from flag_attn import chunk_gla

dev = torch.device("gcu:0")
rows = []
for spec in a.shapes.split(","):
    K, V = (int(x) for x in spec.split("x"))
    torch.manual_seed(7)
    B, T, H = 1, a.T, a.H
    dt = torch.float16
    q = torch.rand(B, T, H, K, dtype=dt, device=dev).requires_grad_()
    k = torch.rand(B, T, H, K, dtype=dt, device=dev).requires_grad_()
    v = torch.rand(B, T, H, V, dtype=dt, device=dev).requires_grad_()
    g = (F.logsigmoid(torch.rand(B, T, H, K, dtype=dt, device=dev)) / a.gln).requires_grad_()
    h0 = torch.rand(B, H, K, V, dtype=torch.float32, device=dev).requires_grad_()
    do = torch.randn_like(v)
    dht = torch.randn(B, H, K, V, dtype=torch.float32, device=dev)
    scale = K ** -0.5

    o, ht = chunk_gla(q=q, k=k, v=v, g=g, scale=scale, initial_state=h0,
                      output_final_state=True, cu_seqlens=None)
    ((o * do).sum() + (ht * dht).sum().to(o.dtype)).backward()
    got = dict(o=o.detach().float(), ht=ht.detach().float(), dq=q.grad.float(),
               dk=k.grad.float(), dv=v.grad.float(), dg=g.grad.float(),
               dh0=h0.grad.float())

    # fp64 recurrence with analytic gradients via autograd
    qr = q.detach().double().requires_grad_()
    kr = k.detach().double().requires_grad_()
    vr = v.detach().double().requires_grad_()
    gr = g.detach().double().requires_grad_()
    hr = h0.detach().double().requires_grad_()
    state = hr.clone()
    outs = []
    for i in range(T):
        state = state * gr[:, i].exp()[..., None]
        state = state + kr[:, i].unsqueeze(-1) * vr[:, i].unsqueeze(-2)
        outs.append(torch.einsum("bhk,bhkv->bhv", qr[:, i] * scale, state))
    o_ref = torch.stack(outs, dim=1)
    ht_ref = state
    ((o_ref * do.double()).sum() + (ht_ref * dht.double()).sum()).backward()
    ref = dict(o=o_ref.detach(), ht=ht_ref.detach(), dq=qr.grad, dk=kr.grad,
               dv=vr.grad, dg=gr.grad, dh0=hr.grad)

    row = {"shape": [K, V], "T": T, "H": H, "gln": a.gln, "tensors": {}}
    for n in ["o", "ht", "dq", "dk", "dv", "dg", "dh0"]:
        x = got[n]; y = ref[n].float()
        max_abs = (x - y).abs().max().item()
        rel = ((x - y).square().mean().sqrt()
               / x.square().mean().sqrt().clamp_min(1e-8)).item()
        bad = bool(torch.isnan(x).any() or torch.isinf(x).any())
        row["tensors"][n] = {"max_abs": max_abs, "rel_rms": rel, "nan_inf": bad,
                             "ok": (not bad) and (max_abs <= 1e-6 or rel < 5e-3)}
    row["ALL_PASS"] = all(v_["ok"] for v_ in row["tensors"].values())
    rows.append(row)
    print(spec, json.dumps({n: [round(t["max_abs"], 6), round(t["rel_rms"], 6)] for n, t in row["tensors"].items()}),
          "PASS", row["ALL_PASS"], flush=True)

print("GLA004_REC " + json.dumps(rows))
