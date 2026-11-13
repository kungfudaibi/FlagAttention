"""Ad-hoc diagnostic: numerical floor of the GDN2 recurrence in reduced precision.

Computes the SAME recurrence three ways on identical inputs:
  * FP64 (reference)
  * FP32 accumulation with dtype-quantized inputs (mimics a device that keeps
    state in FP32 but reads FP16/BF16 operands)
Then compares the real device kernel against FP64 and against the FP32 floor.

If device-vs-FP64 error is comparable to floor-vs-FP64 error, the residual is
input/accumulation quantization, not a kernel defect.
"""
import math
import sys
from pathlib import Path

import torch

ROOT = Path("/workspace/agent-workspaces/gdn2-backward-reference")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests/flag_attn"))

from flag_attn import chunk_gdn2  # noqa: E402
from gdn2_backward_ref import gdn2_forward  # noqa: E402

torch.set_grad_enabled(False)


def gdn2_forward_fp32(q, k, v, g, b, w, scale, h0):
    """Same equations as the FP64 reference but with FP32 accumulation."""
    q, k, v, g, b, w = (x.float() for x in (q, k, v, g, b, w))
    h = h0.float().clone()
    outs = []
    for t in range(q.shape[1]):
        kt, bt, gt = k[:, t], b[:, t], g[:, t]
        vt, wt = v[:, t], w[:, t]
        hd = h * torch.exp(gt)[..., :, None]
        e = torch.einsum("bhk,bhkv->bhv", bt * kt, hd)
        h = hd + torch.einsum("bhk,bhv->bhkv", kt, wt * vt - e)
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], h) * scale)
    return torch.stack(outs, dim=1), h


def rel_stats(ref, act):
    e = ref.double().flatten()
    a = act.double().flatten()
    d = (e - a).abs()
    return d.max().item(), (d.square().mean().sqrt() / (e.square().mean().sqrt() + 1e-12)).item()


for (B, T, H, K, V) in [(2, 130, 3, 64, 64), (2, 65, 2, 128, 128), (1, 65, 1, 64, 64)]:
    for name, dt in [("bf16", torch.bfloat16), ("fp16", torch.float16)]:
        torch.manual_seed(0)
        scale = K ** -0.5
        dev = "gcu:0"
        q = (torch.randn(B, T, H, K) / math.sqrt(K)).to(dt)
        k = (torch.randn(B, T, H, K) / math.sqrt(K)).to(dt)
        v = torch.randn(B, T, H, V).to(dt)
        b = torch.rand(B, T, H, K).to(dt)
        w = torch.rand(B, T, H, V).to(dt)
        g = (-torch.rand(B, T, H, K) * 0.1).to(dt)
        h0 = torch.randn(B, H, K, V) * 0.01

        o, ht = chunk_gdn2(
            q.to(dev), k.to(dev), v.to(dev), g.to(dev), b.to(dev), w.to(dev),
            scale=scale, initial_state=h0.to(dev), output_final_state=True, chunk_size=64,
        )
        eo, eht = gdn2_forward(q, k, v, g, b, w, scale, h0)
        fo, fht = gdn2_forward_fp32(q, k, v, g, b, w, scale, h0)

        f_omax, f_or = rel_stats(eo, fo)
        f_hmax, f_hr = rel_stats(eht, fht)
        d_omax, d_or = rel_stats(eo, o.cpu())
        d_hmax, d_hr = rel_stats(eht, ht.cpu())
        print(
            f"{name} B{B}T{T}H{H}K{K}V{V} | "
            f"FP32floor o(max={f_omax:.3e} rr={f_or:.3e}) ht(max={f_hmax:.3e} rr={f_hr:.3e}) | "
            f"device   o(max={d_omax:.3e} rr={d_or:.3e}) ht(max={d_hmax:.3e} rr={d_hr:.3e})"
        )
