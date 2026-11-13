"""Ad-hoc diagnostic: is the BF16 final-state error consistent with BF16 ULP?

Not a correctness test; not committed as evidence of pass/fail.
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
        ref = eht.float()
        eps = torch.finfo(dt).eps
        ulp = torch.where(ref.abs() > 0, ref.abs() * eps, torch.zeros_like(ref))
        err = (ref - ht.cpu().float()).abs()
        ratio = err / ulp.clamp_min(torch.finfo(dt).tiny)
        k999 = max(1, int(0.999 * ratio.numel()))
        print(
            f"{name} B{B}T{T}H{H}K{K}V{V} max|ht|={ref.abs().max():.4f} "
            f"ulp_max={ulp.max():.3e} err_max={err.max():.3e} "
            f"err/ulp max={ratio.max():.3f} p99.9={ratio.flatten().kthvalue(k999)[0]:.3f} "
            f"frac_gt_1ulp={(ratio > 1).float().mean():.4f}"
        )
