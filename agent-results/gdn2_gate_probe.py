"""Diagnostic: how do the failing BF16 cases fare under standard elementwise
allclose (atol + rtol*|expected|), versus the legacy max_abs-vs-global-RMS gate?
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
ATOL, RTOL = 1e-3, 2e-2

for (B, T, H, K, V) in [(2, 130, 3, 64, 64), (2, 65, 2, 128, 128), (1, 130, 2, 64, 64)]:
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
        for label, ref, act in (("o", eo, o.cpu()), ("ht", eht, ht.cpu())):
            e = ref.double()
            a = act.double()
            d = (e - a).abs()
            # standard elementwise allclose margin: <=1 means pass
            marg = (d / (ATOL + RTOL * e.abs())).max().item()
            rms = (d.square().mean().sqrt() / (e.square().mean().sqrt() + 1e-12)).item()
            legacy = d.max().item() <= ATOL + RTOL * e.square().mean().sqrt().item()
            at_max = e.flatten()[d.flatten().argmax()].abs().item()
            print(
                f"{name} B{B}T{T}H{H}K{K}V{V} {label:2s} | elem_margin={marg:.3f} "
                f"rms_rel={rms:.3e} max|e|={e.abs().max():.3f} "
                f"rms={e.square().mean().sqrt():.3f} |e|@maxerr={at_max:.4f} "
                f"legacy_abs_gate={'PASS' if legacy else 'FAIL'}"
            )
