#!/usr/bin/env python3
"""Record what happens when one process touches both GCUs.

SAGE-002 is restricted to GCU 1.  The existing Sage test file, however, builds
tensors on the repository default device (GCU 0) unless it is overridden, so a
combined pytest session touches both.  On this image that aborts inside
``topsMemcpyAsync``; this probe reproduces it in isolation so the guard in
``tests/flag_attn/test_sage_bottleneck.py`` is backed by evidence rather than
a guess.

Usage: sage002_two_device_probe.py {single,first_zero,first_one}
"""

import sys

import torch

import torch_gcu  # noqa: F401

from flag_attn import sage_attention
from flag_attn.sage_attention import per_block_int8


def run(index: int) -> None:
    torch.gcu.set_device(index)
    device = f"gcu:{index}"
    q = torch.randn(1, 2, 256, 64, device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    q8, q_scale, k8, k_scale = per_block_int8(q, k)
    out, _ = sage_attention(q8, k8, v, q_scale, k_scale)
    torch.gcu.synchronize()
    print(f"DEV {device} sage_ok value={float(out[0, 0, 0, 0]):.6f}", flush=True)


mode = sys.argv[1] if len(sys.argv) > 1 else "single"
if mode == "first_zero":
    run(0)
    run(1)
elif mode == "first_one":
    run(1)
    run(0)
else:
    run(1)
print(f"DEV done mode={mode}", flush=True)
