"""PARALLAX-004 steady-state A/B: baseline vs fixed dense D64, GCU 1.

Alternating rounds, warm kernels, separate forward/backward synchronized wall
times. Baseline = PARALLAX_FIXED_D64=0, candidate = 1 (read per call).
Run only when GLA-005 is not benchmarking.
"""

import json
import os
import statistics
import time

import torch
import torch_gcu  # noqa: F401

from flag_attn import parallel_parallax

torch.gcu.set_device(1)
assert torch.gcu.current_device() == 1
dtype = torch.bfloat16
B, T, H, HQ, D = 2, 256, 2, 8, 64
torch.manual_seed(42)
q, r = [torch.randn(B, T, HQ, D, dtype=dtype, device="gcu:1").requires_grad_() for _ in range(2)]
k, v = [torch.randn(B, T, H, D, dtype=dtype, device="gcu:1").requires_grad_() for _ in range(2)]
grad_output = torch.randn(B, T, HQ, D, dtype=dtype, device="gcu:1")
inputs = (q, r, k, v)
rounds = int(os.environ.get("P004_ROUNDS", "3"))

# Warm both paths once (compilation is excluded from the timed rounds).
for mode in ("0", "1"):
    os.environ["PARALLAX_FIXED_D64"] = mode
    out = parallel_parallax(*inputs, scale=D**-0.5)
    torch.autograd.grad(out, inputs, grad_output)
torch.gcu.synchronize()

records = []
for rnd in range(rounds):
    for mode in ("0", "1"):
        os.environ["PARALLAX_FIXED_D64"] = mode
        torch.gcu.synchronize()
        t0 = time.perf_counter()
        out = parallel_parallax(*inputs, scale=D**-0.5)
        torch.gcu.synchronize()
        fwd = time.perf_counter() - t0
        t0 = time.perf_counter()
        torch.autograd.grad(out, inputs, grad_output)
        torch.gcu.synchronize()
        bwd = time.perf_counter() - t0
        rec = {"round": rnd, "mode": "candidate" if mode == "1" else "baseline",
               "forward_s": round(fwd, 6), "backward_s": round(bwd, 6)}
        records.append(rec)
        print(json.dumps(rec), flush=True)

for mode in ("baseline", "candidate"):
    sel = [x for x in records if x["mode"] == mode]
    print(json.dumps({
        "summary": mode,
        "rounds": len(sel),
        "forward_median_s": statistics.median(x["forward_s"] for x in sel),
        "backward_median_s": statistics.median(x["backward_s"] for x in sel),
        "forward_all_s": [x["forward_s"] for x in sel],
        "backward_all_s": [x["backward_s"] for x in sel],
    }), flush=True)
