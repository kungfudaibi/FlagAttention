#!/usr/bin/env python3
"""Record which input/output dtype combinations the Enflame Sage kernel takes.

The kernel converts the probability tile and V to FP16 before the second dot,
so FP16 Q/K/V is the supported input dtype.  The output dtype is independent:
FP16 input with BF16 output is accepted and accurate.  BF16 input fails at
Triton compile time, which is recorded here instead of being inferred.
"""

import torch
import torch.nn.functional as F

import torch_gcu  # noqa: F401

from flag_attn import sage_attention
from flag_attn.sage_attention import per_block_int8


torch.gcu.set_device(1)
torch.manual_seed(3)
for in_dtype, out_dtype in (
    (torch.float16, torch.float16),
    (torch.float16, torch.bfloat16),
    (torch.bfloat16, torch.bfloat16),
    (torch.bfloat16, torch.float16),
):
    q = torch.randn(1, 2, 1024, 64, device="gcu:1", dtype=in_dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    try:
        q8, qs, k8, ks = per_block_int8(q, k)
        out, _ = sage_attention(q8, k8, v, qs, ks, output_dtype=out_dtype)
        torch.gcu.synchronize()
        ref = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        ).to(out_dtype)
        a = out.float().flatten()
        b = ref.float().flatten()
        print(
            f"DT in={in_dtype} out={out_dtype} result=PASS out_dtype={out.dtype} "
            f"cosine={F.cosine_similarity(a, b, dim=0).item():.6f} "
            f"relative_l2={((a - b).norm() / b.norm()).item():.5f}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"DT in={in_dtype} out={out_dtype} result=FAIL "
            f"{type(exc).__name__}: {str(exc).splitlines()[-1]}",
            flush=True,
        )
