"""PARALLAX-003: static routing divergence probe (no kernels launched).

Prints which backward branch would be taken for FP16 vs BF16 at the target
shape, and the autotune space sizes, on GCU 0.
"""

import json

import torch
import torch_gcu  # noqa: F401

from flag_attn.runtime.backend._enflame.FLA.parallax import parallel as p
from flag_attn.runtime.backend._enflame.FLA.utils import get_num_sms


torch.gcu.set_device(0)
B, T, H, HQ, D = 2, 256, 2, 8, 64
G = HQ // H
BT = p._BLOCK_T
NQ = T // BT
num_sms = get_num_sms(0)
print(json.dumps({"device": torch.gcu.current_device(), "num_sms": num_sms, "BT": BT, "NQ": NQ,
                  "G": G, "D": D, "T": T}))

for name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
    use_dense_aligned = (
        True and dtype in (torch.bfloat16, torch.float16) and
        ((D == 128 and NQ > 16) or (dtype == torch.bfloat16 and D == 64 and T == 256 and G == 4))
        and T % BT == 0 and T <= p._DENSE_ALIGNED_MAX_T
    )
    use_split_dqr = D >= 128
    dkv_programs = NQ * B * H
    use_grouped_dkv = G in (4, 8) and NQ > p._SHORT_MULTI_MAX_TILES and num_sms > 0 and dkv_programs >= num_sms
    if use_dense_aligned:
        dqr = "parallel_parallax_bwd_kernel_dqr_dense_aligned" if use_split_dqr else "parallel_parallax_bwd_kernel_dqr_fused"
    else:
        dqr = "parallel_parallax_bwd_kernel_dqr" if use_split_dqr else "parallel_parallax_bwd_kernel_dqr_fused"
    # Mirrors the real dispatch order: dense-aligned DKV is shadowed by grouped
    # DKV whenever grouped is eligible (T=256, G=4, NQ=4 > 2, 2 SMs).
    dkv = ("parallel_parallax_bwd_kernel_dkv_dense_aligned" if (use_dense_aligned and not use_grouped_dkv)
           else "parallel_parallax_bwd_kernel_dkv_grouped" if use_grouped_dkv
           else "parallel_parallax_bwd_kernel_dkv")
    dqr_configs = p.PARALLAX_DQR_SPLIT_CONFIGS if use_split_dqr else p.PARALLAX_DQR_CONFIGS
    print(json.dumps({
        "dtype": name, "use_dense_aligned": use_dense_aligned, "use_split_dqr": use_split_dqr,
        "use_grouped_dkv": use_grouped_dkv, "dqr_kernel": dqr, "dkv_kernel": dkv,
        "dqr_autotune_configs": len(dqr_configs),
        "dqr_effective_configs_for_D64": len([c for c in dqr_configs if c.kwargs.get("BK") == 64]),
        "dkv_autotune_configs": len(p.PARALLAX_DKV_CONFIGS),
        "dkv_effective_configs_for_D64": len([c for c in p.PARALLAX_DKV_CONFIGS if c.kwargs.get("BK") == 64]),
        "note": "fused DQR, grouped DKV, and FWD are autotuned",
    }))
