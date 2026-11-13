"""Exploratory probe: is the ``total_q >= 512`` floor load-bearing?

Run with the floor temporarily relaxed to ``total_q >= 1`` in
``sparse_attn.py`` (reverted immediately after). Not acceptance evidence: it
only decides whether the shipped floor hides a win at multi-KV-head shapes
where ``cdiv(total_q,16)*num_kv_heads`` already reaches 32 below total_q=512.
"""

import json
import statistics

import torch
import torch_gcu  # noqa: F401
from triton.runtime.jit import JITFunction

from flag_attn import minimax_m3_sparse_attn
from flag_attn.minimax_sparse_attention import (
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
import flag_attn.runtime.backend._enflame.msa.sparse_attn as es
from tests.flag_attn import test_minimax_sparse_attention as refmod

torch.gcu.set_device(2)

LAUNCH = []
_ORIG = JITFunction.run


def _run(self, *a, **kw):
    if kw.get("warmup") is False:
        LAUNCH.append((self.__name__, kw.get("GROUP_SIZE_Q")))
    return _ORIG(self, *a, **kw)


JITFunction.run = _run

CASES = [
    # kv=1, total_q in the 497..511 sliver the 512 floor currently excludes.
    ((4096,), (3600,), 1, 8, 16, "kv1_q496"),
    ((4096,), (3596,), 1, 8, 16, "kv1_q500"),
    ((4096,), (3585,), 1, 8, 16, "kv1_q511"),
    # kv=2: programs>=32 already at total_q=256, but the 512 floor blocks it.
    ((4096,), (3840,), 2, 8, 16, "kv2_q256"),
    ((4096,), (3712,), 2, 8, 16, "kv2_q384"),
    ((4096, 4096), (3904, 3904), 2, 8, 16, "kv2_batch2_q192x2"),
    # kv=4: programs>=32 at total_q=128.
    ((4096,), (3968,), 4, 2, 16, "kv4_q128"),
    ((4096,), (3840,), 4, 2, 16, "kv4_q256"),
    # kv=8: programs>=32 at total_q=64.
    ((4096,), (4032,), 8, 1, 16, "kv8_q64"),
]


def call(data, topk, out):
    qlen = max(data.query_lens)
    kv = data.kv_cache.shape[1]
    scores = minimax_m3_index_score(
        data.idx_q, data.index_kv_cache, data.block_table, data.cu_q,
        data.seq_lens, data.prefix_lens, qlen, data.max_seq_len, kv,
    )
    idx = minimax_m3_index_topk(scores, data.cu_q, data.prefix_lens, qlen, topk, 1, 2)
    minimax_m3_sparse_attn(
        data.q, data.kv_cache, idx, data.block_table, data.cu_q, data.seq_lens,
        data.prefix_lens, qlen, kv, data.sm_scale, out,
    )
    return scores, idx


def measure(data, topk, enabled, n=15):
    es._MSA002_GROUP16_EXPERIMENT = enabled
    out = torch.empty_like(data.q)
    for _ in range(3):
        call(data, topk, out)
    torch.gcu.synchronize()
    LAUNCH.clear()
    call(data, topk, out)
    torch.gcu.synchronize()
    group = [g for name, g in LAUNCH if "sparse_fwd" in name]
    vals = []
    for _ in range(n):
        s = torch.gcu.Event(enable_timing=True)
        e = torch.gcu.Event(enable_timing=True)
        torch.gcu.synchronize()
        s.record()
        call(data, topk, out)
        e.record()
        torch.gcu.synchronize()
        vals.append(s.elapsed_time(e))
    return statistics.median(vals), out, group


for seqs, prefixes, kv, group, topk, cid in CASES:
    data = refmod.make_data(seqs, kv, group, decode=False, prefix_lens=prefixes, mode="bf16")
    ref_scores = refmod._ref_index_score(data)
    ref_topk = refmod._ref_topk(ref_scores, data, topk, 1, 2)
    ref_out = refmod._ref_sparse_attn(data, ref_topk)
    a_ms, a_out, a_group = measure(data, topk, False)
    b_ms, b_out, b_group = measure(data, topk, True)
    record = {
        "id": cid,
        "total_q": int(data.q.shape[0]),
        "batch": len(seqs),
        "kv": kv,
        "programs": (-(-int(data.q.shape[0]) // 16)) * kv,
        "A_ms": a_ms,
        "B_ms": b_ms,
        "speedup": a_ms / b_ms,
        "A_group": a_group[0] if a_group else None,
        "B_group": b_group[0] if b_group else None,
        "refA": bool(torch.allclose(a_out, ref_out, atol=2e-2, rtol=2e-2)),
        "refB": bool(torch.allclose(b_out, ref_out, atol=2e-2, rtol=2e-2)),
        "AB_max_abs": float((a_out.float() - b_out.float()).abs().max()),
    }
    print(json.dumps(record), flush=True)
    torch.gcu.empty_cache()
