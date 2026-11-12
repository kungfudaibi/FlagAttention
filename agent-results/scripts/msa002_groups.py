"""MSA-002 phase 3 experiment: prefill grouped-page union group size.

The shipped Enflame prefill kernel `_gqa_sparse_fwd_grouped8` merges the pages
selected by GROUP_SIZE_Q adjacent queries and runs attention over the union.
For Q=512, S=4096, topk=16 the shipped GROUP_SIZE_Q is 16 (because
block_table.shape[1] == 32 <= 2*topk and parallelism test passes). A group of
16 queries has a mean union of 30.5 distinct pages vs 16 selected per query,
i.e. ~1.91x redundant KV work. Smaller groups reduce the union waste at the
cost of more CTAs.

This probe launches the real kernel with GROUP_SIZE_Q in {1,2,4,8,16,32},
verifies the output is bit-identical to the shipped launch, and times each.
No source change yet; this is evidence for/against the hypothesis.
"""

import json
import statistics
import sys

import torch
import torch_gcu  # noqa: F401

ROOT = "/workspace/agent-workspaces/msa-stage-profile"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/src")
torch.gcu.set_device(1)

import triton  # noqa: E402

from benchmark import test_minimax_sparse_attention_benchmark as bench  # noqa: E402
from flag_attn.minimax_sparse_attention import (  # noqa: E402
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
from flag_attn.runtime.backend._enflame.msa.sparse_attn import (  # noqa: E402
    _gqa_sparse_fwd_grouped8,
    _gqa_sparse_fwd_direct,
)
from flag_attn.minimax_sparse_attention import SPARSE_BLOCK_SIZE  # noqa: E402

for _name in ("randn", "rand", "randperm", "randint"):
    _original = getattr(torch, _name)

    def _make(orig):
        def w(*a, **kw):
            gen = kw.get("generator")
            dev = kw.get("device")
            if gen is not None and dev is not None and str(gen.device) == "cpu" and str(dev).startswith("gcu"):
                kw["device"] = "cpu"
                return orig(*a, **kw).to(dev)
            return orig(*a, **kw)
        return w

    setattr(torch, _name, _make(_original))


def timed(fn, samples=15):
    fn(); torch.gcu.synchronize()
    vals = []
    for _ in range(samples):
        a = torch.gcu.Event(enable_timing=True)
        b = torch.gcu.Event(enable_timing=True)
        torch.gcu.synchronize()
        a.record(); fn(); b.record(); torch.gcu.synchronize()
        vals.append(a.elapsed_time(b))
    vals.sort()
    return statistics.median(vals)


def grouped_launch(data, topk_idx, out, qlen, group_q):
    grid = (triton.cdiv(qlen, group_q), 1, 1)
    _gqa_sparse_fwd_grouped8[grid](
        data.q, data.kv_cache, topk_idx, out, data.block_table,
        data.cu_q, data.seq_lens, data.prefix_lens,
        0, 8, 128, data.sm_scale,
        data.q.stride(0), data.q.stride(1), data.q.stride(2),
        data.kv_cache.stride(0), data.kv_cache.stride(1),
        data.kv_cache.stride(2), data.kv_cache.stride(3),
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        data.block_table.stride(0),
        GROUP_SIZE_Q=group_q,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_D=128,
        BLOCK_SIZE_H=8,
        MAX_TOPK=topk_idx.shape[-1],
        BLOCK_TOPK=triton.next_power_of_2(topk_idx.shape[-1]),
        DECODE_MODE=False,
        num_warps=1,
        num_stages=1,
    )


for qlen, seq_len, topk in ((512, 4096, 16), (1024, 4096, 16), (512, 16384, 16), (512, 4096, 32)):
    g = torch.Generator(device="cpu").manual_seed(0)
    data = bench.make_data(1, seq_len, 1, 8, torch.device("gcu:1"), "bf16",
                           decode=False, decode_qlen=1, query_len=qlen,
                           randomize_pages=True, generator=g)
    scores = minimax_m3_index_score(
        data.idx_q, data.index_kv_cache, data.block_table, data.cu_q,
        data.seq_lens, data.prefix_lens, qlen, seq_len, 1)
    topk_idx = minimax_m3_index_topk(scores, data.cu_q, data.prefix_lens, qlen, topk, 1, 2)
    ref = torch.empty_like(data.q)
    shipped = 16 if (topk <= 16 and (seq_len // 128) <= 2 * topk) else 4
    grouped_launch(data, topk_idx, ref, qlen, shipped)
    torch.gcu.synchronize()
    base = ref.clone()

    for gq in (1, 2, 4, 8, 16, 32):
        out = torch.empty_like(data.q)
        grouped_launch(data, topk_idx, out, qlen, gq)
        torch.gcu.synchronize()
        same = torch.equal(out, base)
        max_abs = float((out.float() - base.float()).abs().max().item())
        ms = timed(lambda out=out, gq=gq: grouped_launch(data, topk_idx, out, qlen, gq))
        print(json.dumps({"qlen": qlen, "seq_len": seq_len, "topk": topk,
                          "shipped_group": shipped, "group_q": gq,
                          "ms": ms, "bit_exact": bool(same), "max_abs": max_abs}), flush=True)
    del data, scores, topk_idx, ref, base
    torch.gcu.empty_cache()
print("DONE", flush=True)
