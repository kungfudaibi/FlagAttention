"""MSA-002 scaling study: where prefill attention time goes.

All measurements are stage-only event-pair medians with sync at both ends,
3 rounds x 15 samples, on GCU 1. Not a whole-call number.
"""

import statistics
import sys

import torch
import torch_gcu  # noqa: F401

ROOT = "/workspace/agent-workspaces/msa-stage-profile"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/src")
torch.gcu.set_device(1)

from benchmark import test_minimax_sparse_attention_benchmark as bench  # noqa: E402
from flag_attn import minimax_m3_sparse_attn  # noqa: E402
from flag_attn.minimax_sparse_attention import (  # noqa: E402
    minimax_m3_index_score,
    minimax_m3_index_topk,
)

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
    fn()
    torch.gcu.synchronize()
    vals = []
    for _ in range(samples):
        a = torch.gcu.Event(enable_timing=True)
        b = torch.gcu.Event(enable_timing=True)
        torch.gcu.synchronize()
        a.record()
        fn()
        b.record()
        torch.gcu.synchronize()
        vals.append(a.elapsed_time(b))
    vals.sort()
    return statistics.median(vals)


def prefill_attn(qlen, seq_len, topk, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    data = bench.make_data(1, seq_len, 1, 8, torch.device("gcu:1"), "bf16",
                           decode=False, decode_qlen=1, query_len=qlen,
                           randomize_pages=True, generator=gen)
    scores = minimax_m3_index_score(
        data.idx_q, data.index_kv_cache, data.block_table, data.cu_q,
        data.seq_lens, data.prefix_lens, qlen, seq_len, 1)
    topk_idx = minimax_m3_index_topk(
        scores, data.cu_q, data.prefix_lens, qlen, topk, 1, 2)
    out = torch.empty_like(data.q)

    def attn():
        minimax_m3_sparse_attn(
            data.q, data.kv_cache, topk_idx, data.block_table, data.cu_q,
            data.seq_lens, data.prefix_lens, qlen, 1, data.sm_scale, out)

    def score():
        minimax_m3_index_score(
            data.idx_q, data.index_kv_cache, data.block_table, data.cu_q,
            data.seq_lens, data.prefix_lens, qlen, seq_len, 1)

    def sel():
        minimax_m3_index_topk(scores, data.cu_q, data.prefix_lens, qlen, topk, 1, 2)

    attn(); torch.gcu.synchronize()
    r_attn = timed(attn)
    r_score = timed(score, 10)
    r_topk = timed(sel, 10)
    # unique selected pages actually visited (diagnostic, CPU check)
    uniq = int(torch.unique(topk_idx[topk_idx >= 0]).numel())
    print(
        "PREFILL Q=%-5d S=%-6d topk=%-3d attn=%.4f score=%.4f topk=%.4f "
        "n_pages=%d total_rows=%d distinct_selected=%d"
        % (qlen, seq_len, topk, r_attn, r_score, r_topk,
           (seq_len + 127) // 128, qlen * 8, uniq),
        flush=True,
    )
    del data, scores, topk_idx, out
    torch.gcu.empty_cache()


print("DEVICE", torch.gcu.current_device(), torch.gcu.get_device_name(1), flush=True)
for qlen in (128, 256, 512, 1024, 2048):
    prefill_attn(qlen, 4096, 16)
for seq_len in (1024, 4096, 16384):
    prefill_attn(512, seq_len, 16)
for topk in (4, 8, 16, 32):
    prefill_attn(512, 4096, topk)
print("DONE", flush=True)
