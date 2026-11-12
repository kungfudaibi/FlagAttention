"""MSA-002 phase 3 experiment: decode index-score chunk count sweep.

Hypothesis: for decode_qlen == 1 the Enflame score kernel is launched with
num_kv_chunks == min(128, max_block) == one 128-token page per CTA. Each CTA
then performs a single tiny [8,128]x[128,128] dot and exits, so the kernel is
latency/occupancy bound rather than bandwidth bound. Coarsening the chunking
(fewer, longer-running CTAs) should cut that overhead without changing results.

This script calls the real kernel directly; scores are compared bit-exactly
against the shipped chunking before any timing claim.
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

from benchmark import test_minimax_sparse_attention_benchmark as bench  # noqa: E402
from flag_attn.runtime.backend._enflame.msa.index_topk import (  # noqa: E402
    _decode_index_score_kernel_enflame,
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

BLOCK = 128


def launch(data, seq_len, score, num_kv_chunks, block_size_q=1):
    max_block = (seq_len + BLOCK - 1) // BLOCK
    grid = (data.seq_lens.shape[0], num_kv_chunks, 1)
    _decode_index_score_kernel_enflame[grid](
        data.idx_q, data.index_kv_cache, score, data.block_table, data.seq_lens,
        1, 128, 1, 2, 1,
        data.idx_q.stride(0), data.idx_q.stride(1), data.idx_q.stride(2),
        data.index_kv_cache.stride(0), data.index_kv_cache.stride(1),
        data.index_kv_cache.stride(2),
        score.stride(0), score.stride(1), score.stride(2),
        data.block_table.stride(0),
        BLOCK_SIZE_K=BLOCK, BLOCK_SIZE_Q=block_size_q,
        num_kv_chunks=num_kv_chunks, num_warps=1, num_stages=1,
    )


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


for seq_len in (4096, 16384, 65536):
    g = torch.Generator(device="cpu").manual_seed(3)
    data = bench.make_data(1, seq_len, 1, 8, torch.device("gcu:1"), "bf16",
                           decode=True, decode_qlen=1, query_len=1,
                           randomize_pages=True, generator=g)
    max_block = (seq_len + BLOCK - 1) // BLOCK
    stride_k = max(16, (max_block + 15) // 16 * 16)
    ref = torch.empty((1, 1, stride_k), dtype=torch.float32, device="gcu:1")
    launch(data, seq_len, ref, max_block)
    torch.gcu.synchronize()
    base = ref.clone()

    for chunks in (128, 64, 32, 16, 8, 4, 2, 1):
        if chunks > max_block:
            continue
        out = torch.empty_like(ref)
        launch(data, seq_len, out, chunks)
        torch.gcu.synchronize()
        same = torch.equal(out[:, :, :max_block], base[:, :, :max_block])
        ms = timed(lambda out=out, chunks=chunks: launch(data, seq_len, out, chunks))
        print(json.dumps({"seq_len": seq_len, "pages": max_block, "num_kv_chunks": chunks,
                          "ms": ms, "bit_exact_vs_shipped": bool(same)}), flush=True)
    del data, ref, base
    torch.gcu.empty_cache()
print("DONE", flush=True)
