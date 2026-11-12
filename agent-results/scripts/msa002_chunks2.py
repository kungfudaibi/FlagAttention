"""Decode index-score chunk coarsening: reconfirm post-fault, S=16384 only.

Direct kernel launches (same as msa002_chunks.py); results compared bit-exactly
against the shipped 128-chunk launch. Score kernel only, no merge kernel.
"""
import json, statistics, sys
import torch, torch_gcu  # noqa: F401

ROOT = "/workspace/agent-workspaces/msa-stage-profile"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/src")
torch.gcu.set_device(1)

from benchmark import test_minimax_sparse_attention_benchmark as bench  # noqa: E402
from flag_attn.runtime.backend._enflame.msa.index_topk import (  # noqa: E402
    _decode_index_score_kernel_enflame)

for _n in ("randn", "rand", "randperm", "randint"):
    _o = getattr(torch, _n)
    def _mk(orig):
        def w(*a, **kw):
            g, d = kw.get("generator"), kw.get("device")
            if g is not None and d is not None and str(g.device) == "cpu" and str(d).startswith("gcu"):
                kw["device"] = "cpu"
                return orig(*a, **kw).to(d)
            return orig(*a, **kw)
        return w
    setattr(torch, _n, _mk(_o))

BLOCK = 128
print(json.dumps({"probe": "health", "sum": float(torch.ones(1024, device="gcu:1").sum().item())}), flush=True)

def launch(data, score, chunks):
    grid = (data.seq_lens.shape[0], chunks, 1)
    _decode_index_score_kernel_enflame[grid](
        data.idx_q, data.index_kv_cache, score, data.block_table, data.seq_lens,
        1, 128, 1, 2, 1,
        data.idx_q.stride(0), data.idx_q.stride(1), data.idx_q.stride(2),
        data.index_kv_cache.stride(0), data.index_kv_cache.stride(1),
        data.index_kv_cache.stride(2),
        score.stride(0), score.stride(1), score.stride(2),
        data.block_table.stride(0),
        BLOCK_SIZE_K=BLOCK, BLOCK_SIZE_Q=1, num_kv_chunks=chunks,
        num_warps=1, num_stages=1)

def timed(fn, n=50):
    fn(); torch.gcu.synchronize()
    v = []
    for _ in range(n):
        a = torch.gcu.Event(enable_timing=True); b = torch.gcu.Event(enable_timing=True)
        torch.gcu.synchronize(); a.record(); fn(); b.record(); torch.gcu.synchronize()
        v.append(a.elapsed_time(b))
    v.sort(); return statistics.median(v)

seq_len = 16384
g = torch.Generator(device="cpu").manual_seed(3)
data = bench.make_data(1, seq_len, 1, 8, torch.device("gcu:1"), "bf16", decode=True,
                       decode_qlen=1, query_len=1, randomize_pages=True, generator=g)
max_block = (seq_len + BLOCK - 1) // BLOCK
stride_k = max(16, (max_block + 15) // 16 * 16)
ref = torch.empty((1, 1, stride_k), dtype=torch.float32, device="gcu:1")
launch(data, ref, 128); torch.gcu.synchronize()
base = ref.clone()
rows = []
for chunks in (128, 64, 32, 16):
    out = torch.empty_like(ref)
    launch(data, out, chunks); torch.gcu.synchronize()
    same = torch.equal(out[:, :, :max_block], base[:, :, :max_block])
    ms = timed(lambda out=out, c=chunks: launch(data, out, c))
    rows.append({"chunks": chunks, "ms": ms, "bit_exact": bool(same)})
    print(json.dumps({"probe": "chunk", "seq_len": seq_len, "pages": max_block,
                      "num_kv_chunks": chunks, "ms": ms,
                      "bit_exact_vs_shipped": bool(same)}), flush=True)
best = min(rows, key=lambda r: r["ms"])
shipped = [r for r in rows if r["chunks"] == 128][0]
print(json.dumps({"probe": "summary", "seq_len": seq_len,
                  "best_chunks": best["chunks"], "best_ms": best["ms"],
                  "shipped_ms": shipped["ms"],
                  "speedup": shipped["ms"] / best["ms"],
                  "all_bit_exact": all(r["bit_exact"] for r in rows)}), flush=True)
print("DONE", flush=True)
