"""MSA-002 phase 2b: amortized vs per-call timing + page-union structure."""

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
from flag_attn import minimax_m3_sparse_attn  # noqa: E402
from flag_attn.minimax_sparse_attention import (  # noqa: E402
    minimax_m3_index_decode,
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


def per_call(fn, n=1, samples=15):
    def run():
        for _ in range(n):
            fn()
    run()
    torch.gcu.synchronize()
    vals = []
    for _ in range(samples):
        a = torch.gcu.Event(enable_timing=True)
        b = torch.gcu.Event(enable_timing=True)
        torch.gcu.synchronize()
        a.record()
        run()
        b.record()
        torch.gcu.synchronize()
        vals.append(a.elapsed_time(b) / n)
    vals.sort()
    return statistics.median(vals)


def report(label, fn):
    one = per_call(fn, 1)
    n10 = per_call(fn, 10)
    n50 = per_call(fn, 50)
    print(json.dumps({"probe": label, "per_call_ms": one, "amortized_n10_ms": n10,
                      "amortized_n50_ms": n50, "fixed_overhead_ms": one - n10}), flush=True)


# ---------- prefill ----------
g = torch.Generator(device="cpu").manual_seed(0)
shape = (1, 512, 4096, 1, 8)
data = bench.make_data(1, 4096, 1, 8, torch.device("gcu:1"), "bf16", decode=False,
                       decode_qlen=1, query_len=512, randomize_pages=True, generator=g)
out = torch.empty_like(data.q)
cache = {}


def index_score():
    cache["s"] = minimax_m3_index_score(
        data.idx_q, data.index_kv_cache, data.block_table, data.cu_q,
        data.seq_lens, data.prefix_lens, 512, 4096, 1)


def index_topk():
    cache["t"] = minimax_m3_index_topk(cache["s"], data.cu_q, data.prefix_lens, 512, 16, 1, 2)


def attention():
    minimax_m3_sparse_attn(data.q, data.kv_cache, cache["t"], data.block_table, data.cu_q,
                           data.seq_lens, data.prefix_lens, 512, 1, data.sm_scale, out)


def whole():
    index_score(); index_topk(); attention()


for label, fn in (("prefill_score", index_score), ("prefill_topk", index_topk),
                  ("prefill_attn", attention), ("prefill_whole", whole)):
    report(label, fn)

# page-union structure for groups of G queries
topk_idx = cache["t"].cpu()
nq = topk_idx.shape[1]
for G in (2, 4, 8, 16, 32):
    unions = []
    for kh in range(topk_idx.shape[0]):
        for start in range(0, nq, G):
            sel = topk_idx[kh, start:start + G]
            sel = sel[sel >= 0]
            unions.append(int(torch.unique(sel).numel()))
    print(json.dumps({"probe": "page_union", "group": G, "groups": len(unions),
                      "mean_union": statistics.mean(unions),
                      "max_union": max(unions),
                      "useful_per_query": 16,
                      "waste_ratio": statistics.mean(unions) / 16}), flush=True)

# ---------- decode ----------
g = torch.Generator(device="cpu").manual_seed(100000)
d = bench.make_data(1, 16384, 1, 8, torch.device("gcu:1"), "bf16", decode=True,
                    decode_qlen=1, query_len=1, randomize_pages=True, generator=g)
dout = torch.empty_like(d.q)
dc = {}


def index_decode():
    dc["t"] = minimax_m3_index_decode(d.idx_q, d.index_kv_cache, d.block_table, d.seq_lens,
                                      16384, 16, 1, 2, 1, 1, 1)


def dattention():
    minimax_m3_sparse_attn(d.q, d.kv_cache, dc["t"], d.block_table, d.seq_lens,
                           1, d.sm_scale, dout, 1, stage="decode")


def dwhole():
    index_decode(); dattention()


for label, fn in (("decode_index", index_decode), ("decode_attn", dattention),
                  ("decode_whole", dwhole)):
    report(label, fn)
print("DONE", flush=True)
