"""MSA-002 decode scaling: isolate index-score vs topk vs attention."""

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
)
from flag_attn.runtime.backend._enflame.msa.index_topk import (  # noqa: E402
    minimax_m3_index_decode_score,
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


def timed(fn, samples=15, n=1):
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


for batch, seq_len in ((1, 2048), (1, 4096), (1, 16384), (1, 65536),
                       (4, 4096), (4, 16384), (16, 4096)):
    g = torch.Generator(device="cpu").manual_seed(3)
    data = bench.make_data(batch, seq_len, 1, 8, torch.device("gcu:1"), "bf16",
                           decode=True, decode_qlen=1, query_len=1,
                           randomize_pages=True, generator=g)
    out = torch.empty_like(data.q)
    box = {}

    def score_only():
        box["s"] = minimax_m3_index_decode_score(
            data.idx_q, data.index_kv_cache, data.block_table, data.seq_lens,
            seq_len, 1, 2, 1, 1, 1)

    score_only(); torch.gcu.synchronize()

    def topk_only():
        minimax_m3_index_decode(
            data.idx_q, data.index_kv_cache, data.block_table, data.seq_lens,
            seq_len, 16, 1, 2, 1, 1, 1, score_out=box["s"])

    def index():
        box["t"] = minimax_m3_index_decode(
            data.idx_q, data.index_kv_cache, data.block_table, data.seq_lens,
            seq_len, 16, 1, 2, 1, 1, 1)

    def attn():
        minimax_m3_sparse_attn(data.q, data.kv_cache, box["t"], data.block_table,
                               data.seq_lens, 1, data.sm_scale, out, 1, stage="decode")

    index(); torch.gcu.synchronize()
    print(json.dumps({
        "batch": batch, "seq_len": seq_len, "pages": (seq_len + 127) // 128,
        "score_only_ms": timed(score_only),
        "topk_with_scorebuf_ms": timed(topk_only),
        "index_ms": timed(index),
        "attn_ms": timed(attn),
    }), flush=True)
    del data, out, box
    torch.gcu.empty_cache()
print("DONE", flush=True)
