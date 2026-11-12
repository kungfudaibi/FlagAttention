"""Decode chunk-cap end-to-end check through the public API.

For each S: run shipped (cap 0) and capped (32) variants; require the score
tensor and the selected topk_idx to be bit-identical, then time the index stage
and the whole decode call with interleaved rounds.
"""
import json, statistics, sys
import torch, torch_gcu  # noqa: F401

ROOT = "/workspace/agent-workspaces/msa-stage-profile"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/src")
torch.gcu.set_device(1)

from benchmark import test_minimax_sparse_attention_benchmark as bench  # noqa: E402
from flag_attn import minimax_m3_sparse_attn_decode  # noqa: E402
import flag_attn.runtime.backend._enflame.msa.index_topk as ei  # noqa: E402

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

print(json.dumps({"probe": "health", "sum": float(torch.ones(1024, device="gcu:1").sum().item())}), flush=True)

def timed(fn, n=20):
    v = []
    for _ in range(n):
        a = torch.gcu.Event(enable_timing=True); b = torch.gcu.Event(enable_timing=True)
        torch.gcu.synchronize(); a.record(); fn(); b.record(); torch.gcu.synchronize()
        v.append(a.elapsed_time(b))
    v.sort(); return statistics.median(v)

for seq_len in (4096, 16384):
    g = torch.Generator(device="cpu").manual_seed(3)
    data = bench.make_data(1, seq_len, 1, 8, torch.device("gcu:1"), "bf16", decode=True,
                           decode_qlen=1, query_len=1, randomize_pages=True, generator=g)
    out = torch.empty_like(data.q)
    res = {}
    for tag, cap in (("A_shipped", 0), ("B_cap32", 32)):
        ei._MSA002_DECODE_CHUNK_CAP = cap
        score = ei.minimax_m3_index_decode_score(
            data.idx_q, data.index_kv_cache, data.block_table, data.seq_lens,
            seq_len, 1, 2, 1, 1, 1)
        tk = ei.minimax_m3_index_decode(
            data.idx_q, data.index_kv_cache, data.block_table, data.seq_lens,
            seq_len, 16, 1, 2, 1, 1, 1)
        torch.gcu.synchronize()
        res[tag] = (score.clone(), tk.clone())
    for tag in res:
        s, tk = res[tag]
        bs, btk = res["A_shipped"]
        print(json.dumps({"probe": "equiv", "seq_len": seq_len, "variant": tag,
                          "score_bit_exact": bool(torch.equal(s, bs)),
                          "topk_bit_exact": bool(torch.equal(tk, btk))}), flush=True)
    del res
    ei._MSA002_DECODE_CHUNK_CAP = 0
    torch.gcu.empty_cache()
print("DONE", flush=True)
