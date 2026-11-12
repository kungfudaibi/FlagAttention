"""Post-fault reproducibility: group-16 A/B, 2 rounds, one case."""
import json, statistics, sys
import torch, torch_gcu  # noqa: F401

ROOT = "/workspace/agent-workspaces/msa-stage-profile"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/src")
torch.gcu.set_device(1)

from benchmark import test_minimax_sparse_attention_benchmark as bench  # noqa: E402
from flag_attn import minimax_m3_sparse_attn  # noqa: E402
from flag_attn.minimax_sparse_attention import (  # noqa: E402
    minimax_m3_index_score, minimax_m3_index_topk)
import flag_attn.runtime.backend._enflame.msa.sparse_attn as es  # noqa: E402

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

def timed(fn, n=15):
    v = []
    for _ in range(n):
        a = torch.gcu.Event(enable_timing=True); b = torch.gcu.Event(enable_timing=True)
        torch.gcu.synchronize(); a.record(); fn(); b.record(); torch.gcu.synchronize()
        v.append(a.elapsed_time(b))
    v.sort(); return statistics.median(v)

qlen, seq_len, b = 512, 4096, 1
g = torch.Generator(device="cpu").manual_seed(0)
data = bench.make_data(b, seq_len, 1, 8, torch.device("gcu:1"), "bf16", decode=False,
                       decode_qlen=1, query_len=qlen, randomize_pages=True, generator=g)
out = torch.empty_like(data.q)
cache = {}
def whole():
    cache["s"] = minimax_m3_index_score(data.idx_q, data.index_kv_cache, data.block_table,
                                        data.cu_q, data.seq_lens, data.prefix_lens,
                                        qlen, seq_len, 1)
    cache["t"] = minimax_m3_index_topk(cache["s"], data.cu_q, data.prefix_lens, qlen, 16, 1, 2)
    minimax_m3_sparse_attn(data.q, data.kv_cache, cache["t"], data.block_table, data.cu_q,
                           data.seq_lens, data.prefix_lens, qlen, 1, data.sm_scale, out)

rounds = {"A": [], "B": []}
for r in range(2):
    for tag, en in (("A", False), ("B", True)):
        es._MSA002_GROUP16_EXPERIMENT = en
        rounds[tag].append(timed(whole))
        print(json.dumps({"probe": "round", "round": r, "variant": tag, "ms": rounds[tag][-1]}), flush=True)
es._MSA002_GROUP16_EXPERIMENT = False
a, bb = statistics.median(rounds["A"]), statistics.median(rounds["B"])
print(json.dumps({"probe": "postfault_ab", "qlen": qlen, "seq_len": seq_len,
                  "A_ms": a, "B_ms": bb, "speedup": a / bb}), flush=True)
print("DONE", flush=True)
