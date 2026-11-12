"""Quiet-window A/B (5 interleaved rounds) for the relaxed group-16 gate.

Correctness vs the independent reference was already established; here each
variant's output is compared with the other variant and checked finite, and the
launch grid is captured so the dispatch actually taken is recorded per row.
"""
import json, statistics, sys
import torch, torch_gcu  # noqa: F401

ROOT = "/workspace/agent-workspaces/msa-stage-profile"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/src")
torch.gcu.set_device(1)

from triton.runtime.jit import JITFunction  # noqa: E402
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

L = []
_orig = JITFunction.run
def _run(self, *a, **kw):
    if kw.get("warmup") is False and "grouped8" in self.__name__:
        g = kw.get("grid")
        L.append((self.__name__, list(g) if isinstance(g, (list, tuple)) else str(g)))
    return _orig(self, *a, **kw)
JITFunction.run = _run

print(json.dumps({"probe": "health", "sum": float(torch.ones(1024, device="gcu:1").sum().item())}), flush=True)

def timed(fn, n=20):
    v = []
    for _ in range(n):
        a = torch.gcu.Event(enable_timing=True); b = torch.gcu.Event(enable_timing=True)
        torch.gcu.synchronize(); a.record(); fn(); b.record(); torch.gcu.synchronize()
        v.append(a.elapsed_time(b))
    v.sort(); return statistics.median(v)

CASES = [(1, 512, 4096), (1, 640, 4096), (1, 768, 4096), (2, 256, 4096),
         (4, 128, 4096), (1, 512, 2048)]
ROUNDS = 5
rows = []
for batch, qlen, seq_len in CASES:
    g = torch.Generator(device="cpu").manual_seed(0)
    data = bench.make_data(batch, seq_len, 1, 8, torch.device("gcu:1"), "bf16",
                           decode=False, decode_qlen=1, query_len=qlen,
                           randomize_pages=True, generator=g)
    out = torch.empty_like(data.q)
    cache = {}
    def whole():
        cache["s"] = minimax_m3_index_score(data.idx_q, data.index_kv_cache,
                                            data.block_table, data.cu_q, data.seq_lens,
                                            data.prefix_lens, qlen, seq_len, 1)
        cache["t"] = minimax_m3_index_topk(cache["s"], data.cu_q, data.prefix_lens,
                                           qlen, 16, 1, 2)
        minimax_m3_sparse_attn(data.q, data.kv_cache, cache["t"], data.block_table,
                               data.cu_q, data.seq_lens, data.prefix_lens, qlen, 1,
                               data.sm_scale, out)

    outs, grids = {}, {}
    for tag, en in (("A", False), ("B", True)):
        es._MSA002_GROUP16_EXPERIMENT = en
        L.clear()
        whole(); torch.gcu.synchronize()
        outs[tag] = out.clone()
        grids[tag] = [x for x in L if "topk" not in x[0] and "index_block" not in x[0]]
    es._MSA002_GROUP16_EXPERIMENT = False
    finite = bool(torch.isfinite(outs["B"].float()).all().item())
    cross = float((outs["A"].float() - outs["B"].float()).abs().max().item())

    r = {"A": [], "B": []}
    for _ in range(ROUNDS):
        for tag, en in (("A", False), ("B", True)):
            es._MSA002_GROUP16_EXPERIMENT = en
            r[tag].append(timed(whole))
    es._MSA002_GROUP16_EXPERIMENT = False
    ma, mb = statistics.median(r["A"]), statistics.median(r["B"])
    row = {"batch": batch, "qlen": qlen, "seq_len": seq_len,
           "total_q": data.q.shape[0], "A_ms": ma, "B_ms": mb, "speedup": ma / mb,
           "A_min": min(r["A"]), "A_max": max(r["A"]),
           "B_min": min(r["B"]), "B_max": max(r["B"]),
           "grid_A": grids["A"], "grid_B": grids["B"],
           "finite_B": finite, "cross_max_abs": cross}
    rows.append(row)
    print(json.dumps({"probe": "quiet_ab", **row}), flush=True)
    del data, out, outs, cache
    torch.gcu.empty_cache()

print(json.dumps({"probe": "quiet_summary",
                  "rows": [{k: v for k, v in x.items() if k != "grid_A" and k != "grid_B"}
                           for x in rows]}), flush=True)
print("DONE", flush=True)
