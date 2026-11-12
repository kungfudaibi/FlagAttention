"""Decode index-score chunk-cap A/B: index stage and whole decode call.

Cap 0 = shipped launch; cap 32/16 = experiment. Equivalence (score + topk
bit-exact) was established separately in phase3_decode_cap_equiv.log; here the
whole-call output is also compared bit-exactly for every cap.
Rounds are interleaved to cancel drift.
"""
import json, statistics, sys
import torch, torch_gcu  # noqa: F401

ROOT = "/workspace/agent-workspaces/msa-stage-profile"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/src")
torch.gcu.set_device(1)

from triton.runtime.jit import JITFunction  # noqa: E402
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

LAUNCH = []
_orig = JITFunction.run
def _run(self, *a, **kw):
    if kw.get("warmup") is False and "decode_index_score" in self.__name__:
        g = kw.get("grid")
        LAUNCH.append((self.__name__, list(g) if isinstance(g, (list, tuple)) else str(g)))
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

ROUNDS = 3
for seq_len in (4096, 16384):
    g = torch.Generator(device="cpu").manual_seed(3)
    data = bench.make_data(1, seq_len, 1, 8, torch.device("gcu:1"), "bf16", decode=True,
                           decode_qlen=1, query_len=1, randomize_pages=True, generator=g)
    out = torch.empty_like(data.q)

    def index():
        return ei.minimax_m3_index_decode(
            data.idx_q, data.index_kv_cache, data.block_table, data.seq_lens,
            seq_len, 16, 1, 2, 1, 1, 1)

    def whole():
        tk = index()
        minimax_m3_sparse_attn_decode(data.q, data.kv_cache, tk, data.block_table,
                                      data.seq_lens, 1, data.sm_scale, out, 1)

    variants = [("A_shipped", 0), ("B_cap32", 32), ("B_cap16", 16)]
    base = None
    for tag, cap in variants:
        ei._MSA002_DECODE_CHUNK_CAP = cap
        LAUNCH.clear()
        whole(); torch.gcu.synchronize()
        o = out.clone()
        if base is None:
            base = o
        print(json.dumps({"probe": "variant", "seq_len": seq_len, "variant": tag,
                          "cap": cap, "grid": list(LAUNCH),
                          "output_bit_exact_vs_shipped": bool(torch.equal(o, base))}), flush=True)

    rounds = {t: [] for t, _ in variants}
    for r in range(ROUNDS):
        for tag, cap in variants:
            ei._MSA002_DECODE_CHUNK_CAP = cap
            rounds[tag].append(timed(whole))
    stage = {}
    for tag, cap in variants:
        ei._MSA002_DECODE_CHUNK_CAP = cap
        stage[tag] = timed(index, 30)
    ei._MSA002_DECODE_CHUNK_CAP = 0
    med = {t: statistics.median(v) for t, v in rounds.items()}
    print(json.dumps({"probe": "ab", "seq_len": seq_len, "rounds": rounds,
                      "whole_ms": med, "index_stage_ms": stage,
                      "whole_speedup": {t: med["A_shipped"] / med[t] for t in med if t != "A_shipped"},
                      "index_speedup": {t: stage["A_shipped"] / stage[t] for t in stage if t != "A_shipped"}}),
          flush=True)
    del data, out
    torch.gcu.empty_cache()

print("DONE", flush=True)
