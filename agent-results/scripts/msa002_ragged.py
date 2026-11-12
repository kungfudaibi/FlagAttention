"""MSA-002 phase 3: correctness + A/B on ragged prefill inside the new band.

The relaxed gate only changes launches where cdiv(total_q,16) is 32..63, i.e.
total_q in [512, 1023]. These ragged cases land there with max_query_len >= 16
so the grouped-16 kernel is actually selected. Both variants are compared with
the repo's independent per-query reference (prefix/causal handling included).
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

from triton.runtime.jit import JITFunction  # noqa: E402

from flag_attn import minimax_m3_sparse_attn  # noqa: E402
from flag_attn.minimax_sparse_attention import (  # noqa: E402
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
import flag_attn.runtime.backend._enflame.msa.sparse_attn as enflame_sparse  # noqa: E402
from tests.flag_attn import test_minimax_sparse_attention as t  # noqa: E402

LAUNCHES = []
_orig_run = JITFunction.run


def _run(self, *a, **kw):
    if kw.get("warmup") is False and "grouped8" in self.__name__:
        grid = kw.get("grid")
        LAUNCHES.append((self.__name__, list(grid) if isinstance(grid, (list, tuple)) else str(grid)))
    return _orig_run(self, *a, **kw)


JITFunction.run = _run


def timed(fn, samples=20):
    vals = []
    for _ in range(samples):
        a = torch.gcu.Event(enable_timing=True)
        b = torch.gcu.Event(enable_timing=True)
        torch.gcu.synchronize()
        a.record(); fn(); b.record(); torch.gcu.synchronize()
        vals.append(a.elapsed_time(b))
    vals.sort()
    return statistics.median(vals)


CASES = [
    # (seq_lens, prefixes)
    ((4096, 4096), (3840, 3584)),   # total 768, max_q 512
    ((2048, 2048, 2048), (1536, 1792, 1920)),  # total 896, max_q 512
    ((1024, 1024), (512, 640)),     # total 896, max_q 512
]

for seq_lens, prefixes in CASES:
    data = t.make_data(seq_lens, 1, 8, decode=False, prefix_lens=prefixes,
                       mode="bf16", seed=42)
    total_q = data.q.shape[0]
    qlens = tuple(s - p for s, p in zip(seq_lens, prefixes))
    qlen = max(qlens)
    out = torch.empty_like(data.q)
    cache = {}

    def run_pipeline():
        cache["s"] = minimax_m3_index_score(
            data.idx_q, data.index_kv_cache, data.block_table, data.cu_q,
            data.seq_lens, data.prefix_lens, qlen, data.max_seq_len, 1)
        cache["t"] = minimax_m3_index_topk(
            cache["s"], data.cu_q, data.prefix_lens, qlen, 16, 1, 2)
        minimax_m3_sparse_attn(
            data.q, data.kv_cache, cache["t"], data.block_table, data.cu_q,
            data.seq_lens, data.prefix_lens, qlen, 1, data.sm_scale, out)

    for enabled, tag in ((False, "A_shipped"), (True, "B_group16")):
        enflame_sparse._MSA002_GROUP16_EXPERIMENT = enabled
        LAUNCHES.clear()
        run_pipeline()
        torch.gcu.synchronize()
        print(json.dumps({"probe": "variant", "seq_lens": list(seq_lens),
                          "prefixes": list(prefixes), "total_q": total_q,
                          "variant": tag, "kernels": list(LAUNCHES)}), flush=True)
        ref_scores = t._ref_index_score(data)
        ref_topk = t._ref_topk(ref_scores, data, 16, 1, 2)
        ref_out = t._ref_sparse_attn(data, ref_topk)
        torch.gcu.synchronize()
        try:
            t._assert_score_match(cache["s"], ref_scores, data, decode=False)
            t._assert_topk_match(cache["t"], ref_topk, data, 16, decode=False)
            t._assert_attention_match(out, ref_out, data)
            ok = True
            err = ""
        except AssertionError as exc:
            ok = False
            err = str(exc)
        print(json.dumps({"probe": "correctness", "seq_lens": list(seq_lens),
                          "prefixes": list(prefixes), "variant": tag, "pass": ok,
                          "error": err}), flush=True)

    rounds = {"A_shipped": [], "B_group16": []}
    for rnd in range(3):
        for tag, enabled in (("A_shipped", False), ("B_group16", True)):
            enflame_sparse._MSA002_GROUP16_EXPERIMENT = enabled
            rounds[tag].append(timed(run_pipeline, 10))
    enflame_sparse._MSA002_GROUP16_EXPERIMENT = False
    a = statistics.median(rounds["A_shipped"])
    b = statistics.median(rounds["B_group16"])
    print(json.dumps({"probe": "ragged_ab", "seq_lens": list(seq_lens),
                      "prefixes": list(prefixes), "total_q": total_q,
                      "A_ms": a, "B_ms": b, "speedup": a / b,
                      "A_rounds": rounds["A_shipped"], "B_rounds": rounds["B_group16"]}),
          flush=True)
    del data, out, cache
    torch.gcu.empty_cache()
print("DONE", flush=True)
