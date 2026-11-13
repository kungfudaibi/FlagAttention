"""MSA-004 exploratory screen: group16 vs default prefill in the affected band.

No timing claim is made here; this only finds candidate shapes for the
reference-checked, interleaved 3-round A/B. Records the real kernel name/grid
for each variant so a shape where the dispatch did not change is obvious.
"""

import json
import os
import statistics
import sys

import torch
import torch_gcu  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch.gcu.set_device(2)

from triton.runtime.jit import JITFunction  # noqa: E402

from benchmark import test_minimax_sparse_attention_benchmark as bench  # noqa: E402
from flag_attn import minimax_m3_sparse_attn  # noqa: E402
from flag_attn.minimax_sparse_attention import (  # noqa: E402
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
import flag_attn.runtime.backend._enflame.msa.sparse_attn as es  # noqa: E402

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

LAUNCH = []
_orig_run = JITFunction.run


def _run(self, *a, **kw):
    if kw.get("warmup") is False:
        grid = kw.get("grid")
        LAUNCH.append((self.__name__, list(grid) if isinstance(grid, (list, tuple)) else str(grid)))
    return _orig_run(self, *a, **kw)


JITFunction.run = _run


def timed(fn, samples):
    vals = []
    for _ in range(samples):
        start = torch.gcu.Event(enable_timing=True)
        end = torch.gcu.Event(enable_timing=True)
        torch.gcu.synchronize()
        start.record()
        fn()
        end.record()
        torch.gcu.synchronize()
        vals.append(start.elapsed_time(end))
    return vals


def make_case(case):
    qlen, seq_len, kv, group, topk, batch = case
    g = torch.Generator(device="cpu").manual_seed(0)
    dev = torch.device("gcu", torch.gcu.current_device())
    data = bench.make_data(batch, seq_len, kv, kv * group, dev,
                           "bf16", decode=False, decode_qlen=1, query_len=qlen,
                           randomize_pages=True, generator=g)
    return data


def run_variant(data, topk, qlen, seq_len, kv, group, enabled, samples):
    es._MSA002_GROUP16_EXPERIMENT = enabled
    scores = minimax_m3_index_score(
        data.idx_q, data.index_kv_cache, data.block_table, data.cu_q,
        data.seq_lens, data.prefix_lens, qlen, seq_len, kv)
    topk_idx = minimax_m3_index_topk(scores, data.cu_q, data.prefix_lens, qlen,
                                     topk, 1, 2)
    out = torch.empty_like(data.q)

    def full():
        s = minimax_m3_index_score(
            data.idx_q, data.index_kv_cache, data.block_table, data.cu_q,
            data.seq_lens, data.prefix_lens, qlen, seq_len, kv)
        ti = minimax_m3_index_topk(s, data.cu_q, data.prefix_lens, qlen,
                                   topk, 1, 2)
        minimax_m3_sparse_attn(data.q, data.kv_cache, ti, data.block_table,
                               data.cu_q, data.seq_lens, data.prefix_lens,
                               qlen, kv, data.sm_scale, out)
        return ti

    LAUNCH.clear()
    full()
    torch.gcu.synchronize()
    launch = list(LAUNCH)
    vals = timed(full, samples)
    return vals, launch


def report(case, samples=9):
    qlen, seq_len, kv, group, topk, batch = case
    data = make_case(case)
    result = {}
    for tag, enabled in (("A", False), ("B", True)):
        vals, launch = run_variant(data, topk, qlen, seq_len, kv, group,
                                   enabled, samples)
        result[tag] = {"samples": vals, "median": statistics.median(vals),
                       "launch": launch,
                       "peak_mib": torch.gcu.max_memory_allocated(2) / 2**20}
        torch.gcu.reset_peak_memory_stats(2)
    print(json.dumps({"probe": "screen", "case": case,
                      "qlen": qlen, "seq_len": seq_len, "kv": kv,
                      "group": group, "topk": topk, "batch": batch,
                      "A_ms": result["A"]["median"],
                      "B_ms": result["B"]["median"],
                      "speedup": result["A"]["median"] / result["B"]["median"],
                      "A_launch": result["A"]["launch"],
                      "B_launch": result["B"]["launch"],
                      "A_peak_mib": result["A"]["peak_mib"],
                      "B_peak_mib": result["B"]["peak_mib"]}), flush=True)
    del data
    torch.gcu.empty_cache()
    return result


CASES = [
    # GQA group sweep at the gate boundary.
    (512, 4096, 1, 1, 16, 1),
    (512, 4096, 1, 2, 16, 1),
    (512, 4096, 1, 4, 16, 1),
    (512, 4096, 1, 8, 16, 1),
    # topk / small-pool variants.
    (512, 512, 1, 8, 16, 1),
    (512, 512, 1, 8, 2, 1),
    (512, 1024, 1, 8, 4, 1),
    (512, 1024, 1, 8, 8, 1),
    (512, 2048, 1, 8, 8, 1),
    (512, 2048, 1, 8, 16, 1),
    (512, 4000, 1, 8, 16, 1),
    (512, 4097, 1, 8, 16, 1),
    # Q boundary / non-multiple-of-16.
    (513, 4096, 1, 8, 16, 1),
    (600, 4096, 1, 8, 16, 1),
    (768, 4096, 1, 8, 16, 1),
    (1008, 4096, 1, 8, 16, 1),
    (1009, 4096, 1, 8, 16, 1),
    (1023, 4096, 1, 8, 16, 1),
    (1024, 4096, 1, 8, 16, 1),
    (384, 4096, 1, 8, 16, 1),
    # Batch / ragged-ish total_q.
    (256, 4096, 1, 8, 16, 2),
    (128, 4096, 1, 8, 16, 4),
    # Fallback controls: pool gate, topk gate, head gate, kv>1.
    (512, 16384, 1, 8, 16, 1),
    (512, 4096, 1, 8, 32, 1),
    (512, 4096, 1, 16, 16, 1),
    (512, 4096, 2, 8, 16, 1),
]

for c in CASES:
    report(c)
print("DONE", flush=True)
