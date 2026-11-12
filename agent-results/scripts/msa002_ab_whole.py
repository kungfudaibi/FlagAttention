"""MSA-002 phase 3: whole-call A/B for the relaxed group-16 gate.

Whole-call = score + topk + attention with one event pair and sync at both
ends. Stage-only numbers are NOT summed here. Interleaved A/B rounds.
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
from flag_attn import minimax_m3_sparse_attn  # noqa: E402
from flag_attn.minimax_sparse_attention import (  # noqa: E402
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
import flag_attn.runtime.backend._enflame.msa.sparse_attn as enflame_sparse  # noqa: E402
from tests.flag_attn import test_minimax_sparse_attention as t  # noqa: E402

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


class Shim:
    pass


def ref_attn(data, topk_idx, qlen, seq_len):
    shim = Shim()
    for attr in ("q", "idx_q", "kv_cache", "index_kv_cache", "block_table",
                 "cu_q", "seq_lens", "prefix_lens", "sm_scale", "k_scale", "v_scale"):
        setattr(shim, attr, getattr(data, attr))
    shim.query_lens = (qlen,) * batch
    shim.max_seq_len = seq_len
    return t._ref_sparse_attn(shim, topk_idx)


# (batch, query_len, seq_len, topk, tag)
CASES = [(1, 512, 4096, 16, "batch1"), (2, 256, 4096, 16, "batch2"),
         (4, 128, 4096, 16, "batch4"), (8, 128, 4096, 16, "batch8"),
         (1, 384, 4096, 16, "batch1_384"), (1, 640, 4096, 16, "batch1_640")]
ROUNDS = 3
out_rows = []

for batch, qlen, seq_len, topk, tag in CASES:
    g = torch.Generator(device="cpu").manual_seed(0)
    data = bench.make_data(batch, seq_len, 1, 8, torch.device("gcu:1"), "bf16",
                           decode=False, decode_qlen=1, query_len=qlen,
                           randomize_pages=True, generator=g)
    out = torch.empty_like(data.q)
    cache = {}

    def score():
        cache["s"] = minimax_m3_index_score(
            data.idx_q, data.index_kv_cache, data.block_table, data.cu_q,
            data.seq_lens, data.prefix_lens, qlen, seq_len, 1)

    def select_topk():
        cache["t"] = minimax_m3_index_topk(
            cache["s"], data.cu_q, data.prefix_lens, qlen, topk, 1, 2)

    def attn():
        minimax_m3_sparse_attn(data.q, data.kv_cache, cache["t"], data.block_table,
                               data.cu_q, data.seq_lens, data.prefix_lens, qlen, 1,
                               data.sm_scale, out)

    def whole():
        score(); select_topk(); attn()

    # correctness of the full pipeline for both variants vs independent reference
    checks = {}
    for enabled, tag in ((False, "A_shipped"), (True, "B_group16")):
        enflame_sparse._MSA002_GROUP16_EXPERIMENT = enabled
        whole()
        torch.gcu.synchronize()
        shim = Shim()
        for attr in ("q", "idx_q", "kv_cache", "index_kv_cache", "block_table",
                     "cu_q", "seq_lens", "prefix_lens", "sm_scale", "k_scale", "v_scale"):
            setattr(shim, attr, getattr(data, attr))
        shim.query_lens = (qlen,) * batch
        shim.max_seq_len = seq_len
        ref_scores = t._ref_index_score(shim)
        ref_topk = t._ref_topk(ref_scores, shim, topk, 1, 2)
        ref_out = t._ref_sparse_attn(shim, ref_topk)
        torch.gcu.synchronize()
        try:
            t._assert_score_match(cache["s"], ref_scores, shim, decode=False)
            t._assert_topk_match(cache["t"], ref_topk, shim, topk, decode=False)
            t._assert_attention_match(out, ref_out, shim)
            ok = True
        except AssertionError as exc:
            ok = False
        checks[tag] = ok
        print(json.dumps({"probe": "correctness", "qlen": qlen, "seq_len": seq_len,
                          "variant": tag, "pipeline_pass": ok, "batch": batch, "topk": topk}), flush=True)
    enflame_sparse._MSA002_GROUP16_EXPERIMENT = False

    rounds = {"A_shipped": [], "B_group16": []}
    for rnd in range(ROUNDS):
        for tag, enabled in (("A_shipped", False), ("B_group16", True)):
            enflame_sparse._MSA002_GROUP16_EXPERIMENT = enabled
            ms = timed(whole)
            rounds[tag].append(ms)
    enflame_sparse._MSA002_GROUP16_EXPERIMENT = False
    a = statistics.median(rounds["A_shipped"])
    b = statistics.median(rounds["B_group16"])
    row = {"batch": batch, "qlen": qlen, "seq_len": seq_len, "topk": topk, "A_ms": a, "B_ms": b,
           "A_rounds": rounds["A_shipped"], "B_rounds": rounds["B_group16"],
           "speedup_B_over_A": a / b, "correct_A": checks["A_shipped"],
           "correct_B": checks["B_group16"]}
    out_rows.append(row)
    print(json.dumps({"probe": "whole_ab", **row}), flush=True)
    del data, out, cache
    torch.gcu.empty_cache()

print(json.dumps({"probe": "summary", "rows": out_rows}), flush=True)
print("DONE", flush=True)
