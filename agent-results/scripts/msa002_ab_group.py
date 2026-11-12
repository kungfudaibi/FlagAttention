"""MSA-002 phase 3: prefill GROUP_SIZE_Q A/B (8 shipped vs 16 candidate).

Interleaved rounds (not a sequential sweep) to cancel clock/thermal drift.
Each variant is checked against the repo's independent PyTorch reference
before any timing claim.
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

import triton  # noqa: E402

from benchmark import test_minimax_sparse_attention_benchmark as bench  # noqa: E402
from flag_attn.minimax_sparse_attention import (  # noqa: E402
    SPARSE_BLOCK_SIZE,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
from flag_attn.runtime.backend._enflame.msa.sparse_attn import (  # noqa: E402
    _gqa_sparse_fwd_grouped8,
)
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


def launch(data, topk_idx, out, qlen, group_q):
    grid = (triton.cdiv(qlen, group_q), 1, 1)
    _gqa_sparse_fwd_grouped8[grid](
        data.q, data.kv_cache, topk_idx, out, data.block_table,
        data.cu_q, data.seq_lens, data.prefix_lens,
        0, 8, 128, data.sm_scale,
        data.q.stride(0), data.q.stride(1), data.q.stride(2),
        data.kv_cache.stride(0), data.kv_cache.stride(1),
        data.kv_cache.stride(2), data.kv_cache.stride(3),
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        data.block_table.stride(0),
        GROUP_SIZE_Q=group_q,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_D=128,
        BLOCK_SIZE_H=8,
        MAX_TOPK=topk_idx.shape[-1],
        BLOCK_TOPK=triton.next_power_of_2(topk_idx.shape[-1]),
        DECODE_MODE=False,
        num_warps=1,
        num_stages=1,
    )


def timed(fn, samples=15):
    vals = []
    for _ in range(samples):
        a = torch.gcu.Event(enable_timing=True)
        b = torch.gcu.Event(enable_timing=True)
        torch.gcu.synchronize()
        a.record(); fn(); b.record(); torch.gcu.synchronize()
        vals.append(a.elapsed_time(b))
    vals.sort()
    return statistics.median(vals)


def metrics(out, ref):
    d = (out.float() - ref.float()).abs()
    rms = float(torch.sqrt(torch.mean((out.float() - ref.float()) ** 2)).item())
    rrms = float(torch.sqrt(torch.mean(ref.float() ** 2)).item())
    return {"max_abs": float(d.max().item()), "relative_rms": rms / rrms if rrms else 0.0,
            "finite": bool(torch.isfinite(out.float()).all().item())}


RESULTS = {}
for qlen, seq_len in ((128, 4096), (512, 4096), (1024, 4096), (512, 16384)):
    g = torch.Generator(device="cpu").manual_seed(0)
    data = bench.make_data(1, seq_len, 1, 8, torch.device("gcu:1"), "bf16",
                           decode=False, decode_qlen=1, query_len=qlen,
                           randomize_pages=True, generator=g)
    scores = minimax_m3_index_score(
        data.idx_q, data.index_kv_cache, data.block_table, data.cu_q,
        data.seq_lens, data.prefix_lens, qlen, seq_len, 1)
    topk_idx = minimax_m3_index_topk(scores, data.cu_q, data.prefix_lens, qlen, 16, 1, 2)
    outs = {}
    for gq in (8, 16):
        o = torch.empty_like(data.q)
        launch(data, topk_idx, o, qlen, gq)
        torch.gcu.synchronize()
        outs[gq] = o.clone()
    class _Shim:
        pass

    shim = _Shim()
    for attr in ("q", "idx_q", "kv_cache", "index_kv_cache", "block_table",
                 "cu_q", "seq_lens", "prefix_lens", "sm_scale", "k_scale", "v_scale"):
        setattr(shim, attr, getattr(data, attr))
    shim.query_lens = (qlen,)
    shim.max_seq_len = seq_len
    ref = t._ref_sparse_attn(shim, topk_idx)
    torch.gcu.synchronize()
    for gq in (8, 16):
        m = metrics(outs[gq], ref)
        m["within_2e-2"] = bool(torch.allclose(outs[gq], ref, atol=2e-2, rtol=2e-2))
        m["cross_variant_max_abs"] = float((outs[gq].float() - outs[16 if gq == 8 else 8].float()).abs().max().item())
        print(json.dumps({"probe": "correctness", "qlen": qlen, "seq_len": seq_len,
                          "group_q": gq, **m}), flush=True)

    rounds = {}
    for rnd in range(3):
        for gq in (8, 16):
            o = torch.empty_like(data.q)
            ms = timed(lambda gq=gq, o=o: launch(data, topk_idx, o, qlen, gq))
            rounds.setdefault(gq, []).append(ms)
        print(json.dumps({"probe": "ab_round", "qlen": qlen, "seq_len": seq_len,
                          "round": rnd, "group8_ms": rounds[8][-1],
                          "group16_ms": rounds[16][-1]}), flush=True)
    for gq in (8, 16):
        print(json.dumps({"probe": "ab_summary", "qlen": qlen, "seq_len": seq_len,
                          "group_q": gq, "rounds": rounds[gq],
                          "median": statistics.median(rounds[gq])}), flush=True)
    RESULTS[(qlen, seq_len)] = rounds
    del data, scores, topk_idx, outs, ref
    torch.gcu.empty_cache()

for (qlen, seq_len), rounds in RESULTS.items():
    s8 = statistics.median(rounds[8])
    s16 = statistics.median(rounds[16])
    print(json.dumps({"probe": "ab_ratio", "qlen": qlen, "seq_len": seq_len,
                      "group8_ms": s8, "group16_ms": s16, "speedup_16_vs_8": s8 / s16}), flush=True)
print("DONE", flush=True)
