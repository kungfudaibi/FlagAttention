"""MSA-002 phase 1: representative-shape correctness against repo references.

Runs on container GCU 1 only (caller verifies PCI and pins CPUs).
Outputs one JSON object per case on stdout, plus human-readable checks.
"""

import json
import math
import sys
import time

import torch
import torch_gcu  # noqa: F401

ROOT = "/workspace/agent-workspaces/msa-stage-profile"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/src")

torch.gcu.set_device(1)

from flag_attn.minimax_sparse_attention import (  # noqa: E402
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
    minimax_m3_sparse_attn,
)
from tests.flag_attn import test_minimax_sparse_attention as t  # noqa: E402


def summarize(name, actual, expected, *, topk_set=False):
    a = actual.float()
    e = expected.float()
    diff = (a - e).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    rms = float(torch.sqrt(torch.mean((a - e) ** 2)).item()) if diff.numel() else 0.0
    ref_rms = float(torch.sqrt(torch.mean(e ** 2)).item()) if diff.numel() else 0.0
    rel_rms = rms / ref_rms if ref_rms else 0.0
    finite = bool(torch.isfinite(a).all().item())
    row = {
        "case": name,
        "max_abs": max_abs,
        "rms": rms,
        "ref_rms": ref_rms,
        "relative_rms": rel_rms,
        "finite": finite,
        "nan": int(torch.isnan(a).sum().item()),
        "inf": int(torch.isinf(a).sum().item()),
    }
    if topk_set:
        row["exact_equal"] = bool(torch.equal(actual.to(torch.int32), expected.to(torch.int32)))
    print("RESULT " + json.dumps(row), flush=True)
    return row


def run_prefill(seq_len, prefix_len, label):
    t0 = time.perf_counter()
    data = t.make_data(
        (seq_len,), 1, 8, decode=False, prefix_lens=(prefix_len,), mode="bf16", seed=42
    )
    qlen = seq_len - prefix_len
    scores = minimax_m3_index_score(
        data.idx_q, data.index_kv_cache, data.block_table, data.cu_q,
        data.seq_lens, data.prefix_lens, qlen, data.max_seq_len, 1,
    )
    topk_idx = minimax_m3_index_topk(
        scores, data.cu_q, data.prefix_lens, qlen, 16, 1, 2
    )
    out = torch.empty_like(data.q)
    minimax_m3_sparse_attn(
        data.q, data.kv_cache, topk_idx, data.block_table, data.cu_q,
        data.seq_lens, data.prefix_lens, qlen, 1, data.sm_scale, out,
    )
    torch.gcu.synchronize()
    op_done = time.perf_counter()

    ref_scores = t._ref_index_score(data)
    ref_topk = t._ref_topk(ref_scores, data, 16, 1, 2)
    ref_out = t._ref_sparse_attn(data, ref_topk)
    torch.gcu.synchronize()
    ref_done = time.perf_counter()

    print(f"CASE {label} shape=(1,{qlen},{seq_len},1,8) op_s={op_done - t0:.1f} ref_s={ref_done - op_done:.1f}", flush=True)
    try:
        t._assert_score_match(scores, ref_scores, data, decode=False)
        t._assert_topk_match(topk_idx, ref_topk, data, 16, decode=False)
        t._assert_attention_match(out, ref_out, data)
        passed = True
        err = ""
    except AssertionError as exc:
        passed = False
        err = str(exc)
    summarize(f"{label}_score", scores, ref_scores)
    summarize(f"{label}_topk", topk_idx, ref_topk, topk_set=True)
    summarize(f"{label}_attn", out, ref_out)
    print("CHECK " + json.dumps({"case": label, "passed": passed, "error": err}), flush=True)


def run_decode(seq_len, label):
    t0 = time.perf_counter()
    data = t.make_data((seq_len,), 1, 8, decode=True, decode_qlen=1, mode="bf16", seed=42)
    topk_idx = minimax_m3_index_decode(
        data.idx_q, data.index_kv_cache, data.block_table, data.seq_lens,
        data.max_seq_len, 16, 1, 2, 1, 1, 1,
    )
    out = torch.empty_like(data.q)
    minimax_m3_sparse_attn(
        data.q, data.kv_cache, topk_idx, data.block_table, data.seq_lens,
        1, data.sm_scale, out, 1, stage="decode",
    )
    torch.gcu.synchronize()
    op_done = time.perf_counter()

    ref_topk = t._ref_decode_index(data, 16, 1, 2, 1)
    ref_out = t._ref_decode_attn(data, ref_topk, 1)
    torch.gcu.synchronize()
    ref_done = time.perf_counter()

    print(f"CASE {label} shape=(1,1,{seq_len},1,8) op_s={op_done - t0:.1f} ref_s={ref_done - op_done:.1f}", flush=True)
    try:
        t._assert_topk_match(topk_idx, ref_topk, data, 16, decode=True)
        t._assert_attention_match(out, ref_out, data)
        passed = True
        err = ""
    except AssertionError as exc:
        passed = False
        err = str(exc)
    summarize(f"{label}_topk", topk_idx, ref_topk, topk_set=True)
    summarize(f"{label}_attn", out, ref_out)
    print("CHECK " + json.dumps({"case": label, "passed": passed, "error": err}), flush=True)


print("DEVICE", torch.gcu.current_device(), torch.gcu.get_device_name(1), flush=True)
run_prefill(4096, 3584, "prefill_Q512_S4096")
run_prefill(4096, 3840, "prefill_Q256_S4096")
run_decode(16384, "decode_S16384")
run_decode(8192, "decode_S8192")
print("OK", flush=True)
