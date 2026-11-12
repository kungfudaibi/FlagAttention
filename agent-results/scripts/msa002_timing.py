"""MSA-002 phase 2: stage-only vs whole-call timing, launches, memory.

Protocol
--------
* One case per process invocation (avoids cross-shape allocator/compiler noise).
* Device: torch.gcu device 1 == efsmi DEV 1 == PCI 0000:83:00.0 (preflight checked).
* Stage-only: Event pair around exactly one stage call, sync at both ends,
  3 rounds x 20 samples; median and raw samples recorded.
* Whole-call: Event pair around score+topk+attn (prefill) / index+attn (decode),
  same protocol.
* Benchmark per_step numbers from bench.do_bench are collected in a separate
  script for cross-checking; they are NOT summed here.

Usage: msa002_timing.py prefill|decode
"""

import json
import statistics
import sys
import time

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
    minimax_m3_index_score,
    minimax_m3_index_topk,
)

MODE = sys.argv[1]
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
SAMPLES = int(sys.argv[3]) if len(sys.argv) > 3 else 20

# Bridge CPU generators exactly like experiments/msa_eval.py.
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


def timed(fn, samples=SAMPLES):
    """Event-pair timing with a synchronize at both ends."""
    fn()
    torch.gcu.synchronize()
    vals = []
    for _ in range(samples):
        a = torch.gcu.Event(enable_timing=True)
        b = torch.gcu.Event(enable_timing=True)
        torch.gcu.synchronize()
        a.record()
        fn()
        b.record()
        torch.gcu.synchronize()
        vals.append(a.elapsed_time(b))
    vals.sort()
    return {
        "median": statistics.median(vals),
        "min": vals[0],
        "p10": vals[max(0, len(vals) // 10)],
        "p90": vals[min(len(vals) - 1, 9 * len(vals) // 10)],
        "max": vals[-1],
        "samples": vals,
    }


def make(shape, decode, seed):
    batch, query_len, seq_len, kvh, heads = shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return bench.make_data(
        batch, seq_len, kvh, heads, torch.device("gcu:1"), "bf16",
        decode=decode, decode_qlen=1, query_len=query_len,
        randomize_pages=True, generator=gen,
    )


def main():
    if MODE == "prefill":
        shape = (1, 512, 4096, 1, 8)
        query_len = shape[1]
        data = make(shape, False, 0)
        out = torch.empty_like(data.q)
        scores = {}
        topk_idx = {}

        def index_score():
            scores["v"] = minimax_m3_index_score(
                data.idx_q, data.index_kv_cache, data.block_table, data.cu_q,
                data.seq_lens, data.prefix_lens, query_len, shape[2], 1)

        def index_topk():
            topk_idx["v"] = minimax_m3_index_topk(
                scores["v"], data.cu_q, data.prefix_lens, query_len, 16, 1, 2)

        def attention():
            minimax_m3_sparse_attn(
                data.q, data.kv_cache, topk_idx["v"], data.block_table, data.cu_q,
                data.seq_lens, data.prefix_lens, query_len, 1, data.sm_scale, out)

        def whole():
            index_score()
            index_topk()
            attention()

        stages = [("index_score", index_score), ("index_topk", index_topk),
                  ("attention", attention), ("whole_call", whole)]
    else:
        shape = (1, 1, 16384, 1, 8)
        data = make(shape, True, 100000)
        out = torch.empty_like(data.q)
        topk_idx = {}

        def index_decode():
            topk_idx["v"] = minimax_m3_index_decode(
                data.idx_q, data.index_kv_cache, data.block_table, data.seq_lens,
                shape[2], 16, 1, 2, 1, 1, 1)

        def attention():
            minimax_m3_sparse_attn(
                data.q, data.kv_cache, topk_idx["v"], data.block_table,
                data.seq_lens, 1, data.sm_scale, out, 1, stage="decode")

        def whole():
            index_decode()
            attention()

        stages = [("index_decode", index_decode), ("attention", attention),
                  ("whole_call", whole)]

    # Compile / first-call latency, then a warm-up pass.
    print(json.dumps({"mode": MODE, "shape": shape, "kind": "meta"}), flush=True)
    for name, fn in stages:
        torch.gcu.synchronize()
        torch.gcu.reset_peak_memory_stats(1)
        t0 = time.perf_counter()
        fn()
        torch.gcu.synchronize()
        first = (time.perf_counter() - t0) * 1000
        peak = torch.gcu.max_memory_allocated(1) / (1024 ** 2)
        print(json.dumps({"mode": MODE, "stage": name, "kind": "first_call",
                          "first_call_ms": first, "peak_mib": peak}), flush=True)

    for r in range(ROUNDS):
        for name, fn in stages:
            res = timed(fn, SAMPLES)
            res.update({"mode": MODE, "stage": name, "kind": "steady", "round": r})
            print(json.dumps(res), flush=True)

    # Launch counts (one call each, after compilation).
    from triton.runtime.jit import JITFunction
    launches = []
    orig = JITFunction.run

    def wrapped(self, *a, **kw):
        if kw.get("warmup") is None:
            grid = kw.get("grid")
            launches.append((self.__name__, list(grid) if isinstance(grid, (tuple, list)) else str(grid)))
        return orig(self, *a, **kw)

    JITFunction.run = wrapped
    try:
        for name, fn in stages:
            launches.clear()
            fn()
            torch.gcu.synchronize()
            print(json.dumps({"mode": MODE, "stage": name, "kind": "launches",
                              "launches": launches}), flush=True)
    finally:
        JITFunction.run = orig


main()
