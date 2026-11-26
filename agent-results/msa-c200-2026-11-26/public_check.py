import functools
import json
import os
import statistics
from contextlib import contextmanager

import torch
import torch_gcu
import triton
from benchmark import test_minimax_sparse_attention_benchmark as b

DEV = 5
torch.gcu.set_device(DEV)
WORKLOAD = (1, 1024, 16384, 1, 8)
PACKED = "FLAG_ATTN_MSA_PACKED_K"
EARLY = "FLAG_ATTN_MSA_EARLY_V_DMA"

@contextmanager
def mode(name):
    old = {k: os.environ.get(k) for k in (PACKED, EARLY)}
    try:
        for k in old:
            os.environ.pop(k, None)
        if name == "packed":
            os.environ[PACKED] = "1"
        elif name == "early":
            os.environ[EARLY] = "1"
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

def make_case(shape):
    batch, q_len, seq, kvh, heads = shape
    args = b.MSABenchmarkArgs()
    index = b.PREFILL_SHAPES.index(shape)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + index)
    data = b.make_data(batch, seq, kvh, heads, torch.device(f"gcu:{DEV}"), "bf16",
                       decode=False, decode_qlen=args.decode_qlen, query_len=q_len,
                       randomize_pages=not args.identity_pages, generator=generator)
    args_tuple = (data, seq, kvh, args.topk, args.init_blocks, args.local_blocks)
    return data, lambda out: b.run_prefill(
        b.minimax_m3_index_score, b.minimax_m3_index_topk,
        b.minimax_m3_sparse_attn, *args_tuple, out,
    )

def run_case(shape, measure):
    data, call = make_case(shape)
    modes = ("stock", "early", "packed")
    outs = {name: torch.empty_like(data.q) for name in modes}
    routes = {name: [] for name in modes}
    active_mode = None
    orig = triton.runtime.jit.JITFunction.run
    def record(self, *a, **kw):
        n = getattr(self, "__name__", "")
        if active_mode is not None and n in (
            "_gqa_sparse_common_prefix", "_gqa_sparse_fwd_direct", "_gqa_sparse_fwd_grouped8"
        ) and not kw.get("warmup", False):
            routes[active_mode].append((n, kw.get("USE_PACKED_K"), kw.get("EARLY_V_DMA")))
        return orig(self, *a, **kw)
    triton.runtime.jit.JITFunction.run = record
    try:
        for name in modes:
            active_mode = name
            with mode(name):
                call(outs[name])
                torch.gcu.synchronize()
    finally:
        triton.runtime.jit.JITFunction.run = orig
    print("routes", shape, routes, flush=True)
    assert all(torch.isfinite(out).all().item() for out in outs.values())
    assert torch.equal(outs["stock"], outs["early"])
    assert torch.equal(outs["stock"], outs["packed"])
    if measure:
        assert ("_gqa_sparse_fwd_direct", False, False) in routes["stock"]
        assert ("_gqa_sparse_fwd_direct", False, True) in routes["early"]
        assert ("_gqa_sparse_fwd_direct", True, True) in routes["packed"]
    else:
        assert all(any(k == "_gqa_sparse_fwd_grouped8" for k, *_ in route) for route in routes.values())
        return
    for name in modes:
        with mode(name):
            for _ in range(3):
                call(outs[name])
            torch.gcu.synchronize()
    samples = {name: [] for name in modes}
    for i in range(9):
        for name in modes[i % 3:] + modes[:i % 3]:
            start = torch.gcu.Event(enable_timing=True)
            end = torch.gcu.Event(enable_timing=True)
            with mode(name):
                start.record()
                call(outs[name])
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end))
    print("SAMPLES", json.dumps(samples), flush=True)
    print("MEDIANS", json.dumps({n: statistics.median(s) for n, s in samples.items()}), flush=True)
    spans = {}
    def time_kernel(self, *a, **kw):
        n = getattr(self, "__name__", "")
        if n not in ("_gqa_sparse_common_prefix", "_gqa_sparse_fwd_direct") or kw.get("warmup", False):
            return orig(self, *a, **kw)
        start = torch.gcu.Event(enable_timing=True)
        end = torch.gcu.Event(enable_timing=True)
        start.record()
        result = orig(self, *a, **kw)
        end.record()
        spans.setdefault(n, []).append((start, end))
        return result
    triton.runtime.jit.JITFunction.run = time_kernel
    try:
        with mode("packed"):
            for _ in range(5):
                call(outs["packed"])
            torch.gcu.synchronize()
    finally:
        triton.runtime.jit.JITFunction.run = orig
    print("STAGES_PACKED", json.dumps({k:[a.elapsed_time(z) for a,z in events]
                                       for k,events in spans.items()}),flush=True)

if __name__ == "__main__":
    print("DEVICE",torch.gcu.current_device(),torch.__version__,triton.__version__,flush=True)
    run_case(WORKLOAD, True)
    run_case((1,512,4096,1,8), False)
    print("PASS", flush=True)
