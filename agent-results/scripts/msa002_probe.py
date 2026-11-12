"""MSA-002 probe: profiler activities, event resolution, launch logging."""

import sys
import time

import torch
import torch_gcu  # noqa: F401

sys.path.insert(0, "/workspace/agent-workspaces/msa-stage-profile")
sys.path.insert(0, "/workspace/agent-workspaces/msa-stage-profile/src")

torch.gcu.set_device(1)
from torch.profiler import ProfilerActivity, profile  # noqa: E402

print("activities:", [n for n in dir(ProfilerActivity) if not n.startswith("_")])

# Event resolution / overhead
for _ in range(5):
    x = torch.ones(1024, device="gcu:1")
torch.gcu.synchronize()
empty = []
for _ in range(20):
    a = torch.gcu.Event(enable_timing=True)
    b = torch.gcu.Event(enable_timing=True)
    a.record()
    b.record()
    torch.gcu.synchronize()
    empty.append(a.elapsed_time(b))
print("empty event pair ms:", sorted(empty)[:3], sorted(empty)[-3:])

# One tiny op, warm
x = torch.randn(1024, 1024, device="gcu:1", dtype=torch.bfloat16)
for _ in range(5):
    y = x * 2
torch.gcu.synchronize()
samples = []
for _ in range(20):
    a = torch.gcu.Event(enable_timing=True)
    b = torch.gcu.Event(enable_timing=True)
    a.record()
    y = x * 2
    b.record()
    torch.gcu.synchronize()
    samples.append(a.elapsed_time(b))
print("tiny mul event ms median:", sorted(samples)[len(samples) // 2], "min", min(samples), "max", max(samples))

# Launch logging check on one prefill call
from triton.runtime.jit import JITFunction  # noqa: E402

launches = []
orig = JITFunction.run


def wrapped(self, *args, **kwargs):
    if kwargs.get("warmup") is False or kwargs.get("warmup") is None:
        grid = kwargs.get("grid")
        launches.append((self.__name__, tuple(grid) if isinstance(grid, (tuple, list)) else grid))
    return orig(self, *args, **kwargs)


JITFunction.run = wrapped

from benchmark import test_minimax_sparse_attention_benchmark as bench  # noqa: E402
from flag_attn import minimax_m3_sparse_attn  # noqa: E402
from flag_attn.minimax_sparse_attention import (  # noqa: E402
    minimax_m3_index_score,
    minimax_m3_index_topk,
)

for name in ("randn", "rand", "randperm", "randint"):
    original = getattr(torch, name)

    def make_wrapped(_original):
        def w(*a, **kw):
            gen = kw.get("generator")
            dev = kw.get("device")
            if gen is not None and dev is not None and str(gen.device) == "cpu" and str(dev).startswith("gcu"):
                kw["device"] = "cpu"
                return _original(*a, **kw).to(dev)
            return _original(*a, **kw)
        return w

    setattr(torch, name, make_wrapped(original))

gen = torch.Generator(device="cpu").manual_seed(0)
data = bench.make_data(1, 4096, 1, 8, torch.device("gcu:1"), "bf16", decode=False,
                       decode_qlen=1, query_len=512, randomize_pages=True, generator=gen)
launches.clear()
scores = minimax_m3_index_score(data.idx_q, data.index_kv_cache, data.block_table,
                                data.cu_q, data.seq_lens, data.prefix_lens, 512, 4096, 1)
topk_idx = minimax_m3_index_topk(scores, data.cu_q, data.prefix_lens, 512, 16, 1, 2)
out = torch.empty_like(data.q)
minimax_m3_sparse_attn(data.q, data.kv_cache, topk_idx, data.block_table, data.cu_q,
                       data.seq_lens, data.prefix_lens, 512, 1, 128 ** -0.5, out)
torch.gcu.synchronize()
print("launches:")
for name, grid in launches:
    print("  ", name, grid)

print("state:", data.q.shape, out.shape, out.dtype)
print("finite:", torch.isfinite(out.float()).all().item())

# profiler smoke on one call
JITFunction.run = orig
try:
    with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
        minimax_m3_sparse_attn(data.q, data.kv_cache, topk_idx, data.block_table, data.cu_q,
                               data.seq_lens, data.prefix_lens, 512, 1, 128 ** -0.5, out)
        torch.gcu.synchronize()
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
    ev = list(prof.events())
    print("event count:", len(ev))
    print("names:", sorted({e.name for e in ev})[:20])
    print("device durations:", [getattr(e, "device_time", None) for e in ev][:10])
except Exception as exc:  # noqa: BLE001
    print("PROFILER_FAILED", type(exc).__name__, exc)
