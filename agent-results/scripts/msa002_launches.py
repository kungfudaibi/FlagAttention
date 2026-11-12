"""MSA-002 launch accounting: real Triton kernel names/grids per stage.

Uses JITFunction.run interception with warmup=False (a real launch). Also
records the host-side enqueue time per stage so wrapper overhead is visible.
"""

import json
import sys
import time

import torch
import torch_gcu  # noqa: F401

ROOT = "/workspace/agent-workspaces/msa-stage-profile"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/src")
torch.gcu.set_device(1)

from triton.runtime.jit import JITFunction  # noqa: E402

from benchmark import test_minimax_sparse_attention_benchmark as bench  # noqa: E402
from flag_attn import minimax_m3_sparse_attn  # noqa: E402
from flag_attn.minimax_sparse_attention import (  # noqa: E402
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)

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

RECORDS = []
_orig_run = JITFunction.run


def run(self, *args, **kwargs):
    warmup = kwargs.get("warmup")
    if warmup is False:
        grid = kwargs.get("grid")
        RECORDS.append({"kernel": self.__name__,
                        "grid": list(grid) if isinstance(grid, (tuple, list)) else str(grid)})
    return _orig_run(self, *args, **kwargs)


JITFunction.run = run


def count(label, fn):
    fn()
    torch.gcu.synchronize()
    RECORDS.clear()
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    torch.gcu.synchronize()
    t2 = time.perf_counter()
    print(json.dumps({"probe": label, "kernels": list(RECORDS),
                      "n_kernels": len(RECORDS),
                      "host_enqueue_ms": (t1 - t0) * 1000,
                      "sync_wait_ms": (t2 - t1) * 1000}), flush=True)


# prefill Q512/S4096/topk16
g = torch.Generator(device="cpu").manual_seed(0)
data = bench.make_data(1, 4096, 1, 8, torch.device("gcu:1"), "bf16", decode=False,
                       decode_qlen=1, query_len=512, randomize_pages=True, generator=g)
out = torch.empty_like(data.q)
c = {}


def p_score():
    c["s"] = minimax_m3_index_score(data.idx_q, data.index_kv_cache, data.block_table,
                                    data.cu_q, data.seq_lens, data.prefix_lens, 512, 4096, 1)


def p_topk():
    c["t"] = minimax_m3_index_topk(c["s"], data.cu_q, data.prefix_lens, 512, 16, 1, 2)


def p_attn():
    minimax_m3_sparse_attn(data.q, data.kv_cache, c["t"], data.block_table, data.cu_q,
                           data.seq_lens, data.prefix_lens, 512, 1, data.sm_scale, out)


def p_whole():
    p_score(); p_topk(); p_attn()


for label, fn in (("prefill_score", p_score), ("prefill_topk", p_topk),
                  ("prefill_attn", p_attn), ("prefill_whole", p_whole)):
    count(label, fn)

# topk variations to expose the path switch
for tk in (4, 8, 16, 32):
    scores = minimax_m3_index_score(data.idx_q, data.index_kv_cache, data.block_table,
                                    data.cu_q, data.seq_lens, data.prefix_lens, 512, 4096, 1)
    ti = minimax_m3_index_topk(scores, data.cu_q, data.prefix_lens, 512, tk, 1, 2)
    o2 = torch.empty_like(data.q)
    fn = lambda ti=ti, o2=o2: minimax_m3_sparse_attn(
        data.q, data.kv_cache, ti, data.block_table, data.cu_q, data.seq_lens,
        data.prefix_lens, 512, 1, data.sm_scale, o2)
    count(f"prefill_attn_topk{tk}", fn)

# decode
gd = torch.Generator(device="cpu").manual_seed(100000)
d = bench.make_data(1, 16384, 1, 8, torch.device("gcu:1"), "bf16", decode=True,
                    decode_qlen=1, query_len=1, randomize_pages=True, generator=gd)
dout = torch.empty_like(d.q)
dc = {}


def d_index():
    dc["t"] = minimax_m3_index_decode(d.idx_q, d.index_kv_cache, d.block_table, d.seq_lens,
                                      16384, 16, 1, 2, 1, 1, 1)


def d_attn():
    minimax_m3_sparse_attn(d.q, d.kv_cache, dc["t"], d.block_table, d.seq_lens,
                           1, d.sm_scale, dout, 1, stage="decode")


def d_whole():
    d_index(); d_attn()


for label, fn in (("decode_index", d_index), ("decode_attn", d_attn), ("decode_whole", d_whole)):
    count(label, fn)

# decode path variants by sequence length (identity vs scored topk vs rows)
for s in (2048, 4096, 16384):
    gx = torch.Generator(device="cpu").manual_seed(7)
    dx = bench.make_data(1, s, 1, 8, torch.device("gcu:1"), "bf16", decode=True,
                         decode_qlen=1, query_len=1, randomize_pages=True, generator=gx)
    ox = torch.empty_like(dx.q)
    box = {}
    f1 = lambda dx=dx, box=box: box.update(t=minimax_m3_index_decode(
        dx.idx_q, dx.index_kv_cache, dx.block_table, dx.seq_lens, s, 16, 1, 2, 1, 1, 1))
    f1(); torch.gcu.synchronize()
    count(f"decode_index_S{s}", f1)
    count(f"decode_attn_S{s}", lambda dx=dx, box=box, ox=ox: minimax_m3_sparse_attn(
        dx.q, dx.kv_cache, box["t"], dx.block_table, dx.seq_lens, 1, dx.sm_scale, ox, 1,
        stage="decode"))
print("DONE", flush=True)
