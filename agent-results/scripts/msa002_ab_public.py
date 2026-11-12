"""MSA-002 phase 3: A/B of the relaxed group-16 gate via the public API.

Variant A (shipped): MSA002_GROUP16_EXPERIMENT off -> GROUP_SIZE_Q = 8 at
Q=512/S=4096/topk=16 because the dense-pool gate requires
cdiv(total_q,16)*num_kv_heads >= 64 programs (this GCU has 2 multi-processors).
Variant B (experiment): same code with the opt-in gate -> GROUP_SIZE_Q = 16.

Rounds are interleaved A,B,A,B,... to cancel drift. Each variant's full output
is checked against the repo's independent PyTorch reference and against the
other variant. Kernel name/grid is captured for both so the switch is proven.
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

LAUNCHES = []
_orig_run = JITFunction.run


def _run(self, *a, **kw):
    if kw.get("warmup") is False and "grouped8" in self.__name__:
        grid = kw.get("grid")
        LAUNCHES.append((self.__name__, list(grid) if isinstance(grid, (list, tuple)) else str(grid)))
    return _orig_run(self, *a, **kw)


JITFunction.run = _run


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


def set_variant(enabled):
    enflame_sparse._MSA002_GROUP16_EXPERIMENT = bool(enabled)


CASES = [(128, 4096), (512, 4096), (1024, 4096), (512, 16384), (2048, 4096)]
ROUNDS = 3

for qlen, seq_len in CASES:
    g = torch.Generator(device="cpu").manual_seed(0)
    data = bench.make_data(1, seq_len, 1, 8, torch.device("gcu:1"), "bf16",
                           decode=False, decode_qlen=1, query_len=qlen,
                           randomize_pages=True, generator=g)
    scores = minimax_m3_index_score(
        data.idx_q, data.index_kv_cache, data.block_table, data.cu_q,
        data.seq_lens, data.prefix_lens, qlen, seq_len, 1)
    topk_idx = minimax_m3_index_topk(scores, data.cu_q, data.prefix_lens, qlen, 16, 1, 2)
    outs = {}

    def attn(out):
        minimax_m3_sparse_attn(data.q, data.kv_cache, topk_idx, data.block_table,
                               data.cu_q, data.seq_lens, data.prefix_lens, qlen, 1,
                               data.sm_scale, out)

    for enabled, tag in ((False, "A_shipped"), (True, "B_group16")):
        set_variant(enabled)
        out = torch.empty_like(data.q)
        LAUNCHES.clear()
        attn(out)
        torch.gcu.synchronize()
        outs[tag] = out.clone()
        print(json.dumps({"probe": "variant", "qlen": qlen, "seq_len": seq_len,
                          "variant": tag, "kernels": list(LAUNCHES)}), flush=True)

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

    for tag in ("A_shipped", "B_group16"):
        o = outs[tag]
        d = (o.float() - ref.float()).abs()
        rms = float(torch.sqrt(torch.mean((o.float() - ref.float()) ** 2)).item())
        rrms = float(torch.sqrt(torch.mean(ref.float() ** 2)).item())
        print(json.dumps({"probe": "correctness", "qlen": qlen, "seq_len": seq_len,
                          "variant": tag, "max_abs": float(d.max().item()),
                          "relative_rms": rms / rrms if rrms else 0.0,
                          "finite": bool(torch.isfinite(o.float()).all().item()),
                          "allclose_2e-2": bool(torch.allclose(o, ref, atol=2e-2, rtol=2e-2))}),
              flush=True)
    cross = float((outs["A_shipped"].float() - outs["B_group16"].float()).abs().max().item())
    print(json.dumps({"probe": "cross_variant", "qlen": qlen, "seq_len": seq_len,
                      "max_abs": cross}), flush=True)

    rounds = {"A_shipped": [], "B_group16": []}
    for rnd in range(ROUNDS):
        for tag, enabled in (("A_shipped", False), ("B_group16", True)):
            set_variant(enabled)
            out = torch.empty_like(data.q)
            ms = timed(lambda out=out: attn(out))
            rounds[tag].append(ms)
        print(json.dumps({"probe": "ab_round", "qlen": qlen, "seq_len": seq_len,
                          "round": rnd, "A_ms": rounds["A_shipped"][-1],
                          "B_ms": rounds["B_group16"][-1]}), flush=True)
    a = statistics.median(rounds["A_shipped"])
    b = statistics.median(rounds["B_group16"])
    print(json.dumps({"probe": "ab_summary", "qlen": qlen, "seq_len": seq_len,
                      "A_ms": a, "B_ms": b, "A_rounds": rounds["A_shipped"],
                      "B_rounds": rounds["B_group16"], "speedup_B_over_A": a / b}),
          flush=True)
    set_variant(False)
    del data, scores, topk_idx, outs, ref
    torch.gcu.empty_cache()

print("DONE", flush=True)
