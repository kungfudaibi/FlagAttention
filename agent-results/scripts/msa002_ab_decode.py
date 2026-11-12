"""MSA-002 phase 3: decode index-score chunk-cap A/B (default off vs on).

Variant A = shipped (cap 0 -> num_kv_chunks = min(128, pages, 512//batch)).
Variant B = cap 32 (or 16) -> fewer, longer-running CTAs on the score kernel.

Checks, in order, before any timing claim:
  * score tensor bit-exact vs shipped
  * selected top-k indices identical
  * final attention output identical
Rounds are interleaved A,B,A,B,... to cancel drift.
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
from flag_attn import minimax_m3_sparse_attn_decode  # noqa: E402
import flag_attn.runtime.backend._enflame.msa.index_topk as enflame_index  # noqa: E402

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
    if kw.get("warmup") is False and "decode" in self.__name__ and "score" in self.__name__:
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


CASES = [(1, 16384)]
CAPS = [32, 16]
ROUNDS = 3

# Health probe: fail fast if the device did not recover from the earlier SIP
# exception instead of producing a misleading partial row.
torch.gcu.synchronize()
_h = torch.ones(1024, device="gcu:1")
assert float(_h.sum().item()) == 1024.0
print(json.dumps({"probe": "health", "sum": float(_h.sum().item())}), flush=True)

for batch, seq_len in CASES:
    g = torch.Generator(device="cpu").manual_seed(3)
    data = bench.make_data(batch, seq_len, 1, 8, torch.device("gcu:1"), "bf16",
                           decode=True, decode_qlen=1, query_len=1,
                           randomize_pages=True, generator=g)
    out = torch.empty_like(data.q)
    ref = {}

    def run_index():
        return enflame_index.minimax_m3_index_decode(
            data.idx_q, data.index_kv_cache, data.block_table, data.seq_lens,
            seq_len, 16, 1, 2, 1, 1, 1)

    def run_whole():
        tk = enflame_index.minimax_m3_index_decode(
            data.idx_q, data.index_kv_cache, data.block_table, data.seq_lens,
            seq_len, 16, 1, 2, 1, 1, 1)
        minimax_m3_sparse_attn_decode(
            data.q, data.kv_cache, tk, data.block_table, data.seq_lens,
            8, data.sm_scale, out, 1)

    enflame_index._MSA002_DECODE_CHUNK_CAP = 0
    LAUNCHES.clear()
    base_score = run_index().clone()
    torch.gcu.synchronize()
    print(json.dumps({"probe": "shipped", "batch": batch, "seq_len": seq_len,
                      "grid": list(LAUNCHES)}), flush=True)
    base_topk = None
    enflame_index._MSA002_DECODE_CHUNK_CAP = 0
    run_whole(); torch.gcu.synchronize()
    base_out = out.clone()

    variants = [("A_shipped", 0)] + [(f"B_cap{c}", c) for c in CAPS]
    for tag, cap in variants:
        enflame_index._MSA002_DECODE_CHUNK_CAP = cap
        LAUNCHES.clear()
        score = run_index()
        torch.gcu.synchronize()
        run_whole(); torch.gcu.synchronize()
        grids = list(LAUNCHES)
        same_score = torch.equal(score[:, :, :score.shape[-1]], base_score)
        same_out = torch.equal(out, base_out)
        print(json.dumps({"probe": "variant", "batch": batch, "seq_len": seq_len,
                          "variant": tag, "cap": cap, "grid": grids,
                          "score_bit_exact": bool(same_score),
                          "whole_output_bit_exact": bool(same_out)}), flush=True)

    rounds = {tag: [] for tag, _ in variants}
    for rnd in range(ROUNDS):
        for tag, cap in variants:
            enflame_index._MSA002_DECODE_CHUNK_CAP = cap
            rounds[tag].append(timed(run_whole))
    for tag, cap in variants:
        enflame_index._MSA002_DECODE_CHUNK_CAP = cap
        idx_rounds = [timed(run_index) for _ in range(2)]
        print(json.dumps({"probe": "stage", "batch": batch, "seq_len": seq_len,
                          "variant": tag, "cap": cap,
                          "index_ms": statistics.median(idx_rounds)}), flush=True)
    enflame_index._MSA002_DECODE_CHUNK_CAP = 0
    row = {"batch": batch, "seq_len": seq_len,
           "rounds": {tag: rounds[tag] for tag, _ in variants},
           "median": {tag: statistics.median(rounds[tag]) for tag, _ in variants}}
    row["speedup"] = {tag: row["median"]["A_shipped"] / row["median"][tag]
                      for tag, _ in variants if tag != "A_shipped"}
    print(json.dumps({"probe": "ab", **row}), flush=True)
    del data, out, base_score, base_out
    torch.gcu.empty_cache()

print("DONE", flush=True)
