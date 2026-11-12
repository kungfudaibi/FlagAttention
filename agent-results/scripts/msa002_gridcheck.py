"""Verify the actual launch grid of A vs B for the multi-batch A/B rows."""
import json
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

LAUNCHES = []
_orig_run = JITFunction.run


def _run(self, *a, **kw):
    if kw.get("warmup") is False:
        grid = kw.get("grid")
        LAUNCHES.append((self.__name__, list(grid) if isinstance(grid, (list, tuple)) else str(grid)))
    return _orig_run(self, *a, **kw)


JITFunction.run = _run

CASES = [(1, 128, 4096), (1, 384, 4096), (1, 512, 4096), (1, 1024, 4096),
         (1, 512, 16384), (2, 256, 4096), (4, 128, 4096), (8, 128, 4096)]

for batch, qlen, seq_len in CASES:
    g = torch.Generator(device="cpu").manual_seed(0)
    data = bench.make_data(batch, seq_len, 1, 8, torch.device("gcu:1"), "bf16",
                           decode=False, decode_qlen=1, query_len=qlen,
                           randomize_pages=True, generator=g)
    total_q = data.q.shape[0]
    out = torch.empty_like(data.q)
    for enabled, tag in ((False, "A_shipped"), (True, "B_group16")):
        enflame_sparse._MSA002_GROUP16_EXPERIMENT = enabled
        s = minimax_m3_index_score(data.idx_q, data.index_kv_cache, data.block_table,
                                   data.cu_q, data.seq_lens, data.prefix_lens, qlen,
                                   seq_len, 1)
        tk = minimax_m3_index_topk(s, data.cu_q, data.prefix_lens, qlen, 16, 1, 2)
        LAUNCHES.clear()
        minimax_m3_sparse_attn(data.q, data.kv_cache, tk, data.block_table, data.cu_q,
                               data.seq_lens, data.prefix_lens, qlen, 1,
                               data.sm_scale, out)
        torch.gcu.synchronize()
        attn = [x for x in LAUNCHES if "topk_rows" not in x[0] and "index_block" not in x[0]]
        print(json.dumps({"probe": "grid", "batch": batch, "qlen": qlen,
                          "seq_len": seq_len, "total_q": total_q, "variant": tag,
                          "grid": attn}), flush=True)
    enflame_sparse._MSA002_GROUP16_EXPERIMENT = False
    del data, out, s, tk
    torch.gcu.empty_cache()
print("DONE", flush=True)
