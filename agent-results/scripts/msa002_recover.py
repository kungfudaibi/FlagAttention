"""Post-fault health probe: does the shipped public decode path still run?"""
import json
import sys
import torch
import torch_gcu  # noqa: F401

ROOT = "/workspace/agent-workspaces/msa-stage-profile"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/src")
torch.gcu.set_device(1)

from benchmark import test_minimax_sparse_attention_benchmark as bench  # noqa: E402
from flag_attn import minimax_m3_sparse_attn_decode  # noqa: E402
from flag_attn.minimax_sparse_attention import minimax_m3_index_decode  # noqa: E402

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

torch.gcu.synchronize()
print(json.dumps({"probe": "health", "sum": float(torch.ones(1024, device="gcu:1").sum().item())}), flush=True)

for seq_len in (4096, 16384):
    g = torch.Generator(device="cpu").manual_seed(3)
    data = bench.make_data(1, seq_len, 1, 8, torch.device("gcu:1"), "bf16",
                           decode=True, decode_qlen=1, query_len=1,
                           randomize_pages=True, generator=g)
    out = torch.empty_like(data.q)
    tk = minimax_m3_index_decode(
        data.idx_q, data.index_kv_cache, data.block_table, data.seq_lens,
        seq_len, 16, 1, 2, 8, 1, 1)
    torch.gcu.synchronize()
    print(json.dumps({"probe": "index_ok", "seq_len": seq_len, "topk_shape": list(tk.shape)}), flush=True)
    minimax_m3_sparse_attn_decode(
        data.q, data.kv_cache, tk, data.block_table, data.seq_lens,
        8, data.sm_scale, out, 1)
    torch.gcu.synchronize()
    print(json.dumps({"probe": "attn_ok", "seq_len": seq_len,
                      "finite": bool(torch.isfinite(out.float()).all().item()),
                      "absmax": float(out.float().abs().max().item())}), flush=True)
    del data, out, tk
    torch.gcu.empty_cache()
print("DONE", flush=True)
