import sys, torch, torch_gcu  # noqa: F401
ROOT = "/workspace/agent-workspaces/msa-stage-profile"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/src")
torch.gcu.set_device(1)
from benchmark import test_minimax_sparse_attention_benchmark as bench
for _n in ("randn", "rand", "randperm", "randint"):
    _o = getattr(torch, _n)
    def _mk(orig):
        def w(*a, **kw):
            g, d = kw.get("generator"), kw.get("device")
            if g is not None and d is not None and str(g.device) == "cpu" and str(d).startswith("gcu"):
                kw["device"] = "cpu"
                return orig(*a, **kw).to(d)
            return orig(*a, **kw)
        return w
    setattr(torch, _n, _mk(_o))
g = torch.Generator(device="cpu").manual_seed(3)
d = bench.make_data(1, 16384, 1, 8, torch.device("gcu:1"), "bf16", decode=True,
                    decode_qlen=1, query_len=1, randomize_pages=True, generator=g)
for a in ("q", "idx_q", "kv_cache", "index_kv_cache", "block_table", "seq_lens", "prefix_lens"):
    t = getattr(d, a)
    print(a, tuple(t.shape), t.dtype)
print("sm_scale", d.sm_scale)
