import json, statistics
import torch, torch_gcu
import torch.nn.functional as F
from flag_attn.testing import backend
from flag_attn.sage_attention import per_block_int8
torch.gcu.set_device(1)

def metrics(actual, expected):
    a=actual.float().flatten(); e=expected.float().flatten()
    return {"cosine": F.cosine_similarity(a,e,dim=0).item(),
            "rel_l2": ((a-e).norm()/e.norm().clamp_min(1e-8)).item()}

for T in (512, 4096):
    torch.manual_seed(42)
    q=torch.randn(1,2,T,64,device="gcu:1",dtype=torch.float16)
    k=torch.randn_like(q); v=torch.randn_like(q)
    f=lambda: F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)
    vals=backend.do_bench(f,warmup=10,rep=50,return_mode="all")
    print(json.dumps({"T":T,"do_bench_n":len(vals),"median":statistics.median(vals),"min":min(vals),"max":max(vals),"mean":statistics.mean(vals)}),flush=True)
    allv=backend.do_bench(f,warmup=10,rep=50,return_mode="all")
    print(json.dumps({"T":T,"second_run_median":statistics.median(allv),"n":len(allv)}),flush=True)
    # explicit flash and math
    from torch.nn.attention import sdpa_kernel, SDPBackend
    for b in (SDPBackend.FLASH_ATTENTION, SDPBackend.MATH):
        try:
            with sdpa_kernel(b):
                vv=backend.do_bench(f,warmup=10,rep=50,return_mode="all")
            print(json.dumps({"T":T,"backend":str(b),"median":statistics.median(vv),"min":min(vv),"n":len(vv)}),flush=True)
        except Exception as exc:
            print(json.dumps({"T":T,"backend":str(b),"error":repr(exc)}),flush=True)
