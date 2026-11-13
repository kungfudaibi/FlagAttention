import json, statistics, torch, torch_gcu
import torch.nn.functional as F
from flag_attn import sage_attention
from flag_attn.sage_attention import per_block_int8

torch.gcu.set_device(1)

def timed(fn, n=20, warm=5):
    for _ in range(warm): fn()
    torch.gcu.synchronize()
    vals=[]
    for _ in range(n):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        s.record(); out=fn(); e.record(); torch.gcu.synchronize(); vals.append(s.elapsed_time(e))
    return vals, out

def acc(a, b):
    a=a.float().flatten(); b=b.float().flatten()
    if a.numel()>1_048_576:
        st=(a.numel()+1_048_576-1)//1_048_576; a=a[::st]; b=b[::st]
    d=a-b
    return {"cosine": F.cosine_similarity(a,b,dim=0).item(),
            "rel_l2": (d.norm()/b.norm().clamp_min(1e-8)).item(),
            "max_abs": d.abs().max().item(),
            "finite": bool(torch.isfinite(a).all().item())}

def sdpa_ref(q,k,v):
    return F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)

def flash(q,k,v):
    return torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)[0]

def efficient(q,k,v):
    return torch.ops.aten._scaled_dot_product_efficient_attention(q,k,v,None,False,0.0,False)[0]

for T in (512,1024,2048,4096):
    torch.manual_seed(42)
    q=torch.randn(1,2,T,64,device="gcu:1",dtype=torch.float16)
    k=torch.randn_like(q); v=torch.randn_like(q)
    ref=sdpa_ref(q,k,v)
    fo=flash(q,k,v); eo=efficient(q,k,v)
    q8,qs,k8,ks=per_block_int8(q,k)
    so,_=sage_attention(q8,k8,v,qs,ks,output_dtype=torch.float16)
    def full():
        a,b,c,d=per_block_int8(q,k); return sage_attention(a,c,v,b,d,output_dtype=torch.float16)[0]
    res={"T":T,"ref":"F.sdpa","acc":{
        "flash_vs_ref":acc(fo,ref),"eff_vs_ref":acc(eo,ref),
        "sage_vs_ref":acc(so,ref),"flash_vs_eff":acc(fo,eo),"sage_vs_flash":acc(so,fo)}}
    for name,fn in (("F_sdpa",lambda: sdpa_ref(q,k,v)),("flash",lambda: flash(q,k,v)),
                    ("efficient",lambda: efficient(q,k,v)),
                    ("sage_full",full)):
        vals,_=timed(fn)
        res[name]={"median":statistics.median(vals),"min":min(vals),"max":max(vals),
                   "samples":[round(x,6) for x in vals]}
    print(json.dumps(res),flush=True)
