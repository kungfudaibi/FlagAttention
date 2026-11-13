import statistics, sys
import torch, torch_gcu
import torch.nn.functional as F
from flag_attn import sage_attention
from flag_attn.sage_attention import per_block_int8

torch.gcu.set_device(1)

def bench(fn, n=30, warm=10):
    for _ in range(warm): fn()
    torch.gcu.synchronize()
    vals=[]
    for _ in range(n):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.gcu.synchronize(); vals.append(s.elapsed_time(e))
    return [statistics.median(vals), min(vals), max(vals)]

def sdpa(q,k,v):
    return F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)

for T in (512,1024,2048,4096):
    q=torch.randn(1,2,T,64,device="gcu:1",dtype=torch.float16)
    k=torch.randn_like(q); v=torch.randn_like(q)
    q8,qs,k8,ks=per_block_int8(q,k)
    def attn(): return sage_attention(q8,k8,v,qs,ks,output_dtype=torch.float16)
    def full():
        a,b,c,d=per_block_int8(q,k); return sage_attention(a,c,v,b,d,output_dtype=torch.float16)
    print("PROBE",T,"quant",bench(lambda: per_block_int8(q,k)),
          "attn",bench(attn),"full",bench(full),"sdpa",bench(lambda: sdpa(q,k,v)),
          "sdpa_nocast",bench(lambda: sdpa(q,k,v)), flush=True)
