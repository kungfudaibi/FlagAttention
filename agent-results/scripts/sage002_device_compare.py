import os, time, statistics, torch, torch_gcu
import torch.nn.functional as F
from flag_attn.testing import backend
torch.gcu.set_device(1)
def bench(fn,n,warm=5):
    for _ in range(warm): fn()
    torch.gcu.synchronize()
    vals=[]
    for _ in range(n):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.gcu.synchronize(); vals.append(s.elapsed_time(e))
    return round(statistics.median(vals),4)
T=512
q=torch.randn(1,2,T,64,device="gcu:1",dtype=torch.float16)
k=torch.randn_like(q); v=torch.randn_like(q)
sdpa=lambda: F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)
print("DEV1 sdpa512", bench(sdpa,10), flush=True)
qc=q.to("cpu"); kc=k.to("cpu"); vc=v.to("cpu")
try:
    s=time.perf_counter(); out=F.scaled_dot_product_attention(qc,kc,vc,dropout_p=0.0,is_causal=False); print("CPU sdpa512 wall_ms", (time.perf_counter()-s)*1000, out.dtype, flush=True)
except Exception as exc:
    print("CPU sdpa ERR", repr(exc), flush=True)
