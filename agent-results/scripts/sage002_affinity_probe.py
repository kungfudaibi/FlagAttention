import os, time, statistics, torch, torch_gcu
import torch.nn.functional as F
torch.gcu.set_device(1)
print("AFF cpus", len(os.sched_getaffinity(0)), sorted(os.sched_getaffinity(0))[:4], flush=True)
q=torch.randn(1,2,4096,64,device="gcu:1",dtype=torch.float16)
k=torch.randn_like(q); v=torch.randn_like(q)
fn=lambda: F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)
fn(); torch.gcu.synchronize()
vals=[]
for _ in range(20):
    s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
    s.record(); fn(); e.record(); torch.gcu.synchronize(); vals.append(s.elapsed_time(e))
print("AFF sdpa4096 median_ms", round(statistics.median(vals),4), "min", round(min(vals),4), flush=True)
