import statistics, time
import torch, torch_gcu
import torch.nn.functional as F
torch.gcu.set_device(1)
def bench(fn,n=20,warm=5):
    for _ in range(warm): fn()
    torch.gcu.synchronize()
    ev=[]; wall=[]
    for _ in range(n):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        t0=time.perf_counter(); s.record(); fn(); e.record(); torch.gcu.synchronize()
        wall.append((time.perf_counter()-t0)*1000); ev.append(s.elapsed_time(e))
    return round(statistics.median(ev),4), round(statistics.median(wall),4)
for T in (512,1024,2048,4096):
    q=torch.randn(1,2,T,64,device="gcu:1",dtype=torch.float16)
    k=torch.randn_like(q); v=torch.randn_like(q)
    print("T",T,"sdpa",bench(lambda: F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)),flush=True)
    print("T",T,"add",bench(lambda: torch.add(q,q)),flush=True)
