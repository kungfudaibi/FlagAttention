import os, time, statistics, torch, torch_gcu
import torch.nn.functional as F
torch.gcu.set_device(1)
def cpu_s():
    p=open("/proc/self/stat").read().split(); return (int(p[13])+int(p[14]))/os.sysconf("SC_CLK_TCK")
def bench(name, fn, n=10, warm=3):
    for _ in range(warm): fn()
    torch.gcu.synchronize()
    w=[]; c=[]; ev=[]
    for _ in range(n):
        c0=cpu_s(); s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        t0=time.perf_counter(); s.record(); fn(); e.record(); torch.gcu.synchronize()
        w.append((time.perf_counter()-t0)*1000); c.append((cpu_s()-c0)*1000); ev.append(s.elapsed_time(e))
    print(f"DT {name} wall={statistics.median(w):.3f} cpu={statistics.median(c):.3f} event={statistics.median(ev):.3f} min_ev={min(ev):.3f}", flush=True)
for dt in (torch.float16, torch.bfloat16):
    for T in (512, 4096):
        q=torch.randn(1,2,T,64,device="gcu:1",dtype=dt)
        k=torch.randn_like(q); v=torch.randn_like(q)
        bench(f"sdpa_{dt}_T{T}", lambda: F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False), n=5)
    x=torch.randn(512,512,device="gcu:1",dtype=dt); y=torch.randn_like(x)
    bench(f"mm512_{dt}", lambda: x@y, n=5)
