import os, time, statistics, torch, torch_gcu
import torch.nn.functional as F
torch.gcu.set_device(1)
def cpu_s():
    p=open("/proc/self/stat").read().split(); return (int(p[13])+int(p[14]))/os.sysconf("SC_CLK_TCK")
def bench(fn, n, warm=5):
    for _ in range(warm): fn()
    torch.gcu.synchronize()
    vals=[]
    for _ in range(n):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.gcu.synchronize(); vals.append(s.elapsed_time(e))
    return statistics.median(vals)
q=torch.randn(1,2,512,64,device="gcu:1",dtype=torch.float16)
k=torch.randn_like(q); v=torch.randn_like(q)
f=lambda: F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)
print("PREFIX first", bench(f,1), flush=True)
c0=cpu_s(); t0=time.perf_counter(); f(); torch.gcu.synchronize(); print("PREFIX single cpu_ms",(cpu_s()-c0)*1000,"wall_ms",(time.perf_counter()-t0)*1000, flush=True)
for i in range(6):
    print("PREFIX iter", i, bench(f,5), flush=True)
