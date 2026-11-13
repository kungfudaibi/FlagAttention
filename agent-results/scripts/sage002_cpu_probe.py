import os, time, torch, torch_gcu
import torch.nn.functional as F
torch.gcu.set_device(1)
def stat():
    with open("/proc/self/stat") as f:
        parts=f.read().split()
    return int(parts[13])/os.sysconf("SC_CLK_TCK"), int(parts[14])/os.sysconf("SC_CLK_TCK")
def probe(name, fn, n=10):
    fn(); torch.gcu.synchronize()
    t0=time.perf_counter(); c0=stat(); p0=time.process_time()
    for _ in range(n): fn()
    torch.gcu.synchronize()
    wall=time.perf_counter()-t0; c1=stat(); p1=time.process_time()
    print(f"{name}: wall_ms={wall/n*1000:.3f} cpu_ms={(p1-p0)/n*1000:.3f} "
          f"utime_ms={(c1[0]-c0[0])/n*1000:.3f} stime_ms={(c1[1]-c0[1])/n*1000:.3f}", flush=True)
q=torch.randn(1,2,512,64,device="gcu:1",dtype=torch.float16)
k=torch.randn_like(q); v=torch.randn_like(q)
probe("sdpa512", lambda: F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False))
q4=torch.randn(1,2,4096,64,device="gcu:1",dtype=torch.float16)
k4=torch.randn_like(q4); v4=torch.randn_like(q4)
probe("sdpa4096", lambda: F.scaled_dot_product_attention(q4,k4,v4,dropout_p=0.0,is_causal=False), n=3)
x=torch.randn(512,512,device="gcu:1",dtype=torch.float16); y=torch.randn_like(x)
probe("mm512_fp16", lambda: x@y, n=2)
xb=torch.randn(512,512,device="gcu:1",dtype=torch.bfloat16); yb=torch.randn_like(xb)
probe("mm512_bf16", lambda: xb@yb)
