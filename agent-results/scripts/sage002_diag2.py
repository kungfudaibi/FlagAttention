import statistics, time
import torch, torch_gcu
torch.gcu.set_device(1)

def bench(fn,n=10,warm=5):
    for _ in range(warm): fn()
    torch.gcu.synchronize()
    ev=[]; wall=[]
    for _ in range(n):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        torch.gcu.synchronize(); t0=time.perf_counter()
        s.record(); fn(); e.record(); torch.gcu.synchronize()
        wall.append((time.perf_counter()-t0)*1000); ev.append(s.elapsed_time(e))
    return {"ev_med":round(statistics.median(ev),4),"wall_med":round(statistics.median(wall),4),"ev_min":round(min(ev),4),"wall_min":round(min(wall),4)}

for n in (256,512,1024,2048):
    x=torch.randn(n,n,device="gcu:1",dtype=torch.float16)
    y=torch.randn(n,n,device="gcu:1",dtype=torch.float16)
    r=bench(lambda: x@y)
    print("MM",n,r,"tflops_ev",round(2*n**3/(r["ev_med"]*1e-3)/1e12,2),flush=True)
a=torch.randn(1<<24,device="gcu:1",dtype=torch.float16)
b=torch.empty_like(a)
print("COPY64MB",bench(lambda: b.copy_(a)),flush=True)
print("ADD",bench(lambda: torch.add(a,a)),flush=True)
