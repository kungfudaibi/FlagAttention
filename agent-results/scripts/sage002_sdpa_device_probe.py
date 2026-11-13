import statistics, torch, torch_gcu
import torch.nn.functional as F
torch.gcu.set_device(1)
def bench(fn,n=10,warm=3):
    for _ in range(warm): fn()
    torch.gcu.synchronize()
    vals=[]
    for _ in range(n):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        s.record(); o=fn(); e.record(); torch.gcu.synchronize(); vals.append(s.elapsed_time(e))
    return round(statistics.median(vals),4), o.device
for dev in ("gcu:0","gcu:1"):
    for T in (512,4096):
        q=torch.randn(1,2,T,64,device=dev,dtype=torch.float16)
        k=torch.randn_like(q); v=torch.randn_like(q)
        print("PROBE",dev,T,"F.sdpa",bench(lambda: F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)),flush=True)
        print("PROBE",dev,T,"flash",bench(lambda: torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)[0]),flush=True)
