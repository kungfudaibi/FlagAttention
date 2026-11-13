import time, torch, torch_gcu
import torch.nn.functional as F
torch.gcu.set_device(0)
print("VIS count", torch.gcu.device_count(), "current", torch.gcu.current_device(), flush=True)
q=torch.randn(1,2,4096,64,device="gcu:0",dtype=torch.float16)
k=torch.randn_like(q); v=torch.randn_like(q)
fn=lambda: F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)
fn(); torch.gcu.synchronize()
t0=time.time(); n=0
while time.time()-t0<15:
    fn(); n+=1
torch.gcu.synchronize()
print("VIS SDPA per_ms", (time.time()-t0)/n*1000, "iters", n, "outdev", fn().device, flush=True)
