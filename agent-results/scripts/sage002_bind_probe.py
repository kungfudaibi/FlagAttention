import torch, torch_gcu
import torch.nn.functional as F
torch.gcu.set_device(1)
print("current after set_device", torch.gcu.current_device(), flush=True)
for name in ("device","Device","device_of","set_device"):
    print("API",name, hasattr(torch.gcu,name), flush=True)
q=torch.randn(1,2,512,64,device="gcu:1",dtype=torch.float16)
k=torch.randn_like(q); v=torch.randn_like(q)
out=F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)
print("out device", out.device, "current", torch.gcu.current_device(), flush=True)
try:
    ctx=torch.gcu.device(1)
    print("device ctx", ctx, flush=True)
except Exception as exc:
    print("device ctx ERR", repr(exc), flush=True)
