import sys, time, torch, torch_gcu
import torch.nn.functional as F
torch.gcu.set_device(1)
which=sys.argv[1]
T=int(sys.argv[2]); secs=float(sys.argv[3])
q=torch.randn(1,2,T,64,device="gcu:1",dtype=torch.float16)
k=torch.randn_like(q); v=torch.randn_like(q)
if which=="sdpa":
    fn=lambda: F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)
elif which=="add":
    fn=lambda: torch.add(q,q)
elif which=="mm":
    x=torch.randn(1024,1024,device="gcu:1",dtype=torch.float16); y=torch.randn_like(x)
    fn=lambda: x@y
fn(); torch.gcu.synchronize()
t0=time.time(); n=0
while time.time()-t0<secs:
    fn(); n+=1
torch.gcu.synchronize()
print("LOOP",which,T,"iters",n,"per_ms",(time.time()-t0)/n*1000,flush=True)
