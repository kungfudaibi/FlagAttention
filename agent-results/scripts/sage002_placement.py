import sys, time, torch, torch_gcu
import torch.nn.functional as F
from flag_attn import sage_attention
from flag_attn.sage_attention import per_block_int8
torch.gcu.set_device(1)
which=sys.argv[1]; secs=float(sys.argv[2])
T=4096
q=torch.randn(1,2,T,64,device="gcu:1",dtype=torch.float16)
k=torch.randn_like(q); v=torch.randn_like(q)
if which=="sdpa":
    fn=lambda: F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)
else:
    q8,qs,k8,ks=per_block_int8(q,k)
    fn=lambda: sage_attention(q8,k8,v,qs,ks,output_dtype=torch.float16)
fn(); torch.gcu.synchronize()
t0=time.time(); n=0
while time.time()-t0<secs:
    fn(); n+=1
torch.gcu.synchronize()
print("LOOP",which,"iters",n,"per_ms",(time.time()-t0)/n*1000,flush=True)
