import torch, torch_gcu
import torch.nn.functional as F
from flag_attn import sage_attention
from flag_attn.sage_attention import per_block_int8
torch.gcu.set_device(1)
print("ATTR current", torch.gcu.current_device(), "stream", torch.gcu.current_stream(), flush=True)
for d in (0,1):
    torch.gcu.reset_peak_memory_stats(d)
T=4096
q=torch.randn(1,2,T,64,device="gcu:1",dtype=torch.float16)
k=torch.randn_like(q); v=torch.randn_like(q)
print("ATTR after alloc", {d: torch.gcu.memory_allocated(d)/2**20 for d in (0,1)}, flush=True)
for d in (0,1):
    torch.gcu.reset_peak_memory_stats(d)
out=F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)
torch.gcu.synchronize()
print("ATTR sdpa peak", {d: torch.gcu.max_memory_allocated(d)/2**20 for d in (0,1)},
      {d: torch.gcu.max_memory_reserved(d)/2**20 for d in (0,1)}, "out", out.device, flush=True)
for d in (0,1):
    torch.gcu.reset_peak_memory_stats(d)
q8,qs,k8,ks=per_block_int8(q,k)
o,_=sage_attention(q8,k8,v,qs,ks,output_dtype=torch.float16)
torch.gcu.synchronize()
print("ATTR sage peak", {d: torch.gcu.max_memory_allocated(d)/2**20 for d in (0,1)},
      {d: torch.gcu.max_memory_reserved(d)/2**20 for d in (0,1)}, "out", o.device, flush=True)
