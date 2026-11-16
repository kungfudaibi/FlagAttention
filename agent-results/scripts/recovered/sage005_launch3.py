import torch, torch_gcu, collections
torch.gcu.set_device(5)
from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8
from flag_attn.runtime.backend._enflame.sage_attention import attn_qk_int8_per_block as A
def trace(tag, shape, fn):
    q=torch.randn(shape,device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
    for _ in range(3): fn(q,k,v)
    torch.gcu.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.GCU]) as prof:
        fn(q,k,v); torch.gcu.synchronize()
    c=collections.Counter((str(e.device_type),e.name) for e in prof.events() for _ in range(e.count))
    print(f"### {tag}")
    for (dt,nm),n in sorted(c.items(), key=lambda x:(-x[1],x[0][1])):
        print(f"   {dt:12s} n={n:3d} {nm[:66]!r}")
    del q,k,v; torch.gcu.empty_cache()
nat=lambda q,k,v: torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)
def sage(q,k,v):
    qi,qs,ki,ks=per_block_int8(q,k,sm_scale=q.shape[-1]**-0.5)
    return A.forward(qi,ki,v,qs,ks,output_dtype=torch.float16)
trace("sage_d64",(1,2,4096,64),sage)
