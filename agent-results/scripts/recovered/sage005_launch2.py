import torch, torch_gcu, json
torch.gcu.set_device(5)
from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8
from flag_attn.runtime.backend._enflame.sage_attention import attn_qk_int8_per_block as A
def trace(tag, shape, fn):
    q=torch.randn(shape,device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
    for _ in range(3): fn(q,k,v)
    torch.gcu.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.GCU]) as prof:
        fn(q,k,v); torch.gcu.synchronize()
    evs=prof.events()
    print(f"### {tag} {shape} kineto_events={len(evs)}")
    agg={}
    for e in evs:
        if e.device_type != torch.autograd.DeviceType.CPU and "gcu" not in str(e.device_type).lower():
            pass
        key=(str(e.device_type), e.name[:70])
        d=agg.setdefault(key, [0,0,0])
        d[0]+=e.count; d[1]+=e.device_time_total; d[2]+=e.cpu_time_total
    for key,(cnt,dev,cpu) in sorted(agg.items(), key=lambda x:-x[1][1]):
        if cnt>=1 and ("aunch" in key[1] or "emcpy" in key[1] or "emset" in key[1] or "Synchronize" in key[1] or "flash" in key[1] or key[0]!="DeviceType.CPU"):
            print(f"   {key[0]:12s} n={cnt:3d} dev_us={dev/1000:8.3f} cpu_us={cpu/1000:8.3f} {key[1]!r}")
    del q,k,v; torch.gcu.empty_cache()
nat=lambda q,k,v: torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)
def sage(q,k,v):
    qi,qs,ki,ks=per_block_int8(q,k,sm_scale=q.shape[-1]**-0.5)
    return A.forward(qi,ki,v,qs,ks,output_dtype=torch.float16)
trace("native_d64",(1,2,4096,64),nat); trace("native_d128",(4,32,1024,128),nat)
trace("sage_d64",(1,2,4096,64),sage);   trace("sage_d128",(4,32,1024,128),sage)
