import json, collections, os, torch, torch_gcu
torch.gcu.set_device(5)
from torch.profiler import profile, ProfilerActivity
from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8
from flag_attn.runtime.backend._enflame.sage_attention import attn_qk_int8_per_block as A

def trace(tag, shape, fn):
    q=torch.randn(shape,device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
    fn(q,k,v); torch.gcu.synchronize()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.GCU]) as prof:
        fn(q,k,v); torch.gcu.synchronize()
    p=f"/tmp/lc_{tag}.json"; prof.export_chrome_trace(p)
    ev=json.load(open(p))["traceEvents"]
    rt=collections.Counter(); 
    for e in ev:
        if e.get("cat")=="gcu_runtime":
            rt[e["name"]]+=1
    print(f"### {tag} {shape}")
    for name,n in sorted(rt.items(), key=lambda x:-x[1]):
        tot=sum(e.get("dur",0) for e in ev if e.get("cat")=="gcu_runtime" and e["name"]==name)
        if n>=1 and ("Launch" in name or "Memcpy" in name or "Memset" in name or "Synchronize" in name):
            print(f"   {name:32s} n={n:3d} cpu_us={tot:9.2f}")
    print("   total gcu_runtime events:", sum(rt.values()))
    del q,k,v; torch.gcu.empty_cache()

nat=lambda q,k,v: torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)
def sage(q,k,v):
    qi,qs,ki,ks=per_block_int8(q,k,sm_scale=q.shape[-1]**-0.5)
    return A.forward(qi,ki,v,qs,ks,output_dtype=torch.float16)
trace("native_d64",(1,2,4096,64),nat); trace("native_d128",(4,32,1024,128),nat)
trace("sage_d64",(1,2,4096,64),sage);   trace("sage_d128",(4,32,1024,128),sage)
