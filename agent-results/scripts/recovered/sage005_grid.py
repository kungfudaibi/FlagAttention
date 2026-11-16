import torch, torch_gcu, json, os
torch.gcu.set_device(5)
from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8
from flag_attn.runtime.backend._enflame.sage_attention import attn_qk_int8_per_block as A
from torch.profiler import profile, ProfilerActivity

def run(tag, shape, fn):
    q=torch.randn(shape,device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
    fn(q,k,v); torch.gcu.synchronize()
    ref=fn(q,k,v)[0] if isinstance(fn(q,k,v),tuple) else fn(q,k,v)
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.GCU]) as prof:
        out=fn(q,k,v); torch.gcu.synchronize()
    p=f"/tmp/grid_{tag}.json"; prof.export_chrome_trace(p)
    ev=json.load(open(p))["traceEvents"]
    seen={}
    for e in ev:
        if e.get("ph")=="X" and e.get("cat") in ("kernel","Kernel","gpu_memcpy","Memcpy","gpu_memset"):
            nm=e["name"][:70]; d=e.get("dur",0)
            seen.setdefault((e["cat"],nm),[]).append((d,e.get("args",{})))
    print("###",tag,shape)
    for kk in sorted(seen,key=lambda x:-len(seen[x])):
        ds=[d for d,_ in seen[kk]]
        a=seen[kk][0][1]
        grid=a.get("grid") or a.get("Grid") or [x for x in a.values() if isinstance(x,list)][:1]
        print(f"  {kk[0]:10s} n={len(ds):3d} tot={sum(ds)/1000:8.4f}ms med={sorted(ds)[len(ds)//2]:7.1f}us name={kk[1]!r} grid={grid} args_keys={sorted(a)[:8]}")
    del q,k,v,out,ref
    torch.gcu.empty_cache()

for shp in [(1,2,4096,64),(4,32,1024,128)]:
    run("native", shp, lambda q,k,v: torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False))
for shp in [(1,2,4096,64),(4,32,1024,128)]:
    def sage(q,k,v):
        qi,qs,ki,ks=per_block_int8(q,k,sm_scale=q.shape[-1]**-0.5)
        return A.forward(qi,ki,v,qs,ks,output_dtype=torch.float16)
    run("sage", shp, sage)
