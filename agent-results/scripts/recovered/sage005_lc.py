import torch, torch_gcu, collections, json, sys
import torch.nn.functional as F
torch.gcu.set_device(5)
from flag_attn.runtime.backend._enflame.sage_attention import attn_qk_int8_per_block as attn
from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8
from torch.profiler import profile, ProfilerActivity
out=open(sys.argv[1],"w")
def one(path, shp):
    D=shp[-1]
    q=torch.randn(shp,device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
    qi,qs,ki,ks=per_block_int8(q,k,sm_scale=D**-0.5); torch.gcu.synchronize()
    fn={"native":lambda: torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False),
        "host_F_sdpa":lambda: F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False),
        "sage_full":lambda: attn.forward(qi,ki,v,qs,ks,output_dtype=torch.float16)}[path]
    for _ in range(3): fn()
    torch.gcu.synchronize()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.GCU]) as prof:
        fn(); torch.gcu.synchronize()
    c=collections.Counter()
    for e in prof.events(): c[e.name]+=e.count
    rec={"path":path,"shape":list(shp),"n_events":sum(c.values()),
         "launches":{n:ct for n,ct in c.items() if "Launch" in n},
         "memcpy":{n:ct for n,ct in c.items() if "emcpy" in n or "_copy" in n},
         "sync":{n:ct for n,ct in c.items() if "Synchronize" in n},
         "host_fallback":{n:ct for n,ct in c.items() if "for_cpu" in n or "_to_cpu" in n},
         "full_census":dict(c)}
    out.write(json.dumps(rec,sort_keys=True)+"\n"); out.flush()
    print("%-12s %-20s launches=%s fallback=%s n_events=%d" % (path,shp,rec["launches"],list(rec["host_fallback"]),rec["n_events"]))
    del q,k,v,qi,qs,ki,ks; torch.gcu.empty_cache()
for rep in range(3):
    for path in ("native","host_F_sdpa","sage_full"):
        for shp in ((1,2,4096,64),(4,32,1024,128)):
            one(path,shp)
out.close()
