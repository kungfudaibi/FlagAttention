import torch, torch_gcu, json, sys
import torch.nn.functional as F
torch.gcu.set_device(5)
sys.path.insert(0,"/workspace/agent-workspaces/sage-fused-sdpa-study-5/src")
from flag_attn import sage_attention
from flag_attn.sage_attention import per_block_int8
shape=(4,32,1024,128)
q=torch.randn(shape,device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
def native(): return torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)[0]
def host(): return F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)
def sage():
    qi,qs,ki,ks=per_block_int8(q,k)
    return sage_attention(qi,ki,v,qs,ks,output_dtype=torch.float16)[0]
for name,fn in (("native",native),("host",host),("sage",sage)):
    for _ in range(3): fn()
    torch.gcu.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.GCU]) as prof:
        fn(); torch.gcu.synchronize()
    evs=prof.events()
    print("===",name,"events",len(evs))
    for e in evs:
        print(json.dumps({"name":e.name,"device_type":str(e.device_type),"dur_us":e.device_time_total,"cpu_us":e.cpu_time_total,"count":e.count,"self_device":e.self_device_time_total}))
