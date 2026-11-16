import os, sys, json, time, statistics, subprocess
import torch, torch_gcu
import torch.nn.functional as F
sys.path.insert(0, "/workspace/agent-workspaces/sage-fused-sdpa-study-5/src")
from flag_attn import sage_attention
from flag_attn.sage_attention import per_block_int8
torch.gcu.set_device(5)
print("dev", torch.gcu.current_device(), "stream", torch.gcu.current_stream(), "aff", sorted(os.sched_getaffinity(0)))
print("props", torch.gcu.get_device_properties(5))
for shape in [(1,2,4096,64),(4,32,1024,128)]:
    torch.manual_seed(2026)
    q=torch.randn(shape,device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
    def native(): return torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)[0]
    def host(): return F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)
    def sage_full():
        qi,qs,ki,ks=per_block_int8(q,k)
        return sage_attention(qi,ki,v,qs,ks,output_dtype=torch.float16)[0]
    for fn in (native,host,sage_full):
        fn(); torch.gcu.synchronize()
    def bench(fn,n=20):
        vals=[]; hosts=[]
        for _ in range(5): fn()
        torch.gcu.synchronize()
        for _ in range(n):
            s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
            t=time.perf_counter(); s.record(); fn(); e.record(); torch.gcu.synchronize(); h=time.perf_counter()-t
            vals.append(s.elapsed_time(e)); hosts.append(h*1e3)
        return {"median":statistics.median(vals),"min":min(vals),"max":max(vals),"host_med":statistics.median(hosts)}
    print(shape, "native", bench(native), "hostF", bench(host), "sage", bench(sage_full))
    print("native sum", native().float().sum().item(), "host sum", host().float().sum().item())
