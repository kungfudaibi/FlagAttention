import torch, torch_gcu, json, sys, os
torch.gcu.set_device(5)
sys.path.insert(0,"/workspace/agent-workspaces/sage-fused-sdpa-study-5/src")
from flag_attn import sage_attention
from flag_attn.sage_attention import per_block_int8
shape=(1,2,4096,64)
q=torch.randn(shape,device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
def native(): return torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)[0]
def full():
    p=per_block_int8(q,k); return sage_attention(p[0],p[2],v,p[1],p[3],output_dtype=torch.float16)[0]
native(); full(); torch.gcu.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.GCU], record_shapes=False) as prof:
    native(); full(); torch.gcu.synchronize()
path="/tmp/sage005_trace.json"
prof.export_chrome_trace(path)
print("size",os.path.getsize(path))
d=json.load(open(path))
print("keys",d.keys())
from collections import Counter
print(Counter(e.get("cat") for e in d["traceEvents"]))
print(Counter(e.get("ph") for e in d["traceEvents"]))
for e in d["traceEvents"]:
    if e.get("cat") in ("kernel","Kernel","gpu_op","GCU") or ("args" in e and any(k in e.get("args",{}) for k in ("grid","block","registers per thread","shared memory"))):
        print(json.dumps(e)[:800])
