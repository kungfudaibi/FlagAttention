import os, sys, json, time, argparse
import torch, torch.nn.functional as F
p=argparse.ArgumentParser()
p.add_argument("--dtype",default="fp16"); p.add_argument("--T",type=int,default=512)
p.add_argument("--B",type=int,default=1); p.add_argument("--H",type=int,default=4)
p.add_argument("--K",type=int,default=64); p.add_argument("--V",type=int,default=64)
p.add_argument("--rounds",type=int,default=3); p.add_argument("--warmup",type=int,default=3)
p.add_argument("--mode",default="both") # fwd | fwd_bwd | both
p.add_argument("--tag",default="run")
a=p.parse_args()
from flag_attn import chunk_gla
dev=torch.device("gcu:0"); torch.manual_seed(42)
dt=torch.float16 if a.dtype=="fp16" else torch.bfloat16
B,T,H,K,V=a.B,a.T,a.H,a.K,a.V
q=torch.rand(B,T,H,K,dtype=dt,device=dev).requires_grad_()
k=torch.rand(B,T,H,K,dtype=dt,device=dev).requires_grad_()
v=torch.rand(B,T,H,V,dtype=dt,device=dev).requires_grad_()
g=F.logsigmoid(torch.rand(B,T,H,K,dtype=dt,device=dev)).requires_grad_()
h0=torch.rand(B,H,K,V,dtype=torch.float32,device=dev).requires_grad_()
do=torch.randn_like(v); dht=torch.randn(B,H,K,V,dtype=torch.float32,device=dev)

def fwd():
    return chunk_gla(q=q,k=k,v=v,g=g,scale=K**-0.5,initial_state=h0,
        output_final_state=True,cu_seqlens=None)
def step():
    o,ht=fwd()
    ((o*do).sum()+(ht*dht).sum().to(do.dtype)).backward()
    q.grad=k.grad=v.grad=g.grad=h0.grad=None

def sync(): torch.gcu.synchronize()
# cold first call, timed separately
sync(); t0=time.perf_counter(); fwd(); sync(); cold_fwd=(time.perf_counter()-t0)*1000
if a.mode!="fwd":
    q.grad=k.grad=v.grad=g.grad=h0.grad=None
    sync(); t0=time.perf_counter(); step(); sync(); cold_step=(time.perf_counter()-t0)*1000
else: cold_step=None
for _ in range(a.warmup):
    (fwd() if a.mode=="fwd" else step())
sync()
fwd_ts=[]; step_ts=[]
for _ in range(a.rounds):
    sync(); t0=time.perf_counter(); fwd(); sync(); fwd_ts.append((time.perf_counter()-t0)*1000)
    if a.mode!="fwd":
        q.grad=k.grad=v.grad=g.grad=h0.grad=None
        sync(); t0=time.perf_counter(); step(); sync(); step_ts.append((time.perf_counter()-t0)*1000)
print(json.dumps({"tag":a.tag,"dtype":a.dtype,"T":T,"mode":a.mode,
 "cold_fwd_ms":cold_fwd,"cold_fwd_bwd_ms":cold_step,
 "fwd_ms":fwd_ts,"fwd_median":sorted(fwd_ts)[len(fwd_ts)//2],
 "fwd_bwd_ms":step_ts,"fwd_bwd_median":sorted(step_ts)[len(step_ts)//2] if step_ts else None}))
