import os, sys, json, time, argparse
import torch, torch.nn.functional as F
p=argparse.ArgumentParser()
p.add_argument("--T",type=int,default=512); p.add_argument("--H",type=int,default=4)
p.add_argument("--B",type=int,default=1); p.add_argument("--K",type=int,default=64); p.add_argument("--V",type=int,default=64)
p.add_argument("--rounds",type=int,default=5); p.add_argument("--warmup",type=int,default=2)
p.add_argument("--tag",default="run")
a=p.parse_args()
from flag_attn import chunk_gla
import importlib
C=importlib.import_module("flag_attn.runtime.backend._enflame.FLA.gla.chunk_gla")
dev=torch.device("gcu:0"); torch.manual_seed(42)
B,T,H,K,V=a.B,a.T,a.H,a.K,a.V
dt=torch.float16
q=torch.rand(B,T,H,K,dtype=dt,device=dev).requires_grad_()
k=torch.rand(B,T,H,K,dtype=dt,device=dev).requires_grad_()
v=torch.rand(B,T,H,V,dtype=dt,device=dev).requires_grad_()
g=F.logsigmoid(torch.rand(B,T,H,K,dtype=dt,device=dev)).requires_grad_()
h0=torch.rand(B,H,K,V,dtype=torch.float32,device=dev).requires_grad_()
do=torch.randn_like(v); dht=torch.randn(B,H,K,V,dtype=torch.float32,device=dev)
scale=K**-0.5
# grads are manually set to skip autograd graph within stages
q.grad=torch.zeros_like(q); k.grad=torch.zeros_like(k); v.grad=torch.zeros_like(v); g.grad=torch.zeros_like(g); h0.grad=torch.zeros_like(h0)
with torch.no_grad():
    def fwd_all():
        return C.chunk_gla_fwd(q=q,k=k,v=v,g=g,g_cumsum=None,scale=scale,
            initial_state=h0,output_final_state=True,cu_seqlens=None,chunk_size=64,chunk_indices=None)
    g_cumsum,A,h,ht,o=fwd_all()
    def bwd_dh():
        return C.chunk_bwd_dh(q=q,k=k,v=v,g=None,gk=g_cumsum,gv=None,do=do,h0=h0,dht=dht,
            scale=scale,cu_seqlens=None,chunk_size=64,states_in_fp32=True,state_v_first=False)
    dh,dh0=bwd_dh()
    def bwd_dv(): return C.chunk_gla_bwd_dv(k=k,g=g_cumsum,A=A,do=do,dh=dh,cu_seqlens=None,chunk_size=64,chunk_indices=None,state_v_first=False)
    dv=bwd_dv()
    def bwd_dA(): return C.chunk_gla_bwd_dA(v=v,do=do,scale=scale,cu_seqlens=None,chunk_size=64,chunk_indices=None)
    dA=bwd_dA()
    def bwd_intra(): return C.chunk_gla_bwd_dqk_intra(q=q,k=k,g=g_cumsum,dA=dA,cu_seqlens=None,chunk_size=64,chunk_indices=None)
    dqi,dki=bwd_intra()
    def bwd_dqkg(): return C.chunk_gla_bwd_dqkg(q=q,k=k,v=v,h=h,g=g_cumsum,do=do,dh=dh,dq=dqi,dk=dki,scale=scale,cu_seqlens=None,chunk_size=64,chunk_indices=None,state_v_first=False,output_in_input_dtype=True)
    dq,dk,dg=bwd_dqkg()
    def fwd_h(): return C.chunk_fwd_h(k=k,v=v,g=None,gk=g_cumsum,gv=None,h0=h0,output_final_state=True,cu_seqlens=None,chunk_size=64,states_in_fp32=False,state_v_first=False)
    def bwd_dh_fp32():
        return C.chunk_bwd_dh(q=q,k=k,v=v,g=None,gk=g_cumsum,gv=None,do=do,h0=h0,dht=dht,
            scale=scale,cu_seqlens=None,chunk_size=64,states_in_fp32=True,state_v_first=False)
    def bwd_dh_fp16():
        return C.chunk_bwd_dh(q=q,k=k,v=v,g=None,gk=g_cumsum,gv=None,do=do,h0=h0,dht=dht,
            scale=scale,cu_seqlens=None,chunk_size=64,states_in_fp32=False,state_v_first=False)
    # g_cumsum for recompute: stage-level timing only
    stages={"fwd_chunk_gla_fwd":fwd_all,"fwd_chunk_fwd_h":fwd_h,
            "bwd_chunk_bwd_dh(states_fp32)":bwd_dh,"bwd_chunk_gla_bwd_dv":bwd_dv,
            "bwd_chunk_gla_bwd_dA":bwd_dA,"bwd_chunk_gla_bwd_dqk_intra":bwd_intra,
            "bwd_chunk_gla_bwd_dqkg":bwd_dqkg}
    for _ in range(a.warmup):
        for f in stages.values(): f()
    torch.gcu.synchronize()
    res={}
    for name,f in stages.items():
        ts=[]
        for _ in range(a.rounds):
            torch.gcu.synchronize(); t0=time.perf_counter(); f(); torch.gcu.synchronize()
            ts.append((time.perf_counter()-t0)*1000)
        res[name]={"ms":ts,"median":sorted(ts)[len(ts)//2]}
print(json.dumps({"tag":a.tag,"T":T,"H":H,"stages":res},indent=1))
