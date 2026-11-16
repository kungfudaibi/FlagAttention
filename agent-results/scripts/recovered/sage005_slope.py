import torch, torch_gcu, importlib.util, statistics
torch.gcu.set_device(5)
KERNEL="/workspace/agent-workspaces/sage-fused-sdpa-study-5/src/flag_attn/runtime/backend/_enflame/sage_attention/attn_qk_int8_per_block.py"
SRC=open(KERNEL).read()
SB="""            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]"""
SG="""            m_ij = m_i
            qk = qk - m_i[:, None]"""
RS="            l_ij = tl.sum(p, 1).to(tl.float32)"
RG="            l_ij = tl.full([BLOCK_M], 1.0, tl.float32)"
RE="            alpha = tl.math.exp2(m_i - m_ij)\n            acc = acc * alpha[:, None]"
RG2="            alpha = 1.0\n            acc = acc"
PV="            acc = tl.dot(p, v, acc)"
PG="            acc = acc * 1.0"
V={"full":(), "no_row_max":((SB,"            m_ij = m_i"),),
   "both_dots":((SB,SG),(RS,RG),(RE,RG2)),
   "qk_matmul_only":((SB,SG),(RS,RG),(RE,RG2),(PV,PG)),
   "no_pv_dot":((PV,PG),),
   "no_sub":((SB,"            m_ij = m_i\n            qk = qk"),)}
def build(name,reps):
    src=SRC
    for old,new in reps:
        assert src.count(old)==1,(name,old); src=src.replace(old,new,1)
    p=f"/tmp/sl_{name}.py"; open(p,"w").write(src)
    spec=importlib.util.spec_from_file_location(f"sl_{name}",p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
def bench(f,warm=3,it=15):
    f(); torch.gcu.synchronize()
    for _ in range(warm): f()
    torch.gcu.synchronize()
    v=[]
    for _ in range(it):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        s.record(); f(); e.record(); torch.gcu.synchronize(); v.append(s.elapsed_time(e))
    return statistics.median(v)
from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8
mods={k:build(k,r) for k,r in V.items()}
TS=[1024,2048,3072,4096]
res={}
for name,m in mods.items():
    row=[]
    for T in TS:
        q=torch.randn((1,1,T,64),device="gcu:5",dtype=torch.float16)
        k=torch.randn_like(q); v=torch.randn_like(q)
        qi,qs,ki,ks=per_block_int8(q,k,sm_scale=64**-0.5); torch.gcu.synchronize()
        ms=bench(lambda: m.forward(qi,ki,v,qs,ks,output_dtype=torch.float16)[0])
        row.append(ms); del q,k,v,qi,qs,ki,ks; torch.gcu.empty_cache()
    res[name]=row
    # 1 CTA cols: T=1024 ->4 iters ... 4096 ->16, all 16 CTAs concurrent on 24 SIPs
    per=[(row[i]-row[0])/((TS[i]-TS[0])/128) for i in range(1,len(TS))]
    print(f"{name:16s} " + "  ".join(f"{x:8.4f}" for x in row) + "   slope_us_per_iter=" + ",".join(f"{p*1e3:6.2f}" for p in per))
