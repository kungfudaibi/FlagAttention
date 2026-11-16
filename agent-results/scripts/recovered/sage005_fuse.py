import os, sys, importlib.util, statistics, torch, torch_gcu
torch.gcu.set_device(5)
KERNEL="/workspace/agent-workspaces/sage-fused-sdpa-study-5/src/flag_attn/runtime/backend/_enflame/sage_attention/attn_qk_int8_per_block.py"
SRC=open(KERNEL).read()
OLD="""            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]"""
NEWS={
 "base": None,
 "fused_expr": """            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p32 = qk - m_ij[:, None]""",
 "inplace_sub": """            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]""",
}
# fused_expr also needs exp2 to consume p32
def build(name):
    src=SRC
    if NEWS[name] is None: new=OLD
    else: new=NEWS[name]
    assert src.count(OLD)==1
    src=src.replace(OLD,new,1)
    if name=="fused_expr":
        o2="            p = tl.math.exp2(qk).to(tl.float16)"
        assert src.count(o2)==1
        src=src.replace(o2,"            p = tl.math.exp2(p32).to(tl.float16)",1)
    p=f"/tmp/sage005_var_{name}.py"; open(p,"w").write(src)
    spec=importlib.util.spec_from_file_location(f"v_{name}",p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def bench(fn,warm=5,iters=25):
    fn(); torch.gcu.synchronize()
    for _ in range(warm): fn()
    torch.gcu.synchronize()
    v=[]
    for _ in range(iters):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.gcu.synchronize(); v.append(s.elapsed_time(e))
    return statistics.median(v), min(v)

from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8
mods={n:build(n) for n in NEWS}
for shape in [(1,2,4096,64),(4,32,1024,128)]:
    torch.manual_seed(2026)
    q=torch.randn(shape,device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
    qi,qs,ki,ks=per_block_int8(q,k,sm_scale=shape[3]**-0.5); torch.gcu.synchronize()
    nat=bench(lambda: torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)[0])
    print(f"== {shape} native {nat[0]:.4f} ms")
    outs={}
    for n,m in mods.items():
        r=bench(lambda: m.forward(qi,ki,v,qs,ks,output_dtype=torch.float16))
        outs[n]=m.forward(qi,ki,v,qs,ks,output_dtype=torch.float16)
        print(f"   {n:12s} {r[0]:.4f} ms  x{r[0]/nat[0]:.2f}  min {r[1]:.4f}")
    for n in ("inplace_sub","fused_expr"):
        d=(outs[n].float()-outs["base"].float()).abs()
        print(f"   {n}: maxabs {d.max().item():.3e} mean {d.mean().item():.3e} bitwise_equal {torch.equal(outs[n],outs[base])}")
