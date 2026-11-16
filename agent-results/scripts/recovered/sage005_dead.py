import torch, torch_gcu, importlib.util, json
torch.gcu.set_device(5)
KERNEL="/workspace/agent-workspaces/sage-fused-sdpa-study-5/src/flag_attn/runtime/backend/_enflame/sage_attention/attn_qk_int8_per_block.py"
SRC=open(KERNEL).read()
SOFTMAX_BLOCK="""            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]"""
SOFTMAX_GONE="""            m_ij = m_i
            qk = qk - m_i[:, None]"""
ROW_SUM="            l_ij = tl.sum(p, 1).to(tl.float32)"
ROW_SUM_GONE="            l_ij = tl.full([BLOCK_M], 1.0, tl.float32)"
RESCALE="            alpha = tl.math.exp2(m_i - m_ij)\n            acc = acc * alpha[:, None]"
RESCALE_GONE="            alpha = 1.0\n            acc = acc"
PV_DOT="            acc = tl.dot(p, v, acc)"
PV_GONE="            acc = acc * 1.0"
V={
 "full":(),
 "no_row_max":((SOFTMAX_BLOCK,"            m_ij = m_i"),),
 "both_dots":((SOFTMAX_BLOCK,SOFTMAX_GONE),(ROW_SUM,ROW_SUM_GONE),(RESCALE,RESCALE_GONE)),
 "qk_matmul_only":((SOFTMAX_BLOCK,SOFTMAX_GONE),(ROW_SUM,ROW_SUM_GONE),(RESCALE,RESCALE_GONE),(PV_DOT,PV_GONE)),
 "no_pv_dot":((PV_DOT,PV_GONE),),
}
from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8
shape=(1,2,4096,64)
q=torch.randn(shape,device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
qi,qs,ki,ks=per_block_int8(q,k,sm_scale=64**-0.5); torch.gcu.synchronize()
for name,reps in V.items():
    src=SRC
    for old,new in reps:
        assert src.count(old)==1,(name,src.count(old))
        src=src.replace(old,new,1)
    p=f"/tmp/dc_{name}.py"; open(p,"w").write(src)
    spec=importlib.util.spec_from_file_location(f"dc_{name}",p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.forward(qi,ki,v,qs,ks,output_dtype=torch.float16); torch.gcu.synchronize()
    km=m._attn_fwd.device_caches[5][0]
    for key,val in km.items():
        g=val.asm.get("gcuir","")
        n_mat=g.count("gcu.matmul"); n_trans=g.count("transpose_start"); n_vec=g.count("vector.load")
        print(f"{name:16s} gcu.matmul={n_mat:3d} transpose_start={n_trans:3d} vector.load={n_vec:5d} gcuir_bytes={len(g)}")
        break
