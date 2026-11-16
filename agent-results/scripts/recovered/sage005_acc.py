import torch, torch_gcu, json
torch.gcu.set_device(5)
from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8
from flag_attn.runtime.backend._enflame.sage_attention import attn_qk_int8_per_block as A
def stats(a,b):
    a=a.float(); b=b.float()
    d=(a-b).abs()
    return dict(max_abs=d.max().item(), rel_l2=(d.norm()/b.norm()).item(),
                cosine=torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0).item(),
                finite=bool(torch.isfinite(a).all().item()), n=a.numel())
rows=[]
for shp in [(1,2,4096,64),(4,32,1024,128),(1,2,4090,64),(1,2,257,64),(2,3,1000,128),(1,1,129,128)]:
    D=shp[-1]
    torch.manual_seed(7)
    q=torch.randn(shp,device="gcu:5",dtype=torch.float16)
    k=torch.randn_like(q); v=torch.randn_like(q)
    nat=torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)[0]
    qi,qs,ki,ks=per_block_int8(q,k,sm_scale=D**-0.5)
    for odt in (torch.float16, torch.bfloat16):
        out=A.forward(qi,ki,v,qs,ks,output_dtype=odt)[0]
        st=stats(out,nat)
        st.update(shape=list(shp), output_dtype=str(odt).split(".")[-1], ref=str(nat.dtype).split(".")[-1])
        rows.append(st)
        print("SHAPE %-22s out=%-9s max_abs=%.4e rel_l2=%.4e cos=%.6f finite=%s" % (shp,st["output_dtype"],st["max_abs"],st["rel_l2"],st["cosine"],st["finite"]))
    del q,k,v,nat,qi,qs,ki,ks; torch.gcu.empty_cache()
json.dump(rows, open("/tmp/sage005_acc.json","w"), indent=1)
