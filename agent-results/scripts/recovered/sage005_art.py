import torch, torch_gcu, re
torch.gcu.set_device(5)
from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8
from flag_attn.runtime.backend._enflame.sage_attention import attn_qk_int8_per_block as A
shape=(1,2,4096,64)
q=torch.randn(shape,device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
qi,qs,ki,ks=per_block_int8(q,k,sm_scale=64**-0.5); torch.gcu.synchronize()
A.forward(qi,ki,v,qs,ks,output_dtype=torch.float16); torch.gcu.synchronize()
km=A._attn_fwd.device_caches[5][0]
for key,val in km.items():
    print("ASM KEYS:", {k:(len(x) if hasattr(x,"__len__") else type(x).__name__) for k,x in val.asm.items()})
    md=val.metadata
    print("META:", {k:getattr(md,k) for k in dir(md) if not k.startswith("_") and isinstance(getattr(md,k,None),(int,str,bool,float))})
    for kk in val.asm:
        if isinstance(val.asm[kk],(bytes,bytearray)):
            open(f"/tmp/tri_{kk}.bin","wb").write(val.asm[kk]); print("wrote",kk,len(val.asm[kk]))
    g=val.asm.get("gcuir","")
    if g:
        open("/tmp/tri_gcuir.txt","w").write(g)
        for m in sorted(set(re.findall(r"memref<[^>]*>", g))):
            print("MEMREF", m, g.count(m))
        print("alloc lines:")
        for line in g.splitlines():
            if "memref.alloc" in line or "memref_ext.alloc" in line:
                print("   ", line.strip()[:160])
    break
