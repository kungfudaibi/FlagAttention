import sys, time, torch, torch_gcu, json
torch.gcu.set_device(5)
from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8
from flag_attn.runtime.backend._enflame.sage_attention import attn_qk_int8_per_block as A
which=sys.argv[1]; shape=tuple(int(x) for x in sys.argv[2].split(","))
q=torch.randn(shape,device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
torch.gcu.synchronize(); time.sleep(0.3)
t0=time.perf_counter()
if which=="native":
    torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)
else:
    qi,qs,ki,ks=per_block_int8(q,k,sm_scale=shape[-1]**-0.5)
    A.forward(qi,ki,v,qs,ks,output_dtype=torch.float16)
torch.gcu.synchronize()
t1=time.perf_counter()
print(f"COLD {which} {shape} first_call_ms={(t1-t0)*1e3:.2f}")
