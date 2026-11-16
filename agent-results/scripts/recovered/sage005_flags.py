import torch, torch_gcu, statistics, itertools, json
torch.gcu.set_device(5)
from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8
from flag_attn.runtime.backend._enflame.sage_attention import attn_qk_int8_per_block as A
from flag_attn.runtime.backend._enflame.sage_attention import attn_qk_int8_per_block as A2
def bench(f,warm=3,it=20):
    f(); torch.gcu.synchronize()
    for _ in range(warm): f()
    torch.gcu.synchronize()
    v=[]
    for _ in range(it):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        s.record(); f(); e.record(); torch.gcu.synchronize(); v.append(s.elapsed_time(e))
    return statistics.median(v)
for shp in [(1,2,4096,64),(4,32,1024,128)]:
    D=shp[-1]
    q=torch.randn(shp,device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
    qi,qs,ki,ks=per_block_int8(q,k,sm_scale=D**-0.5); torch.gcu.synchronize()
    ref=A.forward(qi,ki,v,qs,ks,output_dtype=torch.float16)[0]
    nat=bench(lambda: torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)[0])
    print("### %s native %.4f ms" % (shp,nat))
    for pd,tm,qsb in itertools.product([True,False],[True,False],[True,False]):
        try:
            out=A.forward(qi,ki,v,qs,ks,output_dtype=torch.float16,predequantize_k=pd,tile_major_predequant_k=tm,qk_scale_before_dot=qsb)[0]
            err=(out.float()-ref.float()).abs().max().item()
            ms=bench(lambda: A.forward(qi,ki,v,qs,ks,output_dtype=torch.float16,predequantize_k=pd,tile_major_predequant_k=tm,qk_scale_before_dot=qsb)[0])
            print("   predequant_k=%-5s tile_major=%-5s qk_scale_before=%-5s  %7.4f ms  %.2fx  maxdiff=%.2e" % (pd,tm,qsb,ms,ms/nat,err))
        except Exception as ex:
            print("   predequant_k=%-5s tile_major=%-5s qk_scale_before=%-5s  ERR %s" % (pd,tm,qsb,str(ex)[:80]))
    del q,k,v,qi,qs,ki,ks,ref; torch.gcu.empty_cache()
