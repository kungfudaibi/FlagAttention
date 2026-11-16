import torch, torch_gcu, statistics, json
import torch.nn.functional as F
torch.gcu.set_device(5)
from flag_attn.runtime.backend._enflame.sage_attention import attn_qk_int8_per_block as attn
from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8
def bench(f,warm=4,it=25):
    f(); torch.gcu.synchronize()
    for _ in range(warm): f()
    torch.gcu.synchronize()
    v=[]
    for _ in range(it):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        s.record(); f(); e.record(); torch.gcu.synchronize(); v.append(s.elapsed_time(e))
    return statistics.median(v)
def sage_full(q,k,v,D):
    qi,qs,ki,ks=per_block_int8(q,k,sm_scale=D**-0.5)
    return attn.forward(qi,ki,v,qs,ks,output_dtype=torch.float16)
Ts=[1024,1536,2048,2560,3072,3584,4096]
for D in (64,128):
    # keep every CTA in ONE wave: nCTA = T/BM must stay <= 24
    BM = 256 if D==64 else 128
    print("### D=%d BM=%d  single-wave sweep (nCTA<=24)" % (D,BM))
    nat={}; sag={}
    for T in Ts:
        q=torch.randn((1,1,T,D),device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
        nat[T]=bench(lambda: torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)[0])
        sag[T]=bench(lambda: sage_full(q,k,v,D)[0])
        print("   T=%4d nCTA=%2d iters=%2d  native %7.4f  sage %7.4f  ratio %5.2f" % (T,T//BM,T//128,nat[T],sag[T],sag[T]/nat[T]))
        del q,k,v; torch.gcu.empty_cache()
    for name,d in (("native",nat),("sage",sag)):
        sl=[1e3*(d[Ts[i]]-d[Ts[0]])/((Ts[i]-Ts[0])/128) for i in range(1,len(Ts))]
        print("   %-7s us per extra 128-KV tile: %s mean=%.2f" % (name,[round(x,2) for x in sl],sum(sl)/len(sl)))
