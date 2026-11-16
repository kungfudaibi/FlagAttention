import torch, torch_gcu, statistics, json
torch.gcu.set_device(5)
from flag_attn.runtime.backend._enflame.sage_attention import attn_qk_int8_per_block as attn
from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8
def bench(f,warm=3,it=20):
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
LEG={}
for (B,H,D,Ts) in [(1,2,64,[1024,2048,3072,4096]),(4,32,128,[512,768,1024])]:
    print("### B=%d H=%d D=%d" % (B,H,D))
    nat={}; sag={}
    for T in Ts:
        q=torch.randn((B,H,T,D),device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
        nat[T]=bench(lambda: torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)[0])
        sag[T]=bench(lambda: sage_full(q,k,v,D)[0])
        print("   T=%4d native %7.4f ms  sage %7.4f ms  ratio %5.2f" % (T,nat[T],sag[T],sag[T]/nat[T]))
        del q,k,v; torch.gcu.empty_cache()
    for name,d in (("native",nat),("sage",sag)):
        sl=[1e3*(d[Ts[i]]-d[Ts[0]])/((Ts[i]-Ts[0])/128) for i in range(1,len(Ts))]
        print("   %-7s marginal us per extra 128-KV tile: %s  mean=%.2f" % (name,[round(x,2) for x in sl],sum(sl)/len(sl)))
        LEG["%dx%dx%d_%s"%(B,H,D,name)]=sl
json.dump(LEG,open("/tmp/marg.json","w"),indent=1)
