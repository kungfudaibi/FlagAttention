import torch, torch_gcu, statistics
torch.gcu.set_device(5)
from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8
from flag_attn.runtime.backend._enflame.sage_attention import attn_qk_int8_per_block as A

def bench(fn,warm=5,iters=25):
    fn(); torch.gcu.synchronize()
    for _ in range(warm): fn()
    torch.gcu.synchronize()
    v=[]
    for _ in range(iters):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.gcu.synchronize(); v.append(s.elapsed_time(e))
    return statistics.median(v)

for shape in [(1,2,4096,64),(4,32,1024,128)]:
    b,h,t,d=shape
    torch.manual_seed(2026)
    q=torch.randn(shape,device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
    qi,qs,ki,ks=per_block_int8(q,k,sm_scale=d**-0.5); torch.gcu.synchronize()
    nat=bench(lambda: torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)[0])
    print(f"== {shape} native {nat:.4f} ms  (SIPs=24)")
    ref=None
    for bm in (64,128,256):
        for bn in (128,):
            for nw in (1,):
                try:
                    grid_blocks=(t + (bm if not (d==64 and bm==128) else 2*bm) - 1)//(bm if not (d==64 and bm==128) else 2*bm)
                    eff_bm = 2*bm if (d==64 and bm==128) else bm
                    ctas=grid_blocks*b*h
                    ms=bench(lambda: A.forward(qi,ki,v,qs,ks,output_dtype=torch.float16,block_m=bm,block_n=bn,num_warps=nw))
                    out=A.forward(qi,ki,v,qs,ks,output_dtype=torch.float16,block_m=bm,block_n=bn,num_warps=nw)[0]
                    if ref is None: ref=out
                    derr=(out.float()-ref.float()).abs().max().item()
                    print(f"   block_m={bm:3d} (eff {eff_bm:3d}) block_n={bn} warps={nw}  CTAs={ctas:3d} waves={ctas/24:5.2f}  {ms:.4f} ms  x{ms/nat:.2f}  maxdiff_vs_first {derr:.2e}")
                except Exception as ex:
                    print(f"   block_m={bm} FAILED {type(ex).__name__} {str(ex).splitlines()[-1][:90]}")
