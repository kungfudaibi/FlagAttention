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
print("D64 T=4096, BM_eff=256 -> nCTA = BH*16. Per-CTA work identical across BH.")
for bh in (1,2,3,4,6,12):
    shape=(1,bh,4096,64)
    torch.manual_seed(2026)
    q=torch.randn(shape,device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
    qi,qs,ki,ks=per_block_int8(q,k,sm_scale=64**-0.5); torch.gcu.synchronize()
    ctas=bh*16; waves=ctas/24
    ms=bench(lambda: A.forward(qi,ki,v,qs,ks,output_dtype=torch.float16)[0])
    nat=bench(lambda: torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)[0])
    per=ms/ctas*1e3
    print(f"  BH={bh:3d} nCTA={ctas:4d} waves={waves:6.2f}  sage {ms:8.4f} ms  per-CTA {per:7.2f} us | native {nat:8.4f}  ratio {ms/nat:5.2f}")
