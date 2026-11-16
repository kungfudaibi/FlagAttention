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

print("D64, B*H=2, BLOCK_M_eff=256 -> CTAs = 2*T/256; native ~0.265 ms at T=4096")
for T in (4096, 3584, 3072, 2560, 2048, 1536, 1024):
    shape=(1,2,T,64); d=64
    torch.manual_seed(2026)
    q=torch.randn(shape,device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
    qi,qs,ki,ks=per_block_int8(q,k,sm_scale=d**-0.5); torch.gcu.synchronize()
    ctas=2*((T+255)//256)
    waves=ctas/24
    ms=bench(lambda: A.forward(qi,ki,v,qs,ks,output_dtype=torch.float16)[0])
    nat=bench(lambda: torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)[0])
    fl=4*1*2*T*T*64
    print(f"  T={T:5d} CTAs={ctas:3d} waves={waves:5.2f}  sage {ms:.4f} ms {fl/ms/1e9:6.2f} TF | native {nat:.4f} ms {fl/nat/1e9:6.2f} TF | ratio {ms/nat:5.2f}")
