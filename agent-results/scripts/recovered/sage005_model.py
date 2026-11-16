import torch, torch_gcu, statistics, sys
torch.gcu.set_device(5)
sys.path.insert(0,"/tmp")
import bw2k2
def bench(f,n=12):
    f(); torch.gcu.synchronize()
    for _ in range(3): f()
    torch.gcu.synchronize()
    v=[]
    for _ in range(n):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        s.record(); f(); e.record(); torch.gcu.synchronize(); v.append(s.elapsed_time(e))
    return statistics.median(v)
# Pure streaming read ceiling for the exact per-iter tile size the kernel loads
NBYTES=512<<20; N=NBYTES//2
print("=== native call latency sweep (CTA count / geometry fixed by vendor)")
for shp in [(1,2,4096,64),(4,32,1024,128)]:
    q=torch.randn(shp,device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
    ms=bench(lambda: torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)[0])
    B,H,T,D=shp
    fl=4*B*H*T*T*D
    print("  %s  %.4f ms  %.2f TF  flops=%.3f GFLOP" % (shp, ms, fl/ms/1e9, fl/1e9))
    del q,k,v; torch.gcu.empty_cache()
