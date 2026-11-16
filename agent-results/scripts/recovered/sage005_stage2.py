import torch, torch_gcu, statistics, sys
torch.gcu.set_device(5); sys.path.insert(0,"/tmp")
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
NI=32
for BM,BN,D in [(256,128,64),(128,128,128)]:
    fl=2.0*BM*BN*D*NI
    print("--- tile-major qk, tile %dx%dx%d" % (BM,BN,D))
    for grid in (24,96):
        K=torch.randn((grid*NI*D*BN,),device="gcu:5",dtype=torch.float16)*0.05
        Q=torch.randn((BM,D),device="gcu:5",dtype=torch.float16)*0.05
        O=torch.empty((grid*BM*BN,),device="gcu:5",dtype=torch.float16)
        for ns in (1,2,3,4):
            for nw in (1,2):
                ms=bench(lambda: bw2k2.k_qk_tm[(grid,)](Q,K,O,NI,BM=BM,BN=BN,D=D,num_warps=nw,num_stages=ns))
                print("  grid=%3d stages=%d warps=%d  %6.2f TF agg  %5.2f TF/CTA" % (grid,ns,nw,grid*fl/ms/1e9,fl/ms/1e9))
        del K,Q,O; torch.gcu.empty_cache()
