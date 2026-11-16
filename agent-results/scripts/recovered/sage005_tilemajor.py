import torch, torch_gcu, triton, triton.language as tl, statistics
torch.gcu.set_device(5)

@triton.jit
def k_qk_tm(Q, K, O, NITER, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid=tl.program_id(0)
    om=tl.arange(0,BM); on=tl.arange(0,BN); ok=tl.arange(0,D)
    q=tl.load(Q+om[:,None]*D+ok[None,:])
    acc=tl.zeros((BM,BN),tl.float32)
    base=K+pid*NITER*D*BN
    for i in range(NITER):
        # tile-major: K_ptrs = base + i*D*BN + on[None,:] + ok[:,None]*BN
        k=tl.load(base+i*D*BN+on[None,:]+ok[:,None]*BN)
        acc+=tl.dot(q,k,out_dtype=tl.float32)
    tl.store(O+pid*BM*BN+om[:,None]*BN+on[None,:],acc.to(tl.float16))

@triton.jit
def k_pv_tm(P, V, O, NITER, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid=tl.program_id(0)
    om=tl.arange(0,BM); on=tl.arange(0,BN); ok=tl.arange(0,D)
    p=tl.load(P+pid*BM*BN+om[:,None]*BN+on[None,:])
    acc=tl.zeros((BM,D),tl.float32)
    base=V+pid*NITER*BN*D
    for i in range(NITER):
        v=tl.load(base+i*BN*D+on[:,None]*D+ok[None,:])
        acc=tl.dot(p,v,acc)
    tl.store(O+pid*BM*D+om[:,None]*D+ok[None,:],acc.to(tl.float16))

def bench(f,n=20):
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
    print(f"--- tile-major f16, tile {BM}x{BN}x{D}, {NI} iters/CTA")
    for grid in (1,24,48,96):
        Q=torch.randn((BM,D),device="gcu:5",dtype=torch.float16)*0.05
        K=torch.randn((grid*NI*D*BN,),device="gcu:5",dtype=torch.float16)*0.05
        P=torch.rand((grid*BM*BN,),device="gcu:5",dtype=torch.float16)
        V=torch.randn((grid*NI*BN*D,),device="gcu:5",dtype=torch.float16)*0.05
        O=torch.empty((grid*BM*max(BN,D),),device="gcu:5",dtype=torch.float16)
        t1=bench(lambda: k_qk_tm[(grid,)](Q,K,O,NI,BM=BM,BN=BN,D=D,num_warps=1,num_stages=1))
        t2=bench(lambda: k_pv_tm[(grid,)](P,V,O,NI,BM=BM,BN=BN,D=D,num_warps=1,num_stages=1))
        print(f"  nCTA={grid:3d}  qk {grid*fl/t1/1e9:6.2f} TF ({fl/t1/1e9:5.2f}/CTA) | pv {grid*fl/t2/1e9:6.2f} TF ({fl/t2/1e9:5.2f}/CTA) | sum est {grid*2*fl/(t1+t2)/1e9:6.2f} TF")
    del Q,K,P,V,O
    torch.gcu.empty_cache()
