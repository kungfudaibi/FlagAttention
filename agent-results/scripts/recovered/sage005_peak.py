import torch, torch_gcu, triton, triton.language as tl, statistics
torch.gcu.set_device(5)

@triton.jit
def k_qk(Q, K, O, NITER, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid=tl.program_id(0)
    om=tl.arange(0,BM); on=tl.arange(0,BN); ok=tl.arange(0,D)
    q=tl.load(Q+om[:,None]*D+ok[None,:])
    acc=tl.zeros((BM,BN),tl.float32)
    for i in range(NITER):
        k=tl.load(K+pid*NITER*BN*D+i*BN*D+on[None,:]*D+ok[:,None])
        acc+=tl.dot(q,k,out_dtype=tl.float32)
    tl.store(O+pid*BM*BN+om[:,None]*BN+on[None,:],acc.to(tl.float16))

@triton.jit
def k_pv(P, V, O, NITER, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid=tl.program_id(0)
    om=tl.arange(0,BM); on=tl.arange(0,BN); ok=tl.arange(0,D)
    p=tl.load(P+pid*BM*BN+om[:,None]*BN+on[None,:])
    pv=tl.zeros((BM,D),tl.float32)
    for i in range(NITER):
        v=tl.load(V+pid*NITER*BN*D+i*BN*D+on[:,None]*D+ok[None,:])
        pv=tl.dot(p,v,pv)
    tl.store(O+pid*BM*D+om[:,None]*D+ok[None,:],pv.to(tl.float16))

def bench(f, n=20):
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
    print(f"--- tile {BM}x{BN}x{D}, {NI} iters/CTA")
    flops=2.0*BM*BN*D*NI
    for grid in (1,2,4,8,16,24,32,48,64,96):
        Q=torch.randn((BM,D),device="gcu:5",dtype=torch.float16)
        K=torch.randn((grid*NI*BN*D,),device="gcu:5",dtype=torch.float16)
        P=torch.rand((grid*BM*BN,),device="gcu:5",dtype=torch.float16)
        V=torch.randn((grid*NI*BN*D,),device="gcu:5",dtype=torch.float16)
        O=torch.empty((grid*BM*max(BN,D),),device="gcu:5",dtype=torch.float16)
        qk=bench(lambda: k_qk[(grid,)](Q,K,O,NI,BM=BM,BN=BN,D=D,num_warps=1,num_stages=1))
        pv=bench(lambda: k_pv[(grid,)](P,V,O,NI,BM=BM,BN=BN,D=D,num_warps=1,num_stages=1))
        print(f"  CTAs={grid:3d}  qk {qk*1e3:7.1f} us {grid*flops/qk/1e9:6.1f} TF  |  pv {pv*1e3:7.1f} us {grid*flops/pv/1e9:6.1f} TF  |  both est {grid*2*flops/(qk+pv)/1e9:6.1f} TF")
    del Q,K,P,V,O
    torch.gcu.empty_cache()
