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
def k_qk_pf(Q, K, O, NITER, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid=tl.program_id(0)
    om=tl.arange(0,BM); on=tl.arange(0,BN); ok=tl.arange(0,D)
    q=tl.load(Q+om[:,None]*D+ok[None,:])
    acc=tl.zeros((BM,BN),tl.float32)
    base=K+pid*NITER*BN*D
    k=tl.load(base+on[None,:]*D+ok[:,None])
    for i in range(NITER):
        nxt=tl.load(base+((i+1)%NITER)*BN*D+on[None,:]*D+ok[:,None])
        acc+=tl.dot(q,k,out_dtype=tl.float32)
        k=nxt
    tl.store(O+pid*BM*BN+om[:,None]*BN+on[None,:],acc.to(tl.float16))

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
for BM,BN,D in [(256,128,64)]:
    flops=2.0*BM*BN*D*NI
    grid=24
    print(f"--- tile {BM}x{BN}x{D} CTAs={grid}")
    for nw in (1,):
        for ns in (1,2,3,4):
            for kern,nm in ((k_qk,"plain"),(k_qk_pf,"manual_prefetch")):
                try:
                    Q=torch.randn((BM,D),device="gcu:5",dtype=torch.float16)
                    K=torch.randn((grid*NI*BN*D,),device="gcu:5",dtype=torch.float16)
                    O=torch.empty((grid*BM*BN,),device="gcu:5",dtype=torch.float16)
                    ms=bench(lambda: kern[(grid,)](Q,K,O,NI,BM=BM,BN=BN,D=D,num_warps=nw,num_stages=ns))
                    print(f"  warps={nw} stages={ns} {nm:15s} {ms*1e3:7.1f} us  {grid*flops/ms/1e9:6.2f} TF  {flops/ms/1e9:5.2f} TF/SIP")
                except Exception as ex:
                    print(f"  warps={nw} stages={ns} {nm:15s} FAILED {type(ex).__name__} {str(ex).splitlines()[-1][:70]}")
