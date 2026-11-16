import torch, torch_gcu, triton, triton.language as tl, statistics
torch.gcu.set_device(5)

@triton.jit
def k_lay(X, O, NITER, BM: tl.constexpr, BN: tl.constexpr, OP: tl.constexpr):
    om=tl.arange(0,BM); on=tl.arange(0,BN)
    x=tl.load(X+om[:,None]*BN+on[None,:])          # [BM,BN] f32
    xt=tl.trans(x)                                  # [BN,BM]
    m=tl.zeros((BM,),tl.float32)
    m0=tl.zeros((BN,),tl.float32)
    for i in range(NITER):
        if OP == 1:      # reduce contiguous axis (axis 1) only
            m += tl.max(x,1)
        elif OP == 2:    # reduce axis 1 + broadcast subtract (shipped pattern)
            m2=tl.maximum(m, tl.max(x,1)); x = x - m2[:,None]; m=m2
        elif OP == 3:    # transposed tile: reduce axis 0 of xt == rows of x
            m0 += tl.max(xt,0)
        elif OP == 4:    # transposed tile: reduce axis 0 + subtract along axis 0
            m2=tl.maximum(m0, tl.max(xt,0)); xt = xt - m2[None,:]; m0=m2
        elif OP == 5:    # explicit transpose then axis-0 reduce + axis-1 subtract
            m2=tl.maximum(m, tl.max(xt,0)); xt = xt - m2[None,:]; m=m2
        elif OP == 6:    # broadcast add along contiguous axis (no reduce)
            x = x + m[:,None]
        elif OP == 7:    # broadcast add along non-contiguous axis
            x = x + m0[None,:]
    acc = tl.sum(x,1) + m
    tl.store(O+om, acc)

def run(BM,BN,OP,nw,n=25,NI=32):
    X=torch.randn((BM,BN),device="gcu:5",dtype=torch.float32)
    O=torch.empty((max(BM,BN),),device="gcu:5",dtype=torch.float32)
    def call(): k_lay[(1,)](X,O,NI,BM=BM,BN=BN,OP=OP,num_warps=nw,num_stages=1)
    for _ in range(5): call()
    torch.gcu.synchronize()
    v=[]
    for _ in range(n):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        s.record(); call(); e.record(); torch.gcu.synchronize(); v.append(s.elapsed_time(e))
    return statistics.median(v)

names={1:"max axis1 only",2:"max axis1 + sub (shipped)",3:"max axis0 of transposed",4:"max axis0 + sub on transposed",
       5:"explicit trans, max axis0",6:"bcast add axis1",7:"bcast add axis0"}
for BM,BN in [(256,128)]:
    print(f"--- tile {BM}x{BN} f32, 32 iters")
    for nw in (1,2,4):
        for op in (1,2,3,4,5,6,7):
            try:
                ms=run(BM,BN,op,nw); print(f"  warps={nw} {names[op]:28s} {ms:8.4f} ms  /iter {ms/32*1000:6.2f} us")
            except Exception as ex:
                print(f"  warps={nw} {names[op]:28s} FAILED {type(ex).__name__} {str(ex).splitlines()[-1][:90]}")
