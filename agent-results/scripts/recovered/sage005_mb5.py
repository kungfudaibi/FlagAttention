import torch, torch_gcu, triton, triton.language as tl, statistics
torch.gcu.set_device(5)

@triton.jit
def k_epi2(X, O, NITER, BM: tl.constexpr, BN: tl.constexpr, OP: tl.constexpr):
    om=tl.arange(0,BM); on=tl.arange(0,BN)
    x=tl.load(X+om[:,None]*BN+on[None,:])
    m=tl.zeros((BM,),tl.float32); l=tl.zeros((BM,),tl.float32)
    for i in range(NITER):
        if OP == 1:   # pure elementwise, no cross-lane reduce
            x = x * 1.0000001
        elif OP == 2: # row max reduce only
            m += tl.max(x,1)
        elif OP == 3: # row max + subtract (as in kernel)
            m2=tl.maximum(m, tl.max(x,1)); x = x - m2[:,None]; m=m2
        elif OP == 4: # column max reduce (axis 0) for comparison
            l += tl.max(x,0)
        elif OP == 5: # row sum reduce
            l += tl.sum(x,1)
        elif OP == 6: # elementwise add of two row vectors
            x = x + m[:,None] + l[None,:]
    tl.store(O+om, m+l+tl.sum(x,1))

def run(BM,BN,OP,nw,n=30,NI=32):
    X=torch.randn((BM,BN),device="gcu:5",dtype=torch.float32)
    O=torch.empty((BM,),device="gcu:5",dtype=torch.float32)
    def call(): k_epi2[(1,)](X,O,NI,BM=BM,BN=BN,OP=OP,num_warps=nw,num_stages=1)
    for _ in range(5): call()
    torch.gcu.synchronize()
    vals=[]
    for _ in range(n):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        s.record(); call(); e.record(); torch.gcu.synchronize(); vals.append(s.elapsed_time(e))
    return statistics.median(vals)

names={1:"elementwise mul",2:"rowmax reduce only",3:"rowmax+subtract",4:"colmax reduce",5:"rowsum reduce",6:"bcast add rows"}
BM,BN=256,128
print(f"--- {BM}x{BN} fp32 tile, 32 passes")
for nw in (1,2,4,8):
    print(f" num_warps={nw}")
    for op in (1,2,3,4,5,6):
        try:
            ms=run(BM,BN,op,nw)
            print(f"   {names[op]:20s} {ms:8.4f} ms  per-pass {ms/32*1000:7.2f} us")
        except Exception as ex:
            print(f"   {names[op]:20s} FAILED {type(ex).__name__} {str(ex)[:120]}")
