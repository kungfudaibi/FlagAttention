import torch, torch_gcu, triton, triton.language as tl
torch.gcu.set_device(5)
@triton.jit
def k(X, O, NITER, BM: tl.constexpr, BN: tl.constexpr, OP: tl.constexpr):
    om=tl.arange(0,BM); on=tl.arange(0,BN)
    x=tl.load(X+om[:,None]*BN+on[None,:])
    m=tl.zeros((BM,),tl.float32)
    for i in range(NITER):
        if OP == 1:
            x = x * 1.0001
        elif OP == 2:
            m += tl.max(x,1)
    tl.store(O+om, m+tl.sum(x,1))
X=torch.randn((256,128),device="gcu:5",dtype=torch.float32); O=torch.empty((256,),device="gcu:5",dtype=torch.float32)
for op in (1,2):
    k[(1,)](X,O,4,BM=256,BN=128,OP=op,num_warps=1,num_stages=1); torch.gcu.synchronize(); print("op",op,"ok")
