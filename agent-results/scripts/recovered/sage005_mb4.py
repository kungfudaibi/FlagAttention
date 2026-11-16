import torch, torch_gcu, triton, triton.language as tl, statistics
torch.gcu.set_device(5)
assert torch.gcu.current_device()==5

# Epilogue cost: pass over a BM x BN fp32 tile, no matmul.
@triton.jit
def k_epi(X, O, NITER, BM: tl.constexpr, BN: tl.constexpr, OP: tl.constexpr):
    om=tl.arange(0,BM); on=tl.arange(0,BN)
    x=tl.load(X+om[:,None]*BN+on[None,:])
    m=tl.zeros((BM,),tl.float32); l=tl.zeros((BM,),tl.float32)
    p=tl.zeros((BM,BN),tl.float16)
    for i in range(NITER):
        if OP == 1:   # row max + subtract
            m2=tl.maximum(m, tl.max(x,1)); x = x - m2[:,None]; m = m2
        elif OP == 2: # exp2 + fp16 convert
            p = tl.math.exp2(x).to(tl.float16)
        elif OP == 3: # row sum of p
            l += tl.sum(p,1).to(tl.float32)
        elif OP == 4: # full softmax pass set (max, sub, exp2, convert, sum)
            m2=tl.maximum(m, tl.max(x,1)); x = x - m2[:,None]
            p = tl.math.exp2(x).to(tl.float16)
            l += tl.sum(p,1).to(tl.float32)
            a=tl.math.exp2(m-m2); m=m2
            x = x * a[:,None]
        elif OP == 5: # fp16 tile row max + sub + exp2 + sum, all fp16
            x16 = x.to(tl.float16)
            m2=tl.maximum(m, tl.max(x16,1).to(tl.float32)); x16 = x16 - m2[:,None].to(tl.float16)
            p = tl.math.exp2(x16)
            l += tl.sum(p,1).to(tl.float32)
            m=m2
    acc = tl.sum(p.to(tl.float32),1) + l + m
    tl.store(O+om, acc)

def run(kern,args,BM,BN,n=30,**kw):
    def call(): kern[(1,)](*args,BM=BM,BN=BN,num_warps=1,num_stages=1,**kw)
    for _ in range(5): call()
    torch.gcu.synchronize()
    vals=[]
    for _ in range(n):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        s.record(); call(); e.record(); torch.gcu.synchronize(); vals.append(s.elapsed_time(e))
    return statistics.median(vals)

names={1:"rowmax+sub",2:"exp2+f16conv",3:"rowsum(f16 p)",4:"full_softmax_f32tile",5:"softmax_f16tile"}
for BM,BN in [(256,128),(128,128)]:
    X=torch.randn((BM,BN),device="gcu:5",dtype=torch.float32)
    O=torch.empty((BM,),device="gcu:5",dtype=torch.float32)
    NI=32
    print(f"--- epilogue tile {BM}x{BN} fp32, {NI} passes, 1 warp/12 threads")
    for op in (1,2,3,4,5):
        ms=run(k_epi,(X,O,NI),BM,BN,OP=op)
        print(f"  {names[op]:20s} {ms:8.4f} ms   per-pass {ms/NI*1000:7.2f} us")
