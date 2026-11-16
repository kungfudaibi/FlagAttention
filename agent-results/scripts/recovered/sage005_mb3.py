import torch, torch_gcu, triton, triton.language as tl, statistics
torch.gcu.set_device(5)
assert torch.gcu.current_device()==5

@triton.jit
def k_chain(Q, K, V, O, NITER, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr, CARRY: tl.constexpr):
    om=tl.arange(0,BM); on=tl.arange(0,BN); ok=tl.arange(0,D)
    q=tl.load(Q+om[:,None]*D+ok[None,:])
    pv=tl.zeros((BM,D),tl.float32)
    for i in range(NITER):
        k=tl.load(K+i*BN*D+on[None,:]*D+ok[:,None])
        v=tl.load(V+i*BN*D+on[:,None]*D+ok[None,:])
        qk=tl.dot(q,k,out_dtype=tl.float32)
        p=qk.to(tl.float16)
        if CARRY:
            pv=tl.dot(p,v,pv)
        else:
            pv=tl.dot(p,v)
            pv=pv+qk[:,0][:,None]
    tl.store(O+om[:,None]*D+ok[None,:],pv.to(tl.float16))

@triton.jit
def k_indep(Q, K, P, V, O, NITER, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr, CARRY: tl.constexpr):
    om=tl.arange(0,BM); on=tl.arange(0,BN); ok=tl.arange(0,D)
    q=tl.load(Q+om[:,None]*D+ok[None,:])
    p=tl.load(P+om[:,None]*BN+on[None,:])
    sq=tl.zeros((BM,BN),tl.float32); pv=tl.zeros((BM,D),tl.float32)
    for i in range(NITER):
        k=tl.load(K+i*BN*D+on[None,:]*D+ok[:,None])
        v=tl.load(V+i*BN*D+on[:,None]*D+ok[None,:])
        sq+=tl.dot(q,k,out_dtype=tl.float32)
        if CARRY:
            pv=tl.dot(p,v,pv)
        else:
            pv=tl.dot(p,v)
            pv=pv+sq[:,0][:,None]
    tl.store(O+om[:,None]*D+ok[None,:],pv.to(tl.float16))

@triton.jit
def k_qkonly(Q, K, O, NITER, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    om=tl.arange(0,BM); on=tl.arange(0,BN); ok=tl.arange(0,D)
    q=tl.load(Q+om[:,None]*D+ok[None,:]); acc=tl.zeros((BM,BN),tl.float32)
    for i in range(NITER):
        k=tl.load(K+i*BN*D+on[None,:]*D+ok[:,None])
        acc+=tl.dot(q,k,out_dtype=tl.float32)
    tl.store(O+om[:,None]*BN+on[None,:],acc.to(tl.float16))

def run(kern,args,BM,BN,D,n=30,**kw):
    def call(): kern[(1,)](*args,BM=BM,BN=BN,D=D,num_warps=1,num_stages=1,**kw)
    for _ in range(5): call()
    torch.gcu.synchronize()
    vals=[]
    for _ in range(n):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        s.record(); call(); e.record(); torch.gcu.synchronize(); vals.append(s.elapsed_time(e))
    return statistics.median(vals)

for BM,BN,D in [(256,128,64),(128,128,128)]:
    NI=32
    Q=torch.randn((BM,D),device="gcu:5",dtype=torch.float16)
    K=torch.randn((NI,BN,D),device="gcu:5",dtype=torch.float16)
    V=torch.randn((NI,BN,D),device="gcu:5",dtype=torch.float16)
    P=torch.rand((BM,BN),device="gcu:5",dtype=torch.float16)
    O=torch.empty((BM,max(BN,D)*2),device="gcu:5",dtype=torch.float16)
    print(f"--- BM={BM} BN={BN} D={D} iters={NI} 1 CTA/1 warp (12 threads)")
    tqk=run(k_qkonly,(Q,K,O,NI),BM,BN,D)
    print(f"  qk only                       {tqk:8.4f} ms")
    tc=run(k_chain,(Q,K,V,O,NI),BM,BN,D,CARRY=True)
    print(f"  chain qk->p->pv carry acc     {tc:8.4f} ms  x{tc/tqk:.2f} vs qk")
    tn=run(k_chain,(Q,K,V,O,NI),BM,BN,D,CARRY=False)
    print(f"  chain qk->p->pv no carry      {tn:8.4f} ms  x{tn/tqk:.2f}")
    ti=run(k_indep,(Q,K,P,V,O,NI),BM,BN,D,CARRY=True)
    print(f"  indep qk + pv carry acc       {ti:8.4f} ms  x{ti/tqk:.2f}")
    tin=run(k_indep,(Q,K,P,V,O,NI),BM,BN,D,CARRY=False)
    print(f"  indep qk + pv no carry        {tin:8.4f} ms  x{tin/tqk:.2f}")
