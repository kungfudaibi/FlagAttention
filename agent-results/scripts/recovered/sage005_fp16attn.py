import torch, torch_gcu, triton, triton.language as tl, statistics
torch.gcu.set_device(5)

@triton.jit
def attn_f16(Q,K,V,O, sm_scale, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, NQB: tl.constexpr):
    pid=tl.program_id(0)
    start_m=pid % NQB; off_hz=pid//NQB; off_h=off_hz % H; off_z=off_hz//H
    om=tl.arange(0,BM); on=tl.arange(0,BN); ok=tl.arange(0,D)
    q=tl.load(Q+off_z*H*T*D+off_h*T*D+(start_m*BM+om)[:,None]*D+ok[None,:])
    m_i=tl.full([BM], float("-inf"), tl.float32); l_i=tl.zeros([BM],tl.float32)
    acc=tl.zeros([BM,D],tl.float32)
    Kp=K+off_z*H*T*D+off_h*T*D; Vp=V+off_z*H*T*D+off_h*T*D
    for start_n in range(0,T,BN):
        k=tl.load(Kp+(start_n+on)[None,:]*D+ok[:,None])
        v=tl.load(Vp+(start_n+on)[:,None]*D+ok[None,:])
        qk=tl.dot(q,k,out_dtype=tl.float32)*sm_scale
        m_ij=tl.maximum(m_i, tl.max(qk,1))
        qk=qk-m_ij[:,None]
        p=tl.math.exp2(qk*1.4426950408889634).to(tl.float16)
        l_ij=tl.sum(p,1).to(tl.float32)
        alpha=tl.math.exp2((m_i-m_ij)*1.4426950408889634)
        acc=acc*alpha[:,None]
        acc=tl.dot(p,v,acc)
        l_i=l_i*alpha+l_ij; m_i=m_ij
    acc=acc/l_i[:,None]
    tl.store(O+off_z*H*T*D+off_h*T*D+(start_m*BM+om)[:,None]*D+ok[None,:], acc.to(tl.float16))

def bench(f,n=25):
    f(); torch.gcu.synchronize()
    for _ in range(5): f()
    torch.gcu.synchronize()
    v=[]
    for _ in range(n):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        s.record(); f(); e.record(); torch.gcu.synchronize(); v.append(s.elapsed_time(e))
    return statistics.median(v)

for (b,h,t,d),bm in [((1,2,4096,64),256),((4,32,1024,128),128)]:
    shape=(b,h,t,d)
    q=torch.randn(shape,device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
    o=torch.empty_like(q); nqb=t//bm; grid=b*h*nqb
    f=lambda: attn_f16[(grid,)](q,k,v,o,d**-0.5,T=t,H=h,D=d,BM=bm,BN=128,NQB=nqb,num_warps=1,num_stages=1)
    ms=bench(f)
    nat=bench(lambda: torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)[0])
    fl=4*b*h*t*t*d
    ref=torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)[0]
    err=(o.float()-ref.float()).abs().max().item()
    print(f"{shape} BM={bm} grid={grid}  triton_fp16 {ms:.4f} ms {fl/ms/1e9:6.2f} TF | native {nat:.4f} ms {fl/nat/1e9:6.2f} TF | ratio {ms/nat:.2f} | maxabs vs native {err:.2e}")
