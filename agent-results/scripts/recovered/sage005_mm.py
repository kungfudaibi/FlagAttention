import torch, torch_gcu, time, statistics
torch.gcu.set_device(5)
def bench(fn, n=30):
    for _ in range(5): fn()
    torch.gcu.synchronize()
    vals=[]
    for _ in range(n):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.gcu.synchronize(); vals.append(s.elapsed_time(e))
    return statistics.median(vals)
shapes=[(4096,4096,64),(4096,64,4096),(1024,1024,128),(1024,128,1024),(4096,4096,128)]
for M,N,K in shapes:
    a=torch.randn((M,K),device="gcu:5",dtype=torch.float16); b=torch.randn((K,N),device="gcu:5",dtype=torch.float16)
    ms=bench(lambda: torch.matmul(a,b))
    fl=2*M*N*K
    print(f"fp16 {M}x{N}x{K} {ms:.4f} ms {fl/ms/1e9:.1f} TFLOPS")
for M,N,K in [(4096,4096,64),(4096,64,4096),(1024,1024,128)]:
    try:
        a=torch.randint(-127,127,(M,K),device="gcu:5",dtype=torch.int8); b=torch.randint(-127,127,(K,N),device="gcu:5",dtype=torch.int8)
        ms=bench(lambda: torch._int_mm(a,b))
        fl=2*M*N*K
        print(f"int8 {M}x{N}x{K} {ms:.4f} ms {fl/ms/1e9:.1f} TFLOPS")
    except Exception as ex:
        print("int8 err",type(ex).__name__,str(ex)[:200])
for nbytes in (64<<20, 256<<20, 1024<<20):
    n=nbytes//2
    a=torch.empty(n,device="gcu:5",dtype=torch.float16); b=torch.empty_like(a)
    ms=bench(lambda: b.copy_(a), n=20)
    print(f"copy {nbytes>>20} MiB {ms:.4f} ms {2*nbytes/ms/1e6:.1f} GB/s")
