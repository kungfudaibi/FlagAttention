import torch, torch_gcu, statistics
torch.gcu.set_device(5)
def bench(f,n=20):
    f(); torch.gcu.synchronize()
    for _ in range(5): f()
    torch.gcu.synchronize()
    v=[]
    for _ in range(n):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        s.record(); f(); e.record(); torch.gcu.synchronize(); v.append(s.elapsed_time(e))
    return statistics.median(v)
for M,N,K in [(4096,4096,64),(4096,4096,128),(8192,8192,128),(4096,4096,4096),(2048,2048,2048)]:
    a=torch.randn((M,K),device="gcu:5",dtype=torch.float16)*0.01
    b=torch.randn((K,N),device="gcu:5",dtype=torch.float16)*0.01
    try:
        ms=bench(lambda: torch.mm(a,b))
        fl=2*M*N*K
        print(f"mm   {M}x{N}x{K:5d}  {ms:9.4f} ms  {fl/ms/1e9:7.1f} TF   {fl/ms/1e9/24:5.2f} TF/SIP")
    except Exception as ex:
        print("mm err",M,N,K,type(ex).__name__,str(ex)[:120])
    bf=b.to(torch.bfloat16); a3=a.to(torch.bfloat16)
    try:
        ms=bench(lambda: torch.mm(a3,bf))
        fl=2*M*N*K
        print(f"bf16 {M}x{N}x{K:5d}  {ms:9.4f} ms  {fl/ms/1e9:7.1f} TF")
    except Exception as ex:
        print("bf16 err",type(ex).__name__,str(ex)[:100])
    del a,b,bf,a3
    torch.gcu.empty_cache()
