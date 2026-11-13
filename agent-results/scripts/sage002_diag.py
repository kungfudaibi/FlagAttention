import statistics
import torch, torch_gcu
import torch.nn.functional as F
torch.gcu.set_device(1)

def bench(fn,n=20,warm=10):
    for _ in range(warm): fn()
    torch.gcu.synchronize()
    vals=[]
    for _ in range(n):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.gcu.synchronize(); vals.append(s.elapsed_time(e))
    return round(statistics.median(vals), 6), round(min(vals), 6), round(max(vals), 6)

print("torch", torch.__version__, "gcu", torch_gcu.__version__)
for flag in ("flash_sdp_enabled","math_sdp_enabled","mem_efficient_sdp_enabled","cudnn_sdp_enabled"):
    print("FLAG", flag, getattr(torch.backends.cuda, flag)())
x=torch.randn(512,512,device="gcu:1",dtype=torch.float16)
y=torch.randn(512,512,device="gcu:1",dtype=torch.float16)
x@y; torch.gcu.synchronize()
print("MM512", bench(lambda: x@y))
q=torch.randn(1,2,512,64,device="gcu:1",dtype=torch.float16)
k=torch.randn_like(q); v=torch.randn_like(q)
print("SDPA512_default", bench(lambda: F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)))
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    for backend in (SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
        try:
            with sdpa_kernel(backend):
                print("SDPA512", backend, bench(lambda: F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)))
        except Exception as exc:
            print("SDPA512", backend, "ERR", repr(exc))
except Exception as exc:
    print("sdpa_kernel ERR", repr(exc))
