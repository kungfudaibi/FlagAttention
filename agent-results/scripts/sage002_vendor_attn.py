import statistics, torch, torch_gcu
import torch.nn.functional as F
torch.gcu.set_device(1)
def bench(name, fn, n=10, warm=3):
    try:
        for _ in range(warm): fn()
        torch.gcu.synchronize()
        vals=[]
        for _ in range(n):
            s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
            s.record(); out=fn(); e.record(); torch.gcu.synchronize(); vals.append(s.elapsed_time(e))
        print(name, "median_ms", round(statistics.median(vals),4), flush=True)
        return out
    except Exception as exc:
        print(name, "ERR", repr(exc), flush=True)
        return None
for dt in (torch.float16, torch.bfloat16):
    q=torch.randn(1,2,512,64,device="gcu:1",dtype=dt)
    k=torch.randn_like(q); v=torch.randn_like(q)
    print("=== dtype", dt, flush=True)
    bench(f"F.sdpa_{dt}", lambda: F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False))
    bench(f"aten_flash_{dt}", lambda: torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False))
    bench(f"aten_efficient_{dt}", lambda: torch.ops.aten._scaled_dot_product_efficient_attention(q,k,v,None,False,0.0,False))
    bench(f"aten_math_{dt}", lambda: torch.ops.aten._scaled_dot_product_attention_math(q,k,v,None,0.0,False,None,False))
