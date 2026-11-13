import sys, torch, torch_gcu
import torch.nn.functional as F
torch.gcu.set_device(1)
dt = getattr(torch, sys.argv[1])
T = int(sys.argv[2])
q=torch.randn(1,2,T,64,device="gcu:1",dtype=dt)
k=torch.randn_like(q); v=torch.randn_like(q)
for _ in range(5): F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)
torch.gcu.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)
    torch.gcu.synchronize()
rows = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12)
print(rows)
