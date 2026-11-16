import torch, torch_gcu, time
torch.gcu.set_device(5)
a=torch.empty(512*1024*1024//2,device="gcu:5",dtype=torch.float16); b=torch.empty_like(a)
for _ in range(3): b.copy_(a)
torch.gcu.synchronize()
t0=time.perf_counter(); n=0
while time.perf_counter()-t0<2.5: b.copy_(a); n+=1
torch.gcu.synchronize(); el=time.perf_counter()-t0
print("copies",n,"ms",round(el/n*1e3,3),"GB/s",round(2*512*n/el/1e3,1))
