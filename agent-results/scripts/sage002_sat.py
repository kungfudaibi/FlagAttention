import time, torch, torch_gcu
torch.gcu.set_device(1)
x=torch.randn(1024,1024,device="gcu:1",dtype=torch.float16)
y=torch.randn(1024,1024,device="gcu:1",dtype=torch.float16)
torch.gcu.synchronize()
t0=time.time(); n=0
while time.time()-t0 < 20:
    z=x@y; n+=1
torch.gcu.synchronize()
print("MM_ITERS",n,"elapsed",time.time()-t0,"per_call_ms",(time.time()-t0)/n*1000,flush=True)
a=torch.randn(1<<24,device="gcu:1",dtype=torch.float16); b=torch.empty_like(a)
torch.gcu.synchronize(); t0=time.time(); n=0
while time.time()-t0 < 5:
    b.copy_(a); n+=1
torch.gcu.synchronize()
print("COPY_ITERS",n,"per_call_ms",(time.time()-t0)/n*1000,flush=True)
