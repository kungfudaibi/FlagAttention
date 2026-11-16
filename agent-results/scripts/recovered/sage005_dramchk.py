import torch, torch_gcu, time
torch.gcu.set_device(5)
X=torch.randn((256<<20,),device="gcu:5",dtype=torch.float16); Y=torch.empty_like(X)
torch.gcu.synchronize()
t0=time.time(); n=0
while time.time()-t0<8.0:
    Y.copy_(X); n+=1
    if n%5==0: torch.gcu.synchronize()
torch.gcu.synchronize()
dt=time.time()-t0
print(f"COPYLOOP n={n} {n*2*(256<<20)/dt/1e9:.1f} GB/s sustained", flush=True)
