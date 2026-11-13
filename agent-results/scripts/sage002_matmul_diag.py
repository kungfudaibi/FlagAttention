import os, time, torch, torch_gcu
torch.gcu.set_device(1)
x=torch.randn(1024,1024,device="gcu:1",dtype=torch.float16)
y=torch.randn(1024,1024,device="gcu:1",dtype=torch.float16)
print("alloc_mib", torch.gcu.memory_allocated(1)/2**20, "x_device", x.device, "x_type", type(x))
torch.gcu.synchronize()
t0=time.process_time(); w0=time.perf_counter()
z=x@y
torch.gcu.synchronize()
print("cpu_time_s", round(time.process_time()-t0,3), "wall_s", round(time.perf_counter()-w0,3), "z_sum", float(z[0,0]))
for dt in (torch.float16, torch.bfloat16, torch.float32):
    a=torch.randn(512,512,device="gcu:1",dtype=dt); b=torch.randn(512,512,device="gcu:1",dtype=dt)
    torch.gcu.synchronize(); t0=time.process_time(); w0=time.perf_counter()
    c=a@b; torch.gcu.synchronize()
    print("dtype",dt,"cpu_s",round(time.process_time()-t0,3),"wall_s",round(time.perf_counter()-w0,3), "val", float(c[0,0]))
