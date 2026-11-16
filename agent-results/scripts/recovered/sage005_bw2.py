import torch, torch_gcu, importlib.util, statistics, os
torch.gcu.set_device(5)
SRC=r"""
import triton, triton.language as tl
@triton.jit
def k_copy(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    for off in range(pid*BLOCK, N, BLOCK*tl.num_programs(0)):
        idx = off + tl.arange(0, BLOCK)
        tl.store(Y + idx, tl.load(X + idx, mask=idx < N), mask=idx < N)
@triton.jit
def k_read(X, O, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), tl.float16)
    for off in range(pid*BLOCK, N, BLOCK*tl.num_programs(0)):
        idx = off + tl.arange(0, BLOCK)
        acc += tl.load(X + idx, mask=idx < N, other=0)
    tl.store(O + pid*BLOCK + tl.arange(0, BLOCK), acc)
"""
open("/tmp/bw2k.py","w").write(SRC)
import sys; sys.path.insert(0,"/tmp")
import bw2k
def bench(f,n=10):
    f(); torch.gcu.synchronize()
    for _ in range(3): f()
    torch.gcu.synchronize()
    v=[]
    for _ in range(n):
        s=torch.gcu.Event(enable_timing=True); e=torch.gcu.Event(enable_timing=True)
        s.record(); f(); e.record(); torch.gcu.synchronize(); v.append(s.elapsed_time(e))
    return statistics.median(v)
NBYTES=512<<20
N=NBYTES//2
X=torch.randn((N,),device="gcu:5",dtype=torch.float16); Y=torch.empty_like(X)
O=torch.empty((1<<20,),device="gcu:5",dtype=torch.float16)
print("torch copy_  %.1f GB/s (r+w)" % (2*NBYTES/1e9/bench(lambda: Y.copy_(X))*1e3))
print("torch add_   %.1f GB/s (r+w)" % (2*NBYTES/1e9/bench(lambda: torch.add(X,X,out=Y))*1e3))
for BLOCK in (2048,8192,16384,32768):
    for grid in (24,48,96,192,384):
        try:
            ms=bench(lambda: bw2k.k_copy[(grid,)](X,Y,N,BLOCK=BLOCK,num_warps=1,num_stages=1))
            msr=bench(lambda: bw2k.k_read[(grid,)](X,O,N,BLOCK=BLOCK,num_warps=1,num_stages=1))
            print("  BLOCK=%5d grid=%3d  copy %6.1f GB/s | read %6.1f GB/s" % (BLOCK,grid,2*NBYTES/1e9/ms*1e3,NBYTES/1e9/msr*1e3))
        except Exception as ex:
            print("  BLOCK=%5d grid=%3d  ERR %s" % (BLOCK,grid,str(ex)[:70]))
