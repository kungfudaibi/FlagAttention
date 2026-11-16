import torch, torch_gcu, triton, triton.language as tl, statistics
torch.gcu.set_device(5)

@triton.jit
def k_read(X, O, NBYTES_PER_CTA, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    n = NBYTES_PER_CTA // 2
    base = X + pid * n
    acc = tl.zeros((BLOCK,), tl.float32)
    for i in range(0, n, BLOCK):
        acc += tl.load(base + i + tl.arange(0, BLOCK)).to(tl.float32)
    tl.store(O + pid * BLOCK + tl.arange(0, BLOCK), acc)

@triton.jit
def k_copy(X, Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK
    tl.store(Y + off + tl.arange(0, BLOCK), tl.load(X + off + tl.arange(0, BLOCK)))

def bench(f, n=15):
    f(); torch.gcu.synchronize()
    for _ in range(3): f()
    torch.gcu.synchronize()
    v = []
    for _ in range(n):
        s = torch.gcu.Event(enable_timing=True); e = torch.gcu.Event(enable_timing=True)
        s.record(); f(); e.record(); torch.gcu.synchronize(); v.append(s.elapsed_time(e))
    return statistics.median(v)

print("--- streaming read bandwidth (GB/s), 128MiB total")
for grid in (24, 48, 96, 192):
    per = (128 << 20) // grid // 2 * 2
    X = torch.randn((per * grid // 2,), device="gcu:5", dtype=torch.float16)
    O = torch.empty((grid * 1024,), device="gcu:5", dtype=torch.float32)
    t = bench(lambda: k_read[(grid,)](X, O, per, BLOCK=1024, num_warps=1, num_stages=1))
    print(f"  nCTA={grid:4d}  {(128) / t * 1e3 / 1024:7.1f} GB/s")
    del X, O; torch.gcu.empty_cache()
print("--- copy bandwidth (GB/s r+w), 128MiB")
for grid in (48, 192):
    n = (128 << 20) // 2 // grid // 4096 * 4096
    X = torch.randn((n * grid,), device="gcu:5", dtype=torch.float16)
    Y = torch.empty_like(X)
    t = bench(lambda: k_copy[(n // 4096 * grid,)](X, Y, BLOCK=4096, num_warps=1, num_stages=1))
    print(f"  nCTA~{n//4096*grid:5d}  {2*128/t*1e3/1024:7.1f} GB/s")
    del X, Y; torch.gcu.empty_cache()
print("--- torch copy_ 128MiB")
X = torch.randn((64 << 20,), device="gcu:5", dtype=torch.float16); Y = torch.empty_like(X)
t = bench(lambda: Y.copy_(X)); print(f"  torch copy_ {2*128/t*1e3/1024:7.1f} GB/s")
