import torch, torch_gcu, triton, triton.language as tl, statistics
torch.gcu.set_device(5)

@triton.jit
def k_res(Q, K, O, NITER, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    om = tl.arange(0, BM); on = tl.arange(0, BN); ok = tl.arange(0, D)
    q = tl.load(Q + om[:, None] * D + ok[None, :])
    k = tl.load(K + ok[:, None] * BN + on[None, :])
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(NITER):
        acc += tl.dot(q, k, out_dtype=tl.float32)
    tl.store(O + pid * BM * BN + om[:, None] * BN + on[None, :], acc.to(tl.float16))

def bench(f, n=15):
    f(); torch.gcu.synchronize()
    for _ in range(3): f()
    torch.gcu.synchronize()
    v = []
    for _ in range(n):
        s = torch.gcu.Event(enable_timing=True); e = torch.gcu.Event(enable_timing=True)
        s.record(); f(); e.record(); torch.gcu.synchronize(); v.append(s.elapsed_time(e))
    return statistics.median(v)

NI = 200
for BM, BN, D in [(256, 128, 64), (128, 128, 128)]:
    fl = 2.0 * BM * BN * D * NI
    print(f"--- resident-K dot, tile {BM}x{BN}x{D}, {NI} dots/CTA, no per-iter loads")
    for grid in (1, 12, 24, 48, 96):
        Q = torch.randn((grid * BM * D,), device="gcu:5", dtype=torch.float16) * 0.05
        K = torch.randn((grid * D * BN,), device="gcu:5", dtype=torch.float16) * 0.05
        O = torch.empty((grid * BM * BN,), device="gcu:5", dtype=torch.float16)
        t = bench(lambda: k_res[(grid,)](Q, K, O, NI, BM=BM, BN=BN, D=D, num_warps=1, num_stages=1))
        print(f"  nCTA={grid:3d}  {grid*fl/t/1e9:7.2f} TF agg   {fl/t/1e9/1e3:6.3f} kFLOP/us/SIP-CTA")
        del Q, K, O
    torch.gcu.empty_cache()
