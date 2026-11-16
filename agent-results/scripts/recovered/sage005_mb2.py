import torch, torch_gcu, triton, triton.language as tl, statistics
torch.gcu.set_device(5)
assert torch.gcu.current_device()==5

@triton.jit
def k_qk32(Q, K, O, NITER, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    om = tl.arange(0, BM); on = tl.arange(0, BN); ok = tl.arange(0, D)
    q = tl.load(Q + om[:, None] * D + ok[None, :])
    acc = tl.zeros((BM, BN), tl.float32)
    for i in range(NITER):
        k = tl.load(K + i * BN * D + on[None, :] * D + ok[:, None])
        acc += tl.dot(q, k, out_dtype=tl.float32)
    tl.store(O + om[:, None] * BN + on[None, :], acc.to(tl.float16))

@triton.jit
def k_qk16(Q, K, O, NITER, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    om = tl.arange(0, BM); on = tl.arange(0, BN); ok = tl.arange(0, D)
    q = tl.load(Q + om[:, None] * D + ok[None, :])
    acc = tl.zeros((BM, BN), tl.float16)
    for i in range(NITER):
        k = tl.load(K + i * BN * D + on[None, :] * D + ok[:, None])
        acc += tl.dot(q, k, out_dtype=tl.float16)
    tl.store(O + om[:, None] * BN + on[None, :], acc)

@triton.jit
def k_pv(P, V, O, NITER, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr):
    om = tl.arange(0, BM); on = tl.arange(0, BN); ok = tl.arange(0, D)
    p = tl.load(P + om[:, None] * BN + on[None, :])
    pv = tl.zeros((BM, D), tl.float32)
    for i in range(NITER):
        v = tl.load(V + i * BN * D + on[:, None] * D + ok[None, :])
        pv = tl.dot(p, v, pv)
        if SCALE:
            pv = pv * 1.000001
    tl.store(O + om[:, None] * D + ok[None, :], pv.to(tl.float16))

def run(kern, args, BM, BN, D, n=30, **kw):
    def call(): kern[(1,)](*args, BM=BM, BN=BN, D=D, num_warps=1, num_stages=1, **kw)
    for _ in range(5): call()
    torch.gcu.synchronize()
    vals = []
    for _ in range(n):
        s = torch.gcu.Event(enable_timing=True); e = torch.gcu.Event(enable_timing=True)
        s.record(); call(); e.record(); torch.gcu.synchronize(); vals.append(s.elapsed_time(e))
    return statistics.median(vals)

for BM, BN, D in [(256, 128, 64), (128, 128, 128)]:
    NI = 32
    Q = torch.randn((BM, D), device="gcu:5", dtype=torch.float16)
    K = torch.randn((NI, BN, D), device="gcu:5", dtype=torch.float16)
    V = torch.randn((NI, BN, D), device="gcu:5", dtype=torch.float16)
    P = torch.rand((BM, BN), device="gcu:5", dtype=torch.float16)
    O = torch.empty((BM, max(BN, D) * 2), device="gcu:5", dtype=torch.float16)
    print(f"--- BM={BM} BN={BN} D={D} iters={NI} 1 CTA/1 warp")
    t32 = run(k_qk32, (Q, K, O, NI), BM, BN, D)
    print(f"  qk f32-out            {t32:8.4f} ms")
    try:
        t16 = run(k_qk16, (Q, K, O, NI), BM, BN, D)
        print(f"  qk f16-out            {t16:8.4f} ms  ratio {t16/t32:.2f}")
    except Exception as ex:
        print("  qk f16-out FAILED", type(ex).__name__, str(ex)[:300])
    tpv = run(k_pv, (P, V, O, NI), BM, BN, D, SCALE=False)
    print(f"  pv on loaded p        {tpv:8.4f} ms")
    tpvs = run(k_pv, (P, V, O, NI), BM, BN, D, SCALE=True)
    print(f"  pv + fp32 scale chain {tpvs:8.4f} ms  (scale cost {tpvs-tpv:+.4f})")
