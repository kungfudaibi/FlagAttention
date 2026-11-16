
import triton, triton.language as tl

@triton.jit
def k_qk_tm(Q, K, O, NITER, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    om = tl.arange(0, BM); on = tl.arange(0, BN); ok = tl.arange(0, D)
    q = tl.load(Q + om[:, None] * D + ok[None, :])
    acc = tl.zeros((BM, BN), tl.float32)
    base = K + pid * NITER * D * BN
    for i in range(NITER):
        k = tl.load(base + i * D * BN + on[None, :] + ok[:, None] * BN)
        acc += tl.dot(q, k, out_dtype=tl.float32)
    tl.store(O + pid * BM * BN + om[:, None] * BN + on[None, :], acc.to(tl.float16))

@triton.jit
def k_pv_tm(P, V, O, NITER, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    om = tl.arange(0, BM); on = tl.arange(0, BN); ok = tl.arange(0, D)
    p = tl.load(P + pid * BM * BN + om[:, None] * BN + on[None, :])
    acc = tl.zeros((BM, D), tl.float32)
    base = V + pid * NITER * BN * D
    for i in range(NITER):
        v = tl.load(base + i * BN * D + on[:, None] * D + ok[None, :])
        acc = tl.dot(p, v, acc)
    tl.store(O + pid * BM * D + om[:, None] * D + ok[None, :], acc.to(tl.float16))
