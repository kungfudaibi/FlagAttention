
import triton, triton.language as tl

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
