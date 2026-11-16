
import triton, triton.language as tl

@triton.jit
def k_rowmax(X, OUT, PASSES, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    om = tl.arange(0, BM); on = tl.arange(0, BN)
    x = tl.load(X + pid * BM * BN + om[:, None] * BN + on[None, :])
    m = tl.zeros((BM,), tl.float32)
    for _ in range(PASSES):
        m += tl.max(x, 1)
    tl.store(OUT + pid * BM + om, m)

@triton.jit
def k_rowmax_sub(X, OUT, PASSES, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    om = tl.arange(0, BM); on = tl.arange(0, BN)
    x = tl.load(X + pid * BM * BN + om[:, None] * BN + on[None, :])
    m = tl.zeros((BM,), tl.float32)
    for _ in range(PASSES):
        mi = tl.max(x, 1)
        m += mi
        x = x - mi[:, None]
    tl.store(OUT + pid * BM + om, m)

@triton.jit
def k_rowsum(X, OUT, PASSES, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    om = tl.arange(0, BM); on = tl.arange(0, BN)
    x = tl.load(X + pid * BM * BN + om[:, None] * BN + on[None, :])
    s = tl.zeros((BM,), tl.float32)
    for _ in range(PASSES):
        s += tl.sum(x, 1)
    tl.store(OUT + pid * BM + om, s)

@triton.jit
def k_mul(X, OUT, PASSES, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    om = tl.arange(0, BM); on = tl.arange(0, BN)
    x = tl.load(X + pid * BM * BN + om[:, None] * BN + on[None, :])
    m = tl.zeros((BM,), tl.float32)
    for _ in range(PASSES):
        x = x * 1.0001
        m += tl.max(x, 1)
    tl.store(OUT + pid * BM + om, m)
