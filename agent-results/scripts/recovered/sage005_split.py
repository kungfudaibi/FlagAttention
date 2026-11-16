import torch, torch_gcu, triton, triton.language as tl, statistics, sys

@triton.jit
def _probe(Q, K, V, P, Out, NITER, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr, MODE: tl.constexpr):
    offs_m = tl.arange(0, BM); offs_n = tl.arange(0, BN); offs_k = tl.arange(0, D)
    q = tl.load(Q + offs_m[:, None] * D + offs_k[None, :])
    acc = tl.zeros((BM, BN), tl.float32)
    acc16 = tl.zeros((BM, BN), tl.float16)
    pv = tl.zeros((BM, D), tl.float32)
    for i in range(0, NITER):
        k = tl.load(K + i * BN * D + offs_n[None, :] * D + offs_k[:, None])
        if MODE == 1:      # qk only
            qk = tl.dot(q, k, out_dtype=tl.float32)
            acc += qk
        elif MODE == 2:    # qk only, fp16 dot output
            qk16 = tl.dot(q, k, out_dtype=tl.float16)
            acc16 += qk16
        elif MODE == 3:    # qk fp32 -> fp16 convert, no PV
            qk = tl.dot(q, k, out_dtype=tl.float32)
            acc += qk
            acc16 += qk.to(tl.float16)
        elif MODE == 4:    # chained: qk then PV on converted tile
            qk = tl.dot(q, k, out_dtype=tl.float32)
            v = tl.load(V + i * BN * D + offs_n[:, None] * D + offs_k[None, :])
            pv = tl.dot(qk.to(tl.float16), v, pv)
        elif MODE == 5:    # independent: PV on loaded fp16 tile
            p = tl.load(P + offs_m[:, None] * BN + offs_n[None, :])
            v = tl.load(V + i * BN * D + offs_n[:, None] * D + offs_k[None, :])
            pv = tl.dot(p, v, pv)
        elif MODE == 6:    # both dots, no conversion (qk fp16 out feeds PV)
            qk16 = tl.dot(q, k, out_dtype=tl.float16)
            v = tl.load(V + i * BN * D + offs_n[:, None] * D + offs_k[None, :])
            pv = tl.dot(qk16, v, pv)
        elif MODE == 7:    # PV dot accumulating into fp32 acc from previous PV (dependency chain)
            p = tl.load(P + offs_m[:, None] * BN + offs_n[None, :])
            v = tl.load(V + i * BN * D + offs_n[:, None] * D + offs_k[None, :])
            pv = tl.dot(p, v, pv)
            pv = pv * 1.000001
    if MODE in (1, 3):
        tl.store(Out + offs_m[:, None] * BN + offs_n[None, :], acc)
    if MODE in (2,):
        tl.store(Out + offs_m[:, None] * BN + offs_n[None, :], acc16)
    if MODE in (4, 5, 6, 7):
        tl.store(Out + offs_m[:, None] * D + offs_k[None, :], pv)

def run(mode, BM, BN, D, N, n=30):
    Q = torch.randn((BM, D), device="gcu:5", dtype=torch.float16)
    K = torch.randn((N, D, BN), device="gcu:5", dtype=torch.float16)  # [i][k][n]
    V = torch.randn((N, BN, D), device="gcu:5", dtype=torch.float16)
    P = torch.rand((BM, BN), device="gcu:5", dtype=torch.float16)
    Out = torch.empty((BM, max(BN, D)), device="gcu:5", dtype=torch.float16)
    def call():
        _probe[(1,)](Q, K, V, P, Out, N, BM=BM, BN=BN, D=D, MODE=mode, num_warps=1, num_stages=1)
    for _ in range(5): call()
    torch.gcu.synchronize()
    vals = []
    for _ in range(n):
        s = torch.gcu.Event(enable_timing=True); e = torch.gcu.Event(enable_timing=True)
        s.record(); call(); e.record(); torch.gcu.synchronize(); vals.append(s.elapsed_time(e))
    return statistics.median(vals)

names = {1: "qk_dot_f32out", 2: "qk_dot_f16out", 3: "qk_f32+f16convert", 4: "qk_f32->f16+PV(chained)",
         5: "PV_on_loaded_p", 6: "qk_f16out+PV", 7: "PV_loaded+scale_chain"}
for BM, BN, D in [(256, 128, 64), (128, 128, 128)]:
    print(f"--- tile BM={BM} BN={BN} D={D} iters=32 (one CTA, num_warps=1)")
    base = None
    for mode in (1, 2, 3, 4, 5, 6, 7):
        try:
            ms = run(mode, BM, BN, D, 32)
            if mode == 1: base = ms
            print(f"  {names[mode]:28s} {ms:8.4f} ms   x{ms/base:5.2f} vs qk")
        except Exception as ex:
            print(f"  {names[mode]:28s} FAILED {type(ex).__name__}: {str(ex)[:160]}")
