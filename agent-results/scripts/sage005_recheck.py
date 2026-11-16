#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SAGE-006 short rechecks for the SAGE-005 claims that shipped without raw JSONL.

Every probe pins GCU 5 by PCI address *and* serial before the first device
action, writes one JSON object per measurement, and is deliberately short: the
point is to turn a surviving /tmp script into committed raw evidence, not to
redo the SAGE-005 performance study.

Probes
------
``fp16``      pure-FP16 Triton flash attention versus the device fused op.
``fuse``      the two rejected row-max-subtract rewrites, plus bitwise equality.
``flags``     ``predequantize_k`` x ``tile_major_predequant_k`` x
              ``qk_scale_before_dot`` on both headline shapes.
``stages``    ``num_stages`` 1-4 on the tile-fed QK loop, plain versus manual
              double-buffered prefetch.
``waves``     fixed per-CTA work at 16..192 CTAs, plus the ``block_m`` sweep.
``bw``        streaming copy/read bandwidth at 2-32 KiB blocks.
``marginal``  single-wave marginal cost per extra 128-wide KV tile.  Only
              points with ``nCTA <= 24`` enter the slope, which the original
              /tmp script got wrong for D128 at T=3584/4096.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import itertools
import json
import os
import statistics
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sage005_guard import bench, device_guard, kernel_path, repo_root, writer  # noqa: E402

FP16_SRC = r'''
import triton, triton.language as tl

@triton.jit
def attn_f16(Q, K, V, O, sm_scale, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, NQB: tl.constexpr):
    pid = tl.program_id(0)
    start_m = pid % NQB; off_hz = pid // NQB; off_h = off_hz % H; off_z = off_hz // H
    om = tl.arange(0, BM); on = tl.arange(0, BN); ok = tl.arange(0, D)
    q = tl.load(Q + off_z*H*T*D + off_h*T*D + (start_m*BM+om)[:, None]*D + ok[None, :])
    m_i = tl.full([BM], float("-inf"), tl.float32); l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    Kp = K + off_z*H*T*D + off_h*T*D; Vp = V + off_z*H*T*D + off_h*T*D
    for start_n in range(0, T, BN):
        k = tl.load(Kp + (start_n+on)[None, :]*D + ok[:, None])
        v = tl.load(Vp + (start_n+on)[:, None]*D + ok[None, :])
        qk = tl.dot(q, k, out_dtype=tl.float32)*sm_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk*1.4426950408889634).to(tl.float16)
        l_ij = tl.sum(p, 1).to(tl.float32)
        alpha = tl.math.exp2((m_i-m_ij)*1.4426950408889634)
        acc = acc*alpha[:, None]
        acc = tl.dot(p, v, acc)
        l_i = l_i*alpha + l_ij; m_i = m_ij
    acc = acc/l_i[:, None]
    tl.store(O + off_z*H*T*D + off_h*T*D + (start_m*BM+om)[:, None]*D + ok[None, :],
             acc.to(tl.float16))
'''

STAGES_SRC = r'''
import triton, triton.language as tl

@triton.jit
def k_qk(Q, K, O, NITER, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    om = tl.arange(0, BM); on = tl.arange(0, BN); ok = tl.arange(0, D)
    q = tl.load(Q + om[:, None]*D + ok[None, :])
    acc = tl.zeros((BM, BN), tl.float32)
    for i in range(NITER):
        k = tl.load(K + pid*NITER*BN*D + i*BN*D + on[None, :]*D + ok[:, None])
        acc += tl.dot(q, k, out_dtype=tl.float32)
    tl.store(O + pid*BM*BN + om[:, None]*BN + on[None, :], acc.to(tl.float16))

@triton.jit
def k_qk_tm(Q, K, O, NITER, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    om = tl.arange(0, BM); on = tl.arange(0, BN); ok = tl.arange(0, D)
    q = tl.load(Q + om[:, None]*D + ok[None, :])
    acc = tl.zeros((BM, BN), tl.float32)
    base = K + pid*NITER*D*BN
    for i in range(NITER):
        k = tl.load(base + i*D*BN + on[None, :] + ok[:, None]*BN)
        acc += tl.dot(q, k, out_dtype=tl.float32)
    tl.store(O + pid*BM*BN + om[:, None]*BN + on[None, :], acc.to(tl.float16))

@triton.jit
def k_qk_pf(Q, K, O, NITER, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    om = tl.arange(0, BM); on = tl.arange(0, BN); ok = tl.arange(0, D)
    q = tl.load(Q + om[:, None]*D + ok[None, :])
    acc = tl.zeros((BM, BN), tl.float32)
    base = K + pid*NITER*BN*D
    k = tl.load(base + on[None, :]*D + ok[:, None])
    for i in range(NITER):
        nxt = tl.load(base + ((i+1) % NITER)*BN*D + on[None, :]*D + ok[:, None])
        acc += tl.dot(q, k, out_dtype=tl.float32)
        k = nxt
    tl.store(O + pid*BM*BN + om[:, None]*BN + on[None, :], acc.to(tl.float16))
'''

BW_SRC = r'''
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
'''


def load_kernels(tmpdir, name, source):
    path = os.path.join(tmpdir, f"sage006_{name}_kernels.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)
    spec = importlib.util.spec_from_file_location(f"sage006_{name}_kernels", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_head():
    return subprocess.run(["git", "-C", str(repo_root()), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=False).stdout.strip()


def emit_env(emit, probe, extra=None):
    import torch
    import torch_gcu  # noqa: F401
    import triton

    fields = dict(
        probe="env",
        target=probe,
        when=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        host=os.uname().nodename,
        python=sys.version.split()[0],
        torch=torch.__version__,
        torch_gcu=torch_gcu.__version__ if hasattr(torch_gcu, "__version__") else "unknown",
        triton=triton.__version__,
        commit=_git_head(),
        loadavg=os.getloadavg(),
    )
    if extra:
        fields.update(extra)
    emit(**fields)


def probe_fp16(torch, tmpdir, emit):
    mod = load_kernels(tmpdir, "fp16", FP16_SRC)
    for (b, h, t, d), bm in (((1, 2, 4096, 64), 256), ((4, 32, 1024, 128), 128)):
        shape = (b, h, t, d)
        torch.manual_seed(2026)
        q = torch.randn(shape, device="gcu:5", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        o = torch.empty_like(q)
        nqb = t // bm
        grid = b * h * nqb
        ref = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False)[0]
        call = lambda: mod.attn_f16[(grid,)](  # noqa: E731
            q, k, v, o, d ** -0.5, T=t, H=h, D=d, BM=bm, BN=128, NQB=nqb,
            num_warps=1, num_stages=1)
        ms, samples = bench(torch, call, warmup=3, iters=20)
        call()
        torch.gcu.synchronize()
        err = (o.float() - ref.float()).abs().max().item()
        nat, nat_samples = bench(
            torch,
            lambda: torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False)[0],
            warmup=3, iters=20)
        flops = 4.0 * b * h * t * t * d
        emit(probe="fp16", shape=list(shape), block_m=bm, block_n=128, grid=grid,
             triton_ms=ms, triton_min_ms=min(samples), native_ms=nat,
             native_min_ms=min(nat_samples), ratio=ms / nat,
             triton_tflops=flops / ms / 1e9, native_tflops=flops / nat / 1e9,
             maxabs_vs_native=err, samples=samples, native_samples=nat_samples)
        del q, k, v, o, ref
        torch.gcu.empty_cache()


FUSE_OLD = """            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]"""
FUSE_NEW = {
    "base": None,
    "fused_expr": """            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p32 = qk - m_ij[:, None]""",
    "inplace_sub": """            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]""",
}


def build_fuse_variant(name, tmpdir):
    src = kernel_path().read_text(encoding="utf-8")
    if FUSE_NEW[name] is not None:
        assert src.count(FUSE_OLD) == 1, f"{name}: anchor not unique"
        src = src.replace(FUSE_OLD, FUSE_NEW[name], 1)
    if name == "fused_expr":
        old = "            p = tl.math.exp2(qk).to(tl.float16)"
        assert src.count(old) == 1, "fused_expr: exp2 anchor not unique"
        src = src.replace(old, "            p = tl.math.exp2(p32).to(tl.float16)", 1)
    path = os.path.join(tmpdir, f"sage006_var_{name}.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(src)
    spec = importlib.util.spec_from_file_location(f"sage006_var_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe_fuse(torch, tmpdir, emit):
    from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8

    mods = {name: build_fuse_variant(name, tmpdir) for name in FUSE_NEW}
    for shape in ((1, 2, 4096, 64), (4, 32, 1024, 128)):
        torch.manual_seed(2026)
        q = torch.randn(shape, device="gcu:5", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        qi, qs, ki, ks = per_block_int8(q, k, sm_scale=shape[3] ** -0.5)
        torch.gcu.synchronize()
        nat, _ = bench(
            torch,
            lambda: torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False)[0],
            warmup=3, iters=20)
        outs = {}
        for name, mod in mods.items():
            call = lambda m=mod: m.forward(qi, ki, v, qs, ks, output_dtype=torch.float16)[0]  # noqa: E731
            ms, samples = bench(torch, call, warmup=3, iters=20)
            outs[name] = call()
            torch.gcu.synchronize()
            emit(probe="fuse", shape=list(shape), variant=name, ms=ms,
                 min_ms=min(samples), native_ms=nat, ratio=ms / nat, samples=samples)
        base = outs["base"]
        for name in ("fused_expr", "inplace_sub"):
            diff = (outs[name].float() - base.float()).abs()
            emit(probe="fuse_equiv", shape=list(shape), variant=name,
                 maxabs_vs_base=diff.max().item(), meanabs_vs_base=diff.mean().item(),
                 bitwise_equal=bool(torch.equal(outs[name], base)))
        del q, k, v, qi, qs, ki, ks, outs, base
        torch.gcu.empty_cache()


def probe_flags(torch, tmpdir, emit):
    from flag_attn.runtime.backend._enflame.sage_attention import attn_qk_int8_per_block as attn
    from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8

    for shape in ((1, 2, 4096, 64), (4, 32, 1024, 128)):
        d = shape[-1]
        torch.manual_seed(2026)
        q = torch.randn(shape, device="gcu:5", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        qi, qs, ki, ks = per_block_int8(q, k, sm_scale=d ** -0.5)
        torch.gcu.synchronize()
        ref = attn.forward(qi, ki, v, qs, ks, output_dtype=torch.float16)[0]
        nat, _ = bench(
            torch,
            lambda: torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False)[0],
            warmup=3, iters=20)
        for pd, tm, qsb in itertools.product((True, False), (True, False), (True, False)):
            kwargs = dict(output_dtype=torch.float16, predequantize_k=pd,
                          tile_major_predequant_k=tm, qk_scale_before_dot=qsb)
            out = attn.forward(qi, ki, v, qs, ks, **kwargs)[0]
            err = (out.float() - ref.float()).abs().max().item()
            call = lambda kw=kwargs: attn.forward(qi, ki, v, qs, ks, **kw)[0]  # noqa: E731
            ms, samples = bench(torch, call, warmup=3, iters=20)
            emit(probe="flags", shape=list(shape), predequantize_k=pd,
                 tile_major_predequant_k=tm, qk_scale_before_dot=qsb,
                 ms=ms, min_ms=min(samples), native_ms=nat, ratio=ms / nat,
                 maxdiff_vs_default=err, samples=samples)
        del q, k, v, qi, qs, ki, ks, ref
        torch.gcu.empty_cache()


def probe_stages(torch, tmpdir, emit):
    """num_stages 1-4 on the tile-fed QK loop, row-major and tile-major K.

    The SAGE-005 report quoted 28.05/28.41/25.01/24.10 TF for D128; those come
    from the *tile-major* K layout (``k_qk_tm``), while the 186-187 us prefetch
    pair comes from the row-major loop (``k_qk``).  Both are recorded here and
    tagged, because the two layouts do not have the same baseline.
    """
    mod = load_kernels(tmpdir, "stages", STAGES_SRC)
    niter = 32
    for bm, bn, dim, layout, kerns in (
        (256, 128, 64, "row_major", ((mod.k_qk, "plain"), (mod.k_qk_pf, "manual_prefetch"))),
        (128, 128, 128, "tile_major", ((mod.k_qk_tm, "plain"), (mod.k_qk_pf, "manual_prefetch"))),
    ):
        flops = 2.0 * bm * bn * dim * niter
        for grid in (24, 96):
            if layout == "row_major" and grid == 96:
                continue  # the prefetch comparison was measured at 24 CTAs only
            q = torch.randn((bm, dim), device="gcu:5", dtype=torch.float16) * 0.05
            k = torch.randn((grid * niter * bn * dim,), device="gcu:5", dtype=torch.float16) * 0.05
            out = torch.empty((grid * bm * bn,), device="gcu:5", dtype=torch.float16)
            for stages in (1, 2, 3, 4):
                for kern, name in kerns:
                    call = lambda kk=kern, ns=stages: kk[(grid,)](
                        q, k, out, niter, BM=bm, BN=bn, D=dim, num_warps=1, num_stages=ns)
                    ms, samples = bench(torch, call, warmup=2, iters=10)
                    emit(probe="stages", tile=[bm, bn, dim], k_layout=layout, n_cta=grid,
                         num_warps=1, num_stages=stages, kernel=name, ms=ms, us=ms * 1e3,
                         tflops=grid * flops / ms / 1e9, tflops_per_cta=flops / ms / 1e9,
                         samples=samples)
            del q, k, out
            torch.gcu.empty_cache()


def probe_waves(torch, tmpdir, emit):
    from flag_attn.runtime.backend._enflame.sage_attention import attn_qk_int8_per_block as attn
    from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8

    for bh in (1, 2, 3, 4, 6, 12):
        shape = (1, bh, 4096, 64)
        torch.manual_seed(2026)
        q = torch.randn(shape, device="gcu:5", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        qi, qs, ki, ks = per_block_int8(q, k, sm_scale=64 ** -0.5)
        torch.gcu.synchronize()
        ctas = bh * 16
        call = lambda: attn.forward(qi, ki, v, qs, ks, output_dtype=torch.float16)[0]  # noqa: E731
        ms, samples = bench(torch, call, warmup=3, iters=20)
        nat, _ = bench(
            torch,
            lambda: torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False)[0],
            warmup=3, iters=20)
        emit(probe="waves_bh", shape=list(shape), n_cta=ctas, waves=ctas / 24.0,
             ms=ms, us_per_cta=ms / ctas * 1e3, native_ms=nat, ratio=ms / nat,
             samples=samples)
        del q, k, v, qi, qs, ki, ks
        torch.gcu.empty_cache()

    for shape in ((1, 2, 4096, 64), (4, 32, 1024, 128)):
        b, h, t, d = shape
        torch.manual_seed(2026)
        q = torch.randn(shape, device="gcu:5", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        qi, qs, ki, ks = per_block_int8(q, k, sm_scale=d ** -0.5)
        torch.gcu.synchronize()
        nat, _ = bench(
            torch,
            lambda: torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False)[0],
            warmup=3, iters=20)
        ref = None
        for bm in (64, 128):
            eff = 2 * bm if (d == 64 and bm == 128) else bm
            ctas = ((t + eff - 1) // eff) * b * h
            call = lambda m=bm: attn.forward(  # noqa: E731
                qi, ki, v, qs, ks, output_dtype=torch.float16, block_m=m, block_n=128, num_warps=1)[0]
            ms, samples = bench(torch, call, warmup=3, iters=20)
            out = call()
            torch.gcu.synchronize()
            if ref is None:
                ref = out
            emit(probe="waves_block", shape=list(shape), block_m=bm, effective_block_m=eff,
                 n_cta=ctas, waves=ctas / 24.0, ms=ms, native_ms=nat, ratio=ms / nat,
                 maxdiff_vs_first=(out.float() - ref.float()).abs().max().item(),
                 samples=samples)
        del q, k, v, qi, qs, ki, ks, ref
        torch.gcu.empty_cache()


def probe_bw(torch, tmpdir, emit):
    mod = load_kernels(tmpdir, "bw", BW_SRC)
    nbytes = 512 << 20
    n = nbytes // 2
    x = torch.randn((n,), device="gcu:5", dtype=torch.float16)
    y = torch.empty_like(x)
    o = torch.empty((1 << 20,), device="gcu:5", dtype=torch.float16)
    ms, samples = bench(torch, lambda: y.copy_(x), warmup=2, iters=8)
    emit(probe="bw_torch", op="copy_", block=None, grid=None,
         gbps=2 * nbytes / 1e9 / ms * 1e3, ms=ms, samples=samples)
    for block in (2048, 8192, 16384, 32768):
        for grid in (24, 96, 384):
            ms, samples = bench(
                torch, lambda: mod.k_copy[(grid,)](x, y, n, BLOCK=block, num_warps=1, num_stages=1),
                warmup=2, iters=8)
            msr, samples_r = bench(
                torch, lambda: mod.k_read[(grid,)](x, o, n, BLOCK=block, num_warps=1, num_stages=1),
                warmup=2, iters=8)
            emit(probe="bw", op="copy", block=block, grid=grid,
                 gbps=2 * nbytes / 1e9 / ms * 1e3, ms=ms, samples=samples)
            emit(probe="bw", op="read", block=block, grid=grid,
                 gbps=nbytes / 1e9 / msr * 1e3, ms=msr, samples=samples_r)
    del x, y, o
    torch.gcu.empty_cache()


def probe_marginal(torch, tmpdir, emit):
    from flag_attn.runtime.backend._enflame.sage_attention import attn_qk_int8_per_block as attn
    from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8

    def sage_full(q, k, v, d):
        qi, qs, ki, ks = per_block_int8(q, k, sm_scale=d ** -0.5)
        return attn.forward(qi, ki, v, qs, ks, output_dtype=torch.float16)[0]

    for d, bm, ts in ((64, 256, (1024, 1536, 2048, 2560, 3072, 3584, 4096)),
                      (128, 128, (1024, 1536, 2048, 2560, 3072))):
        nat = {}
        sag = {}
        for t in ts:
            assert t // bm <= 24, "point is outside one wave"
            q = torch.randn((1, 1, t, d), device="gcu:5", dtype=torch.float16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            nat[t], _ = bench(
                torch,
                lambda: torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False)[0],
                warmup=3, iters=15)
            sag[t], samples = bench(torch, lambda: sage_full(q, k, v, d)[0], warmup=3, iters=15)
            emit(probe="marginal_point", head_dim=d, block_m=bm, seq_len=t,
                 iters=t // 128, n_cta=t // bm, waves=(t // bm) / 24.0,
                 native_ms=nat[t], sage_ms=sag[t], ratio=sag[t] / nat[t], samples=samples)
            del q, k, v
            torch.gcu.empty_cache()
        for name, table in (("native", nat), ("sage", sag)):
            slopes = [1e3 * (table[ts[i]] - table[ts[0]]) / ((ts[i] - ts[0]) / 128)
                      for i in range(1, len(ts))]
            emit(probe="marginal_summary", head_dim=d, block_m=bm, path=name,
                 slopes_us_per_tile=slopes, mean_us_per_tile=statistics.mean(slopes))


PROBES = dict(fp16=probe_fp16, fuse=probe_fuse, flags=probe_flags, stages=probe_stages,
              waves=probe_waves, bw=probe_bw, marginal=probe_marginal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True,
                        help="comma-separated subset of " + ",".join(PROBES))
    parser.add_argument("--jsonl", required=True)
    args = parser.parse_args()
    probes = [p for p in args.probe.split(",") if p]
    unknown = set(probes) - set(PROBES)
    if unknown:
        parser.error(f"unknown probe(s): {sorted(unknown)}")
    torch = device_guard()
    emit, close = writer(args.jsonl)
    try:
        emit_env(emit, ",".join(probes))
        with tempfile.TemporaryDirectory(prefix="sage006_recheck_") as tmpdir:
            for name in probes:
                PROBES[name](torch, tmpdir, emit)
    finally:
        close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
