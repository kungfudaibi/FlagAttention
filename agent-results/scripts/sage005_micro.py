#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SAGE-005 microbenchmarks: where does the fused-SDPA gap actually live?

Four independent probes, each writing one JSON object per line:

``--probe resident``
    Pure ``tl.dot`` throughput with both operands resident in shared memory and
    no per-iteration loads.  This is the ceiling Triton can reach on this
    device for the exact tile shapes the Sage kernel uses, so it separates
    "the matmul units are slow" from "the loop around them is slow".

``--probe tilemajor``
    The same dots but with a per-iteration tile load in the layout the Sage
    kernel uses (tile-major INT8/FP16 K).  The delta against ``resident`` is
    the cost of the tile feed.

``--probe slope``
    Full ``_attn_fwd`` variants benchmarked across query lengths 1024..4096 at
    one CTA per programme.  Because every CTA sees the same number of KV
    iterations, the *slope* of latency against iteration count is the steady
    state cost of one online-softmax step, with launch, prologue and epilogue
    differenced out.  Ablations are produced by rewriting exactly one source
    fragment of the shipped kernel, so nothing else changes.

``--probe epilogue``
    Standalone kernels that replay the softmax bookkeeping over a tile so the
    reduce / broadcast-subtract / elementwise costs can be priced separately.

Every probe pins GCU 5 by PCI address before the first device action and never
touches another ordinal.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys

DEVICE_INDEX = 5
DEVICE_PCI = "0000:83:00.0"
DEVICE_SERIAL = "A0D5460520864"
KERNEL = os.environ.get(
    "SAGE005_KERNEL",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "src/flag_attn/runtime/backend/_enflame/sage_attention/attn_qk_int8_per_block.py",
    ),
)


def device_guard():
    """Bind GCU 5 and prove the mapping before touching the device."""
    import torch
    import torch_gcu  # noqa: F401

    identity = ""
    for field in ("DEVICE", "PCIE"):
        identity += subprocess.run(
            ["efsmi", "-q", "-d", field, "-i", str(DEVICE_INDEX)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    # efsmi prints the Bus/Dev/Func of a PCIe function on separate labelled rows,
    # so rebuild the canonical BDF and compare it against the assigned address.
    bus = re.search(r"Bus\s*:\s*([0-9a-fA-F]+)", identity)
    dev = re.search(r"Dev\s*:\s*([0-9a-fA-F]+)", identity)
    func = re.search(r"Func\s*:\s*([0-9a-fA-F]+)", identity)
    if not (bus and dev and func):
        raise SystemExit(f"cannot read a PCI address for GCU {DEVICE_INDEX}")
    observed = f"0000:{int(bus.group(1), 16):02x}:{int(dev.group(1), 16):02x}.{int(func.group(1), 16)}"
    if observed != DEVICE_PCI or DEVICE_SERIAL not in identity:
        raise SystemExit(
            f"GCU {DEVICE_INDEX} is {observed}, not {DEVICE_PCI} ({DEVICE_SERIAL}); refusing to run."
        )
    torch.gcu.set_device(DEVICE_INDEX)
    if torch.gcu.current_device() != DEVICE_INDEX:
        raise SystemExit("torch.gcu.set_device did not take effect")
    return torch


def bench(fn, warmup=3, iters=15):
    """Median device-event latency of one call, fully synchronized."""
    torch = bench.torch
    fn()
    torch.gcu.synchronize()
    for _ in range(warmup):
        fn()
    torch.gcu.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.gcu.Event(enable_timing=True)
        end = torch.gcu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.gcu.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), samples


# --------------------------------------------------------------------------
# resident / tilemajor: pure dot throughput
# --------------------------------------------------------------------------

RESIDENT_SRC = r'''
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
'''

TILEMAJOR_SRC = r'''
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
'''


def _load_kernels(name, source, tmpdir):
    """Triton needs @jit definitions in a real file, so materialize one."""
    path = os.path.join(tmpdir, f"sage005_{name}_kernels.py")
    with open(path, "w") as handle:
        handle.write(source)
    spec = importlib.util.spec_from_file_location(f"sage005_{name}_kernels", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe_resident(torch, tmpdir, emit):
    mod = _load_kernels("resident", RESIDENT_SRC, tmpdir)
    niter = 200
    for bm, bn, dim in ((256, 128, 64), (128, 128, 128)):
        flops = 2.0 * bm * bn * dim * niter
        for grid in (1, 12, 24, 48, 96):
            q = torch.randn((grid * bm * dim,), device="gcu:5", dtype=torch.float16) * 0.05
            k = torch.randn((grid * dim * bn,), device="gcu:5", dtype=torch.float16) * 0.05
            out = torch.empty((grid * bm * bn,), device="gcu:5", dtype=torch.float16)
            ms, _ = bench(
                lambda: mod.k_res[(grid,)](
                    q, k, out, niter, BM=bm, BN=bn, D=dim, num_warps=1, num_stages=1
                )
            )
            emit(
                probe="resident",
                block_m=bm,
                block_n=bn,
                head_dim=dim,
                n_cta=grid,
                iters=niter,
                ms=ms,
                tflops=grid * flops / ms / 1e9,
                tflops_per_cta=flops / ms / 1e9,
            )
            del q, k, out
            torch.gcu.empty_cache()


def probe_tilemajor(torch, tmpdir, emit):
    mod = _load_kernels("tilemajor", TILEMAJOR_SRC, tmpdir)
    niter = 32
    for bm, bn, dim in ((256, 128, 64), (128, 128, 128)):
        flops = 2.0 * bm * bn * dim * niter
        for grid in (1, 24, 48, 96):
            q = torch.randn((bm, dim), device="gcu:5", dtype=torch.float16) * 0.05
            k = torch.randn((grid * niter * dim * bn,), device="gcu:5", dtype=torch.float16) * 0.05
            p = torch.rand((grid * bm * bn,), device="gcu:5", dtype=torch.float16)
            v = torch.randn((grid * niter * bn * dim,), device="gcu:5", dtype=torch.float16) * 0.05
            out = torch.empty((grid * bm * max(bn, dim),), device="gcu:5", dtype=torch.float16)
            ms_qk, _ = bench(
                lambda: mod.k_qk_tm[(grid,)](
                    q, k, out, niter, BM=bm, BN=bn, D=dim, num_warps=1, num_stages=1
                )
            )
            ms_pv, _ = bench(
                lambda: mod.k_pv_tm[(grid,)](
                    p, v, out, niter, BM=bm, BN=bn, D=dim, num_warps=1, num_stages=1
                )
            )
            emit(
                probe="tilemajor",
                block_m=bm,
                block_n=bn,
                head_dim=dim,
                n_cta=grid,
                iters=niter,
                qk_ms=ms_qk,
                pv_ms=ms_pv,
                qk_tflops=grid * flops / ms_qk / 1e9,
                pv_tflops=grid * flops / ms_pv / 1e9,
                qk_tflops_per_cta=flops / ms_qk / 1e9,
                pv_tflops_per_cta=flops / ms_pv / 1e9,
                sum_tflops=grid * 2 * flops / (ms_qk + ms_pv) / 1e9,
            )
            del q, k, p, v, out
            torch.gcu.empty_cache()


# --------------------------------------------------------------------------
# slope: per-online-softmax-step cost of the shipped kernel and ablations
# --------------------------------------------------------------------------

# Each ablation rewrites exactly one fragment of the shipped kernel.  The
# fragment must occur exactly once, so a kernel refactor fails loudly instead
# of silently benchmarking something else.
ABLATIONS = {
    "full": (),
    "no_row_max": ((
        "            m_ij = tl.maximum(m_i, tl.max(qk, 1))\n"
        "            qk = qk - m_ij[:, None]",
        "            m_ij = m_i",
    ),),
    "no_sub": ((
        "            m_ij = tl.maximum(m_i, tl.max(qk, 1))\n"
        "            qk = qk - m_ij[:, None]",
        "            m_ij = m_i\n            qk = qk",
    ),),
    "both_dots": (
        (
            "            m_ij = tl.maximum(m_i, tl.max(qk, 1))\n"
            "            qk = qk - m_ij[:, None]",
            "            m_ij = m_i\n            qk = qk - m_i[:, None]",
        ),
        (
            "            l_ij = tl.sum(p, 1).to(tl.float32)",
            "            l_ij = tl.full([BLOCK_M], 1.0, tl.float32)",
        ),
        (
            "            alpha = tl.math.exp2(m_i - m_ij)\n"
            "            acc = acc * alpha[:, None]",
            "            alpha = 1.0\n            acc = acc",
        ),
    ),
    "no_pv_dot": (("            acc = tl.dot(p, v, acc)", "            acc = acc * 1.0"),),
    "qk_matmul_only": (
        (
            "            m_ij = tl.maximum(m_i, tl.max(qk, 1))\n"
            "            qk = qk - m_ij[:, None]",
            "            m_ij = m_i\n            qk = qk - m_i[:, None]",
        ),
        (
            "            l_ij = tl.sum(p, 1).to(tl.float32)",
            "            l_ij = tl.full([BLOCK_M], 1.0, tl.float32)",
        ),
        (
            "            alpha = tl.math.exp2(m_i - m_ij)\n"
            "            acc = acc * alpha[:, None]",
            "            alpha = 1.0\n            acc = acc",
        ),
        ("            acc = tl.dot(p, v, acc)", "            acc = acc * 1.0"),
    ),
}


def build_variant(torch, name, rewrites, tmpdir):
    with open(KERNEL) as handle:
        source = handle.read()
    for old, new in rewrites:
        if source.count(old) != 1:
            raise SystemExit(
                f"ablation {name!r}: fragment occurs {source.count(old)} times, expected 1"
            )
        source = source.replace(old, new, 1)
    path = os.path.join(tmpdir, f"sage005_variant_{name}.py")
    with open(path, "w") as handle:
        handle.write(source)
    spec = importlib.util.spec_from_file_location(f"sage005_variant_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe_slope(torch, tmpdir, emit, variants, lengths):
    from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import (
        per_block_int8,
    )

    shape_head = (1, 1)
    for name in variants:
        module = build_variant(torch, name, ABLATIONS[name], tmpdir)
        wanted = module._attn_fwd
        latencies = []
        for length in lengths:
            q = torch.randn((*shape_head, length, 64), device="gcu:5", dtype=torch.float16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            qi, qs, ki, ks = per_block_int8(q, k, sm_scale=64**-0.5)
            torch.gcu.synchronize()
            ms, samples = bench(
                lambda: module.forward(qi, ki, v, qs, ks, output_dtype=torch.float16)[0]
            )
            latencies.append(ms)
            emit(
                probe="slope",
                variant=name,
                seq_len=length,
                block_m=256,
                block_n=128,
                n_cta=(length // 256),
                iters=(length // 128),
                ms=ms,
                samples=samples,
            )
            del q, k, v, qi, qs, ki, ks
            torch.gcu.empty_cache()
        # Milliseconds per KV iteration, converted to microseconds.
        slopes = [
            1e3 * (latencies[i] - latencies[0]) / ((lengths[i] - lengths[0]) / 128)
            for i in range(1, len(lengths))
        ]
        emit(
            probe="slope_summary",
            variant=name,
            seq_lens=list(lengths),
            latencies_ms=latencies,
            slope_us_per_iter=slopes,
            mean_slope_us_per_iter=sum(slopes) / len(slopes),
        )
        # Prove the ablation did not delete the dot ops it is supposed to keep.
        caches = wanted.device_caches[DEVICE_INDEX][0]
        for _, compiled in caches.items():
            gcuir = compiled.asm.get("gcuir", "")
            emit(
                probe="ablation_artifact",
                variant=name,
                gcu_matmul=gcuir.count("gcu.matmul"),
                gcuir_bytes=len(gcuir),
                n_regs=getattr(compiled, "n_regs", None),
                n_spills=getattr(compiled, "n_spills", None),
                num_warps=compiled.metadata.num_warps,
                num_stages=compiled.metadata.num_stages,
            )
            break


# --------------------------------------------------------------------------
# epilogue: price the softmax bookkeeping primitives
# --------------------------------------------------------------------------

EPILOGUE_SRC = r'''
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
'''

EPILOGUE_KERNELS = {
    "rowmax": "k_rowmax",
    "rowmax_sub": "k_rowmax_sub",
    "rowmax_mul_sub": "k_mul",
    "rowsum": "k_rowsum",
}


def probe_epilogue(torch, tmpdir, emit, warps):
    mod = _load_kernels("epilogue", EPILOGUE_SRC, tmpdir)
    bm, bn, passes = 256, 128, 32
    grid = 24
    x = torch.randn((grid * bm * bn,), device="gcu:5", dtype=torch.float32) * 0.01
    out = torch.empty((grid * bm,), device="gcu:5", dtype=torch.float32)
    for label, kernel_name in EPILOGUE_KERNELS.items():
        kernel = getattr(mod, kernel_name)
        for num_warps in warps:
            ms, _ = bench(
                lambda: kernel[(grid,)](
                    x, out, passes, BM=bm, BN=bn, num_warps=num_warps, num_stages=1
                )
            )
            emit(
                probe="epilogue",
                kernel=label,
                block_m=bm,
                block_n=bn,
                passes=passes,
                n_cta=grid,
                num_warps=num_warps,
                ms=ms,
                us_per_pass=ms * 1e3 / passes,
            )
    del x, out
    torch.gcu.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe",
        action="append",
        required=True,
        help="comma-separated subset of resident,tilemajor,slope,epilogue",
    )
    parser.add_argument("--variants", default="full,no_row_max,no_sub,both_dots,no_pv_dot,qk_matmul_only")
    parser.add_argument("--lengths", default="1024,2048,3072,4096")
    parser.add_argument("--warps", default="1,2,4")
    parser.add_argument("--tmpdir", default="/tmp/sage005_micro")
    parser.add_argument("--jsonl", required=True)
    args = parser.parse_args()

    probes = [p for group in args.probe for p in group.split(",") if p]
    unknown = set(probes) - {"resident", "tilemajor", "slope", "epilogue"}
    if unknown:
        parser.error(f"unknown probe(s): {sorted(unknown)}")

    os.makedirs(args.tmpdir, exist_ok=True)
    torch = device_guard()
    bench.torch = torch
    handle = open(args.jsonl, "w")

    def emit(**fields):
        fields["device_index"] = DEVICE_INDEX
        fields["pci"] = DEVICE_PCI
        handle.write(json.dumps(fields, sort_keys=True) + "\n")
        handle.flush()
        summary = {k: v for k, v in fields.items() if not isinstance(v, list)}
        print(json.dumps(summary, sort_keys=True), flush=True)

    if "resident" in probes:
        probe_resident(torch, args.tmpdir, emit)
    if "tilemajor" in probes:
        probe_tilemajor(torch, args.tmpdir, emit)
    if "slope" in probes:
        probe_slope(
            torch,
            args.tmpdir,
            emit,
            [v for v in args.variants.split(",") if v],
            [int(x) for x in args.lengths.split(",")],
        )
    if "epilogue" in probes:
        probe_epilogue(torch, args.tmpdir, emit, [int(x) for x in args.warps.split(",")])
    handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
