#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SAGE-004 kernel-bottleneck localisation for the Enflame SageAttention kernel.

Method
------
The question is *which stage of the attention kernel* costs the time, so this
script changes one stage at a time in a copy of the real kernel and times what
is left.  Every variant is a single textual replacement of the shipped source,
asserted to match exactly once, so it measures "the shipped kernel minus this
one stage" rather than a hand-written stand-in.

Two things make the numbers trustworthy:

* ``both_dots`` keeps BOTH matmuls and both operand loads and deletes only the
  online-softmax bookkeeping.  That is a hard lower bound: no softmax
  optimisation can beat it.
* ``qk_matmul_only`` additionally deletes the PV matmul, so the QK matrix path
  can be compared against the tile's FLOPs and reported in TFLOPS.

Pre-quantized inputs only: this localises the attention kernel and is never
mixed into the formal full-call A/B (``sage004_ab.py``).

Exit status: 0 all variants measured, 3 a quench failed, 4 wrong device.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys

EXPECTED_PCI = "0000:83:00.0"
EXPECTED_INDEX = 5
KERNEL = "src/flag_attn/runtime/backend/_enflame/sage_attention/attn_qk_int8_per_block.py"

# The shipped inner-loop block.  Each replacement is asserted to occur once.
SOFTMAX_BLOCK = """            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]"""
ROW_MAX_BLOCK = "            m_ij = tl.maximum(m_i, tl.max(qk, 1))"
ROW_SUM = "            l_ij = tl.sum(p, 1).to(tl.float32)"
RESCALE = "            alpha = tl.math.exp2(m_i - m_ij)\n            acc = acc * alpha[:, None]"
EXP2 = "            p = tl.math.exp2(qk).to(tl.float16)"
PV_DOT = "            acc = tl.dot(p, v, acc)"

SOFTMAX_GONE = """            m_ij = m_i
            qk = qk - m_i[:, None]"""
ROW_SUM_GONE = "            l_ij = tl.full([BLOCK_M], 1.0, tl.float32)"
RESCALE_GONE = "            alpha = 1.0\n            acc = acc"
EXP2_GONE = "            p = (qk * 0.5).to(tl.float16)"
PV_GONE = "            acc = acc * 1.0"

VARIANTS = {
    # name: (removed stage, replacements)
    "full": ("none", ()),
    "no_row_max": ("online-softmax row max", ((ROW_MAX_BLOCK, "            m_ij = m_i"),)),
    "no_row_sum": ("probability row sum", ((ROW_SUM, ROW_SUM_GONE),)),
    "no_acc_rescale": ("accumulator rescale", ((RESCALE, RESCALE_GONE),)),
    "no_exp2": ("exp2", ((EXP2, EXP2_GONE),)),
    "no_pv_dot": ("PV matmul", ((PV_DOT, PV_GONE),)),
    "both_dots": ("all online-softmax bookkeeping",
                  ((SOFTMAX_BLOCK, SOFTMAX_GONE), (ROW_SUM, ROW_SUM_GONE),
                   (RESCALE, RESCALE_GONE))),
    "qk_matmul_only": ("all softmax and the PV matmul",
                       ((SOFTMAX_BLOCK, SOFTMAX_GONE), (ROW_SUM, ROW_SUM_GONE),
                        (RESCALE, RESCALE_GONE), (PV_DOT, PV_GONE))),
}


def device_guard() -> list[str]:
    import subprocess
    import re as _re
    try:
        raw = subprocess.run(["efsmi", "-L"], capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ["efsmi -L failed"]
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) < 5 or not fields[0].isdigit():
            continue
        if any(f.lower() == EXPECTED_PCI for f in fields):
            if int(fields[0]) != EXPECTED_INDEX:
                return [f"PCI {EXPECTED_PCI} is DEV {fields[0]}, expected DEV {EXPECTED_INDEX}"]
            return []
    return [f"efsmi does not report PCI {EXPECTED_PCI}"]


def build_variant(name: str, replacements):
    with open(KERNEL) as handle:
        source = handle.read()
    for old, new in replacements:
        count = source.count(old)
        if count != 1:
            raise AssertionError(f"{name}: pattern occurs {count} times, expected 1")
        source = source.replace(old, new, 1)
    path = os.path.join(os.environ.get("SAGE004_VARIANT_DIR", "/tmp/sage004_variants"),
                        f"variant_{name}.py")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(source)
    spec = importlib.util.spec_from_file_location(f"sage004_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bench(fn, warmup: int, iters: int) -> dict:
    import torch
    fn()
    torch.gcu.synchronize()
    for _ in range(warmup):
        fn()
    torch.gcu.synchronize()
    values = []
    for _ in range(iters):
        start = torch.gcu.Event(enable_timing=True)
        end = torch.gcu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.gcu.synchronize()
        values.append(start.elapsed_time(end))
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p10_ms": ordered[int(0.1 * (len(ordered) - 1))],
        "n": len(ordered),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", nargs="+", default=["1x2x4096x64", "4x32x1024x128"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--jsonl", default="")
    args = parser.parse_args()

    problems = device_guard()
    if problems:
        print(json.dumps({"event": "device_guard", "problems": problems}), flush=True)
        return 4

    import torch
    import torch_gcu  # noqa: F401
    from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import (
        per_block_int8,
    )

    torch.gcu.set_device(EXPECTED_INDEX)
    stream = open(args.jsonl, "w") if args.jsonl else None

    def emit(record: dict) -> None:
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if stream is not None:
            stream.write(line + "\n")
            stream.flush()

    def device_op(q, k, v):
        return torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, 0.0, False, False
        )[0]

    failures = []
    for text in args.shapes:
        shape = tuple(int(x) for x in text.replace("x", ",").split(","))
        b, h, t, d = shape
        flops = 4.0 * b * h * t * t * d
        torch.manual_seed(2026)
        q = torch.randn(shape, device=f"gcu:{EXPECTED_INDEX}", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        qi, qs, ki, ks = per_block_int8(q, k, sm_scale=d ** -0.5)
        torch.gcu.synchronize()

        device = bench(lambda: device_op(q, k, v), args.warmup, args.iters)
        emit({
            "event": "device_reference",
            "shape": list(shape),
            "ideal_gflop": flops / 1e9,
            "device_fused_ms": device,
            "device_tflops": flops / (device["median_ms"] * 1e-3) / 1e12,
        })

        results = {}
        for name, (removed, replacements) in VARIANTS.items():
            try:
                module = build_variant(name, replacements)
                stats = bench(
                    lambda: module.forward(qi, ki, v, qs, ks, output_dtype=torch.float16),
                    args.warmup, args.iters,
                )
            except Exception as exc:  # pragma: no cover - diagnostic
                failures.append(f"{shape} {name}: {type(exc).__name__}: {exc}")
                emit({"event": "variant", "shape": list(shape), "variant": name,
                      "removed": removed, "error": f"{type(exc).__name__}: {exc}"})
                continue
            results[name] = stats
            emit({
                "event": "variant",
                "shape": list(shape),
                "variant": name,
                "removed": removed,
                "ideal_gflop": flops / 1e9,
                "timing": stats,
                "tflops": flops / (stats["median_ms"] * 1e-3) / 1e12,
                "over_device": stats["median_ms"] / device["median_ms"],
            })

        if "full" in results:
            base = results["full"]["median_ms"]
            summary = {
                "event": "shape_summary",
                "shape": list(shape),
                "device_fused_ms": device["median_ms"],
                "full_ms": base,
                "full_over_device": base / device["median_ms"],
                "removed_stage_cost": {
                    name: {
                        "ms_saved": base - stats["median_ms"],
                        "fraction_of_full": (base - stats["median_ms"]) / base,
                    }
                    for name, stats in results.items() if name != "full"
                },
            }
            emit(summary)

    emit({"event": "decision", "status": "PASS" if not failures else "FAIL",
          "failures": failures})
    if stream is not None:
        stream.close()
    return 0 if not failures else 3


if __name__ == "__main__":
    sys.exit(main())
