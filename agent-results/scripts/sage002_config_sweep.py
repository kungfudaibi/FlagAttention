#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SAGE-002 tuning sweep: can a launch configuration close the gap?

Before declaring the SageAttention full call a no-go against the GCU-native
fused attention op, this script exhausts the knobs the Enflame kernel already
exposes (``block_m``, ``block_n``, ``num_warps``, ``num_stages``) plus the
optional feature flags, on the same GCU 1 and the same FP16 inputs.  It also
measures the repository benchmark's own workload shape ``(4, 32, T, 128)`` so
the verdict is not an artifact of the narrow SAGE-001 provisional shape.

Every configuration is validated for accuracy against the exact baseline
before its timing is reported; a fast but wrong configuration is not a win.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time

import torch
import torch.nn.functional as F

import torch_gcu  # noqa: F401

from flag_attn import sage_attention
from flag_attn.sage_attention import per_block_int8


def parse_shape(text):
    parts = tuple(int(x) for x in text.replace("x", ",").split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"bad shape {text!r}; want B,H,T,D")
    return parts


def metrics(actual, expected):
    a = actual.float().flatten()
    e = expected.float().flatten()
    if a.numel() > 1_048_576:
        stride = (a.numel() + 1_048_576 - 1) // 1_048_576
        a, e = a[::stride], e[::stride]
    diff = a - e
    return {
        "cosine": F.cosine_similarity(a, e, dim=0).item(),
        "relative_l2": (diff.norm() / e.norm().clamp_min(1e-8)).item(),
        "max_abs": diff.abs().max().item(),
    }


def bench(fn, samples=15, warmup=4):
    for _ in range(warmup):
        fn()
    torch.gcu.synchronize()
    values, out = [], None
    for _ in range(samples):
        start = torch.gcu.Event(enable_timing=True)
        end = torch.gcu.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.gcu.synchronize()
        values.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples_ms": values,
    }, out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes", type=parse_shape, nargs="+",
        default=[(1, 2, 4096, 64), (4, 32, 1024, 128)],
    )
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--out", default="")
    parser.add_argument("--tag", default="sweep")
    args = parser.parse_args()

    torch.gcu.set_device(1)
    stream = open(args.out, "w") if args.out else None

    def emit(record):
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if stream is not None:
            stream.write(line + "\n")
            stream.flush()

    emit({
        "event": "meta",
        "tag": args.tag,
        "device": "gcu:1",
        "pci": "0000:83:00.0",
        "dtype": "float16",
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": torch.__version__,
        "torch_gcu": torch_gcu.__version__,
    })

    configs = [
        {"label": "default"},
        {"label": "warps2", "num_warps": 2},
        {"label": "warps4", "num_warps": 4},
        {"label": "bm64", "block_m": 64},
        {"label": "bn64", "block_n": 64},
        {"label": "bm64_bn64", "block_m": 64, "block_n": 64},
        {"label": "stages2", "num_stages": 2},
        {"label": "no_prefetch_v", "prefetch_v": False},
        {"label": "no_prefetch_k", "prefetch_k_next": False},
        {"label": "k_prequant_off", "predequantize_k": False},
        {"label": "scale_after_dot", "qk_scale_before_dot": False},
    ]

    failures = []
    for shape in args.shapes:
        torch.manual_seed(7)
        q = torch.randn(shape, device="gcu:1", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        exact = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )
        device = torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, 0.0, False, False
        )[0]
        torch.gcu.synchronize()

        exact_bench, _ = bench(lambda: F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False), args.samples, args.warmup)
        device_bench, _ = bench(
            lambda: torch.ops.aten._scaled_dot_product_flash_attention(
                q, k, v, 0.0, False, False)[0], args.samples, args.warmup)
        emit({
            "event": "baseline",
            "shape": list(shape),
            "exact_sdpa": exact_bench,
            "device_sdpa": device_bench,
            "device_vs_exact_cosine": metrics(device, exact)["cosine"],
        })

        rows = []
        for config in configs:
            kwargs = {k2: v2 for k2, v2 in config.items() if k2 != "label"}

            def full(kwargs=kwargs):
                a, b, c, d = per_block_int8(q, k)
                return sage_attention(a, c, v, b, d, **kwargs)[0]

            try:
                result, out = bench(full, args.samples, args.warmup)
            except Exception as exc:  # a rejected configuration is data too
                emit({"event": "config", "shape": list(shape),
                      "label": config["label"], "kwargs": kwargs,
                      "error": f"{type(exc).__name__}: {exc}"})
                continue
            accuracy = metrics(out, exact)
            accuracy_device = metrics(out, device)
            ok = (
                accuracy["cosine"] >= 0.99
                and accuracy["relative_l2"] <= 0.05
                and bool(torch.isfinite(out).all().item())
            )
            if not ok:
                failures.append(
                    f"shape {list(shape)} config {config['label']}: "
                    f"cosine {accuracy['cosine']:.6f} "
                    f"rel_l2 {accuracy['relative_l2']:.6f}"
                )
            row = {
                "event": "config",
                "shape": list(shape),
                "label": config["label"],
                "kwargs": kwargs,
                "accuracy_vs_exact_sdpa": accuracy,
                "accuracy_vs_device_sdpa": accuracy_device,
                "accuracy_ok": ok,
                "timing": result,
                "speedup_vs_exact_sdpa": exact_bench["median_ms"] / result["median_ms"],
                "speedup_vs_device_sdpa": device_bench["median_ms"] / result["median_ms"],
            }
            rows.append(row)
            emit(row)

        if rows:
            best = min(rows, key=lambda r: r["timing"]["median_ms"])
            emit({
                "event": "shape_best",
                "shape": list(shape),
                "best_label": best["label"],
                "best_median_ms": best["timing"]["median_ms"],
                "device_sdpa_ms": device_bench["median_ms"],
                "best_speedup_vs_exact_sdpa": best["speedup_vs_exact_sdpa"],
                "best_speedup_vs_device_sdpa": best["speedup_vs_device_sdpa"],
                "best_accuracy_ok": best["accuracy_ok"],
            })
        del q, k, v, exact, device

    if stream is not None:
        stream.close()
    status = "PASS" if not failures else "FAIL"
    print(json.dumps({"event": "decision", "status": status,
                      "failures": failures}, sort_keys=True), flush=True)
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
