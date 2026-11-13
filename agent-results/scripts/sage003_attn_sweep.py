#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SAGE-003 attention-kernel launch sweep on pre-quantized inputs.

Purpose
-------
The formal A/B shows the Sage full call is ~4x the device fused op and that
quantization is only 15-34% of it, so most of the gap is the attention kernel
itself.  This script answers the obvious follow-up -- "is the kernel just badly
configured?" -- by sweeping every launch knob the Enflame override exposes
(``block_m``, ``block_n``, ``num_warps``, ``num_stages``) on inputs that are
*already* quantized.  Quantization is therefore excluded from every number
here, which makes this the most favourable possible setting for Sage.

Every candidate is validated against the device fused op before it is timed, so
a "fast" miscompiled configuration cannot be reported as an improvement.

Exit status
-----------
0   the sweep completed and every candidate passed the numeric gate
2   environment/backend unusable
3   a candidate produced non-finite or out-of-gate output
4   wrong device identity
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

from sage003_ab import (  # noqa: E402  (same directory, imported after argv check)
    EXPECTED_INDEX,
    EXPECTED_PCI,
    device_guard,
    metrics,
    one_sample,
    parse_shape,
    summarize,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=parse_shape, nargs="+",
                        default=[(1, 2, 4096, 64), (4, 32, 1024, 128)])
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--jsonl", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--accuracy-cosine", type=float, default=0.99)
    parser.add_argument("--accuracy-rel-l2", type=float, default=0.05)
    args = parser.parse_args()

    problems, guard = device_guard(EXPECTED_INDEX, EXPECTED_PCI)
    print(json.dumps(guard, sort_keys=True), flush=True)
    if problems:
        return 4

    import torch
    import torch_gcu  # noqa: F401
    from flag_attn import sage_attention
    from flag_attn.sage_attention import per_block_int8

    torch.gcu.set_device(EXPECTED_INDEX)
    if not torch.gcu.is_available() or torch.gcu.device_count() <= EXPECTED_INDEX:
        print("ERROR: assigned GCU is not visible", file=sys.stderr)
        return 2

    out_stream = open(args.jsonl, "w") if args.jsonl else None

    def emit(record: dict) -> None:
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if out_stream is not None:
            out_stream.write(line + "\n")
            out_stream.flush()

    def device_sdpa(q, k, v):
        return torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, 0.0, False, False
        )[0]

    # The full cross product is 4*3*2*2 = 48 timed variants per shape; the
    # subset below spans every axis while keeping a formal run under a minute.
    configs = [
        {"block_m": 128, "block_n": 128, "num_warps": 1, "num_stages": 1},
        {"block_m": 128, "block_n": 128, "num_warps": 1, "num_stages": 2},
        {"block_m": 128, "block_n": 64, "num_warps": 1, "num_stages": 1},
        {"block_m": 64, "block_n": 128, "num_warps": 1, "num_stages": 1},
        {"block_m": 128, "block_n": 128, "num_warps": 2, "num_stages": 1},
        {"block_m": 128, "block_n": 128, "num_warps": 4, "num_stages": 1},
        {"block_m": 64, "block_n": 64, "num_warps": 1, "num_stages": 1},
    ]

    failures: list[str] = []
    results: dict[str, dict] = {}

    for shape in args.shapes:
        torch.manual_seed(2026)
        q = torch.randn(shape, device=f"gcu:{EXPECTED_INDEX}", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        q8, q_scale, k8, k_scale = per_block_int8(q, k)
        reference = device_sdpa(q, k, v)
        torch.gcu.synchronize()
        reference_ms = statistics.median(
            [one_sample(lambda: device_sdpa(q, k, v))[0] for _ in range(args.samples)]
        )
        key = "x".join(str(x) for x in shape)
        results[key] = {"shape": list(shape), "device_ms": reference_ms, "configs": []}

        for config in configs:
            def run(config=config):
                return sage_attention(
                    q8, k8, v, q_scale, k_scale,
                    output_dtype=torch.float16, **config,
                )[0]

            try:
                out = run()
            except Exception as exc:  # config rejected by the kernel, not a failure
                results[key]["configs"].append({
                    "config": config,
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                })
                continue
            torch.gcu.synchronize()

            accuracy = metrics(out, reference)
            config_failures = []
            if not accuracy["finite"]:
                config_failures.append("non-finite output")
            if accuracy["cosine"] < args.accuracy_cosine:
                config_failures.append(f"cosine {accuracy['cosine']:.6f}")
            if accuracy["relative_l2"] > args.accuracy_rel_l2:
                config_failures.append(f"relative_l2 {accuracy['relative_l2']:.6f}")
            if config_failures:
                failures.append(f"{key} {config}: {config_failures}")

            for _ in range(args.warmup):
                run()
            torch.gcu.synchronize()
            samples = [one_sample(run)[0] for _ in range(args.samples)]
            record = {
                "config": config,
                "accuracy": accuracy,
                "timing": summarize(samples),
                "over_device": statistics.median(samples) / reference_ms,
                "valid": not config_failures,
            }
            results[key]["configs"].append(record)
            torch.gcu.empty_cache()

        best = min(
            (c for c in results[key]["configs"] if c.get("valid")),
            key=lambda c: c["timing"]["median_ms"],
            default=None,
        )
        results[key]["best"] = best
        del q, k, v, q8, q_scale, k8, k_scale, reference
        torch.gcu.empty_cache()

    decision = {
        "event": "decision",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "device_index": EXPECTED_INDEX,
        "pci": EXPECTED_PCI,
        "results": results,
    }
    emit(decision)
    if out_stream is not None:
        out_stream.close()
    if args.summary:
        with open(args.summary, "w") as handle:
            json.dump(decision, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0 if decision["status"] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
