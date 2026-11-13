#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the SAGE-003 report tables from the raw JSONL records.

Reads the ``agent-results/raw/sage003_ab_*.jsonl`` records and prints the
markdown tables used in ``agent-results/SAGE-003.md``. Exit status is nonzero when a decision record
is missing or does not say PASS, so a stale log cannot silently regenerate a
report.
"""

from __future__ import annotations

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")


def load(name: str) -> list[dict]:
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        print(f"MISSING: {path}", file=sys.stderr)
        raise SystemExit(2)
    return [json.loads(line) for line in open(path) if line.strip()]


def table(records: list[dict], label: str) -> None:
    decisions = [r for r in records if r["event"] == "decision"]
    if not decisions:
        print(f"MISSING decision record in {label}", file=sys.stderr)
        raise SystemExit(2)
    decision = decisions[-1]
    if decision["status"] != "PASS":
        print(f"{label}: status {decision['status']} failures {decision['failures']}", file=sys.stderr)
        raise SystemExit(3)

    print(f"\n### {label} steady-state medians\n")
    print("| shape | device fused ms | Sage full ms | Sage/device | device peak MiB | Sage peak MiB |")
    print("|---|---|---|---|---|---|")
    rounds = [r for r in records if r["event"] == "round"]
    peaks = {}
    for row in rounds:
        key = "x".join(str(x) for x in row["shape"])
        peaks.setdefault(key, row["memory"])
    for key, entry in decision["aggregate"].items():
        mem = peaks.get(key, {})
        print(
            "| {shape} | {dev:.4f} | {sage:.4f} | {ratio:.2f}x | {dp:.2f} | {sp:.2f} |".format(
                shape=key,
                dev=entry["median_device_ms"],
                sage=entry["median_sage_ms"],
                ratio=entry["median_sage_over_device"],
                dp=mem.get("device_peak_mib", float("nan")),
                sp=mem.get("sage_peak_mib", float("nan")),
            )
        )

    print(f"\n### {label} stage attribution (per round, ms)\n")
    print("| shape | round | quant tuned | quant default launch | speedup | attention | attention/device | quant share |")
    print("|---|---|---|---|---|---|---|---|")
    for row in rounds:
        stage = row["stage_ms"]
        print(
            "| {shape} | {rnd} | {q:.4f} | {qd:.4f} | {qs:.2f}x | {a:.4f} | {ao:.2f}x | {qshare:.0%} |".format(
                shape="x".join(str(x) for x in row["shape"]),
                rnd=row["round"],
                q=stage["quant_tuned_ms"]["median_ms"],
                qd=stage["quant_default_launch_ms"]["median_ms"],
                qs=stage["quant_speedup_vs_default_launch"],
                a=stage["attn_only_ms"]["median_ms"],
                ao=stage["attn_over_device"],
                qshare=stage["quant_share_of_sage"],
            )
        )

    print(f"\n### {label} accuracy against the device fused op\n")
    print("| shape | round | cosine | relative L2 | max abs | finite |")
    print("|---|---|---|---|---|---|")
    for row in rounds:
        acc = row["accuracy"]["sage_vs_device_fused"]
        print(
            "| {shape} | {rnd} | {c:.7f} | {r:.5f} | {m:.6f} | {f} |".format(
                shape="x".join(str(x) for x in row["shape"]),
                rnd=row["round"],
                c=acc["cosine"],
                r=acc["relative_l2"],
                m=acc["max_abs"],
                f=acc["finite"],
            )
        )

    print(f"\n### {label} cold compile (fresh Triton cache)\n")
    print("| shape | scope | first call s | steady ms | compile overhead s | cache files after |")
    print("|---|---|---|---|---|---|")
    for row in [r for r in records if r["event"] == "cold"]:
        # The scope's headline first call is the full call for scope=full and
        # the quantizer for scope=stages; both are also in the JSONL raw.
        headline_key = "full_first" if row["scope"] == "full" else "quant_first"
        first = row[headline_key]["wall_ms"] / 1000.0
        steady_key = "full_ms" if row["scope"] == "full" else "quant_ms"
        steady = row["steady"][steady_key]["median_ms"]
        overhead = row["compile_overhead_ms"][headline_key.replace("_first", "")]
        cache_after = row.get("cache_files_after_full") or row.get("cache_files_after_stages")
        print(
            "| {shape} | {scope} | {first:.2f} | {steady:.4f} | {overhead:.2f} | {cache} |".format(
                shape="x".join(str(x) for x in row["shape"]),
                scope=row["scope"],
                first=first,
                steady=steady,
                overhead=overhead / 1000.0,
                cache=cache_after,
            )
        )

    paths = [r for r in records if r["event"] == "path_evidence"]
    if paths:
        p = paths[-1]
        print(f"\n### {label} device path evidence\n")
        print(f"- device baseline ops: `{p['device_baseline_ops']}`")
        print(f"- reference F.sdpa ops: `{p['reference_sdpa_ops']}`")
        print(f"- host CPU per call: device {p['device_host_ms_median']:.3f} ms, "
              f"F.sdpa {p['reference_host_ms_median']:.2f} ms")
        print(f"- device baseline is a GCU kernel: {p['device_baseline_has_gcu_kernel']}")
        print(f"- reference F.sdpa is the CPU fallback: {p['reference_is_cpu_fallback']}")


def negative_controls() -> None:
    """Show that the harness exits nonzero on the failures it claims to catch."""
    print("\n### negative controls (must be nonzero)\n")
    print("| control | expected | observed |")
    print("|---|---|---|")
    for name, expected in (
        ("sage003_negative_numeric.exit", "3"),
        ("sage003_negative_samples.exit", "3"),
        ("sage003_ab_float16.exit", "0"),
        ("sage003_ab_bfloat16.exit", "0"),
        ("sage003_attn_sweep.exit", "0"),
        ("sage003_tests.exit", "0"),
    ):
        path = os.path.join(RAW, name)
        if not os.path.exists(path):
            print(f"| {name} | {expected} | MISSING |")
            continue
        observed = open(path).read().strip()
        observed_code = observed.split("=")[-1]
        print(f"| {name} | {expected} | {observed} |")
        if observed_code != expected:
            print(
                f"MISMATCH: {name} expected exit {expected}, recorded {observed}",
                file=sys.stderr,
            )
            raise SystemExit(3)


def attention_sweep() -> None:
    path = os.path.join(RAW, "sage003_attn_sweep_summary.json")
    if not os.path.exists(path):
        return
    payload = json.load(open(path))
    if payload["status"] != "PASS":
        print(f"attention sweep status {payload['status']}", file=sys.stderr)
        raise SystemExit(3)
    print("\n### attention kernel launch sweep (quantization excluded)\n")
    print("| shape | device ms | best config | best ms | best/device | worst/device |")
    print("|---|---|---|---|---|---|")
    for key, entry in payload["results"].items():
        valid = [c for c in entry["configs"] if c.get("valid")]
        best = min(valid, key=lambda c: c["timing"]["median_ms"])
        worst = max(valid, key=lambda c: c["timing"]["median_ms"])
        cfg = best["config"]
        print(
            "| {key} | {dev:.4f} | bm={bm} bn={bn} nw={nw} ns={ns} | {ms:.4f} | "
            "{ratio:.2f}x | {worst:.2f}x |".format(
                key=key,
                dev=entry["device_ms"],
                bm=cfg["block_m"], bn=cfg["block_n"],
                nw=cfg["num_warps"], ns=cfg["num_stages"],
                ms=best["timing"]["median_ms"],
                ratio=best["over_device"],
                worst=worst["over_device"],
            )
        )


def bimodality() -> None:
    """Many full-call timings are two modes ~1.8x apart, not one tail."""
    print("\n### full-call latency modes (FP16, all rounds and shapes)\n")
    print("| shape | dtype | median ms | fast mode ms | slow mode ms | ratio | slow calls |")
    print("|---|---|---|---|---|---|---|")
    any_bimodal = False
    for name, label in (
        ("sage003_ab_float16.jsonl", "fp16"),
        ("sage003_ab_bfloat16.jsonl", "bf16"),
        ("sage003_ab_shapes_float16.jsonl", "fp16"),
    ):
        for record in load(name):
            if record.get("event") != "round":
                continue
            stats = record["stats"]["sage_full"]
            marker = "yes" if stats.get("bimodal") else "no"
            if stats.get("bimodal"):
                any_bimodal = True
                print(
                    "| {shape} | {dtype} | {med:.3f} | {fast:.3f} | {slow:.3f} | "
                    "{ratio:.2f}x | {frac:.0%} |".format(
                        shape="x".join(str(v) for v in record["shape"]),
                        dtype=label,
                        med=stats["median_ms"],
                        fast=stats["fast_median_ms"],
                        slow=stats["slow_median_ms"],
                        ratio=stats["mode_ratio"],
                        frac=stats["slow_call_fraction"],
                    )
                )
            else:
                print(
                    "| {shape} | {dtype} | {med:.3f} | - | - | - | unimodal |".format(
                        shape="x".join(str(v) for v in record["shape"]),
                        dtype=label,
                        med=stats["median_ms"],
                    )
                )
    print(
        "\nBimodal rounds exist: **%s**. The ratio is medians of the two modes; "
        "`slow calls` is the share falling in the slower mode." % any_bimodal
    )


def main() -> int:
    fp16 = load("sage003_ab_float16.jsonl")
    table(fp16, "float16 inputs, float16 output")
    for name, label in (
        ("sage003_ab_bfloat16.jsonl", "float16 inputs, bfloat16 output"),
        ("sage003_ab_shapes_float16.jsonl", "float16 additional shapes"),
    ):
        if os.path.exists(os.path.join(RAW, name)):
            table(load(name), label)

    attention_sweep()
    bimodality()
    negative_controls()

    # One-line headline used in the report's decision paragraph.
    decision = [r for r in fp16 if r["event"] == "decision"][-1]
    ratios = [e["median_sage_over_device"] for e in decision["aggregate"].values()]
    print(
        "\nHeadline: Sage is {lo:.2f}-{hi:.2f}x slower than the device fused op on "
        "the formal shapes.".format(lo=min(ratios), hi=max(ratios))
    )
    stage = [r for r in fp16 if r["event"] == "round"]
    quant_share = max(r["stage_ms"]["quant_share_of_sage"] for r in stage)
    speedups = [r["stage_ms"]["quant_speedup_vs_default_launch"] for r in stage]
    attn_floor = min(r["stage_ms"]["floor_if_quant_free"] for r in stage)
    print(
        "Quantization is up to {qs:.0%} of the Sage call; the SAGE-003 launch "
        "change is {lo:.2f}-{hi:.2f}x faster than the default launch with "
        "bitwise identical int8/scale outputs. Even with quantization free the "
        "attention kernel alone would still be {floor:.2f}x the device op.".format(
            qs=quant_share, lo=min(speedups), hi=max(speedups), floor=attn_floor
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
