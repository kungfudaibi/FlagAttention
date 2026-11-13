#!/usr/bin/env python
"""Summarize a ``gdn2_forward_ab.py drive`` JSONL into markdown tables.

Usage::

    /flagos/bin/python benchmark/gdn2_forward_ab_summarize.py \
        agent-results/GDN2-003-ab.jsonl

The script never touches the device; it only re-reads the recorded samples.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load(path: Path):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args(argv)
    records = load(Path(args.jsonl))

    shapes = [r for r in records if r.get("kind") == "shape" and r.get("dtype") == args.dtype]
    if not shapes:
        print("no shape records for dtype", args.dtype)
        return 1

    driver = next((r for r in records if r.get("kind") == "driver"), {})
    invocations = [r for r in records if r.get("kind") == "invocation"]
    errors = [r for r in records if r.get("kind") == "error"]
    end = next((r for r in records if r.get("kind") == "driver_end"), {})

    print(f"file: `{args.jsonl}`")
    print(f"commits: `{driver.get('commits')}`  rounds: {driver.get('rounds')}  "
          f"device: {driver.get('device')}  pci: {driver.get('pci_expected')}  "
          f"cpu: {driver.get('cpu_list')}")
    print(f"child invocations: {len(invocations)}, non-zero: "
          f"{sum(1 for r in invocations if r.get('child_exit_code'))}, "
          f"recorded errors: {len(errors)}, driver exit: {end.get('exit_code')}")
    print()

    by_variant = defaultdict(list)
    per_round = defaultdict(lambda: defaultdict(dict))
    for record in shapes:
        key = (tuple(record["shape"]), record["variant"])
        by_variant[key].append(record["steady"]["median_ms"])
        per_round[tuple(record["shape"])][record["round"]][record["variant"]] = \
            record["steady"]["median_ms"]

    print("| shape (B,T,H,K,V) | legacy med ms | adaptive med ms | speedup | round medians legacy | round medians adaptive |")
    print("|---|---|---|---|---|---|")
    rows = []
    for shape, rounds in sorted(per_round.items(), key=lambda item: item[0]):
        legacy = by_variant[(shape, "legacy")]
        adaptive = by_variant[(shape, "adaptive")]
        if not legacy or not adaptive:
            continue
        med_legacy = statistics.median(legacy)
        med_adaptive = statistics.median(adaptive)
        speedup = med_legacy / med_adaptive
        rounds_legacy = ", ".join(f"{r['legacy']:.4f}" for r in rounds.values() if "legacy" in r)
        rounds_adaptive = ", ".join(f"{r['adaptive']:.4f}" for r in rounds.values() if "adaptive" in r)
        rows.append((shape, med_legacy, med_adaptive, speedup))
        print(f"| {shape} | {med_legacy:.4f} | {med_adaptive:.4f} | {speedup:.3f}x | "
              f"{rounds_legacy} | {rounds_adaptive} |")
    if rows:
        geometric = statistics.geometric_mean([row[3] for row in rows])
        print(f"\ngeometric-mean speedup over {len(rows)} shapes: **{geometric:.3f}x**")
    print()

    print("| shape | variant | cold compile s | peak alloc MiB | kernels (BK,BV) | o gateA/B | ht gateA/B | ht max_abs | ht rms_rel | finite |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for record in sorted(shapes, key=lambda r: (tuple(r["shape"]), r["variant"], r["round"])):
        launches = record.get("cold_launches", [])
        kernels = sorted({(item["kernel"].replace("_chunk_gdn2_fwd_", "").replace("_infer_kernel", ""),
                           item["cfg"].get("BK"), item["cfg"].get("BV")) for item in launches})
        metrics = record["metrics"]
        print(f"| {tuple(record['shape'])} | {record['variant']} r{record['round']} | "
              f"{record['cold_compile_s']:.3f} | {record['peak_alloc_mib']:.2f} | {kernels} | "
              f"{metrics['o']['gate_a']}/{metrics['o']['gate_b']} | "
              f"{metrics['ht']['gate_a']}/{metrics['ht']['gate_b']} | "
              f"{metrics['ht']['max_abs']:.3e} | {metrics['ht']['rms_rel']:.3e} | "
              f"{metrics['o']['finite'] and metrics['ht']['finite']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
