#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Render the SAGE-002 raw JSONL evidence into markdown tables.

Every number in ``agent-results/SAGE-002.md`` comes from
``agent-results/raw/sage002_*.jsonl`` through this script, so the report can be
regenerated instead of retyped:

    python3 agent-results/scripts/sage002_summarize.py agent-results/raw/*.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys


def load(paths):
    rounds, metas, decisions = {}, {}, {}
    for path in paths:
        name = path.rsplit("/", 1)[-1].replace(".jsonl", "")
        try:
            handle = open(path)
        except OSError:
            continue
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "meta":
                metas[name] = record
            elif record.get("event") == "round":
                rounds.setdefault((name, tuple(record["shape"])), []).append(record)
            elif record.get("event") == "decision":
                decisions[name] = record
        if name not in decisions:
            # Early evidence files closed the stream before writing the
            # decision; the captured stdout log still has it.
            try:
                log = open(path.replace(".jsonl", ".log"))
            except OSError:
                continue
            for line in log:
                index = line.find('{"event": "decision"')
                if index >= 0:
                    try:
                        decisions[name] = json.loads(line[index:])
                    except json.JSONDecodeError:
                        pass
    return rounds, metas, decisions


def median_of_rounds(rows, scope, key="event"):
    """Median across rounds of a scope's per-round median.

    Stage records nest timing kinds (event/wall/host); A/B records store the
    event summary directly per scope.
    """
    values = []
    for row in rows:
        entry = row["stats"][scope]
        values.append(entry[key]["median_ms"] if key in entry else entry["median_ms"])
    return statistics.median(values)


def main(argv):
    rounds, metas, decisions = load(argv[1:])
    print("## A/B full call: candidate vs baselines\n")
    print("| evidence | B,H,T,D | dtype | candidate ms | exact SDPA ms | device SDPA ms "
          "| speedup vs exact | speedup vs device | cosine vs exact | rel L2 |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for (name, shape), rows in sorted(rounds.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if "candidate" not in rows[0]["stats"]:
            continue
        meta = metas.get(name, {})
        cand = median_of_rounds(rows, "candidate")
        acc = rows[0]["accuracy"]["candidate_vs_exact_sdpa"]
        cells = [name, "x".join(str(x) for x in shape), meta.get("output_dtype", "?")]
        cells.append(f"{cand:.4f}")
        speedups = []
        for scope in ("exact_sdpa", "device_sdpa"):
            if scope in rows[0]["stats"]:
                value = median_of_rounds(rows, scope)
                cells.append(f"{value:.4f}")
                speedups.append(f"{value / cand:.2f}x")
            else:
                cells.extend(["n/a", "n/a"])
        cells.extend(speedups)
        cells.extend([f"{acc['cosine']:.6f}", f"{acc['relative_l2']:.5f}"])
        print("| " + " | ".join(cells) + " |")

    print("\n## Stage breakdown (FP16)\n")
    print("| evidence | B,H,T,D | quant ms | attention ms | full ms | exact SDPA ms "
          "| device SDPA ms | quant share | ceiling vs device |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for (name, shape), rows in sorted(rounds.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if "quant" not in rows[0]["stats"]:
            continue
        values = {
            scope: median_of_rounds(rows, scope)
            for scope in ("quant", "attention", "full", "exact_sdpa", "device_sdpa")
        }
        share = values["quant"] / values["full"]
        # Best case for quantization work: remove it completely. The attention
        # kernel alone must then still beat the GCU-native fused baseline.
        ceiling = values["device_sdpa"] / values["attention"]
        print(
            f"| {name} | {'x'.join(str(x) for x in shape)} | {values['quant']:.4f} "
            f"| {values['attention']:.4f} | {values['full']:.4f} "
            f"| {values['exact_sdpa']:.4f} | {values['device_sdpa']:.4f} "
            f"| {share:.1%} | {ceiling:.3f}x |"
        )

    print("\n## Decisions\n")
    for name, decision in sorted(decisions.items()):
        print(f"- `{name}`: **{decision['status']}**"
              + (f" ({'; '.join(decision['failures'])})" if decision.get("failures") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
