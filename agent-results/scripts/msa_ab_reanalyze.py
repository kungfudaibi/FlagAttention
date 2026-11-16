"""Re-analyze committed MSA A/B JSONL without a GCU or torch import.

Usage:
  python3 agent-results/scripts/msa_ab_reanalyze.py \
      --run MSA-004=agent-results/raw/MSA-004/ab.jsonl \
      --run MSA-005-run1=agent-results/raw/MSA-005/run1/ab.jsonl \
      --run MSA-005-run2=agent-results/raw/MSA-005/ab.jsonl \
      --run MSA-005-formal=agent-results/raw/MSA-005/formal/20261116T044252Z-99672caa/stdout.log \
      --json-out agent-results/raw/MSA-005/reanalysis.json

Inputs are read-only.  Non-JSON trailer lines (``WROTE``/``PASSED``) in the
orchestrator stdout log are skipped and counted.  Exit code 0 means the
analysis itself completed; unstable controls are reported as evidence, not
hidden and not turned into a process failure unless --require-stable-controls
is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import msa_ab_stats as stats  # noqa: E402


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(path: str) -> tuple[list[dict], int]:
    records = []
    skipped = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if not line.startswith("{"):
                skipped += 1
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    return records, skipped


def analyze_run(name: str, path: str) -> dict:
    records, skipped = load_run(path)
    headers = [r for r in records if r.get("probe") == "header"]
    cases = [stats.classify_case(r) for r in records if r.get("probe") == "case"]
    summaries = [r for r in records if r.get("probe") == "summary"]
    summary = stats.summarize(cases)
    summary.update(
        {
            "run": name,
            "path": path,
            "sha256": sha256(path),
            "non_json_lines_skipped": skipped,
            "header": headers[0] if headers else None,
            "recorded_summary": summaries[0] if summaries else None,
            "case_verdicts": {
                case["id"]: {
                    "verdict": case["verdict"],
                    "timing_status": case["timing_status"],
                    "path_changed": case["path_changed"],
                    "a_median_ms": case["a_median_ms"],
                    "b_median_ms": case["b_median_ms"],
                    "legacy_speedup": case["legacy_speedup"],
                    "round_median_speedup": case["round_median_speedup"],
                    "round_speedups": case["round_speedups"],
                    "round_ratio_range": case["round_ratio_range"],
                    "reasons": case["reasons"],
                    "A_kind": case["A"]["kind"],
                    "B_kind": case["B"]["kind"],
                    "A_largest_gap": case["A"]["largest_gap"],
                    "B_largest_gap": case["B"]["largest_gap"],
                }
                for case in cases
            },
        }
    )
    return summary


def cross_run(runs: dict[str, dict]) -> dict:
    """Cases whose verdicts or ratios move between independent runs."""
    by_id: dict[str, dict[str, dict]] = {}
    for name, run in runs.items():
        for case_id, verdict in run["case_verdicts"].items():
            by_id.setdefault(case_id, {})[name] = verdict
    unstable_across_runs = []
    for case_id, verdicts in by_id.items():
        ratios = [v["round_median_speedup"] for v in verdicts.values()]
        verdict_set = {v["verdict"] for v in verdicts.values()}
        if len(verdict_set) > 1 or (max(ratios) - min(ratios)) > stats.ROUND_FLIP_REL:
            unstable_across_runs.append(
                {
                    "id": case_id,
                    "verdicts": sorted(verdict_set),
                    "round_median_speedup_min": min(ratios),
                    "round_median_speedup_max": max(ratios),
                    "range": max(ratios) - min(ratios),
                }
            )
    return {
        "cases_seen_in_all_runs": sorted(
            case_id for case_id, verdicts in by_id.items() if len(verdicts) == len(runs)
        ),
        "cross_run_unstable": sorted(unstable_across_runs, key=lambda item: -item["range"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--require-stable-controls",
        action="store_true",
        help="exit 1 if any control's timing is bimodal or round-flipped",
    )
    args = parser.parse_args()
    if not args.run:
        parser.error("at least one --run NAME=PATH is required")

    runs = {}
    for item in args.run:
        if "=" not in item:
            parser.error(f"--run expects NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        if not os.path.isfile(path):
            parser.error(f"missing input for {name}: {path}")
        runs[name] = analyze_run(name, path)

    payload = {
        "schema": "msa-ab-reanalysis-v1",
        "rules": {
            "bimodal_gap_rel": stats.BIMODAL_GAP_REL,
            "bimodal_min_fraction": stats.BIMODAL_MIN_FRACTION,
            "bimodal_min_count": stats.BIMODAL_MIN_COUNT,
            "spike_gap_rel": stats.SPIKE_GAP_REL,
            "round_flip_rel": stats.ROUND_FLIP_REL,
            "ratio_agree_rel": stats.RATIO_AGREE_REL,
            "small_case_ms": stats.SMALL_CASE_MS,
        },
        "runs": runs,
        "cross_run": cross_run(runs),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)

    unstable = sum(len(run["unstable_controls"]) for run in runs.values())
    print(
        f"ANALYZED runs={len(runs)} unstable_controls={unstable} "
        f"cross_run_unstable={len(payload['cross_run']['cross_run_unstable'])}",
        file=sys.stderr,
    )
    if args.require_stable_controls and unstable:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
