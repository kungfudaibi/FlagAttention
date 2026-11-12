"""Rebuild PARALLAX-006 evidence JSONL from raw logs and exit codes."""

import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "experiments" / "logs"
RES = ROOT / "experiments" / "results"
OUT = RES / "PARALLAX-006.jsonl"


def sha16(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def exit_code(tag):
    p = RES / f"p006-{tag}.exit"
    return int(p.read_text().strip()) if p.exists() else None


def main():
    records = []
    for log in sorted(LOGS.glob("p006-*.log")):
        tag = log.stem[len("p006-"):]
        rows = read_jsonl(log)
        entry = {
            "evidence": "log", "tag": tag, "log": str(log.relative_to(ROOT)),
            "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
            "exit_code": exit_code(tag),
            "events": [r for r in rows if r.get("event") in
                       ("import_probe", "setup", "cache_before", "cache_after", "stage_done",
                        "device_call", "launch_summary", "validation", "path_probe", "steady_summary")],
            "launches": [r for r in rows if r.get("event") == "launch"],
            "autotune_configs": [r for r in rows if r.get("event") == "autotune_config"],
            "samples": [r for r in rows if r.get("event") == "sample"],
        }
        records.append(entry)

    # Summaries per dtype/route from steady logs.
    summaries = {}
    for rec in records:
        if not rec["tag"].startswith("ab-"):
            continue
        parts = rec["tag"].split("-")  # ab-<dtype>-r<round>-<route>
        if len(parts) != 4:
            continue
        dtype, rnd, route = parts[1], parts[2], parts[3]
        for ev in rec["events"]:
            if ev.get("event") != "steady_summary":
                continue
            key = f"{dtype}/{route}"
            s = summaries.setdefault(key, {"rounds": [], "peak_alloc": set(), "peak_reserved": set()})
            s["rounds"].append({"round": ev["round"], "summary": ev["summary"]})
            s["peak_alloc"].add(ev["peak_memory_allocated_bytes"])
            s["peak_reserved"].add(ev["peak_memory_reserved_bytes"])
    for key, s in summaries.items():
        s["peak_alloc"] = sorted(s["peak_alloc"])
        s["peak_reserved"] = sorted(s["peak_reserved"])
        med = {}
        for metric in ("forward_s", "backward_s", "full_s"):
            round_medians = [r["summary"][metric]["median"] for r in s["rounds"]]
            med[metric] = {
                "round_medians": round_medians,
                "median_of_round_medians": statistics.median(round_medians),
                "min": min(round_medians), "max": max(round_medians),
            }
        s["metrics"] = med
    records.append({"evidence": "summary", "steady": summaries})

    OUT.write_text("\n".join(json.dumps(r, default=str) for r in records) + "\n")
    print(f"wrote {OUT} ({len(records)} records)")
    for key, s in summaries.items():
        m = s["metrics"]["full_s"]
        print(key, "full median", round(m["median_of_round_medians"] * 1e3, 3), "ms",
              "rounds", [round(x * 1e3, 3) for x in m["round_medians"]])


if __name__ == "__main__":
    main()
