import json, hashlib
from pathlib import Path

root = Path("/workspace/agent-workspaces/parallax-steady-ab")
logs = root / "experiments/logs"
out = root / "experiments/results/PARALLAX-005.jsonl"
rows = []
for f in sorted(logs.glob("p005-*.log")):
    text = f.read_text(errors="replace")
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try: events.append(json.loads(line))
            except Exception: pass
    exits = [int(x.split("=")[1]) for x in text.splitlines() if x.startswith("EXIT=")]
    rows.append({
        "log": f.name,
        "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
        "exit_codes": exits,
        "event_counts": {e: sum(1 for x in events if x.get("event") == e)
                         for e in sorted({x.get("event") for x in events if x.get("event")})},
        "import_probes": [x for x in events if x.get("event") == "import_probe"],
        "path_probes": [x for x in events if x.get("event") == "path_probe"],
        "device_calls": [x for x in events if x.get("event") == "device_call"],
        "validations": [{k: v for k, v in x.items() if k != "results"} | {"results": x.get("results")}
                        for x in events if x.get("event") == "validation"],
        "steady": [x for x in events if x.get("event") == "steady_summary"],
        "stage_done": [x for x in events if x.get("event") == "stage_done"],
        "cache": [x for x in events if x.get("event") in ("cache_before", "cache_after")],
    })
out.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
print(out, len(rows), "logs")
