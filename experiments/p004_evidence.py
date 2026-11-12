"""Extract JSON probe events and exit codes from PARALLAX-004 console logs."""

import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parent
logs = root / "logs"
out = root / "results" / "PARALLAX-004.jsonl"
out.parent.mkdir(exist_ok=True)
names = (
    "p004-probe-cand-fp16.log",
    "p004-cand-fp16-valid.log",
    "p004-cand-bf16-valid.log",
    "p004-cand-fp16-t63.log",
    "p004-cand-bf16-t63.log",
    "p004-base-fp16-valid.log",
    "p004-base-fp16-valid2.log",
    "p004-base-bf16-valid.log",
)
with out.open("w", encoding="utf-8") as dst:
    for name in names:
        path = logs / name
        data = path.read_bytes()
        rows = []
        exit_code = None
        for line in data.decode("utf-8", errors="replace").splitlines():
            if line.startswith("EXIT="):
                exit_code = int(line[5:])
            elif line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        dst.write(json.dumps({
            "source": name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "exit_code": exit_code,
            "events": rows,
        }) + "\n")
print(out)
