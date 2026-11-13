"""PARALLAX-007 opt-in static route validation and steady full-call A/B."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "experiments/parallax_static_tune.py"


def probe(route, dtype, case, mode, round_id=0):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}:" + env.get("PYTHONPATH", "")
    env["PARALLAX_STATIC_D64"] = "1" if route == "candidate" else "0"
    cmd = ["/flagos/bin/python", str(HARNESS), "--route", route, "--dtype", dtype,
           "--case", case, "--mode", mode, "--round", str(round_id), "--samples", "5"]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=1200)
    records = []
    for line in proc.stdout.splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            pass
    if proc.returncode:
        raise RuntimeError(json.dumps({"route": route, "dtype": dtype, "case": case,
                                       "mode": mode, "rc": proc.returncode,
                                       "stdout_tail": proc.stdout[-2000:],
                                       "stderr_tail": proc.stderr[-2000:]}))
    imports = [r for r in records if r.get("event") == "import_probe"]
    if len(imports) != 1 or imports[0].get("device") != 1 or imports[0].get("tree") != str(ROOT):
        raise AssertionError(f"wrong import/device: {imports}")
    if mode == "cold":
        vals = [r for r in records if r.get("event") == "validation"]
        if len(vals) != 1 or not vals[0].get("all_pass"):
            raise AssertionError(f"missing or failed validation: {route}, {dtype}, {case}")
    else:
        vals = [r for r in records if r.get("event") == "sample"]
        if len(vals) != 5 or not all(r.get("full_s", 0) > 0 for r in vals):
            raise AssertionError(f"missing full-call samples: {route}, {dtype}, {case}")
    print(json.dumps({"route": route, "dtype": dtype, "case": case,
                      "mode": mode, "round": round_id, "records": records}), flush=True)


def main():
    for dtype in ("fp16", "bf16"):
        for case in ("target", "nearby", "t63"):
            probe("candidate", dtype, case, "cold")
        for round_id in range(3):
            for route in (("baseline", "candidate") if round_id % 2 == 0
                          else ("candidate", "baseline")):
                probe(route, dtype, "target", "steady", round_id)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PARALLAX007_ERROR: {exc}", file=sys.stderr)
        raise

