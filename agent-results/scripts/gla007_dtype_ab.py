"""GLA-008 formal benchmark entry point: three-dtype full-call A/B.

The scheduler calls this file with no arguments for the ``benchmark.full_call_ab``
gate.  It delegates to ``gla008_drive.py`` (one implementation of the matrix, so
the report and the gate cannot drift) and keeps the process exit contract:

  * 0 only when every (KxV, dtype, gate, round, side) sample is present, the
    requested route really launched on GCU 4, and the parallel side matches the
    fallback side numerically;
  * non-zero for a missing sample, a numeric/NaN failure, a route mismatch, a
    device mismatch, or a crashed child.

The full default matrix is 4 shapes x 3 dtypes x 3 gates x 3 rounds x 2 sides
= 216 timed rows on top of the cold rows.  ``--smoke`` shrinks it to one case
so the pytest gate stays quick; the scheduler always runs the full matrix.
The default output is ``gla008_ab_full.jsonl`` so a scheduler run never mixes
with the narrower hand-run datasets (``gla008_ab.jsonl`` etc.).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "agent-results/scripts/gla008_drive.py"
RAW = ROOT / "agent-results/raw"
PY = "/flagos/bin/python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--out", default=str(RAW / "gla008_ab_full.jsonl"))
    ap.add_argument("--shapes", default=None)
    ap.add_argument("--dtypes", default=None)
    ap.add_argument("--gates", default=None)
    args = ap.parse_args()
    if args.rounds < 1 or args.samples < 1:
        raise SystemExit("GLA007_ERROR: rounds and samples must be positive")

    cmd = [PY, str(DRIVER), "--mode", "ab", "--rounds", str(args.rounds),
           "--samples", str(args.samples), "--out", args.out]
    if args.smoke:
        cmd.append("--smoke")
    else:
        if args.shapes:
            cmd += ["--shapes", args.shapes]
        if args.dtypes:
            cmd += ["--dtypes", args.dtypes]
        if args.gates:
            cmd += ["--gates", args.gates]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}:/opt/flagtree"
    env["TRITON_F32_DEFAULT"] = "ieee"
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True)
    wall = round(time.time() - started, 1)
    meta = {"kind": "benchmark_entry", "cmd": cmd, "rc": proc.returncode,
            "wall_s": wall, "out": args.out, "rounds": args.rounds,
            "samples": args.samples, "smoke": bool(args.smoke)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out + ".meta.jsonl", "a") as f:
        f.write(json.dumps(meta) + "\n")
    print(f"GLA007_BENCHMARK rc={proc.returncode} wall_s={wall}", flush=True)
    if proc.returncode != 0:
        print("GLA007_ERROR: full-call A/B failed; see driver output above "
              "and the raw JSONL", file=sys.stderr)
        return proc.returncode or 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
