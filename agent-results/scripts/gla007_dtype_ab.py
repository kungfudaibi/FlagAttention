"""GLA-007 fallback/parallel full-call probe. Agent expands the declared matrix."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "agent-results/scripts/gla006_case.py"
MATRIX = [(32, 64), (64, 64), (96, 128), (256, 64)]


def run(shape, dtype, gate, side, reference, round_id, samples):
    k, v = shape
    cmd = ["/flagos/bin/python", str(CASE), "--B", "2", "--T", "256", "--H", "2",
           "--K", str(k), "--V", str(v), "--dtype", dtype, "--gln", str(gate),
           "--samples", str(samples), "--warmup", "2", "--mem", "--tag", f"{side}-r{round_id}"]
    if side == "fallback":
        cmd += ["--force-fallback", "--save", str(reference)]
    else:
        cmd += ["--exp-fp16", "1", "--exp-fp32", "1", "--compare", str(reference)]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}:" + env.get("PYTHONPATH", "")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=600)
    lines = [line.removeprefix("GLA006_JSON ") for line in proc.stdout.splitlines()
             if line.startswith("GLA006_JSON ")]
    if proc.returncode or len(lines) != 1:
        raise RuntimeError(json.dumps({"shape": shape, "dtype": dtype, "gate": gate,
                                       "side": side, "rc": proc.returncode,
                                       "stdout_tail": proc.stdout[-1000:],
                                       "stderr_tail": proc.stderr[-2000:]}))
    record = json.loads(lines[0])
    timings = record.get("timing", {}).get("full_fwd_bwd", {}).get("ms", [])
    if len(timings) != samples or not all(isinstance(x, (int, float)) and x > 0 for x in timings):
        raise AssertionError(f"missing full-call samples: {shape}, {dtype}, {gate}, {side}")
    if side == "parallel":
        if not record.get("ALL_PASS"):
            raise AssertionError(f"numeric mismatch: {shape}, {dtype}, {gate}: {record.get('compare')}")
        if not record.get("parallel_fwd_h") or not record.get("parallel_bwd_dh"):
            raise AssertionError(f"parallel route not launched: {shape}, {dtype}, {gate}")
    else:
        if record.get("parallel_fwd_h") or record.get("parallel_bwd_dh"):
            raise AssertionError(f"fallback route not honored: {shape}, {dtype}, {gate}")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    matrix = [(64, 64)] if args.smoke else MATRIX
    dtypes = ["fp16"] if args.smoke else ["fp16", "bf16", "fp32"]
    gates = [1.0] if args.smoke else [0.1, 1.0, 10.0]
    rounds = 1 if args.smoke else args.rounds
    if rounds < 1 or args.samples < 1:
        raise ValueError("rounds and samples must be positive")
    count = 0
    with tempfile.TemporaryDirectory(prefix="gla007-") as temp:
        for shape in matrix:
            for dtype in dtypes:
                for gate in gates:
                    reference = Path(temp) / "reference.pt"
                    for round_id in range(rounds):
                        for side in (["fallback", "parallel"] if round_id % 2 == 0
                                     else ["parallel", "fallback"]):
                            record = run(shape, dtype, gate, side, reference, round_id, args.samples)
                            print(json.dumps({"shape": shape, "dtype": dtype, "gate": gate,
                                              "round": round_id, "side": side, "result": record}), flush=True)
                            count += 1
                    reference.unlink(missing_ok=True)
    expected = len(matrix) * len(dtypes) * len(gates) * rounds * 2
    if count != expected:
        raise AssertionError(f"missing runs: {count}/{expected}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"GLA007_ERROR: {exc}", file=sys.stderr)
        raise

