"""GLA-005 alternating baseline/candidate steady-state timing.

Usage: [ROUND=1] [SAMPLES=5] [B=2] [T=256] [H=2] drive_ab5.py <shape...>
One process per (round, side, shape); each process records cold compile, then
warmup 2 and 5 steady samples per stage, plus full fwd+bwd.
"""
import json, os, subprocess, sys, time

CAND = "/workspace/agent-workspaces/gla-shape-validate"
BASE = CAND + "/.baseline-4f437de"
RAW = CAND + "/agent-results/raw"
case = CAND + "/agent-results/scripts/gla005_case.py"
SH = {"32x32": (32, 32), "64x64": (64, 64), "64x128": (64, 128),
      "128x64": (128, 64), "128x128": (128, 128)}
sel = sys.argv[1:] or ["32x32", "128x128", "64x64"]
B = os.environ.get("B", "2"); T = os.environ.get("T", "256"); H = os.environ.get("H", "2")
samples = os.environ.get("SAMPLES", "5")
rounds = [int(os.environ.get("ROUND", "0"))] if os.environ.get("ROUND") else list(range(1, int(os.environ.get("ROUNDS", "3")) + 1))
out = os.path.join(RAW, os.environ.get("OUT", "ab_b2t256h2.jsonl"))
env = dict(os.environ)
for r in rounds:
    for name in sel:
        K, V = SH[name]
        # alternate which side goes first each round to reduce order bias
        order = [("base", BASE), ("cand", CAND)] if r % 2 else [("cand", CAND), ("base", BASE)]
        for who, tree in order:
            e = dict(env)
            e["PYTHONPATH"] = f"{tree}/src:{tree}:/opt/flagtree"
            cmd = ["/flagos/bin/python", case, "--B", B, "--T", T, "--H", H,
                   "--K", str(K), "--V", str(V), "--tag", f"{who}_{name}_r{r}",
                   "--samples", samples, "--warmup", "2"]
            t0 = time.time()
            try:
                pr = subprocess.run(cmd, capture_output=True, text=True, timeout=170, env=e)
                rc, so, se = pr.returncode, pr.stdout, pr.stderr
            except subprocess.TimeoutExpired as exc:
                rc = 124
                so = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                se = "TIMEOUT"
            rec = {"who": who, "shape": [K, V], "B": int(B), "T": int(T), "H": int(H),
                   "round": r, "rc": rc, "wall_s": round(time.time() - t0, 3),
                   "samples": int(samples)}
            line = ""
            for ln in so.splitlines():
                if ln.startswith("GLA005_JSON "):
                    line = ln[len("GLA005_JSON "):]
            if line:
                rec.update(json.loads(line))
            else:
                rec["stdout_tail"] = so[-3000:]
                rec["stderr_tail"] = se[-3000:]
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            t = rec.get("timing", {})
            def g(stage):
                return round(t.get(stage, {}).get("median", float("nan")), 4)
            print(who, name, "r", r, "rc", rc, "cold", round(rec.get("cold_full_fwd_bwd_ms", -1), 1),
                  "fwd_h", g("fwd_h"), "bwd_dh", g("bwd_dh_fp32"),
                  "fwd", g("whole_fwd"), "e2e", g("full_fwd_bwd"),
                  "pf", rec.get("parallel_fwd_h"), "pb", rec.get("parallel_bwd_dh"), flush=True)
print("DONE")
