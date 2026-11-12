import json, os, subprocess, sys, time
CAND = "/workspace/agent-workspaces/gla-shape-sweep"
BASE = CAND + "/.baseline-4f437de"
case = CAND + "/agent-results/scripts/gla004_case.py"
out = CAND + "/agent-results/raw/ab_T256.jsonl"
env = dict(os.environ)
shape_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
shapes = {"32x32": (32, 32), "64x64": (64, 64), "64x128": (64, 128),
          "128x64": (128, 64), "128x128": (128, 128)}
if shape_arg != "all":
    shapes = {shape_arg: shapes[shape_arg]}
rounds = int(os.environ.get("ROUNDS", "3"))
samples = int(os.environ.get("SAMPLES", "8"))
r0 = int(os.environ.get("ROUND", "0")) or None
for r in ([r0] if r0 else range(1, rounds + 1)):
    for name, (K, V) in shapes.items():
        for who, tree in (("base", BASE), ("cand", CAND)):
            e = dict(env)
            e["PYTHONPATH"] = f"{tree}/src:{tree}:/opt/flagtree"
            cmd = ["/flagos/bin/python", case, "--B", "1", "--T", "256", "--H", "4",
                   "--K", str(K), "--V", str(V), "--tag", f"{who}_{name}_r{r}",
                   "--samples", str(samples), "--warmup", "3"]
            t0 = time.time()
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=e)
                rc, so, se = p.returncode, p.stdout, p.stderr
            except subprocess.TimeoutExpired as exc:
                rc = 124
                so = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                se = "TIMEOUT"
            rec = {"who": who, "shape": [K, V], "T": 256, "round": r, "rc": rc,
                   "wall_s": round(time.time() - t0, 3), "samples": samples}
            line = ""
            for ln in so.splitlines():
                if ln.startswith("GLA004_JSON "):
                    line = ln[len("GLA004_JSON "):]
            if line:
                rec.update(json.loads(line))
            else:
                rec["stdout_tail"] = so[-2000:]
                rec["stderr_tail"] = se[-2000:]
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            t = rec.get("timing", {})
            print(who, name, "r", r, "rc", rc,
                  "fwd_h", round(t.get("fwd_h", {}).get("median", float("nan")), 4),
                  "bwd_dh", round(t.get("bwd_dh_fp32", {}).get("median", float("nan")), 4),
                  "fwd", round(t.get("whole_fwd", {}).get("median", float("nan")), 4),
                  "e2e", round(t.get("full_fwd_bwd", {}).get("median", float("nan")), 4),
                  "pf", rec.get("parallel_fwd_h"), "pb", rec.get("parallel_bwd_dh"),
                  flush=True)
print("DONE")
