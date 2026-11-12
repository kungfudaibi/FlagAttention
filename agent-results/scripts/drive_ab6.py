"""GLA-006 alternating old-vs-experimental steady-state A/B.

One process per (round, side, shape); every device process is pinned with
taskset -c 128-159 and the batch is strictly serial.  Baseline is the exact
read-only 2ed35d3 worktree with flags 0; candidate is the task clone with the
experimental dtype flag requested for this shape family.
"""
import json, os, subprocess, sys, time

ROOT = "/workspace/agent-workspaces/gla-general-local-state"
BASE = ROOT + "/.baseline-2ed"
CAND = ROOT
RAW = ROOT + "/agent-results/raw"
case = ROOT + "/agent-results/scripts/gla006_case.py"

SH = {"32x32": (32, 32), "64x64": (64, 64), "64x128": (64, 128),
      "128x64": (128, 64), "128x128": (128, 128), "32x64": (32, 64),
      "64x32": (64, 32), "256x64": (256, 64), "48x80": (48, 80),
      "96x128": (96, 128), "64x256": (64, 256)}
sel = sys.argv[1:] or ["32x64", "64x32", "256x64", "64x64"]
B = os.environ.get("B", "2"); T = os.environ.get("T", "256"); H = os.environ.get("H", "2")
dtype = os.environ.get("DTYPE", "fp16")
exp16 = os.environ.get("EXP16", "1"); exp32 = os.environ.get("EXP32", "0")
samples = os.environ.get("SAMPLES", "5")
rounds = ([int(os.environ["ROUND"])] if os.environ.get("ROUND")
          else list(range(1, int(os.environ.get("ROUNDS", "3")) + 1)))
out = os.path.join(RAW, os.environ.get("OUT", "phase3_ab_fp16.jsonl"))
mem = os.environ.get("MEM", "0") == "1"
env = dict(os.environ)
for r in rounds:
    for name in sel:
        K, V = SH[name]
        order = [("base", BASE), ("cand", CAND)] if r % 2 else [("cand", CAND), ("base", BASE)]
        for who, tree in order:
            e = dict(env)
            e["PYTHONPATH"] = f"{tree}/src:{tree}:/opt/flagtree"
            e.pop("FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP16_STATE", None)
            e.pop("FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP32_STATE", None)
            if who == "cand":
                e["FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP16_STATE"] = exp16
                e["FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP32_STATE"] = exp32
            cmd = ["taskset", "-c", "128-159", "/flagos/bin/python", case,
                   "--B", B, "--T", T, "--H", H, "--K", str(K), "--V", str(V),
                   "--dtype", dtype, "--tag", f"{who}_{name}_r{r}",
                   "--samples", samples, "--warmup", "2",
                   "--exp-fp16", (exp16 if who == "cand" else "0"),
                   "--exp-fp32", (exp32 if who == "cand" else "0")]
            if mem:
                cmd.append("--mem")
            t0 = time.time()
            try:
                pr = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=e)
                rc, so, se = pr.returncode, pr.stdout, pr.stderr
            except subprocess.TimeoutExpired as exc:
                rc = 124
                so = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                se = "TIMEOUT"
            rec = {"who": who, "shape": [K, V], "B": int(B), "T": int(T), "H": int(H),
                   "dtype": dtype, "round": r, "rc": rc,
                   "wall_s": round(time.time() - t0, 3), "samples": int(samples),
                   "cmd": cmd}
            line = ""
            for ln in so.splitlines():
                if ln.startswith("GLA006_JSON "):
                    line = ln[len("GLA006_JSON "):]
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
            print(who, name, "r", r, "rc", rc,
                  "cold", round(rec.get("cold_full_fwd_bwd_ms", -1), 1),
                  "fwd_h", g("fwd_h"), "bwd_dh", g("bwd_dh_fp32"),
                  "fwd", g("whole_fwd"), "e2e", g("full_fwd_bwd"),
                  "pf", rec.get("parallel_fwd_h"), "pb", rec.get("parallel_bwd_dh"),
                  flush=True)
print("DONE")
