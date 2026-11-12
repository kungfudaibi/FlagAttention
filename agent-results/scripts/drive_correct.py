import json, os, subprocess, sys, time
CAND = "/workspace/agent-workspaces/gla-shape-sweep"
BASE = CAND + "/.baseline-4f437de"
case = CAND + "/agent-results/scripts/gla004_case.py"
out = CAND + "/agent-results/raw/correct_T256.jsonl"
env = dict(os.environ)
results = {}
sel = sys.argv[1:] if len(sys.argv) > 1 else ["32x32","64x64","64x128","128x64","128x128"]
SH = {"32x32":(32,32),"64x64":(64,64),"64x128":(64,128),"128x64":(128,64),"128x128":(128,128)}
for K, V in [SH[x] for x in sel]:
    tag = f"{K}x{V}_T256"
    bpt = f"/tmp/gla004_base_{tag}.pt"
    runs = [("base", BASE, bpt, None), ("cand", CAND, None, bpt)]
    for who, tree, save, cmp_ in runs:
        e = dict(env)
        e["PYTHONPATH"] = f"{tree}/src:{tree}:/opt/flagtree"
        cmd = ["/flagos/bin/python", case, "--B", "1", "--T", "256", "--H", "4",
               "--K", str(K), "--V", str(V), "--tag", f"{who}_{tag}",
               "--no-timing"]
        if save:
            cmd += ["--save", save]
        if cmp_:
            cmd += ["--compare", cmp_]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=e)
            rc, so, se = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as exc:
            rc, so, se = 124, (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""), "TIMEOUT"
        rec = {"who": who, "shape": [K, V], "T": 256, "rc": rc,
               "wall_s": round(time.time() - t0, 3)}
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
        results[(who, K, V)] = rec
        print(who, tag, "rc", rc, "parallel_fwd", rec.get("parallel_fwd_h"),
              "parallel_bwd", rec.get("parallel_bwd_dh"),
              "ALL_PASS", rec.get("ALL_PASS"), "err", rec.get("error"), flush=True)
        if rc != 0 and who == "cand":
            break
print("DONE")
