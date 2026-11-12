"""GLA-004 extra T sweep: correctness vs exact baseline + stage A/B."""
import json, os, subprocess, sys, time
CAND = "/workspace/agent-workspaces/gla-shape-sweep"
BASE = CAND + "/.baseline-4f437de"
case = CAND + "/agent-results/scripts/gla004_case.py"
env = dict(os.environ)
T = int(sys.argv[1])
shapes = [(int(a.split("x")[0]), int(a.split("x")[1])) for a in sys.argv[2:]]
out = CAND + f"/agent-results/raw/extra_T{T}.jsonl"
for K, V in shapes:
    tag = f"{K}x{V}_T{T}"
    bpt = f"/tmp/gla004_base_{tag}.pt"
    for who, tree, save, cmp_ in (("base", BASE, bpt, None),
                                  ("cand", CAND, None, bpt)):
        e = dict(env)
        e["PYTHONPATH"] = f"{tree}/src:{tree}:/opt/flagtree"
        cmd = ["/flagos/bin/python", case, "--B", "1", "--T", str(T), "--H", "4",
               "--K", str(K), "--V", str(V), "--tag", f"{who}_{tag}",
               "--samples", "5", "--warmup", "3"]
        if save:
            cmd += ["--save", save]
        if cmp_:
            cmd += ["--compare", cmp_]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=175, env=e)
            rc, so, se = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as exc:
            rc = 124
            so = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            se = "TIMEOUT"
        rec = {"who": who, "shape": [K, V], "T": T, "rc": rc,
               "wall_s": round(time.time() - t0, 3)}
        line = ""
        for ln in so.splitlines():
            if ln.startswith("GLA004_JSON "):
                line = ln[len("GLA004_JSON "):]
        if line:
            rec.update(json.loads(line))
        else:
            rec["stdout_tail"] = so[-1500:]
            rec["stderr_tail"] = se[-1500:]
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        t = rec.get("timing", {})
        print(who, tag, "rc", rc, "pf", rec.get("parallel_fwd_h"),
              "pb", rec.get("parallel_bwd_dh"), "pass", rec.get("ALL_PASS"),
              "fwdh %.3f bwdh %.3f fwd %.3f" % (
                  t.get("fwd_h", {}).get("median", -1),
                  t.get("bwd_dh_fp32", {}).get("median", -1),
                  t.get("whole_fwd", {}).get("median", -1)),
              rec.get("error", ""), flush=True)
print("DONE")
