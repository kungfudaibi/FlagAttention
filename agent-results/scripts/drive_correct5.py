"""GLA-005 correctness driver: exact baseline vs candidate, same seed/inputs.

Usage: drive_correct5.py <combo> <shape...>
  combo in {b2t256h2, b1t512h8, bf16_control, gate01, gate10, specificity}
Each case runs baseline (save reference) then candidate (compare), separate
processes with tree-specific PYTHONPATH.  Failures/timeouts preserved in raw.
"""
import json, os, subprocess, sys, time

CAND = "/workspace/agent-workspaces/gla-shape-validate"
BASE = CAND + "/.baseline-4f437de"
RAW = CAND + "/agent-results/raw"
case = CAND + "/agent-results/scripts/gla005_case.py"
SH = {"32x32": (32, 32), "64x64": (64, 64), "64x128": (64, 128),
      "128x64": (128, 64), "128x128": (128, 128),
      "32x64": (32, 64), "64x32": (64, 32), "256x64": (256, 64)}
COMBO = {
    "b2t256h2": dict(B=2, T=256, H=2, dtype="fp16", gln=1.0),
    "b1t512h8": dict(B=1, T=512, H=8, dtype="fp16", gln=1.0),
    "bf16_control": dict(B=2, T=256, H=2, dtype="bf16", gln=1.0),
    "gate01": dict(B=2, T=256, H=2, dtype="fp16", gln=0.1),
    "gate10": dict(B=2, T=256, H=2, dtype="fp16", gln=10.0),
    "specificity": dict(B=1, T=128, H=2, dtype="fp16", gln=1.0),
}
combo = sys.argv[1]
sel = sys.argv[2:]
if not sel:
    sel = ["32x32", "64x64", "64x128", "128x64", "128x128"]
cfg = COMBO[combo]
out = os.path.join(RAW, f"correct_{combo}.jsonl")
env = dict(os.environ)
fail = 0
for name in sel:
    K, V = SH[name]
    tag = f"{combo}_{name}"
    bpt = f"/tmp/gla005_base_{tag}.pt"
    runs = [("base", BASE, bpt, None), ("cand", CAND, None, bpt)]
    for who, tree, save, cmp_ in runs:
        e = dict(env)
        e["PYTHONPATH"] = f"{tree}/src:{tree}:/opt/flagtree"
        cmd = ["/flagos/bin/python", case, "--B", str(cfg["B"]), "--T", str(cfg["T"]),
               "--H", str(cfg["H"]), "--K", str(K), "--V", str(V),
               "--dtype", cfg["dtype"], "--gln", str(cfg["gln"]),
               "--tag", f"{who}_{tag}", "--no-timing"]
        if save:
            cmd += ["--save", save]
        if cmp_:
            cmd += ["--compare", cmp_]
        t0 = time.time()
        try:
            pr = subprocess.run(cmd, capture_output=True, text=True, timeout=175, env=e)
            rc, so, se = pr.returncode, pr.stdout, pr.stderr
        except subprocess.TimeoutExpired as exc:
            rc = 124
            so = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            se = "TIMEOUT"
        rec = {"who": who, "combo": combo, "shape": [K, V], "cfg": cfg,
               "rc": rc, "wall_s": round(time.time() - t0, 3)}
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
        print(who, combo, name, "rc", rc, "pf", rec.get("parallel_fwd_h"),
              "pb", rec.get("parallel_bwd_dh"), "pass", rec.get("ALL_PASS"),
              "wall", rec["wall_s"], "err", rec.get("error"), flush=True)
        if rc != 0:
            fail += 1
            if who == "cand":
                break
print("DONE fail", fail)
