"""Run one baseline and one opt-in candidate process for a single case.

Baseline uses the read-only .baseline-2ed tree with experimental flags 0.
Candidate uses the task clone with explicit experimental flags.  Logs the
command, exit code, and parsed JSON row.
"""
import json, os, subprocess, sys, time

ROOT = "/workspace/agent-workspaces/gla-general-local-state"
BASE = ROOT + "/.baseline-2ed"
CAND = ROOT
RAW = ROOT + "/agent-results/raw"
case = ROOT + "/agent-results/scripts/gla006_case.py"

DRY = os.environ.get("DRY", "0") == "1"


def run(who, tree, tag, args, env_extra=None):
    e = dict(os.environ)
    e["PYTHONPATH"] = f"{tree}/src:{tree}:/opt/flagtree"
    e.pop("FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP16_STATE", None)
    e.pop("FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP32_STATE", None)
    if env_extra:
        e.update(env_extra)
    cmd = ["taskset", "-c", "128-159", "/flagos/bin/python", case] + args
    rec = {"who": who, "tree": tree, "cmd": cmd, "ts": time.time()}
    if DRY:
        print("DRY", who, " ".join(cmd)); return rec
    t0 = time.time()
    try:
        pr = subprocess.run(cmd, capture_output=True, text=True, timeout=int(os.environ.get("CASE_TIMEOUT", "300")), env=e)
        rc, so, se = pr.returncode, pr.stdout, pr.stderr
    except subprocess.TimeoutExpired as exc:
        rc = 124
        so = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        se = "TIMEOUT"
    rec.update({"rc": rc, "wall_s": round(time.time() - t0, 3)})
    line = ""
    for ln in so.splitlines():
        if ln.startswith("GLA006_JSON "):
            line = ln[len("GLA006_JSON "):]
    if line:
        rec.update(json.loads(line))
    else:
        rec["stdout_tail"] = so[-4000:]
        rec["stderr_tail"] = se[-4000:]
    return rec


def append(path, rec):
    if DRY:
        return
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
