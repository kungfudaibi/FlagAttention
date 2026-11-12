"""Run baseline(save) + candidate(compare) for one case and append raw JSONL."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drive_case as D

B, T, H, K, V = (int(x) for x in sys.argv[1:6])
dtype = sys.argv[6]
gln = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
tag = sys.argv[8] if len(sys.argv) > 8 else f"B{B}T{T}H{H}_{K}x{V}_{dtype}_g{gln}"
exp16 = os.environ.get("EXP16", "1")
exp32 = os.environ.get("EXP32", "0")
out = os.environ.get("OUT", D.ROOT + "/agent-results/raw/phase1_fp16.jsonl")
samples = os.environ.get("SAMPLES", "0")
mem = os.environ.get("MEM", "0")
pt = f"/tmp/gla006_{tag}.pt"

common = ["--B", str(B), "--T", str(T), "--H", str(H), "--K", str(K),
          "--V", str(V), "--dtype", dtype, "--gln", str(gln), "--seed", "42",
          "--no-timing"]
if mem == "1":
    common.append("--mem")

base_args = common + ["--tag", "base_" + tag, "--exp-fp16", "0", "--exp-fp32", "0",
                      "--save", pt]
cand_args = common + ["--tag", "cand_" + tag, "--exp-fp16", exp16, "--exp-fp32", exp32,
                      "--compare", pt]

rb = D.run("base", D.BASE, tag, base_args)
D.append(out, rb)
print("BASE rc", rb.get("rc"), "parallel", rb.get("parallel_fwd_h"), rb.get("parallel_bwd_dh"),
      "wall", rb.get("wall_s"), "err", rb.get("error"), flush=True)
rc_ = D.run("cand", D.CAND, tag, cand_args)
D.append(out, rc_)
print("CAND rc", rc_.get("rc"), "parallel", rc_.get("parallel_fwd_h"), rc_.get("parallel_bwd_dh"),
      "ALL_PASS", rc_.get("ALL_PASS"), "STRICT", rc_.get("ALL_STRICT"),
      "wall", rc_.get("wall_s"), "err", rc_.get("error"), flush=True)
if rc_.get("compare"):
    for row in rc_["compare"]:
        if not row["ok"] or row["max_abs"] > 1e-6:
            print(f"  {row['name']:10s} max_abs={row['max_abs']:.3e} rel={row['rel_rms']:.3e} ok={row['ok']} strict={row['strict_ok']}")
bad = [n for n in ("path_fwd_h", "path_bwd_dh") if rc_.get(n) is None]
if bad:
    print("MISSING_PATH", bad)
print("DONE")
