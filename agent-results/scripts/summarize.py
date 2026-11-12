"""Aggregate GLA-004 raw JSONL into the report tables."""
import json, os, statistics, sys

root = "/workspace/agent-workspaces/gla-shape-sweep/agent-results"
raw = os.path.join(root, "raw")
shapes = ["32x32", "64x64", "64x128", "128x64", "128x128"]


def load(name):
    p = os.path.join(raw, name)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def med(xs):
    xs = sorted(x for x in xs if x is not None)
    return statistics.median(xs) if xs else float("nan")


print("== correctness/candidate vs exact baseline (T256) ==")
rows = load("correct_T256.jsonl")
by = {}
for r in rows:
    by[(tuple(r["shape"]), r["who"])] = r
for s in shapes:
    K, V = (int(x) for x in s.split("x"))
    b = by.get(((K, V), "base")); c = by.get(((K, V), "cand"))
    if not c:
        print(s, "MISSING"); continue
    print(f"{s}: base_rc={b['rc'] if b else '?'} base_pf={b.get('parallel_fwd_h') if b else '?'} "
          f"cand_rc={c['rc']} pf={c.get('parallel_fwd_h')} pb={c.get('parallel_bwd_dh')} "
          f"pass={c.get('ALL_PASS')} err={c.get('error','')}")
    for t in c.get("compare", []):
        print(f"   {t['name']:10s} max_abs={t['max_abs']:.3e} rel_rms={t['rel_rms']:.3e} nan={t['nan_inf']} ok={t['ok']}")

print("\n== A/B stage medians (T256, per-round median of 6 samples) ==")
ab = load("ab_T256.jsonl")
for s in shapes:
    K, V = (int(x) for x in s.split("x"))
    for stage in ["fwd_h", "bwd_dh_fp32", "whole_fwd", "full_fwd_bwd"]:
        out = {}
        for who in ("base", "cand"):
            for r in range(1, 4):
                v = [x for x in ab if x["who"] == who and tuple(x["shape"]) == (K, V)
                     and x.get("round") == r]
                out[(who, r)] = v[0]["timing"][stage]["median"] if v and "timing" in v[0] else None
        bm = med([out[("base", r)] for r in (1, 2, 3)])
        cm = med([out[("cand", r)] for r in (1, 2, 3)])
        sp = bm / cm if cm and cm == cm else float("nan")
        print(f"{s:8s} {stage:14s} base={bm:8.3f} cand={cm:8.3f} speedup={sp:5.2f} "
              f"rounds base={[round(out[('base',r)],2) if out[('base',r)] else None for r in (1,2,3)]} "
              f"cand={[round(out[('cand',r)],2) if out[('cand',r)] else None for r in (1,2,3)]}")

for extra in ("extra_T64.jsonl", "extra_T512.jsonl"):
    rows = load(extra)
    if not rows:
        continue
    print(f"\n== {extra} ==")
    for r in rows:
        t = r.get("timing", {})
        print(r["who"], r["shape"], "rc", r["rc"], "pf", r.get("parallel_fwd_h"),
              "pb", r.get("parallel_bwd_dh"), "pass", r.get("ALL_PASS"),
              "fwd_h", round(t.get("fwd_h", {}).get("median", -1), 3),
              "bwd_dh", round(t.get("bwd_dh_fp32", {}).get("median", -1), 3),
              "fwd", round(t.get("whole_fwd", {}).get("median", -1), 3),
              "e2e", round(t.get("full_fwd_bwd", {}).get("median", -1), 3),
              r.get("error", ""))
        if r["who"] == "cand" and "compare" in r:
            for x in r["compare"]:
                if x["max_abs"] > 0 or x["nan_inf"] or not x["ok"]:
                    print(f"    {x['name']:10s} max_abs={x['max_abs']:.3e} rel_rms={x['rel_rms']:.3e} ok={x['ok']}")
