import json,sys,statistics as st
rows=[json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
byshape={}
for r in rows:
    byshape.setdefault(r["T"],[]).append(r)
for T,rs in byshape.items():
    print(f"T={T}")
    for r in sorted(rs,key=lambda x:(x["tag"])):
        vals=sorted(r["fwd_bwd_ms"]); med=st.median(vals)
        print(f"  {r[\"tag\"]:10s} n={len(vals)} median={med:.3f} min={vals[0]:.3f} max={vals[-1]:.3f}  fwd_med={r[\"fwd_median\"]:.3f}")
    for tag in ("base","cand"):
        meds=[st.median(r["fwd_bwd_ms"]) for r in rs if r["tag"].startswith(tag)]
        if meds: print(f"  {tag} median-of-round-medians={st.median(meds):.3f} rounds={[round(x,3) for x in meds]}")
