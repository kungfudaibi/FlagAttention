"""GLA-008 report generator: raw JSONL -> markdown tables.

Reads only the raw evidence files, so every number in the report is traceable to
a record.  Prints the tables; the report body is written by hand around them.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "agent-results/raw"


def load(name):
    p = RAW / name
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def kv(r):
    """Records carry either K/V or shape=[K,V] depending on the writer."""
    if "K" in r and "V" in r:
        return r["K"], r["V"]
    sh = r.get("shape") or [None, None]
    return sh[0], sh[1]


def fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        if x == 0:
            return "0"
        if abs(x) < 1e-4 or abs(x) >= 1e4:
            return f"{x:.{nd}e}"
        return f"{x:.{nd}f}"
    return str(x)


def probe_tables():
    rows = load("gla008_probe.jsonl")
    if not rows:
        print("(no probe records)\n")
        return
    def uniq(rs):
        """One row per (tag, side): older writers appended helper copies."""
        seen, out = set(), []
        for r in rs:
            if r.get("kind"):
                continue
            key = (r.get("tag"), r.get("side"))
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    rows = uniq(rows)
    par = [r for r in rows if r.get("side") == "parallel"]
    fb = [r for r in rows if r.get("side") == "fallback"]
    print("### probe records")
    print(f"fallback rows {len(fb)}, parallel rows {len(par)}, "
          f"total {len(rows)}\n")

    print("| KxV | dtype | gate | g range | route fwd/bwd | grade | oracle | "
          "NaN (shared) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in par:
        f = r.get("path_fwd_h", {}).get("done", {})
        b = r.get("path_bwd_dh", {}).get("done", {})
        fwd = "parallel" if f.get("_chunk_fwd_local_state_kernel") else "fallback"
        bwd = "parallel" if b.get("_chunk_bwd_local_state_kernel") else "fallback"
        g = (r.get("input_stats") or {}).get("g", {})
        tag = r.get("tag", "")
        tag = tag[4:] if tag.startswith("par_") else tag
        print(f"| {kv(r)[0]}x{kv(r)[1]} | {r['dtype']} | {r['gln']} | "
              f"[{fmt(g.get('min'), 2)}, {fmt(g.get('max'), 2)}] | "
              f"{fwd}/{bwd} | {r.get('grade')} | "
              f"{r.get('oracle_quant_grade')} | "
              f"{','.join(r.get('nan_shared_names') or []) or '-'} |")

    print("\n### worst tensor error per row (path vs same-commit fallback)")
    print("| KxV | dtype | gate | tensor | max_abs | rel_rms | threshold | OK |")
    print("|---|---|---|---|---|---|---|---|")
    for r in par:
        cmp_rows = [x for x in (r.get("compare") or [])
                    if not x["ok"] or x["name"] in ("o", "dv")]
        if not cmp_rows:
            continue
        worst = max(cmp_rows, key=lambda x: x["rel_rms"])
        th = r.get("thresholds", {})
        print(f"| {kv(r)[0]}x{kv(r)[1]} | {r['dtype']} | {r['gln']} | "
              f"{worst['name']} | {fmt(worst['max_abs'])} | "
              f"{fmt(worst['rel_rms'])} | rel<={fmt(th.get('path_rel'))} "
              f"abs<={fmt(worst['abs_cap'])} | {worst['ok']} |")

    print("\n### local_state buffer")
    print("| KxV | dtype | observed dtype | observed bytes | theory bytes | "
          "match |")
    print("|---|---|---|---|---|---|")
    seen = set()
    for r in par:
        key = (kv(r)[0], kv(r)[1], r["dtype"])
        if key in seen:
            continue
        seen.add(key)
        obs = sorted({x["bytes"] for x in (r.get("local_state_observed") or [])})
        th = r.get("local_state_fp32_bytes_theory", {})
        print(f"| {kv(r)[0]}x{kv(r)[1]} | {r['dtype']} | "
              f"{','.join(r.get('local_state_dtypes') or [])} | "
              f"{obs} | fwd={th.get('fwd')} bwd={th.get('bwd')} | "
              f"{th.get('fwd') in obs and th.get('bwd') in obs} |")

    print("\n### memory (MiB, per stage peak delta)")
    print("| KxV | dtype | gate | fwd_h | bwd_dh | whole_fwd | full_call | "
          "reserved |")
    print("|---|---|---|---|---|---|---|---|")
    for r in par:
        m = r.get("memory") or {}
        full = r.get("full_call_memory") or {}

        def mib(d):
            v = (d or {}).get("peak_delta")
            return fmt(v / 1048576, 2) if v is not None else "-"
        print(f"| {kv(r)[0]}x{kv(r)[1]} | {r['dtype']} | {r['gln']} | "
              f"{mib(m.get('fwd_h'))} | {mib(m.get('bwd_dh_fp32'))} | "
              f"{mib(m.get('whole_fwd'))} | {mib(full)} | "
              f"{fmt((r.get('memory_reserved') or 0) / 1048576, 1)} |")

    print("\n### cold compile (ms, first launch in a fresh process)")
    print("| KxV | dtype | cold fwd_h | cold bwd_dh |")
    print("|---|---|---|---|")
    for r in par:
        print(f"| {kv(r)[0]}x{kv(r)[1]} | {r['dtype']} | "
              f"{fmt(r.get('cold_fwd_h_ms'))} | {fmt(r.get('cold_bwd_dh_ms'))} |")


def ab_tables():
    rows = load("gla008_ab.jsonl")
    if not rows:
        print("(no A/B records)\n")
        return
    steady = [r for r in rows if r.get("phase") == "steady"]
    print(f"### A/B steady records: {len(steady)} "
          f"(rounds x samples per side)\n")
    groups = defaultdict(lambda: defaultdict(list))
    for r in steady:
        ms = (r.get("timing") or {}).get("full_fwd_bwd", {}).get("ms") or []
        key = (kv(r)[0], kv(r)[1], r["dtype"], r["gln"])
        groups[key][r["side"]].extend(ms)
    print("| KxV | dtype | gate | n fb | n par | fb median | par median | "
          "par/fb | fb min | par min |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    ratios = []
    for (K, V, dt, g), sides in sorted(groups.items()):
        fb = sides.get("fallback", [])
        pr = sides.get("parallel", [])
        if not fb or not pr:
            print(f"| {K}x{V} | {dt} | {g} | {len(fb)} | {len(pr)} | "
                  f"MISSING | - | - | - | - |")
            continue
        fm, pm = statistics.median(fb), statistics.median(pr)
        ratios.append(pm / fm)
        print(f"| {K}x{V} | {dt} | {g} | {len(fb)} | {len(pr)} | "
              f"{fmt(fm)} | {fmt(pm)} | {fmt(pm / fm)} | {fmt(min(fb))} | "
              f"{fmt(min(pr))} |")
    if ratios:
        print(f"\nparallel/fallback median ratio: min {fmt(min(ratios))}, "
              f"max {fmt(max(ratios))}, mean {fmt(sum(ratios) / len(ratios))}")


def sci(x):
    return "-" if x is None else f"{x:.3e}"


def quant_tables():
    rows = load("gla008_quant.jsonl")
    sums = [r for r in rows if r.get("kind") == "quant_summary"]
    if not sums:
        print("(no quant-separation records)\n")
        return
    print("### input quantisation vs path error (FP64 recurrence, gate 1.0)\n")
    print("| KxV | dtype | tensor | quant effect (rel RMS) | parallel path "
          "error | total vs FP64 base | path ok |")
    print("|---|---|---|---|---|---|---|")
    for r in sums[-1]["rows"]:
        q = {x["name"]: x for x in r["quant_effect"]}
        pe = {x["name"]: x for x in r["path_error"]}
        te = {x["name"]: x for x in r["total_error"]}
        for name in ("o", "dv"):
            print(f"| {r['K']}x{r['V']} | {r['dtype']} | {name} | "
                  f"{sci(q[name]['rel_rms'])} | {sci(pe[name]['rel_rms'])} | "
                  f"{sci(te[name]['rel_rms'])} | {pe[name]['ok']} |")
    print()


def edge_tables():
    rows = load("gla008_edges.jsonl")
    if not rows:
        print("(no boundary/pressure records)\n")
        return
    seen, keep = set(), []
    for r in rows:
        if r.get("kind"):
            continue
        key = (r.get("tag"), r.get("side"))
        if key in seen:
            continue
        seen.add(key)
        keep.append(r)
    print("### boundary and pressure shapes\n")
    print("| case | B,T,H | KxV | dtype | side | route | grade | local state "
          "dtype/bytes | stage peak MiB (fwd_h/bwd/full) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in keep:
        f = (r.get("path_fwd_h") or {}).get("done", {})
        b = (r.get("path_bwd_dh") or {}).get("done", {})
        fwd = "parallel" if f.get("_chunk_fwd_local_state_kernel") else "fallback"
        bwd = "parallel" if b.get("_chunk_bwd_local_state_kernel") else "fallback"
        m = r.get("memory") or {}
        full = (r.get("full_call_memory") or {}).get("peak_delta")

        def mib(k):
            v = (m.get(k) or {}).get("peak_delta")
            return fmt(v / 1048576, 2) if v is not None else "-"
        obs = sorted({x["bytes"] for x in (r.get("local_state_observed") or [])})
        print(f"| {r.get('tag')} | {r.get('B')},{r.get('T')},{r.get('H')} | "
              f"{kv(r)[0]}x{kv(r)[1]} | {r.get('dtype')} | {r.get('side')} | "
              f"{fwd}/{bwd} | {r.get('grade') or '-'} | "
              f"{','.join(r.get('local_state_dtypes') or []) or '-'} {obs or ''} | "
              f"{mib('fwd_h')}/{mib('bwd_dh_fp32')}/"
              f"{fmt(full / 1048576, 2) if full else '-'} |")
    print()


def guard_table():
    rows = load("gla008_guards.jsonl")
    if not rows:
        print("(no guard records)\n")
        return
    print("### semantic guards (both opt-in flags = 1)\n")
    print("| case | error | launched |")
    print("|---|---|---|")
    for r in rows:
        if r.get("kind") == "guard":
            for row in r["rows"]:
                launched = ", ".join(f"{k}x{v}" for k, v in row["done"].items())
                print(f"| {row['case']} | {row['error'] or '-'} | {launched} |")
    print("\n### production default route (flags = 0)\n")
    print("| case | route | identical to baseline commit |")
    print("|---|---|---|")
    for r in rows:
        if r.get("kind") == "default_route_summary":
            for row in r["rows"]:
                route = ", ".join(f"{k}x{v}" for k, v in row["route"].items())
                print(f"| {row['tag']} | {route} | {row['tensor_identical']} |")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "probe"):
        probe_tables()
    if what in ("all", "ab"):
        ab_tables()
    if what in ("all", "quant"):
        quant_tables()
    if what in ("all", "edges"):
        edge_tables()
    if what in ("all", "guards"):
        guard_table()
