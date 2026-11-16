"""GLA-008 evidence validator: fail closed on incomplete or unproven rows.

Two jobs:

1. Re-validate a raw A/B JSONL as a *dataset* (not as it runs).  The first
   full-matrix run recorded the pre-existing-NaN confirmation in a separate
   ``ab_preexisting_nan_check`` record and did not copy it onto the steady rows;
   this validator joins the two so the NaN evidence is judged on the record that
   actually proves it, instead of on a per-row flag that a writer may omit.

2. Report per-combination coverage so a missing (shape, dtype, gate, side,
   round, sample) is an error, never a silently smaller table.

Exit code is non-zero on any gap, wrong device, wrong route, missing sample,
unconfirmed shared NaN, or numeric failure.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "agent-results/raw"


def load(path):
    return [json.loads(ln) for ln in Path(path).read_text().splitlines()
            if ln.strip()]


def steady_key(r):
    """Identity of one timed sample, so a re-run can supersede a noisy cell."""
    sh = tuple(r.get("shape") or (None, None))
    return (sh, r.get("dtype"), r.get("gln"), r.get("round"), r.get("side"))


def merge(base, supplements):
    """Later files replace the same (shape,dtype,gate,round,side) sample."""
    out, index = [], {}
    for r in base:
        k = steady_key(r) if r.get("phase") == "steady" else None
        if k is None:
            out.append(r)
            continue
        index[k] = len(out)
        out.append(r)
    replaced = 0
    for path in supplements:
        for r in load(path):
            k = steady_key(r) if r.get("phase") == "steady" else None
            if k is None:
                continue
            if k in index:
                out[index[k]] = r
                replaced += 1
            else:
                index[k] = len(out)
                out.append(r)
    return out, replaced


def validate(path, rounds, samples, require_quiet=False, supplements=()):
    rows, replaced = merge(load(path), supplements)
    steady = [r for r in rows if r.get("phase") == "steady"]
    checks = [r for r in rows if r.get("kind") == "ab_preexisting_nan_check"]
    confirmed = {c["tag"] for c in checks if c.get("confirmed")}
    errors = []
    groups = defaultdict(lambda: defaultdict(list))
    for r in steady:
        K, V = r.get("shape") or (None, None)
        key = (K, V, r.get("dtype"), r.get("gln"))
        problems = []
        if r.get("rc") != 0 or r.get("exit") != 0:
            problems.append(f"rc={r.get('rc')} exit={r.get('exit')}")
        if not r.get("device_ok") or r.get("gcu_current_device") != 4 \
                or r.get("stream_device") != "gcu:4":
            problems.append("device")
        fwd = (r.get("path_fwd_h") or {}).get("done", {})
        bwd = (r.get("path_bwd_dh") or {}).get("done", {})
        want_par = r.get("side") == "parallel"
        got_par = bool(fwd.get("_chunk_fwd_local_state_kernel")) and \
            bool(bwd.get("_chunk_bwd_local_state_kernel"))
        if got_par != want_par:
            problems.append(f"route want_parallel={want_par}")
        if want_par and (fwd.get("chunk_fwd_kernel_h")
                         or bwd.get("chunk_bwd_kernel_dh")):
            problems.append("parallel row used fallback kernel")
        if not want_par and (fwd.get("_chunk_fwd_local_state_kernel")
                             or bwd.get("_chunk_bwd_local_state_kernel")):
            problems.append("fallback row used parallel kernel")
        ms = (r.get("timing") or {}).get("full_fwd_bwd", {}).get("ms") or []
        if len(ms) != samples or not all(
                isinstance(x, (int, float)) and x > 0 for x in ms):
            problems.append(f"samples={len(ms)}/{samples}")
        grade = r.get("grade")
        if want_par:
            if grade not in ("pass", "shared_nan"):
                problems.append(f"grade={grade}")
            if r.get("nan_mismatch_names"):
                problems.append(f"nan_mask={r['nan_mismatch_names']}")
            if grade == "shared_nan":
                tag = (r.get("preexisting_nan_check_tag")
                       or f"ab_{K}x{V}_{r.get('dtype')}_g{r.get('gln')}")
                ok = (r.get("preexisting_nan_confirmed") is True
                      or tag in confirmed)
                if not ok:
                    problems.append("shared_nan unconfirmed")
        if require_quiet:
            idle = (r.get("host_load") or {}).get("cpu_idle_pct_128_159")
            if idle is None or idle < 90:
                problems.append(f"noisy idle={idle}")
        if problems:
            errors.append((r.get("tag"), problems))
        groups[key][r.get("side")].append((r.get("round"), len(ms), r))
    print(f"file: {path}")
    if supplements:
        print(f"supplements (override by shape/dtype/gate/round/side): "
              f"{len(supplements)} file(s), {replaced} sample row(s)")
    print(f"rows {len(rows)}, steady {len(steady)}, "
          f"preexisting-NaN checks {len(checks)} "
          f"({len(confirmed)} confirmed)")
    print("\n| KxV | dtype | gate | fb rows/samples | par rows/samples | "
          "fb median ms | par median ms | par/fb |")
    print("|---|---|---|---|---|---|---|---|")
    ratios = []
    for (K, V, dt, g), sides in sorted(groups.items()):
        fb = [x for x in sides.get("fallback", [])]
        pr = [x for x in sides.get("parallel", [])]
        fbms = [v for x in fb for v in ((x[2].get("timing") or {})
                                        .get("full_fwd_bwd", {}).get("ms") or [])]
        prms = [v for x in pr for v in ((x[2].get("timing") or {})
                                        .get("full_fwd_bwd", {}).get("ms") or [])]
        if len(fb) != rounds or len(pr) != rounds:
            errors.append((f"{K}x{V} {dt} g{g}",
                           [f"rounds fb={len(fb)} par={len(pr)} want {rounds}"]))
        if len(fbms) != rounds * samples or len(prms) != rounds * samples:
            errors.append((f"{K}x{V} {dt} g{g}",
                           [f"samples fb={len(fbms)} par={len(prms)} "
                            f"want {rounds * samples}"]))
        if not fbms or not prms:
            print(f"| {K}x{V} | {dt} | {g} | {len(fb)}/{len(fbms)} | "
                  f"{len(pr)}/{len(prms)} | MISSING | MISSING | - |")
            continue
        fm, pm = statistics.median(fbms), statistics.median(prms)
        ratios.append(pm / fm)
        print(f"| {K}x{V} | {dt} | {g} | {len(fb)}/{len(fbms)} | "
              f"{len(pr)}/{len(prms)} | {fm:.2f} | {pm:.2f} | {pm / fm:.3f} |")
    if ratios:
        print(f"\npar/fb median ratio: min {min(ratios):.3f}, "
              f"max {max(ratios):.3f}, "
              f"mean {sum(ratios) / len(ratios):.3f}")
    if errors:
        print(f"\nFAIL: {len(errors)} problem row(s)", file=sys.stderr)
        for tag, probs in errors[:40]:
            print(f"  {tag}: {probs}", file=sys.stderr)
        return 1
    print("\nVALIDATE_OK")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(RAW / "gla008_ab.jsonl"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--require-quiet", action="store_true")
    ap.add_argument("--supplement", action="append", default=[],
                    help="re-run file whose rows supersede the base sample")
    args = ap.parse_args()
    return validate(args.file, args.rounds, args.samples, args.require_quiet,
                    args.supplement)


if __name__ == "__main__":
    sys.exit(main())
