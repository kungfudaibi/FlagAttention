"""Turn PARALLAX-007 raw JSONL into the report tables."""
import json
import statistics
from pathlib import Path

RAW = Path("agent-results/raw/PARALLAX-007")
runs = [json.loads(l) for l in (RAW / "runs.jsonl").read_text().splitlines() if l.strip()]


def by_tag(tag):
    return next(r for r in runs if r["tag"] == tag)


def steady(tag):
    recs = [json.loads(l) for l in (RAW / f"{tag}.jsonl").read_text().splitlines()]
    summ = next(r for r in recs if r["event"] == "steady_summary")
    return summ


def fmt(v):
    return f"{v*1000:.3f}"


print("== cold/probe runs ==")
for r in runs:
    if r["mode"] != "cold":
        continue
    print(f"{r['tag']:<34} wall={r['wall_s']:8.1f}s compiles={r['compiles']} "
          f"tunes={r['autotune_configs']} cache={r['cache_files_before']}->{r['cache_files_after']} "
          f"peak_alloc={r['peak_alloc']} peak_res={r['peak_reserved']} valid={r['validation_pass']}")

print("\n== steady medians (ms) ==")
for dtype in ("fp16", "bf16"):
    for rnd in range(3):
        row = {}
        for route in ("baseline", "candidate"):
            s = steady(f"steady-{route}-{dtype}-r{rnd}")
            sm = s["summary"]
            row[route] = (sm["full_s"]["median"] * 1000, sm["forward_s"]["median"] * 1000,
                          sm["backward_s"]["median"] * 1000, sm["full_s"]["min"] * 1000,
                          sm["full_s"]["max"] * 1000, sm["full_s"]["stdev"] * 1000)
        b, c = row["baseline"], row["candidate"]
        print(f"{dtype} r{rnd}: base full={b[0]:.3f} fwd={b[1]:.3f} bwd={b[2]:.3f} "
              f"range=({b[3]:.3f},{b[4]:.3f}) sd={b[5]:.3f} | "
              f"cand full={c[0]:.3f} fwd={c[1]:.3f} bwd={c[2]:.3f} "
              f"range=({c[3]:.3f},{c[4]:.3f}) sd={c[5]:.3f} | "
              f"delta={((c[0]/b[0])-1)*100:+.2f}%")

print("\n== per-round baseline autotune winner vs shipped static ==")
import re
static = {"fp16": {"fwd": (128, 1, 2), "dqr": (128, 1, 2), "dkv": (64, 1, 1)},
          "bf16": {"fwd": (128, 1, 2), "dqr": (64, 1, 1), "dkv": (64, 1, 2)}}
roles = {"parallel_parallax_fwd_kernel": "fwd", "parallel_parallax_bwd_kernel_dqr_fused": "dqr",
         "parallel_parallax_bwd_kernel_dkv_grouped": "dkv"}
for dtype in ("fp16", "bf16"):
    for rnd in range(3):
        recs = [json.loads(l) for l in (RAW / f"steady-baseline-{dtype}-r{rnd}.jsonl").read_text().splitlines()]
        groups = {}
        for r in recs:
            if r.get("event") != "autotune_config":
                continue
            m = re.search(r"BS: (\d+).*num_warps: (\d+).*num_stages: (\d+)", r["config"])
            groups.setdefault(roles[r["kernel"]], []).append(
                ((int(m.group(1)), int(m.group(2)), int(m.group(3))), r["median_ms"]))
        line = []
        for role in ("fwd", "dqr", "dkv"):
            vals = sorted(groups[role], key=lambda x: x[1])
            want = static[dtype][role]
            ms = next(v for c, v in vals if c == want)
            rank = [c for c, _ in vals].index(want) + 1
            line.append(f"{role}: best={vals[0][0]}@{vals[0][1]:.4f} "
                        f"static={want}@{ms:.4f} gap={(ms/vals[0][1]-1)*100:+.2f}% rank={rank}/{len(vals)}")
        print(f"{dtype} r{rnd}: " + " | ".join(line))

print("\n== config sensitivity (steady, shipped vs one-tuple alternate) ==")
for r in runs:
    if not r["tag"].startswith("config-"):
        continue
    s = steady(r["tag"])
    print(f"{r['tag']:<44} full_med={s['summary']['full_s']['median']*1000:.3f}ms "
          f"sd={s['summary']['full_s']['stdev']*1000:.3f}")

print("\n== disk autotune cache ==")
for r in runs:
    if "autotune" not in r["tag"]:
        continue
    print(f"{r['tag']:<34} wall={r['wall_s']:8.1f}s compiles={r['compiles']} tunes={r['autotune_configs']} "
          f"hits={r['disk_cache_hits']} {r['disk_cache_hit_kernels']} files={r['autotune_files']}")

print("\n== memory (efsmi before/after) ==")
for r in runs:
    if not (r["tag"].startswith("cold-") or r["tag"].startswith("steady-")):
        continue
    recs = [json.loads(l) for l in (RAW / f"{r['tag']}.jsonl").read_text().splitlines()]
    b = next((x for x in recs if x["event"] == "device_memory_before"), {})
    a = next((x for x in recs if x["event"] == "device_memory_after"), {})
    print(f"{r['tag']:<34} alloc={r['peak_alloc']} reserved={r['peak_reserved']} "
          f"efsmi_used={b.get('Used Size')}->{a.get('Used Size')}")
