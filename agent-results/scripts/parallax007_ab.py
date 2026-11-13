"""PARALLAX-007 formal A/B driver.

Every measured process is one invocation of experiments/parallax007_probe.py,
pinned to container GCU 1 (PCI 33:00.0) and CPUs 32-63.  The driver validates
the raw JSONL emitted by the probe and fails (nonzero exit) on:

  * a missing/duplicated record, nonzero probe exit, or missing sample;
  * a device other than 1, CPU affinity outside 32-63, or wrong tree/SHA;
  * a numerical validation failure;
  * an unexpected kernel path, autotune count, or static-config value;
  * a candidate target that still autotunes, or a baseline that does not.

Phases
  probe   candidate cold numerics/paths: target, nearby, t63 for fp16+bf16
  cold    empty-cache cold target: baseline then candidate, alternating dtypes
  steady  three alternating A/B rounds per dtype on a warm process
  cache   TRITON_CACHE_AUTOTUNING=1: empty-cache first process, then a new
          process on the same cache directory (baseline route)

Raw records are written to agent-results/raw/PARALLAX-007/<tag>.jsonl and one
summary line per run to agent-results/raw/PARALLAX-007/runs.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "experiments/parallax007_probe.py"
BASELINE = ROOT / ".baseline-70bdc5"
RAW = ROOT / "agent-results/raw/PARALLAX-007"
BASELINE_SHA = "70bdc5e3ed75e877445b9888250b9155523ef852"
BASELINE_PARALLEL_SHA16 = "51cf806459d5d043"
CANDIDATE_PARALLEL_SHA16 = "5f6fc383ec213bcc"
CPU = "32-63"
EXPECTED_TREES = {"baseline": BASELINE, "candidate": ROOT}
EXPECTED_HEAD = {"baseline": BASELINE_SHA, "candidate": None}
STATIC = {
    "fp16": {"fwd": (128, 1, 2), "dqr": (128, 1, 2), "dkv": (64, 1, 1)},
    "bf16": {"fwd": (128, 1, 2), "dqr": (64, 1, 1), "dkv": (64, 1, 2)},
}
TARGET_KERNELS = {
    "parallel_parallax_fwd_kernel": "fwd",
    "parallel_parallax_bwd_kernel_dqr_fused": "dqr",
    "parallel_parallax_bwd_kernel_dkv_grouped": "dkv",
}
SHORT_KERNELS = {
    "parallel_parallax_fwd_kernel_short",
    "parallel_parallax_bwd_kernel_short",
}
NEARBY_KERNELS = set(TARGET_KERNELS)
FP16, BF16 = "fp16", "bf16"


class CheckError(RuntimeError):
    pass


def check(cond, msg):
    if not cond:
        raise CheckError(msg)


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def git_head(tree: Path) -> str:
    return subprocess.check_output(["git", "-C", str(tree), "rev-parse", "HEAD"], text=True).strip()


def tree_parallel(tree: Path) -> Path:
    return tree / "src/flag_attn/runtime/backend/_enflame/FLA/parallax/parallel.py"


def validate_records(records, *, route, dtype, case, mode, tag, expected_static=None,
                     allow_zero_compiles=False, allow_zero_autotune=False):
    imports = [r for r in records if r.get("event") == "import_probe"]
    check(len(imports) == 1, f"{tag}: expected one import_probe, got {len(imports)}")
    imp = imports[0]
    tree = EXPECTED_TREES[route]
    check(Path(imp["tree"]) == tree, f"{tag}: wrong tree {imp['tree']} != {tree}")
    check(imp["device"] == 1, f"{tag}: wrong device {imp['device']}")
    check(imp["affinity"] == list(range(32, 64)), f"{tag}: wrong affinity {imp['affinity']}")
    check(imp["affinity_count"] == 32, f"{tag}: affinity count {imp['affinity_count']}")
    check(imp["static_d64_env"] == ("1" if route == "candidate" else "0"),
          f"{tag}: static env {imp['static_d64_env']}")
    want_par = BASELINE_PARALLEL_SHA16 if route == "baseline" else CANDIDATE_PARALLEL_SHA16
    check(imp["parallel_py_sha256_16"] == want_par,
          f"{tag}: parallel.py sha {imp['parallel_py_sha256_16']} != {want_par}")
    if EXPECTED_HEAD[route]:
        check(imp["git_head"] == EXPECTED_HEAD[route], f"{tag}: HEAD {imp['git_head']}")
    fatal = [r for r in records if r.get("event") == "fatal"]
    check(not fatal, f"{tag}: fatal event {fatal}")

    launches = [r for r in records if r.get("event") == "launch"]
    check(launches, f"{tag}: no launch records")
    check(all(l["device"] == 1 for l in launches), f"{tag}: launch off device 1")
    summary = [r for r in records if r.get("event") == "launch_summary"]
    check(len(summary) <= 1, f"{tag}: duplicate launch_summary")
    if summary:
        check(summary[0]["devices"] == [1], f"{tag}: launch_summary devices {summary[0]['devices']}")

    static = expected_static or STATIC[dtype]
    autotunes = [r for r in records if r.get("event") == "autotune_config"]
    by_kernel = {}
    for l in launches:
        by_kernel.setdefault(l["kernel"], []).append(l)

    if case == "target":
        check(set(by_kernel) == set(TARGET_KERNELS), f"{tag}: target kernels {sorted(by_kernel)}")
        for kernel, role in TARGET_KERNELS.items():
            recs = by_kernel[kernel]
            check(len(recs) == 1, f"{tag}: {kernel} launched {len(recs)} times")
            rec = recs[0]
            want_grid = [4, 4] if role == "dkv" else [4, 16]
            if route == "baseline" and role == "dkv":
                # Autotuned DKV passes a callable grid, so the harness records
                # no literal grid; the effective grid is always 4 x B*H here.
                check(rec["grid"] in (None, want_grid),
                      f"{tag}: {kernel} grid {rec['grid']}")
            else:
                check(rec["grid"] == want_grid, f"{tag}: {kernel} grid {rec['grid']} != {want_grid}")
            if route == "candidate":
                want = static[role]
                got = (rec.get("BS"), rec.get("num_warps"), rec.get("num_stages"))
                check(got == want, f"{tag}: {kernel} static {got} != {want}")
        if route == "candidate":
            check(not autotunes, f"{tag}: candidate target autotuned {len(autotunes)} configs")
        elif allow_zero_autotune:
            check(len(autotunes) == 0, f"{tag}: cached autotune re-ran {len(autotunes)} configs")
        else:
            check(len(autotunes) == 30, f"{tag}: baseline target autotuned {len(autotunes)} != 30")
    elif case == "nearby":
        check(set(by_kernel) == NEARBY_KERNELS, f"{tag}: nearby kernels {sorted(by_kernel)}")
        check(len(autotunes) == 30, f"{tag}: nearby autotuned {len(autotunes)} != 30")
        for rec in launches:
            check(rec.get("BS") in (16, 32, 64, 128, 256), f"{tag}: nearby BS {rec.get('BS')}")
    elif case == "t63":
        check(set(by_kernel) == SHORT_KERNELS, f"{tag}: t63 kernels {sorted(by_kernel)}")
        check(not autotunes, f"{tag}: t63 autotuned {len(autotunes)}")
        for rec in launches:
            check(rec["grid"] == [1, 1], f"{tag}: t63 grid {rec['grid']}")
            check((rec.get("num_warps"), rec.get("num_stages")) == (4, 1),
                  f"{tag}: t63 warps/stages {(rec.get('num_warps'), rec.get('num_stages'))}")
    else:
        raise CheckError(f"{tag}: unknown case {case}")

    if mode == "cold":
        vals = [r for r in records if r.get("event") == "validation"]
        check(len(vals) == 1, f"{tag}: expected one validation, got {len(vals)}")
        val = vals[0]
        check(len(val["results"]) == 5, f"{tag}: expected 5 tensors, got {len(val['results'])}")
        check(all(r["name"] == n for r, n in zip(val["results"], ("o", "dq", "dr", "dk", "dv"))),
              f"{tag}: metric order {[r['name'] for r in val['results']]}")
        check(val["all_pass"], f"{tag}: numeric failure {val['results']}")
        check(all(r["finite_actual"] for r in val["results"]), f"{tag}: nonfinite output")
        calls = [r for r in records if r.get("event") == "device_call"]
        check(len(calls) == 1, f"{tag}: device_call count {len(calls)}")
        if not allow_zero_compiles:
            check(calls[0]["compiles"] > 0, f"{tag}: no compile event")
        else:
            # JITFunction.run fires once per kernel even on an autotune JSON
            # hit (binary load only), and re-benchmarking a deleted JSON entry
            # also re-enters the JIT for the cached binaries.  What separates
            # a cache hit from a real cold build is time, not call count: a
            # cold DKV compile alone costs ~18-28 s, while every warm path
            # stays below 0.1 s total.
            check(calls[0]["compile_s"] < 2.0,
                  f"{tag}: warm-cache compile time {calls[0]['compile_s']:.3f}s looks like a rebuild")
    else:
        samples = [r for r in records if r.get("event") == "sample"]
        check(len(samples) == 5, f"{tag}: expected 5 samples, got {len(samples)}")
        check([s["sample"] for s in samples] == list(range(5)), f"{tag}: sample order")
        check(all(s["full_s"] > 0 and s["forward_s"] > 0 and s["backward_s"] > 0 for s in samples),
              f"{tag}: zero sample")
        summ = [r for r in records if r.get("event") == "steady_summary"]
        check(len(summ) == 1, f"{tag}: steady_summary count {len(summ)}")
        paths = [r for r in records if r.get("event") == "path_probe"]
        check(len(paths) == 1, f"{tag}: path_probe count {len(paths)}")
        check(paths[0]["devices"] == [1], f"{tag}: path devices {paths[0]['devices']}")
        check(set(by_kernel) == set(TARGET_KERNELS), f"{tag}: steady kernels {sorted(by_kernel)}")
        for kernel, role in TARGET_KERNELS.items():
            recs = by_kernel[kernel]
            check(len(recs) == 1, f"{tag}: steady {kernel} launched {len(recs)} times")
            if route == "candidate":
                got = (recs[0].get("BS"), recs[0].get("num_warps"), recs[0].get("num_stages"))
                check(got == static[role],
                      f"{tag}: steady {kernel} static {got} != {static[role]}")
        if route == "candidate":
            check(not autotunes, f"{tag}: candidate steady autotuned {len(autotunes)} configs")
        else:
            check(len(autotunes) == 30, f"{tag}: baseline steady autotuned {len(autotunes)} != 30")
    return {"import": imp, "launches": launches, "autotunes": autotunes,
            "validation": ([r for r in records if r.get("event") == "validation"] or [None])[0],
            "samples": [r for r in records if r.get("event") == "sample"],
            "steady": ([r for r in records if r.get("event") == "steady_summary"] or [None])[0],
            "disk_cache": [r for r in records if r.get("event") == "disk_cache"]}


def completed(tag):
    path = RAW / "runs.jsonl"
    if not path.exists() or not (RAW / f"{tag}.jsonl").exists():
        return None
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("tag") == tag:
            return rec
    return None


def run_probe(route, dtype, case, mode, round_id, *, tag, cache_dir, samples=5,
              cache_autotuning=False, require_empty=False, config_json=None,
              forward_only=False, check_cache_hit=False, expect_retune=False):
    done = completed(tag)
    if done is not None:
        print(json.dumps({**done, "resumed": True}), flush=True)
        return done
    tree = EXPECTED_TREES[route]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    before = sum(1 for p in cache_dir.rglob("*") if p.is_file())
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{tree / 'src'}:{tree}:/opt/flagtree"
    env["PARALLAX_STATIC_D64"] = "1" if route == "candidate" else "0"
    env["TRITON_CACHE_DIR"] = str(cache_dir)
    env["TRITON_F32_DEFAULT"] = "ieee"
    env.pop("TRITON_CACHE_AUTOTUNING", None)
    if cache_autotuning:
        env["TRITON_CACHE_AUTOTUNING"] = "1"
    cmd = ["timeout", "900", "taskset", "-c", CPU, "/flagos/bin/python", str(PROBE),
           "--route", route, "--dtype", dtype, "--case", case, "--mode", mode,
           "--round", str(round_id), "--samples", str(samples), "--tag", tag]
    if require_empty:
        cmd.append("--require-empty")
    if config_json:
        cmd += ["--config-json", config_json]
    if forward_only:
        cmd.append("--forward-only")
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
    wall = time.time() - started
    records = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    RAW.mkdir(parents=True, exist_ok=True)
    raw_path = RAW / f"{tag}.jsonl"
    with raw_path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str) + "\n")
    (RAW / f"{tag}.log").write_text(
        f"CMD {' '.join(cmd)}\nTREE {tree}\nCACHE {cache_dir}\n"
        f"CACHE_FILES_BEFORE {before}\nWALL_S {wall:.3f}\nRC {proc.returncode}\n"
        f"STDOUT\n{proc.stdout}\nSTDERR\n{proc.stderr}\n"
    )
    check(proc.returncode == 0, f"{tag}: probe rc {proc.returncode}; see {raw_path}")
    check(records, f"{tag}: no JSON records")
    after = sum(1 for p in cache_dir.rglob("*") if p.is_file())
    expected_static = None
    if config_json:
        payload = json.loads(config_json)
        expected_static = {k: tuple(int(x) for x in v) for k, v in payload.items()}
    result = validate_records(
        records, route=route, dtype=dtype, case=case, mode=mode, tag=tag,
        expected_static=expected_static,
        allow_zero_compiles=check_cache_hit or expect_retune,
        allow_zero_autotune=check_cache_hit,
    )
    if check_cache_hit:
        hits = [r for r in result["disk_cache"] if r.get("hit") is True]
        check(len(hits) == 3, f"{tag}: expected 3 disk-cache hits, got {hits}")
        check(len(result["autotunes"]) == 0, f"{tag}: re-tuned despite disk cache")
    if expect_retune:
        check(len(result["autotunes"]) == 30,
              f"{tag}: expected full re-tune, got {len(result['autotunes'])} configs")
        hits = [r for r in result["disk_cache"] if r.get("hit") is True]
        check(not hits, f"{tag}: unexpected disk-cache hit after invalidation: {hits}")
    if mode == "cold":
        check(before == 0 if require_empty else True, f"{tag}: cache not empty ({before})")
    summary = {
        "tag": tag, "route": route, "dtype": dtype, "case": case, "mode": mode,
        "round": round_id, "wall_s": round(wall, 3), "rc": proc.returncode,
        "cache_dir": str(cache_dir), "cache_files_before": before,
        "cache_files_after": after, "raw": str(raw_path),
        "pid": result["import"]["pid"], "git_head": result["import"]["git_head"],
        "parallel_py_sha256_16": result["import"]["parallel_py_sha256_16"],
        "compiles": next((r.get("compiles") for r in records if r.get("event") == "device_call"), None),
        "autotune_configs": len(result["autotunes"]),
        "launches": {k: len(v) for k, v in
                     {k: [l for l in result["launches"] if l["kernel"] == k]
                      for k in {l["kernel"] for l in result["launches"]}}.items()},
        "validation_pass": (result["validation"] or {}).get("all_pass"),
        "peak_alloc": next((r.get("peak_memory_allocated_bytes") for r in records
                            if r.get("event") == "peak_memory"), None)
        or (result["steady"] or {}).get("peak_memory_allocated_bytes"),
        "peak_reserved": next((r.get("peak_memory_reserved_bytes") for r in records
                               if r.get("event") == "peak_memory"), None)
        or (result["steady"] or {}).get("peak_memory_reserved_bytes"),
        "disk_cache_hits": sum(1 for r in result["disk_cache"] if r.get("hit") is True),
        "disk_cache_hit_kernels": sorted({r["kernel"] for r in result["disk_cache"] if r.get("hit") is True}),
        "disk_cache_entries": len(result["disk_cache"]),
        "autotune_files": len(list(cache_dir.rglob("*.autotune.json"))),
        "autotune_file_paths": [str(p) for p in sorted(cache_dir.rglob("*.autotune.json"))],
    }
    with (RAW / "runs.jsonl").open("a") as fh:
        fh.write(json.dumps(summary) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def autotune_cache(dtype):
    return RAW / "cache" / f"autotune-{dtype}"


def phase_probe():
    for dtype in (FP16, BF16):
        for case in ("target", "nearby", "t63"):
            run_probe("candidate", dtype, case, "cold", 0,
                      tag=f"probe-candidate-{dtype}-{case}",
                      cache_dir=RAW / "cache" / f"probe-{dtype}-{case}",
                      require_empty=True)


def phase_cold():
    # The baseline cold process also seeds the cross-process autotune cache:
    # TRITON_CACHE_AUTOTUNING only adds a post-benchmark JSON write, so the
    # first-process compile/tune cost is identical with or without it.
    for dtype in (FP16, BF16):
        run_probe("baseline", dtype, "target", "cold", 0,
                  tag=f"cold-baseline-{dtype}-target",
                  cache_dir=autotune_cache(dtype),
                  cache_autotuning=True, require_empty=True)
        run_probe("candidate", dtype, "target", "cold", 0,
                  tag=f"cold-candidate-{dtype}-target",
                  cache_dir=RAW / "cache" / f"cold-candidate-{dtype}",
                  require_empty=True)


def phase_steady():
    for dtype in (FP16, BF16):
        for round_id in range(3):
            order = ("baseline", "candidate") if round_id % 2 == 0 else ("candidate", "baseline")
            for route in order:
                cache_dir = (autotune_cache(dtype) if route == "baseline"
                             else RAW / "cache" / f"cold-candidate-{dtype}")
                run_probe(route, dtype, "target", "steady", round_id,
                          tag=f"steady-{route}-{dtype}-r{round_id}",
                          cache_dir=cache_dir)


def phase_cache():
    for dtype in (FP16, BF16):
        cache_dir = autotune_cache(dtype)
        entries = sorted(cache_dir.rglob("*.autotune.json"))
        check(len(entries) == 3,
              f"autotune-{dtype}: cold baseline left {len(entries)} cache files")
        run_probe("baseline", dtype, "target", "cold", 0,
                  tag=f"autotune-reuse-{dtype}", cache_dir=cache_dir,
                  cache_autotuning=True, check_cache_hit=True)
        if dtype == FP16 and completed(f"autotune-invalidated-{dtype}") is None:
            # One invalidation run is enough to prove the cache key is content
            # addressed: delete the JSON entries, keep the compiled binaries,
            # and confirm the next process re-tunes and rewrites them.
            doomed = sorted(cache_dir.rglob("*.autotune.json"))
            check(len(doomed) == 3, f"autotune-{dtype}: expected 3 cache files, got {doomed}")
            for path in doomed:
                path.unlink()
            run_probe("baseline", dtype, "target", "cold", 0,
                      tag=f"autotune-invalidated-{dtype}", cache_dir=cache_dir,
                      cache_autotuning=True, expect_retune=True)
        if dtype == FP16:
            check(len(list(cache_dir.rglob("*.autotune.json"))) == 3,
                  f"autotune-{dtype}: cache files not rewritten after invalidation")
        run_probe("candidate", dtype, "target", "cold", 0,
                  tag=f"autotune-static-{dtype}",
                  cache_dir=RAW / "cache" / f"autotune-static-{dtype}",
                  cache_autotuning=True, require_empty=True)
        # A later process with both the binary cache and the autotune JSON
        # warm is the realistic repeat-launch case; the static route pays no
        # tuning but must still load its three binaries.
        run_probe("candidate", dtype, "target", "cold", 0,
                  tag=f"autotune-static-warm-{dtype}",
                  cache_dir=RAW / "cache" / f"cold-candidate-{dtype}",
                  check_cache_hit=False)


def phase_config():
    # Configuration sensitivity: run the same steady full-call loop with a
    # neighbouring tuple substituted for one kernel at a time.  These runs are
    # reported separately and never feed the shipped-config verdict.
    alternates = {
        FP16: {
            "fwd-s1": {"fwd": (128, 1, 1)},
            "dqr-s1": {"dqr": (128, 1, 1)},
            "dkv-s2": {"dkv": (64, 1, 2)},
        },
        BF16: {
            "fwd-s1": {"fwd": (128, 1, 1)},
            "dqr-s2": {"dqr": (128, 1, 2)},
            "dkv-s1": {"dkv": (64, 1, 1)},
        },
    }
    for dtype in (FP16, BF16):
        for name, delta in alternates[dtype].items():
            override = {k: list(v) for k, v in STATIC[dtype].items()}
            override.update({k: list(v) for k, v in delta.items()})
            label = "-".join(f"{k}{v[0]}w{v[1]}s{v[2]}" for k, v in sorted(delta.items()))
            run_probe("candidate", dtype, "target", "steady", 3,
                      tag=f"config-{dtype}-{name}-{label}",
                      cache_dir=RAW / "cache" / f"cold-candidate-{dtype}",
                      config_json=json.dumps(override))


def _mutated(tag, mutate):
    records = [json.loads(l) for l in (RAW / f"{tag}.jsonl").read_text().splitlines()]
    records = json.loads(json.dumps(records))
    mutate(records)
    return records


def _drop(event, predicate=lambda r: True):
    def mutate(records):
        for i, rec in enumerate(records):
            if rec.get("event") == event and predicate(rec):
                del records[i]
                return
        raise CheckError(f"self-check: no {event} record to drop")
    return mutate


def _patch(event, **fields):
    def mutate(records):
        for rec in records:
            if rec.get("event") == event:
                rec.update(fields)
                return
        raise CheckError(f"self-check: no {event} record to patch")
    return mutate


def _patch_launch(kernel, **fields):
    def mutate(records):
        for rec in records:
            if rec.get("event") == "launch" and rec.get("kernel") == kernel:
                rec.update(fields)
                return
        raise CheckError(f"self-check: no launch for {kernel}")
    return mutate


def _rename_kernel(old, new):
    def mutate(records):
        for rec in records:
            if rec.get("event") == "launch" and rec.get("kernel") == old:
                rec["kernel"] = new
                return
        raise CheckError(f"self-check: no launch for {old}")
    return mutate


SELF_CHECKS = (
    ("steady-candidate-fp16-r0", "candidate", "fp16", "target", "steady", _drop("sample"), {}),
    ("steady-candidate-fp16-r0", "candidate", "fp16", "target", "steady",
     _patch_launch("parallel_parallax_bwd_kernel_dkv_grouped", device=0), {}),
    ("cold-candidate-fp16-target", "candidate", "fp16", "target", "cold",
     _patch_launch("parallel_parallax_fwd_kernel", device=0), {}),
    ("cold-candidate-fp16-target", "candidate", "fp16", "target", "cold",
     _patch("validation", all_pass=False), {}),
    ("cold-candidate-fp16-target", "candidate", "fp16", "target", "cold",
     _patch_launch("parallel_parallax_fwd_kernel", BS=64), {}),
    ("cold-candidate-fp16-target", "candidate", "fp16", "target", "cold",
     _rename_kernel("parallel_parallax_fwd_kernel", "parallel_parallax_fwd_kernel_short"), {}),
    ("cold-baseline-fp16-target", "baseline", "fp16", "target", "cold",
     _drop("autotune_config"), {}),
    ("cold-candidate-fp16-target", "candidate", "fp16", "target", "cold",
     _patch("import_probe", affinity=list(range(32, 60))), {}),
)


def self_check():
    """Prove the validator rejects each failure class on real recorded output."""
    for tag, route, dtype, case, mode, mutate, kwargs in SELF_CHECKS:
        records = _mutated(tag, mutate)
        try:
            validate_records(records, route=route, dtype=dtype, case=case,
                             mode=mode, tag=f"selfcheck-{tag}", **kwargs)
        except CheckError as exc:
            print(json.dumps({"self_check": tag, "rejected": True, "reason": str(exc)}))
        else:
            raise CheckError(f"self-check: validator accepted mutated record for {tag}")
    print(json.dumps({"self_check": "ok", "cases": len(SELF_CHECKS)}))


PHASES = {"probe": phase_probe, "cold": phase_cold, "steady": phase_steady,
          "cache": phase_cache, "config": phase_config}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=tuple(PHASES) + ("all",))
    ap.add_argument("--self-check", action="store_true",
                    help="replay the validator on mutated copies of recorded output")
    args = ap.parse_args()
    if args.self_check:
        check(args.phase is None, "--self-check does not take --phase")
        self_check()
        return
    check(args.phase is not None, "one of --phase/--self-check is required")
    if git_head(ROOT) is None:
        raise CheckError("cannot read candidate HEAD")
    check(tree_parallel(BASELINE).exists(), "baseline worktree missing")
    phases = list(PHASES) if args.phase == "all" else [args.phase]
    for name in phases:
        print(json.dumps({"phase": name, "start": time.strftime("%FT%TZ", time.gmtime())}), flush=True)
        PHASES[name]()
    print(json.dumps({"status": "ok", "phases": phases}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except CheckError as exc:
        print(f"PARALLAX007_CHECK_ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(5)
