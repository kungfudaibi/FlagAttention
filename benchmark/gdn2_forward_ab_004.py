#!/usr/bin/env python3
"""GDN2-004 reproducible full-forward A/B for the Enflame S60 GDN-2 operator.

The candidate is the K2 state-V-tile split: ``chunk_gdn2_fwd_h_o_infer`` walks
chunks serially per (batch, head, V-tile) and is only parallel over (V-tile,
batch*head), so a 256-wide tile over V in (128, 256] leaves half the machine
idle.  The candidate uses two V tiles there and is bit-identical because V is
never an accumulation axis.

Both sides come from immutable commits via ``git archive``:

* ``base`` -- the accepted GDN2-003 commit c2f42ac07fc15e96f748e1a819fb32a554b11911,
  with ``FLAG_ATTN_GDN2_STATE_BV_SPLIT=0`` injected so the knob cannot differ;
* ``cand`` -- the candidate commit with the default (split enabled).

Every (round, variant, dtype, shape) runs in its own process with a private
Triton cache, so ``cold_compile_s`` is a real first compile and the following
synchronized event samples are steady state.  Each record carries the launched
kernels and their BK/BV/BT, peak device allocation, per-sample timings, finite
checks and the error against an independent CPU FP64 recurrence.

Usage (inside the task clone):

    /flagos/bin/python benchmark/gdn2_forward_ab_004.py drive \
        --workspace . --base-ref c2f42ac07fc15e96f748e1a819fb32a554b11911 \
        --cand-ref HEAD --rounds 3 --out agent-results/GDN2-004-ab.jsonl

    /flagos/bin/python benchmark/gdn2_forward_ab_004.py check \
        --out agent-results/GDN2-004-ab.jsonl

``drive`` exits non-zero if any child fails, if any shape is missing from any
round, or if any candidate output is not bit-identical to the base side.
``check`` re-validates an existing JSONL offline and exits non-zero on any
inconsistency, so a stale or partial file cannot be mistaken for a pass.
The whole run uses GCU 0 (PCI 23:00.0) and CPUs 0-31 only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEVICE = "gcu:0"
EXPECTED_PCI = "23:00.0"
CPU_LIST = "0-31"
SEED = 42
CHUNK_SIZE = 64
STEADY_SAMPLES = 15
WARMUP_ITERS = 5
CLOCK_WARMUP_S = 3.0
TASK = "GDN2-004"

# The accepted GDN2-003 baseline and the knob that reproduces its K2 V tile.
BASE_REF = "c2f42ac07fc15e96f748e1a819fb32a554b11911"

# Shapes whose V lands in (128, 256] are where the candidate should win; the
# V <= 128 and V > 256 cases are controls that must stay bit-identical and
# neutral.  Irregular K (96) and K != V are covered because GDN2-004 is the
# irregular-dimension task.
BF16_SHAPES = [
    (1, 1024, 4, 256, 256),
    (2, 512, 4, 160, 160),
    (2, 2048, 4, 256, 256),
    (2, 512, 4, 96, 128),
    (2, 512, 4, 96, 96),
    (2, 512, 4, 64, 128),
    (2, 512, 4, 128, 64),
    (2, 512, 8, 64, 64),
]
FP16_SHAPES = [
    (1, 1024, 4, 256, 256),
    (2, 512, 4, 160, 160),
    (2, 512, 4, 64, 128),
    (2, 512, 4, 128, 64),
]

GATE_ATOL = 1e-3
GATE_RTOL = 2e-2


def _jsonl_write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _loadavg() -> str:
    try:
        return Path("/proc/loadavg").read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"


def _efsmi_used_mib(index: int = 0):
    try:
        out = subprocess.run(
            ["efsmi", "-q", "-i", str(index)],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"Used Size\s*:\s*(\d+)\s*MiB", out)
    return float(match.group(1)) if match else None


def _efsmi_pci(index: int = 0):
    try:
        out = subprocess.run(
            ["efsmi", "-q", "-d", "PCIE", "--json-format"],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        data = json.loads(out)
    except ValueError:
        return None
    for item in data.get("devices", []):
        if str(item.get("id")) == str(index):
            pcie = item.get("PCIe Info", {})
            return f"{pcie.get('Bus')}:{pcie.get('Dev')}.{pcie.get('Func')}"
    return None


def _build_inputs(B, T, H, K, V, dtype, device):
    import torch

    torch.manual_seed(SEED)
    q = (torch.randn(B, T, H, K) / math.sqrt(K)).to(dtype)
    k = (torch.randn(B, T, H, K) / math.sqrt(K)).to(dtype)
    v = torch.randn(B, T, H, V).to(dtype)
    g = (-torch.rand(B, T, H, K) * 0.1).to(dtype)
    b = torch.rand(B, T, H, K).to(dtype)
    w = torch.rand(B, T, H, V).to(dtype)
    h0 = torch.randn(B, H, K, V) * 0.01
    tensors = [x.to(device) for x in (q, k, v, g, b, w, h0)]
    cpu = [x.cpu() for x in (q, k, v, g, b, w, h0)]
    return tensors, cpu


def _metrics(expected, actual):
    e = expected.double().flatten().cpu()
    a = actual.double().flatten().cpu()
    diff = (e - a).abs()
    rms = diff.square().mean().sqrt().item()
    ref_rms = e.square().mean().sqrt().item()
    return {
        "max_abs": diff.max().item(),
        "rms_rel": rms / (ref_rms + 1e-12),
        "ref_rms": ref_rms,
        "ref_max": e.abs().max().item(),
    }


def _gates(metrics):
    strict = (
        metrics["max_abs"] <= GATE_ATOL + GATE_RTOL * metrics["ref_rms"]
        and metrics["rms_rel"] <= GATE_RTOL
    )
    magnitude = (
        metrics["max_abs"] <= GATE_ATOL + GATE_RTOL * metrics["ref_max"]
        and metrics["rms_rel"] <= GATE_RTOL
    )
    return {"gate_a": bool(strict), "gate_b": bool(magnitude)}


def _install_kernel_tracer():
    """Record every Triton launch: kernel name, grid and BK/BV/BT."""
    from triton.runtime.jit import JITFunction

    records = []
    original = JITFunction.run

    def traced(self, *args, **kwargs):
        grid = kwargs.get("grid")
        if callable(grid):
            grid_repr = "callable"
        elif isinstance(grid, (tuple, list)):
            grid_repr = [int(x) for x in grid]
        else:
            grid_repr = repr(grid)
        cfg = {}
        for key in ("BK", "BV", "BT", "GRID_MODE", "STATE_V_FIRST",
                    "num_warps", "num_stages"):
            if key in kwargs:
                value = kwargs[key]
                cfg[key] = value if isinstance(value, (int, float, bool, str)) else repr(value)
        records.append({
            "kernel": getattr(self.fn, "__name__", str(self.fn)),
            "grid": grid_repr,
            "cfg": cfg,
        })
        return original(self, *args, **kwargs)

    JITFunction.run = traced
    return records


def _clock_warmup(call, seconds=CLOCK_WARMUP_S):
    """Keep the S60 clocks boosted before sampling.

    A cold process starts at low clocks; timing those samples would compare
    clock ramp-up, not the kernel.  This loop is timed separately from the
    steady samples and its iteration count is recorded.
    """
    import torch

    deadline = time.perf_counter() + seconds
    iterations = 0
    while iterations < WARMUP_ITERS or time.perf_counter() < deadline:
        call()
        iterations += 1
    torch.gcu.synchronize()
    return iterations


def _time_call(call, samples=STEADY_SAMPLES):
    import torch

    for _ in range(WARMUP_ITERS):
        call()
    torch.gcu.synchronize()
    begin = torch.gcu.Event(enable_timing=True)
    end = torch.gcu.Event(enable_timing=True)
    values = []
    for _ in range(samples):
        begin.record()
        call()
        end.record()
        torch.gcu.synchronize()
        values.append(begin.elapsed_time(end))
    ordered = sorted(values)
    return {
        "samples_ms": [round(x, 6) for x in values],
        "median_ms": round(statistics.median(values), 6),
        "mean_ms": round(statistics.fmean(values), 6),
        "min_ms": round(ordered[0], 6),
        "p25_ms": round(ordered[len(ordered) // 4], 6),
        "max_ms": round(ordered[-1], 6),
    }


def _digest(tensor) -> str:
    """Exact bit pattern of a tensor, comparable across processes."""
    import hashlib

    import torch

    raw = tensor.detach().to("cpu").contiguous()
    if raw.dtype == torch.bfloat16:
        # numpy has no bfloat16; reinterpret the 16-bit payload.
        raw = raw.view(torch.int16)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def _state_tiles(impl, V):
    """The K2 V/BK tiles this tree will actually launch for ``V``."""
    state_bv = impl._s60_state_bv(V) if hasattr(impl, "_s60_state_bv") else impl._s60_tile(V, impl._S60_STATE_BV)
    return {"state_bv": state_bv, "n_v_tiles": -(-V // state_bv)}


def command_run(args) -> int:
    import torch

    tree = Path(args.tree).resolve()
    sys.path.insert(0, str(tree / "src"))
    sys.path.insert(0, str(tree / "tests" / "flag_attn"))

    from gdn2_backward_ref import gdn2_forward  # noqa: E402
    from flag_attn import chunk_gdn2  # noqa: E402
    impl = __import__(
        "flag_attn.runtime.backend._enflame.FLA.gdn2.chunk_gdn2", fromlist=["x"]
    )

    out = Path(args.out)
    dtype_name = args.dtype
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    shapes = BF16_SHAPES if dtype_name == "bfloat16" else FP16_SHAPES
    if args.shape:
        shapes = [tuple(int(x) for x in item.split(",")) for item in args.shape]

    pci = _efsmi_pci(0)
    _jsonl_write(out, {
        "schema": 1, "task": TASK, "kind": "meta",
        "variant": args.variant, "round": args.round,
        "tree": str(tree), "commit": args.commit, "dtype": dtype_name,
        "hostname": platform.node(), "torch": torch.__version__,
        "device_count_visible": torch.gcu.device_count(),
        "current_device": torch.gcu.current_device(),
        "pci0": pci, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "triton_cache": os.environ.get("TRITON_CACHE_DIR"),
        "tile_adapt_default": getattr(impl, "_S60_TILES_ADAPT", None),
        "bv_split_default": getattr(impl, "_S60_STATE_BV_SPLIT_ENABLED", None),
        "loadavg": _loadavg(), "efsmi_used_mib": _efsmi_used_mib(0),
        "started_at": time.time(),
    })
    if pci is None or pci.strip().lower() != EXPECTED_PCI:
        _jsonl_write(out, {"kind": "error", "task": TASK, "variant": args.variant,
                           "round": args.round,
                           "error": f"unexpected GCU0 PCI {pci}"})
        return 2
    if torch.gcu.device_count() != 1:
        _jsonl_write(out, {"kind": "error", "task": TASK, "variant": args.variant,
                           "round": args.round,
                           "error": f"TOPS_VISIBLE_DEVICES did not pin one device: "
                                    f"{torch.gcu.device_count()}"})
        return 2

    tracer = _install_kernel_tracer()
    exit_code = 0
    for shape in shapes:
        B, T, H, K, V = shape
        scale = K ** -0.5
        try:
            inputs, cpu_inputs = _build_inputs(B, T, H, K, V, dtype, DEVICE)
            ref_o, ref_ht = gdn2_forward(*cpu_inputs[:6], scale, cpu_inputs[6])

            def call():
                with torch.inference_mode():
                    return chunk_gdn2(
                        *inputs[:6], scale=scale, initial_state=inputs[6],
                        output_final_state=True, chunk_size=CHUNK_SIZE,
                    )

            tracer.clear()
            torch.gcu.synchronize()
            wall_start = time.perf_counter()
            o, ht = call()
            torch.gcu.synchronize()
            cold_s = time.perf_counter() - wall_start
            cold_launches = [dict(item) for item in tracer]

            torch.gcu.synchronize()
            if hasattr(torch.gcu, "reset_peak_memory_stats"):
                torch.gcu.reset_peak_memory_stats()
            base_alloc = torch.gcu.memory_allocated()
            o, ht = call()
            torch.gcu.synchronize()
            peak_mib = (torch.gcu.max_memory_allocated() - base_alloc) / 2**20

            clock_iters = _clock_warmup(call)

            tracer.clear()
            steady = _time_call(call)
            steady_launches = [dict(item) for item in tracer]

            metrics = {}
            for label, ref, act in (("o", ref_o, o), ("ht", ref_ht, ht)):
                item = _metrics(ref, act)
                item.update(_gates(item))
                item["finite"] = bool(torch.isfinite(act).all().item())
                metrics[label] = item

            # A float32 checksum of the exact bits: two variants that agree here
            # and in every metric are bit-identical for all practical purposes,
            # and `check` compares the full tensors recorded as raw words.
            _jsonl_write(out, {
                "schema": 1, "kind": "shape", "task": TASK,
                "variant": args.variant, "round": args.round,
                "commit": args.commit, "dtype": dtype_name,
                "shape": [B, T, H, K, V],
                "tiles": _state_tiles(impl, V),
                "cold_compile_s": round(cold_s, 6),
                "cold_launches": cold_launches,
                "steady_launches": steady_launches,
                "steady": steady,
                "clock_warmup_iters": clock_iters,
                "clock_warmup_s": CLOCK_WARMUP_S,
                "peak_alloc_mib": round(peak_mib, 6),
                "peak_reserved_mib": round(
                    (torch.gcu.max_memory_reserved() - torch.gcu.memory_reserved()) / 2**20, 6
                ) if hasattr(torch.gcu, "max_memory_reserved") else None,
                "efsmi_used_mib": _efsmi_used_mib(0),
                "metrics": metrics,
                "exact": {
                    "o_sha256": _digest(o),
                    "ht_sha256": _digest(ht),
                },
                "loadavg": _loadavg(),
                "wall_clock_s": round(time.perf_counter() - wall_start, 6),
            })
            del inputs, cpu_inputs, o, ht
        except Exception as error:  # noqa: BLE001 - one bad shape must be recorded
            import traceback
            traceback.print_exc()
            exit_code = 1
            _jsonl_write(out, {
                "schema": 1, "kind": "error", "task": TASK,
                "variant": args.variant, "round": args.round,
                "commit": args.commit, "dtype": dtype_name, "shape": list(shape),
                "error": f"{type(error).__name__}: {error}",
            })
    _jsonl_write(out, {"schema": 1, "kind": "meta_end", "task": TASK,
                       "variant": args.variant, "round": args.round,
                       "exit_code": exit_code, "ended_at": time.time(),
                       "loadavg": _loadavg(), "efsmi_used_mib": _efsmi_used_mib(0)})
    return exit_code


def _extract(workspace: Path, ref: str, dest: Path) -> str:
    sha = subprocess.run(["git", "-C", str(workspace), "rev-parse", ref],
                         capture_output=True, text=True, check=True).stdout.strip()
    archive = subprocess.run(["git", "-C", str(workspace), "archive", sha],
                             capture_output=True, check=True).stdout
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
    return sha


def command_drive(args) -> int:
    workspace = Path(args.workspace).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        out.unlink()
    root = Path(tempfile.mkdtemp(prefix="gdn2-004-ab-"))
    code = 0
    script = Path(__file__).resolve()
    try:
        paths = {"base": root / "base", "cand": root / "cand"}
        trees = {
            "base": _extract(workspace, args.base_ref, paths["base"]),
            "cand": _extract(workspace, args.cand_ref, paths["cand"]),
        }
        if trees["base"] == trees["cand"]:
            _jsonl_write(out, {"kind": "error", "task": TASK,
                               "error": f"base and cand resolve to the same commit {trees['base']}"})
            return 2
        _jsonl_write(out, {
            "schema": 1, "kind": "driver", "task": TASK,
            "workspace": str(workspace), "base_ref": args.base_ref,
            "cand_ref": args.cand_ref, "commits": trees,
            "rounds": args.rounds, "dtypes": args.dtypes,
            "cpu_list": CPU_LIST, "device": DEVICE, "pci_expected": EXPECTED_PCI,
            "hostname": platform.node(), "started_at": time.time(),
            "loadavg": _loadavg(),
            "shape_sets": {"bfloat16": BF16_SHAPES, "float16": FP16_SHAPES},
        })
        for round_index in range(1, args.rounds + 1):
            order = ("base", "cand") if round_index % 2 else ("cand", "base")
            for variant in order:
                for dtype_name in args.dtypes:
                    shapes = list(args.shape or [
                        ",".join(str(x) for x in item)
                        for item in (BF16_SHAPES if dtype_name == "bfloat16" else FP16_SHAPES)
                    ])
                    for index, shape in enumerate(shapes):
                        # One process and one fresh Triton cache per shape, so
                        # cold_compile_s always means a real first compile.
                        cache = root / f"triton-{round_index}-{variant}-{dtype_name}-{index}"
                        cache.mkdir()
                        env = os.environ.copy()
                        env["PYTHONPATH"] = os.pathsep.join(
                            [str(paths[variant] / "src"), str(paths[variant]),
                             env.get("PYTHONPATH", "")]
                        ).rstrip(os.pathsep)
                        env["TOPS_VISIBLE_DEVICES"] = "0"
                        env["TRITON_CACHE_DIR"] = str(cache)
                        env.pop("CUDA_VISIBLE_DEVICES", None)
                        # The baseline tree predates the knob, so setting it is
                        # harmless there and pins the baseline semantics anyway.
                        env["FLAG_ATTN_GDN2_STATE_BV_SPLIT"] = "0" if variant == "base" else "1"
                        per_run = root / f"records-{round_index}-{variant}-{dtype_name}-{index}.jsonl"
                        argv = [
                            "taskset", "-c", CPU_LIST, sys.executable, str(script), "run",
                            "--tree", str(paths[variant]), "--variant", variant,
                            "--commit", trees[variant], "--round", str(round_index),
                            "--dtype", dtype_name, "--shape", shape,
                            "--out", str(per_run),
                        ]
                        started = time.time()
                        proc = subprocess.run(argv, env=env, capture_output=True, text=True)
                        log = root / f"log-{round_index}-{variant}-{dtype_name}-{index}.txt"
                        log.write_text(
                            f"$ {' '.join(argv)}\n\n[stdout]\n{proc.stdout}\n[stderr]\n{proc.stderr}",
                            encoding="utf-8",
                        )
                        if proc.returncode != 0:
                            code = 1
                        if per_run.exists():
                            with out.open("a", encoding="utf-8") as sink, per_run.open(encoding="utf-8") as src:
                                for line in src:
                                    record = json.loads(line)
                                    record["round"] = round_index
                                    record["variant"] = variant
                                    record["commit"] = trees[variant]
                                    record["child_exit_code"] = proc.returncode
                                    record["child_log"] = str(log)
                                    sink.write(json.dumps(record, sort_keys=True) + "\n")
                        else:
                            code = 1
                        _jsonl_write(out, {
                            "schema": 1, "kind": "invocation", "task": TASK,
                            "round": round_index, "variant": variant, "dtype": dtype_name,
                            "shape": shape, "child_exit_code": proc.returncode, "argv": argv,
                            "env": {k: env.get(k) for k in
                                    ("PYTHONPATH", "TOPS_VISIBLE_DEVICES", "TRITON_CACHE_DIR",
                                     "FLAG_ATTN_GDN2_STATE_BV_SPLIT")},
                            "duration_s": round(time.time() - started, 3),
                            "loadavg": _loadavg(),
                            "efsmi_used_mib": _efsmi_used_mib(0),
                        })
        _jsonl_write(out, {"schema": 1, "kind": "driver_end", "task": TASK,
                           "exit_code": code, "ended_at": time.time(),
                           "loadavg": _loadavg()})
    finally:
        shutil.rmtree(root, ignore_errors=True)
    if code == 0:
        code = command_check(argparse.Namespace(out=str(out), rounds=args.rounds))
    return code


def command_check(args) -> int:
    """Offline validation: completeness, bit-identity, gates and invariants."""
    out = Path(args.out)
    records = _read_jsonl(out)
    if not records:
        print(f"FAIL: {out} is empty")
        return 1
    problems = []
    shapes = {}
    errors = [r for r in records if r.get("kind") == "error"]
    if errors:
        problems.append(f"{len(errors)} error record(s): "
                        f"{errors[0].get('error')}")
    child_fail = [r for r in records
                  if r.get("kind") == "invocation" and r.get("child_exit_code") != 0]
    if child_fail:
        problems.append(f"{len(child_fail)} non-zero child exit(s)")
    driver = [r for r in records if r.get("kind") == "driver"]
    if not driver:
        problems.append("missing driver record")
        rounds = None
    else:
        rounds = driver[0]["rounds"]
    for rec in records:
        if rec.get("kind") != "shape":
            continue
        key = (rec["dtype"], tuple(rec["shape"]))
        shapes.setdefault(key, {}).setdefault(rec["variant"], {})[rec["round"]] = rec

    if rounds is not None:
        expected_rounds = set(range(1, rounds + 1))
        for key, variants in sorted(shapes.items()):
            for variant in ("base", "cand"):
                got = set(variants.get(variant, {}))
                if got != expected_rounds:
                    problems.append(f"{key} {variant}: rounds {sorted(got)} != {sorted(expected_rounds)}")

    def _launch_signature(rec):
        """Kernel name + tile config for each distinct launch in a record."""
        seen = set()
        for item in rec.get("steady_launches") or []:
            seen.add((item.get("kernel"), tuple(sorted(item.get("cfg", {}).items()))))
        return tuple(sorted(seen, key=repr))

    for key, variants in sorted(shapes.items()):
        base = variants.get("base", {})
        cand = variants.get("cand", {})
        for rnd in sorted(set(base) & set(cand)):
            b, c = base[rnd], cand[rnd]
            eb = (b["exact"]["o_sha256"], b["exact"]["ht_sha256"])
            ec = (c["exact"]["o_sha256"], c["exact"]["ht_sha256"])
            if eb != ec:
                problems.append(f"{key} round {rnd}: not exact-identical {eb} != {ec}")
            for label in ("o", "ht"):
                bm, cm = b["metrics"][label], c["metrics"][label]
                if not (bm["finite"] and cm["finite"]):
                    problems.append(f"{key} round {rnd}: {label} not finite")
                if bm["gate_a"] != cm["gate_a"] or bm["gate_b"] != cm["gate_b"]:
                    problems.append(f"{key} round {rnd}: {label} gate verdict changed")
                if not cm["gate_b"]:
                    problems.append(f"{key} round {rnd}: {label} fails Gate B")
            if _launch_signature(b) != _launch_signature(c):
                problems.append(f"{key} round {rnd}: kernel path differs "
                                f"{_launch_signature(b)} != {_launch_signature(c)}")
            # Controls: where the rule does not fire, the tile and time must not move.
            if c["tiles"]["state_bv"] == b["tiles"]["state_bv"]:
                ratio = b["steady"]["median_ms"] / c["steady"]["median_ms"]
                if not (0.85 <= ratio <= 1.18):
                    problems.append(f"{key} round {rnd}: neutral control moved {ratio:.3f}x")

    if problems:
        for item in problems:
            print("FAIL:", item)
        print(f"FAIL: {len(problems)} problem(s) in {out}")
        return 1

    # Summary table (informational; the assertions above are the gate).
    print(f"OK: {len(shapes)} (dtype, shape) groups validated in {out}")
    for key in sorted(shapes):
        base = shapes[key].get("base", {})
        cand = shapes[key].get("cand", {})
        bs = [base[r]["steady"]["median_ms"] for r in sorted(base)]
        cs = [cand[r]["steady"]["median_ms"] for r in sorted(cand)]
        bmedian = statistics.median(bs) if bs else float("nan")
        cmedian = statistics.median(cs) if cs else float("nan")
        ratios = [b / c for b, c in zip(bs, cs)] if cs else []
        bcold = [base[r]["cold_compile_s"] for r in sorted(base)]
        ccold = [cand[r]["cold_compile_s"] for r in sorted(cand)]
        print(f"  {key[0]:9s} {str(key[1]):26s} base={bmedian:.4f}ms cand={cmedian:.4f}ms "
              f"speedup={bmedian / cmedian:.3f}x rounds={[round(x, 3) for x in ratios]} "
              f"cold={statistics.median(bcold):.2f}/{statistics.median(ccold):.2f}s "
              f"bv={sorted({base[r]['tiles']['state_bv'] for r in base})}->"
              f"{sorted({cand[r]['tiles']['state_bv'] for r in cand})} "
              f"peak={max(base[r]['peak_alloc_mib'] for r in base):.2f}/"
              f"{max(cand[r]['peak_alloc_mib'] for r in cand):.2f}MiB")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="measure one variant (single process)")
    run.add_argument("--tree", required=True)
    run.add_argument("--variant", choices=["base", "cand"], required=True)
    run.add_argument("--commit", default="unknown")
    run.add_argument("--round", type=int, default=0)
    run.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    run.add_argument("--shape", action="append", help="B,T,H,K,V; repeatable")
    run.add_argument("--out", required=True)
    run.set_defaults(func=command_run)

    drive = sub.add_parser("drive", help="extract both commits and run alternating rounds")
    drive.add_argument("--workspace", default=".")
    drive.add_argument("--base-ref", default=BASE_REF)
    drive.add_argument("--cand-ref", default="HEAD")
    drive.add_argument("--rounds", type=int, default=3)
    drive.add_argument("--dtypes", nargs="+", default=["bfloat16", "float16"],
                       choices=["bfloat16", "float16"])
    drive.add_argument("--shape", action="append", help="override B,T,H,K,V; repeatable")
    drive.add_argument("--out", required=True)
    drive.set_defaults(func=command_drive)

    check = sub.add_parser("check", help="validate an existing JSONL offline")
    check.add_argument("--out", required=True)
    check.add_argument("--rounds", type=int, default=None)
    check.set_defaults(func=command_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
