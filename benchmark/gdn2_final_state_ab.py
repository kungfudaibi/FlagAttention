#!/usr/bin/env python3
"""PROFILEGDN2-0001: full-step A/B of the opt-in K2 final-state allocation.

Baseline is this round's fixed commit ``1320878ddada3b3d29738d1e5f5c0bcc9385def7``.
The candidate is the same tree plus one explicit opt-in experiment: when
``FLAG_ATTN_GDN2_STATE_EMPTY_HT=1``, K2 allocates the returned ``final_state``
with ``new_empty`` instead of ``new_zeros``.  The kernel writes every element
of that tensor (grid covers all V tiles, the K-block chain covers K <= 4*BK),
which the stage profile proves bit-exactly, so the only difference is that the
0.25-0.56 ms device memset is skipped.  Default behaviour is untouched: with
the variable unset the operator is byte-for-byte the baseline.

Three sub-commands, same shape as the GDN2-004 harness:

    drive   extract both trees with ``git archive``, run the alternating rounds
            -- exactly the shipped tree (baseline semantics) versus the same
            tree with ``FLAG_ATTN_GDN2_STATE_EMPTY_HT=1``;
    run     measure one variant in a fresh process (used by ``drive``);
    check   re-validate an existing JSONL offline and reprint the summary.

Each (round, variant, dtype, shape) is its own process with a private
``TRITON_CACHE_DIR``, so ``cold_compile_s`` is a real first compile.  Every
record carries the launches that actually ran (kernel, grid, BK/BV/BT), the
peak device allocation, all raw samples, finiteness, the CPU FP64 reference
error and the exact bit digests of ``o`` and ``ht``.

Usage (inside the task clone):

    TOPS_VISIBLE_DEVICES=5 taskset -c 160-191 /flagos/bin/python \
        benchmark/gdn2_final_state_ab.py drive --rounds 3 \
        --out agent-results/PROFILEGDN2-0001-ab.jsonl

The baseline commit is the default ``--base-ref``; the candidate is the
working tree (``--cand-ref HEAD``), which must contain the opt-in switch.
``drive`` exits non-zero unless every shape is present in every round, the
candidate is bit-identical to the baseline, and both stay inside the numeric
gates.  ``check`` re-derives all of that from the file alone.
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

TASK = "PROFILEGDN2-0001"
BASE_REF = "1320878ddada3b3d29738d1e5f5c0bcc9385def7"
DEVICE = "gcu:0"
HOST_DEVICE_INDEX = 5
EXPECTED_PCI = "83:00.0"
CPU_LIST = "160-191"
SEED = 42
CHUNK_SIZE = 64
STEADY_SAMPLES = 15
WARMUP_ITERS = 5
CLOCK_WARMUP_S = 2.0
GATE_ATOL = 1e-3
GATE_RTOL = 2e-2
ENV_SWITCH = "FLAG_ATTN_GDN2_STATE_EMPTY_HT"

# V brackets the 128 boundary: 96/128 never split, 160/192/256 split, 320 does
# not.  K=96 and K != V keep the irregular dimension in scope, 256/256 is the
# wide-K control, and the small (8,64,64) case is the cheap control.
BF16_SHAPES = [
    (2, 512, 4, 128, 96),
    (2, 512, 4, 128, 128),
    (2, 512, 4, 128, 160),
    (2, 512, 4, 128, 192),
    (2, 512, 4, 128, 256),
    (2, 512, 4, 128, 320),
    (1, 1024, 4, 256, 256),
    (2, 512, 4, 96, 128),
    (2, 512, 8, 64, 64),
]
FP16_SHAPES = [
    (2, 512, 4, 128, 128),
    (2, 512, 4, 128, 160),
    (2, 512, 4, 128, 256),
    (1, 1024, 4, 256, 256),
]


def _jsonl_write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _loadavg() -> str:
    try:
        return Path("/proc/loadavg").read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"


def _efsmi_pci(index: int = HOST_DEVICE_INDEX):
    try:
        out = subprocess.run(["efsmi", "-q", "-d", "PCIE", "--json-format"],
                             capture_output=True, text=True, timeout=30).stdout
        data = json.loads(out)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    for item in data.get("devices", []):
        if str(item.get("id")) == str(index):
            p = item.get("PCIe Info", {})
            return f"{p.get('Bus')}:{p.get('Dev')}.{p.get('Func')}"
    return None


def _efsmi_used_mib(index: int = HOST_DEVICE_INDEX):
    try:
        out = subprocess.run(["efsmi", "-q", "-i", str(index)],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"Used Size\s*:\s*(\d+)\s*MiB", out)
    return float(match.group(1)) if match else None


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
    return {"max_abs": diff.max().item(), "rms_rel": rms / (ref_rms + 1e-12),
            "ref_rms": ref_rms, "ref_max": e.abs().max().item()}


def _gates(metrics):
    strict = (metrics["max_abs"] <= GATE_ATOL + GATE_RTOL * metrics["ref_rms"]
              and metrics["rms_rel"] <= GATE_RTOL)
    magnitude = (metrics["max_abs"] <= GATE_ATOL + GATE_RTOL * metrics["ref_max"]
                 and metrics["rms_rel"] <= GATE_RTOL)
    return {"gate_a": bool(strict), "gate_b": bool(magnitude)}


def _install_kernel_tracer():
    from triton.runtime.jit import JITFunction

    records = []
    original = JITFunction.run

    def traced(self, *args, **kwargs):
        grid = kwargs.get("grid")
        if callable(grid):
            grid_repr = "callable"
        else:
            try:
                grid_repr = [int(x) for x in grid]
            except TypeError:
                grid_repr = repr(grid)
        cfg = {}
        for key in ("BK", "BV", "BT", "GRID_MODE", "STATE_V_FIRST",
                    "num_warps", "num_stages"):
            if key in kwargs:
                value = kwargs[key]
                cfg[key] = value if isinstance(value, (int, float, bool, str)) else repr(value)
        records.append({"kernel": getattr(self.fn, "__name__", str(self.fn)),
                        "grid": grid_repr, "cfg": cfg})
        return original(self, *args, **kwargs)

    JITFunction.run = traced
    return records


def _clock_warmup(call, seconds=CLOCK_WARMUP_S):
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
    return {"samples_ms": [round(x, 6) for x in values],
            "median_ms": round(statistics.median(values), 6),
            "mean_ms": round(statistics.fmean(values), 6),
            "min_ms": round(ordered[0], 6),
            "p25_ms": round(ordered[len(ordered) // 4], 6),
            "max_ms": round(ordered[-1], 6)}


def _digest(tensor) -> str:
    import hashlib

    import torch

    raw = tensor.detach().to("cpu").contiguous()
    if raw.dtype == torch.bfloat16:
        raw = raw.view(torch.int16)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def _ptx_digest():
    """Fingerprint of the Triton kernel cache: proves which code was compiled."""
    import os
    import hashlib

    root = os.environ.get("TRITON_CACHE_DIR")
    if not root or not os.path.isdir(root):
        return None
    hasher = hashlib.sha256()
    count = 0
    for base, _dirs, files in os.walk(root):
        for name in sorted(files):
            path = os.path.join(base, name)
            hasher.update(os.path.relpath(path, root).encode())
            with open(path, "rb") as handle:
                hasher.update(handle.read())
            count += 1
    return {"files": count, "sha256": hasher.hexdigest()}


def command_run(args) -> int:
    import torch

    tree = Path(args.tree).resolve()
    sys.path.insert(0, str(tree / "src"))
    sys.path.insert(0, str(tree / "tests" / "flag_attn"))

    from gdn2_backward_ref import gdn2_forward  # noqa: E402
    from flag_attn import chunk_gdn2  # noqa: E402
    impl = __import__("flag_attn.runtime.backend._enflame.FLA.gdn2.chunk_gdn2",
                      fromlist=["x"])

    out = Path(args.out)
    dtype_name = args.dtype
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    shapes = BF16_SHAPES if dtype_name == "bfloat16" else FP16_SHAPES
    if args.shape:
        shapes = [tuple(int(x) for x in item.split(",")) for item in args.shape]

    pci = _efsmi_pci()
    meta = {
        "schema": 1, "task": TASK, "kind": "meta",
        "variant": args.variant, "round": args.round,
        "tree": str(tree), "commit": args.commit, "dtype": dtype_name,
        "hostname": platform.node(), "torch": torch.__version__,
        "device_count_visible": torch.gcu.device_count(),
        "current_device": torch.gcu.current_device(),
        "pci_gcu5": pci, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "triton_cache": os.environ.get("TRITON_CACHE_DIR"),
        "ptx": _ptx_digest(),
        "tile_adapt_default": getattr(impl, "_S60_TILES_ADAPT", None),
        "bv_split_default": getattr(impl, "_S60_STATE_BV_SPLIT_ENABLED", None),
        "empty_ht_default": getattr(impl, "_S60_STATE_EMPTY_HT_ENABLED", "absent"),
        "env_switch": {ENV_SWITCH: os.environ.get(ENV_SWITCH)},
        "loadavg": _loadavg(), "efsmi_used_mib": _efsmi_used_mib(),
        "started_at": time.time(),
    }
    _jsonl_write(out, meta)
    if pci is None or pci.strip().lower() != EXPECTED_PCI:
        _jsonl_write(out, {"kind": "error", "task": TASK, "variant": args.variant,
                           "round": args.round,
                           "error": f"unexpected GCU5 PCI {pci}"})
        return 2
    if torch.gcu.device_count() != 1:
        _jsonl_write(out, {"kind": "error", "task": TASK, "variant": args.variant,
                           "round": args.round,
                           "error": f"TOPS_VISIBLE_DEVICES did not pin one device: "
                                    f"{torch.gcu.device_count()}"})
        return 2
    if args.variant == "cand":
        if not hasattr(impl, "_S60_STATE_EMPTY_HT_ENABLED"):
            _jsonl_write(out, {"kind": "error", "task": TASK, "variant": args.variant,
                               "round": args.round,
                               "error": "candidate tree has no empty-ht switch"})
            return 2
        if not impl._S60_STATE_EMPTY_HT_ENABLED:
            _jsonl_write(out, {"kind": "error", "task": TASK, "variant": args.variant,
                               "round": args.round,
                               "error": f"{ENV_SWITCH} did not enable the experiment"})
            return 2
    else:
        if getattr(impl, "_S60_STATE_EMPTY_HT_ENABLED", False):
            _jsonl_write(out, {"kind": "error", "task": TASK, "variant": args.variant,
                               "round": args.round,
                               "error": "baseline must not enable the experiment"})
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
                    return chunk_gdn2(*inputs[:6], scale=scale, initial_state=inputs[6],
                                      output_final_state=True, chunk_size=CHUNK_SIZE)

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
            peak_mib = (torch.gcu.max_memory_allocated() - base_alloc) / 2 ** 20

            clock_iters = _clock_warmup(call)

            tracer.clear()
            steady = _time_call(call)
            steady_launches = [dict(x) for x in tracer]

            metrics = {}
            for label, ref, act in (("o", ref_o, o), ("ht", ref_ht, ht)):
                item = _metrics(ref, act)
                item.update(_gates(item))
                item["finite"] = bool(torch.isfinite(act).all().item())
                metrics[label] = item

            bv = None
            if hasattr(impl, "_s60_state_bv"):
                bv = impl._s60_state_bv(V)
            _jsonl_write(out, {
                "schema": 1, "kind": "shape", "task": TASK,
                "variant": args.variant, "round": args.round,
                "commit": args.commit, "dtype": dtype_name,
                "shape": [B, T, H, K, V],
                "tiles": {"state_bv": bv, "n_v_tiles": (-(-V // bv) if bv else None)},
                "cold_compile_s": round(cold_s, 6),
                "cold_launches": cold_launches,
                "steady_launches": steady_launches,
                "steady": steady,
                "clock_warmup_iters": clock_iters,
                "clock_warmup_s": CLOCK_WARMUP_S,
                "peak_alloc_mib": round(peak_mib, 6),
                "efsmi_used_mib": _efsmi_used_mib(),
                "metrics": metrics,
                "exact": {"o_sha256": _digest(o), "ht_sha256": _digest(ht)},
                "loadavg": _loadavg(),
                "wall_clock_s": round(time.perf_counter() - wall_start, 6),
            })
            del inputs, cpu_inputs, o, ht
        except Exception as error:  # noqa: BLE001 - one bad shape must still be recorded
            import traceback
            traceback.print_exc()
            exit_code = 1
            _jsonl_write(out, {
                "schema": 1, "kind": "error", "task": TASK,
                "variant": args.variant, "round": args.round,
                "commit": args.commit, "dtype": dtype_name, "shape": list(shape),
                "error": f"{type(error).__name__}: {error}"})
    _jsonl_write(out, {"schema": 1, "kind": "meta_end", "task": TASK,
                       "variant": args.variant, "round": args.round,
                       "exit_code": exit_code, "ended_at": time.time(),
                       "loadavg": _loadavg(), "efsmi_used_mib": _efsmi_used_mib()})
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
    root = Path(tempfile.mkdtemp(prefix="profilegdn2-0001-ab-"))
    code = 0
    script = Path(__file__).resolve()
    try:
        paths = {"base": root / "base", "cand": root / "cand"}
        trees = {"base": _extract(workspace, args.base_ref, paths["base"]),
                 "cand": _extract(workspace, args.cand_ref, paths["cand"])}
        if args.cand_ref == args.base_ref:
            _jsonl_write(out, {"kind": "error", "task": TASK,
                               "error": "base and cand are the same ref"})
            return 2
        if not (paths["cand"] / "src/flag_attn/runtime/backend/_enflame/FLA/gdn2/chunk_gdn2.py").read_text(
                encoding="utf-8", errors="replace").find(ENV_SWITCH) >= 0:
            _jsonl_write(out, {"kind": "error", "task": TASK,
                               "error": f"candidate tree does not contain {ENV_SWITCH}"})
            return 2
        _jsonl_write(out, {
            "schema": 1, "kind": "driver", "task": TASK,
            "workspace": str(workspace), "base_ref": args.base_ref,
            "cand_ref": args.cand_ref, "commits": trees,
            "rounds": args.rounds, "dtypes": args.dtypes,
            "cpu_list": CPU_LIST, "device": DEVICE,
            "pci_expected": EXPECTED_PCI, "host_device_index": HOST_DEVICE_INDEX,
            "env_switch": ENV_SWITCH,
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
                        cache = root / f"triton-{round_index}-{variant}-{dtype_name}-{index}"
                        cache.mkdir()
                        env = os.environ.copy()
                        env["PYTHONPATH"] = os.pathsep.join(
                            [str(paths[variant] / "src"), str(paths[variant]),
                             env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
                        env["TOPS_VISIBLE_DEVICES"] = str(HOST_DEVICE_INDEX)
                        env["TRITON_CACHE_DIR"] = str(cache)
                        env.pop("CUDA_VISIBLE_DEVICES", None)
                        if variant == "cand":
                            env[ENV_SWITCH] = "1"
                        else:
                            env.pop(ENV_SWITCH, None)
                        per_run = root / f"records-{round_index}-{variant}-{dtype_name}-{index}.jsonl"
                        argv = ["taskset", "-c", CPU_LIST, sys.executable, str(script), "run",
                                "--tree", str(paths[variant]), "--variant", variant,
                                "--commit", trees[variant], "--round", str(round_index),
                                "--dtype", dtype_name, "--shape", shape, "--out", str(per_run)]
                        started = time.time()
                        proc = subprocess.run(argv, env=env, capture_output=True, text=True)
                        log = root / f"log-{round_index}-{variant}-{dtype_name}-{index}.txt"
                        log.write_text(f"$ {' '.join(argv)}\n\n[stdout]\n{proc.stdout}\n"
                                       f"[stderr]\n{proc.stderr}", encoding="utf-8")
                        if proc.returncode != 0:
                            code = 1
                        if per_run.exists():
                            with out.open("a", encoding="utf-8") as sink, \
                                    per_run.open(encoding="utf-8") as src:
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
                                     ENV_SWITCH)},
                            "duration_s": round(time.time() - started, 3),
                            "loadavg": _loadavg(), "efsmi_used_mib": _efsmi_used_mib()})
        _jsonl_write(out, {"schema": 1, "kind": "driver_end", "task": TASK,
                           "exit_code": code, "ended_at": time.time(),
                           "loadavg": _loadavg()})
    finally:
        shutil.rmtree(root, ignore_errors=True)
    if code == 0:
        code = command_check(argparse.Namespace(out=str(out), rounds=args.rounds))
    return code


def _kernel_path(rec):
    names = []
    for item in rec.get("steady_launches") or []:
        if item.get("kernel") not in names:
            names.append(item.get("kernel"))
    return tuple(names)


def _launch_geometry(rec):
    """(kernel, grid, BK, BV, BT) of every distinct launch, order preserved."""
    out = []
    for item in rec.get("steady_launches") or []:
        cfg = item.get("cfg", {})
        entry = (item.get("kernel"), str(item.get("grid")),
                 cfg.get("BK"), cfg.get("BV"), cfg.get("BT"))
        if entry not in out:
            out.append(entry)
    return tuple(out)


def _k2_bv(rec):
    return tuple(sorted({item.get("cfg", {}).get("BV")
                         for item in (rec.get("steady_launches") or [])
                         if item.get("kernel") == "_chunk_gdn2_fwd_h_o_infer_kernel"}))


def command_check(args) -> int:
    out = Path(args.out)
    records = _read_jsonl(out)
    if not records:
        print(f"FAIL: {out} is empty")
        return 1
    problems, warnings = [], []
    shapes = {}
    errors = [r for r in records if r.get("kind") == "error"]
    if errors:
        problems.append(f"{len(errors)} error record(s): {errors[0].get('error')}")
    if [r for r in records if r.get("kind") == "invocation"
            and r.get("child_exit_code") != 0]:
        problems.append("non-zero child exit(s)")
    driver = [r for r in records if r.get("kind") == "driver"]
    rounds = driver[0]["rounds"] if driver else None
    if not driver:
        problems.append("missing driver record")
    for rec in records:
        if rec.get("kind") != "shape":
            continue
        key = (rec["dtype"], tuple(rec["shape"]))
        shapes.setdefault(key, {}).setdefault(rec["variant"], {})[rec["round"]] = rec

    if rounds is not None:
        for key, variants in sorted(shapes.items()):
            for variant in ("base", "cand"):
                if set(variants.get(variant, {})) != set(range(1, rounds + 1)):
                    problems.append(f"{key} {variant}: rounds "
                                    f"{sorted(variants.get(variant, {}))} "
                                    f"!= {list(range(1, rounds + 1))}")

    for key, variants in sorted(shapes.items()):
        base, cand = variants.get("base", {}), variants.get("cand", {})
        for rnd in sorted(set(base) & set(cand)):
            b, c = base[rnd], cand[rnd]
            if (b["exact"]["o_sha256"], b["exact"]["ht_sha256"]) != \
                    (c["exact"]["o_sha256"], c["exact"]["ht_sha256"]):
                problems.append(f"{key} round {rnd}: not exact-identical")
            for label in ("o", "ht"):
                bm, cm = b["metrics"][label], c["metrics"][label]
                if not (bm["finite"] and cm["finite"]):
                    problems.append(f"{key} r{rnd}: {label} not finite")
                if bm["gate_a"] != cm["gate_a"] or bm["gate_b"] != cm["gate_b"]:
                    problems.append(f"{key} r{rnd}: {label} gate verdict changed")
                if not cm["gate_b"]:
                    problems.append(f"{key} r{rnd}: {label} fails Gate B")
            if _kernel_path(b) != _kernel_path(c):
                problems.append(f"{key} r{rnd}: kernel path differs")
            if _k2_bv(b) != _k2_bv(c):
                problems.append(f"{key} r{rnd}: K2 BV changed {_k2_bv(b)} -> {_k2_bv(c)}"
                                " (this experiment must not touch the tile)")
            if _launch_geometry(b) != _launch_geometry(c):
                problems.append(f"{key} r{rnd}: launch geometry changed "
                                f"{_launch_geometry(b)} -> {_launch_geometry(c)}")
            # The candidate only removes a device memset, so the expected
            # ratio is bounded by (memset + kernel) / kernel.  On this host the
            # memset is ~0.2-0.5 ms against a ~0.14-0.45 ms K2 body, so a ~2x
            # whole-step speedup is the *expected* result, not an anomaly; the
            # band only has to catch a broken measurement (wrong variant, a
            # stale tree, a mis-bound device), which would sit near 1.0 or
            # below.  The report compares the measured ratio with the stage
            # profile's own prediction.
            ratio = b["steady"]["median_ms"] / c["steady"]["median_ms"]
            if not (0.85 <= ratio <= 3.5):
                problems.append(f"{key} r{rnd}: implausible ratio {ratio:.3f}x")
            if ratio < 1.05:
                warnings.append(f"{key} r{rnd}: candidate only {ratio:.3f}x "
                                f"(expected >= 1.05x from the removed memset)")

    for item in warnings:
        print("WARN:", item)
    if problems:
        for item in problems:
            print("FAIL:", item)
        print(f"FAIL: {len(problems)} problem(s) in {out}")
        return 1

    print(f"OK: {len(shapes)} (dtype, shape) groups validated in {out}")
    all_ratios = []
    for key in sorted(shapes):
        base, cand = shapes[key]["base"], shapes[key]["cand"]
        bs = [base[r]["steady"]["median_ms"] for r in sorted(base)]
        cs = [cand[r]["steady"]["median_ms"] for r in sorted(cand)]
        bmed, cmed = statistics.median(bs), statistics.median(cs)
        ratios = [b / c for b, c in zip(bs, cs)]
        all_ratios.extend(ratios)
        print(f"  {key[0]:9s} {str(key[1]):26s} base={bmed:.4f}ms cand={cmed:.4f}ms "
              f"speedup={bmed / cmed:.3f}x rounds={[round(x, 3) for x in ratios]} "
              f"bv={sorted({base[r]['tiles']['state_bv'] for r in base})} "
              f"peak={max(base[r]['peak_alloc_mib'] for r in base):.2f}/"
              f"{max(cand[r]['peak_alloc_mib'] for r in cand):.2f}MiB")
    if all_ratios:
        print(f"  paired rounds: n={len(all_ratios)} median={statistics.median(all_ratios):.4f}x "
              f"geomean={math.exp(statistics.fmean([math.log(x) for x in all_ratios])):.4f}x "
              f"min={min(all_ratios):.4f}x max={max(all_ratios):.4f}x")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="measure one variant (single process)")
    run.add_argument("--tree", required=True)
    run.add_argument("--variant", choices=["base", "cand"], required=True)
    run.add_argument("--commit", default="unknown")
    run.add_argument("--round", type=int, default=0)
    run.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    run.add_argument("--shape", action="append", help="B,T,H,K,V; repeatable")
    run.add_argument("--out", required=True)
    run.set_defaults(func=command_run)

    drive = sub.add_parser("drive", help="extract both trees and run alternating rounds")
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

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["drive", "--out", "agent-results/PROFILEGDN2-0001-ab.jsonl"]
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.error("a subcommand is required")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
