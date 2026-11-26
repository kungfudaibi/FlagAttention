#!/usr/bin/env python3
"""PROFILEGDN2-0001: stage attribution for the fused Enflame S60 GDN-2 forward.

A single ``chunk_gdn2`` inference call launches exactly two Triton kernels:

* K1 ``_chunk_gdn2_fwd_intra_infer_kernel`` -- intra-chunk WY/score prep,
  grid ``(NT, B*H)``, parallel over every chunk;
* K2 ``_chunk_gdn2_fwd_h_o_infer_kernel``  -- sequential state walk + output,
  grid ``(n_v_tiles, B*H)``, parallel only over the V tile and batch*head.

The task hypothesis is that K1/K2 time attribution and, inside K2, the V-tile
parallelism must be separated from the state read/write traffic.  This script
does that with a 2x2x2 ablation on identical inputs:

    split   in {off, on}    -- K2 V tile 256 vs the tuned rule (128 for
                               128 < V <= 256), i.e. the parallelism axis;
    ht      in {zeros, empty} -- ``output_final_state`` allocates the returned
                               state with ``new_zeros`` (default) or, under the
                               opt-in experiment, ``new_empty``; the kernel
                               stores every element either way, so this is
                               purely the device memset of the state traffic;
    h0      in {off, on}    -- whether the initial state is read.

Every cell also has the K2 sub-variants (h0 only, ht only, neither) so the
state read and the state write can be priced separately from the loop itself,
plus a bare ``new_zeros``/``new_empty`` of the exact state shape so the memset
is measured with no kernel attached.

Kernel dispatch (names, grid, BK/BV/BT) is recorded from the ``JITFunction.run``
hook for every measured cell, so a timing is never reported without the launch
that produced it.

Usage (inside the task clone, physical GCU 5 == PCI 83:00.0):

    TOPS_VISIBLE_DEVICES=5 taskset -c 160-191 /flagos/bin/python \
        benchmark/gdn2_stage_profile.py --out agent-results/PROFILEGDN2-0001-stage.jsonl

``--dtype bfloat16|float16|both`` selects the input dtype; ``--v-list``
overrides the V sweep.  The process refuses to run unless it is bound to the
expected PCI address, and exits non-zero if any cell is not finite or a
dispatch is not the expected two-kernel path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

TASK = "PROFILEGDN2-0001"
EXPECTED_PCI = "83:00.0"
DEVICE = "gcu:0"          # ordinal inside a single-visible-device process
HOST_DEVICE_INDEX = 5     # physical GCU 5 on this host
CHUNK_SIZE = 64
SEED = 42
WARMUP_S = 1.0
SAMPLES = 11
BATCH = 8

# B, T, H, K are held fixed so V is the only moving axis; the sweep brackets
# the 128 boundary where the tuned K2 rule changes.
BASE_SHAPE = (2, 512, 4, 128)
V_LIST = [96, 128, 160, 192, 256, 320]
CONTROL_SHAPES = [(1, 1024, 4, 256, 256)]


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
        import re
        m = re.search(r"Used Size\s*:\s*(\d+)\s*MiB", out)
        return float(m.group(1)) if m else None
    except (OSError, subprocess.SubprocessError):
        return None


def _loadavg():
    try:
        return Path("/proc/loadavg").read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"


def _build(B, T, H, K, V, dtype, device):
    import torch

    torch.manual_seed(SEED)
    q = (torch.randn(B, T, H, K) / math.sqrt(K)).to(dtype)
    k = (torch.randn(B, T, H, K) / math.sqrt(K)).to(dtype)
    v = torch.randn(B, T, H, V).to(dtype)
    g = (-torch.rand(B, T, H, K) * 0.1).to(dtype)
    b = torch.rand(B, T, H, K).to(dtype)
    w = torch.rand(B, T, H, V).to(dtype)
    h0 = torch.randn(B, H, K, V) * 0.01
    return [x.to(device) for x in (q, k, v, g, b, w, h0)]


def _tracer():
    """Record (kernel, grid, BK, BV, BT) for every launch."""
    from triton.runtime.jit import JITFunction

    seen = []
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
        cfg = {key: kwargs[key] for key in ("BK", "BV", "BT", "GRID_MODE",
                                            "num_warps", "num_stages")
               if key in kwargs}
        seen.append({
            "kernel": getattr(self.fn, "__name__", str(self.fn)),
            "grid": grid_repr, "cfg": cfg,
        })
        return original(self, *args, **kwargs)

    JITFunction.run = traced
    return seen


def _time(fn, samples=SAMPLES, warm_s=WARMUP_S, tracer=None, batch=1):
    """Device-event timing of ``batch`` consecutive calls, per-call normalised.

    A single-call sample is not stable on this host: a scheduling preemption
    between ``begin.record()`` and the launch lands inside the event pair and
    shows up as a ~5.8 ms sample (one scheduler slice) even for a 0.4 MiB
    operator that launches no memset at all.  Recording ``batch`` back-to-back
    calls inside one event pair and dividing by ``batch`` amortises such a
    stall instead of letting it masquerade as kernel time; the raw normalised
    samples are still reported, together with the fraction above 3x the minimum
    so a contaminated cell is visible rather than hidden by the median.

    Host wall time is recorded as well: a large gap between the device and host
    medians means host launch or allocation cost, not kernel time.
    """
    import torch

    deadline = time.perf_counter() + warm_s
    iters = 0
    while iters < max(8, 2 * batch) or time.perf_counter() < deadline:
        fn()
        iters += 1
    torch.gcu.synchronize()
    if tracer is not None:
        tracer.clear()
    dev, host = [], []
    for _ in range(samples):
        torch.gcu.synchronize()
        begin = torch.gcu.Event(enable_timing=True)
        end = torch.gcu.Event(enable_timing=True)
        t0 = time.perf_counter()
        begin.record()
        for _ in range(batch):
            fn()
        end.record()
        t1 = time.perf_counter()
        torch.gcu.synchronize()
        dev.append(begin.elapsed_time(end) * 1000.0 / batch)
        host.append((t1 - t0) * 1e6 / batch)
    launches = None
    if tracer is not None:
        launches, total = [], 0
        for item in tracer:
            total += 1
            entry = dict(item)
            if entry not in launches:
                launches.append(entry)
        launches = {"total_launches": total,
                    "distinct": launches,
                    "per_call": round(total / (samples * batch), 3)}
    order = sorted(dev)
    return {
        "batch": batch,
        "device_us": [round(x, 2) for x in dev],
        "host_us": [round(x, 2) for x in host],
        "median_us": round(statistics.median(dev), 2),
        "min_us": round(order[0], 2),
        "p25_us": round(order[len(order) // 4], 2),
        "max_us": round(order[-1], 2),
        "host_median_us": round(statistics.median(host), 2),
        "outlier_fraction": round(sum(1 for x in dev if x > 3 * order[0]) / len(dev), 3),
        "warmup_iters": iters,
        "launches": launches,
    }


def _kernel_names(cell):
    """Ordered distinct kernel names launched by one cell."""
    if not cell:
        return []
    return [item["kernel"] for item in cell["distinct"]]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "both"], default="both")
    parser.add_argument("--v-list", help="comma separated V values (default %s)" % V_LIST)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--warmup-s", type=float, default=WARMUP_S)
    parser.add_argument("--batch", type=int, default=BATCH,
                        help="calls per timed sample (host-stall amortisation)")
    args = parser.parse_args(argv)

    import torch

    pci = _efsmi_pci()
    if pci is None or pci.lower() != EXPECTED_PCI:
        print(f"FAIL: host GCU {HOST_DEVICE_INDEX} PCI is {pci!r}, "
              f"expected {EXPECTED_PCI!r}", file=sys.stderr)
        return 4
    if torch.gcu.current_device() != 0:
        print(f"FAIL: current device is {torch.gcu.current_device()}, "
              f"expected the single visible device 0", file=sys.stderr)
        return 4

    from flag_attn import chunk_gdn2
    impl = __import__("flag_attn.runtime.backend._enflame.FLA.gdn2.chunk_gdn2",
                      fromlist=["x"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    v_list = [int(x) for x in args.v_list.split(",")] if args.v_list else list(V_LIST)
    dtypes = (["bfloat16", "float16"] if args.dtype == "both" else [args.dtype])
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}

    props = {}
    try:
        p = torch.gcu.get_device_properties(0)
        props = {k: getattr(p, k) for k in dir(p)
                 if not k.startswith("_") and isinstance(getattr(p, k), (int, str))}
    except Exception as error:  # noqa: BLE001 - properties are informational
        props = {"error": f"{type(error).__name__}: {error}"}

    with out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "schema": 1, "task": TASK, "kind": "meta",
            "hostname": platform.node(), "torch": torch.__version__,
            "python": sys.version.split()[0], "device": DEVICE,
            "host_device_index": HOST_DEVICE_INDEX, "pci": pci,
            "device_count_visible": torch.gcu.device_count(),
            "current_device": torch.gcu.current_device(),
            "cpu_affinity": [min(os.sched_getaffinity(0)), max(os.sched_getaffinity(0))],
            "cpu_count": len(os.sched_getaffinity(0)),
            "device_properties": props,
            "efsmi_used_mib": _efsmi_used_mib(),
            "loadavg": _loadavg(), "samples": args.samples,
            "warmup_s": args.warmup_s, "batch": args.batch, "chunk_size": CHUNK_SIZE,
            "dtypes": dtypes, "v_list": v_list,
            "base_shape": list(BASE_SHAPE), "control_shapes": CONTROL_SHAPES,
            "tiles": {
                "intra_bk": impl._S60_INTRA_BK, "intra_bv": impl._S60_INTRA_BV,
                "state_bk": [impl._S60_STATE_BK_SMALL, impl._S60_STATE_BK_MEDIUM,
                             impl._S60_STATE_BK_LARGE],
                "state_bv": impl._S60_STATE_BV,
                "state_bv_split": impl._S60_STATE_BV_SPLIT,
                "tiles_adapt": impl._S60_TILES_ADAPT,
                "bv_split_enabled": impl._S60_STATE_BV_SPLIT_ENABLED,
                "empty_ht_enabled": getattr(impl, "_S60_STATE_EMPTY_HT_ENABLED", None),
            },
            "started_at": time.time(),
        }, sort_keys=True) + "\n")

    tracer = _tracer()
    failures = []
    cases = [(list(BASE_SHAPE) + [v]) for v in v_list] + [list(x) for x in CONTROL_SHAPES]

    for dtype_name in dtypes:
        dtype = torch_dtype[dtype_name]
        for shape in cases:
            B, T, H, K, V = shape
            scale = K ** -0.5
            args7 = _build(B, T, H, K, V, dtype, DEVICE)
            with torch.inference_mode():
                w_wy, u_wy, qg, kg, Aqk, _Akk, g_cumsum = impl.chunk_gdn2_fwd_intra_infer(
                    q=args7[0], k=args7[1], v=args7[2], g=args7[3], b=args7[4],
                    w_gate=args7[5], scale=scale, chunk_size=CHUNK_SIZE)

            def k1():
                with torch.inference_mode():
                    return impl.chunk_gdn2_fwd_intra_infer(
                        q=args7[0], k=args7[1], v=args7[2], g=args7[3], b=args7[4],
                        w_gate=args7[5], scale=scale, chunk_size=CHUNK_SIZE)

            def k2(h0, ht):
                with torch.inference_mode():
                    return impl.chunk_gdn2_fwd_h_o_infer(
                        kg=kg, w_wy=w_wy, u_wy=u_wy, gk=g_cumsum, qg=qg, Aqk=Aqk,
                        scale=scale, initial_state=h0, output_final_state=ht,
                        chunk_size=CHUNK_SIZE)

            def full(empty_ht):
                impl._S60_STATE_EMPTY_HT_ENABLED = empty_ht
                with torch.inference_mode():
                    return chunk_gdn2(*args7[:6], scale=scale, initial_state=args7[6],
                                      output_final_state=True, chunk_size=CHUNK_SIZE)

            state_shape = (B, H, K, V)
            state = torch.empty(state_shape, dtype=torch.float32, device=DEVICE)

            # Measure the allocation exactly as the operator performs it: the
            # returned tensor is dropped, so the caching allocator recycles one
            # block.  Holding tensors alive would force the allocator to grow
            # and would price allocation, not the zero fill.
            def zeros_state():
                return state.new_zeros(state_shape, dtype=torch.float32)

            def empty_state():
                return state.new_empty(state_shape, dtype=torch.float32)

            drv = {"schema": 1, "kind": "cell", "task": TASK,
                   "dtype": dtype_name, "shape": shape,
                   "state_shape": list(state_shape),
                   "state_mib": round(B * H * K * V * 4 / 2 ** 20, 4),
                   "state_bv_off": impl._s60_state_bv(V),
                   "loadavg": _loadavg(), "efsmi_used_mib": _efsmi_used_mib()}
            impl._S60_STATE_EMPTY_HT_ENABLED = False
            drv["k1_intra"] = _time(k1, args.samples, args.warmup_s, tracer, args.batch)

            # The bare memset / allocation of the exact final-state shape.
            drv["ht_new_zeros"] = _time(zeros_state, args.samples, args.warmup_s, batch=args.batch)
            drv["ht_new_empty"] = _time(empty_state, args.samples, args.warmup_s, batch=args.batch)

            drv["k2"] = {}
            for split in (False, True):
                impl._S60_STATE_BV_SPLIT_ENABLED = split
                bv = impl._s60_state_bv(V)
                key = f"split_{int(split)}_bv{bv}"
                cell = {"state_bv": bv, "n_v_tiles": -(-V // bv)}
                cell["h0_ht"] = _time(lambda: k2(args7[6], True), args.samples, args.warmup_s, tracer, args.batch)
                cell["h0_only"] = _time(lambda: k2(args7[6], False), args.samples, args.warmup_s, tracer, args.batch)
                cell["ht_only"] = _time(lambda: k2(None, True), args.samples, args.warmup_s, tracer, args.batch)
                cell["neither"] = _time(lambda: k2(None, False), args.samples, args.warmup_s, tracer, args.batch)
                for ht in (False, True):
                    impl._S60_STATE_EMPTY_HT_ENABLED = ht
                    cell[f"full_empty_ht_{int(ht)}"] = _time(
                        lambda: full(ht), args.samples, args.warmup_s, tracer, args.batch)
                impl._S60_STATE_EMPTY_HT_ENABLED = False
                drv["k2"][key] = cell
            impl._S60_STATE_BV_SPLIT_ENABLED = True

            # Validation: the two ht paths and the two V-tile paths must agree
            # bit for bit, else a timing difference is meaningless.
            import hashlib

            def digest(t):
                raw = t.detach().to("cpu").contiguous()
                if raw.dtype == torch.bfloat16:
                    raw = raw.view(torch.int16)
                return hashlib.sha256(raw.numpy().tobytes()).hexdigest()

            digests = {}
            for split in (False, True):
                impl._S60_STATE_BV_SPLIT_ENABLED = split
                for ht in (False, True):
                    impl._S60_STATE_EMPTY_HT_ENABLED = ht
                    o, h = full(ht)
                    digests[f"split{int(split)}_empty{int(ht)}"] = {
                        "o": digest(o), "ht": digest(h),
                        "finite": bool(torch.isfinite(o).all().item()
                                       and torch.isfinite(h).all().item())}
            impl._S60_STATE_BV_SPLIT_ENABLED = True
            impl._S60_STATE_EMPTY_HT_ENABLED = False
            drv["bit_identity"] = {
                "o_all_equal": len({v["o"] for v in digests.values()}) == 1,
                "ht_all_equal": len({v["ht"] for v in digests.values()}) == 1,
                "all_finite": all(v["finite"] for v in digests.values()),
                "digests": digests,
            }
            if not (drv["bit_identity"]["o_all_equal"]
                    and drv["bit_identity"]["ht_all_equal"]
                    and drv["bit_identity"]["all_finite"]):
                failures.append(f"{dtype_name} {shape}: bit identity / finite check failed")
            for label, item in drv["k2"].items():
                for name, timing in item.items():
                    if isinstance(timing, dict):
                        names = _kernel_names(timing["launches"])
                        expected = (["_chunk_gdn2_fwd_intra_infer_kernel",
                                     "_chunk_gdn2_fwd_h_o_infer_kernel"]
                                    if name.startswith("full_")
                                    else ["_chunk_gdn2_fwd_h_o_infer_kernel"])
                        if names != expected:
                            failures.append(f"{dtype_name} {shape} {label}.{name}: "
                                            f"unexpected launches {names}")
            for timing in (drv["k1_intra"],):
                names = _kernel_names(timing["launches"])
                if names != ["_chunk_gdn2_fwd_intra_infer_kernel"]:
                    failures.append(f"{dtype_name} {shape} k1: unexpected launches {names}")

            with out.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(drv, sort_keys=True) + "\n")
            print(json.dumps({
                "dtype": dtype_name, "shape": shape,
                "k1_us": drv["k1_intra"]["median_us"],
                "ht_zeros_us": drv["ht_new_zeros"]["median_us"],
                "ht_empty_us": drv["ht_new_empty"]["median_us"],
                "k2": {k: {kk: vv["median_us"] if isinstance(vv, dict) else vv
                           for kk, vv in v.items() if isinstance(vv, dict)}
                       for k, v in drv["k2"].items()},
                "bit_identical": drv["bit_identity"]["o_all_equal"]
                                 and drv["bit_identity"]["ht_all_equal"],
            }), flush=True)
            del args7, w_wy, u_wy, qg, kg, Aqk, _Akk, g_cumsum, state

    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema": 1, "kind": "end", "task": TASK,
                                 "failures": failures, "ended_at": time.time(),
                                 "loadavg": _loadavg()}, sort_keys=True) + "\n")
    if failures:
        for item in failures:
            print("FAIL:", item, file=sys.stderr)
        return 3
    print(f"OK: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
