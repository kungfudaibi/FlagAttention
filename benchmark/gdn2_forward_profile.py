#!/usr/bin/env python
"""GDN2-003 stage profile of the fused Enflame GDN-2 forward.

The public ``chunk_gdn2`` is one Python call that launches two Triton kernels:

* K1 ``_chunk_gdn2_fwd_intra_infer_kernel`` -- intra-chunk WY/score prep,
* K2 ``_chunk_gdn2_fwd_h_o_infer_kernel``  -- sequential state + output.

This script measures the full call and both stages separately on the same
inputs, over the T/state axes used by the report, and records the launched
kernel names plus their BK/BV tiles.  Stage timings come from the same code
path the public entry point uses, so the split adds information without a
profiler.

Usage::

    TOPS_VISIBLE_DEVICES=0 taskset -c 0-31 /flagos/bin/python \
        benchmark/gdn2_forward_profile.py --out agent-results/GDN2-003-profile.jsonl
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

DEVICE = "gcu:0"
CHUNK_SIZE = 64
WARMUP_S = 2.0

SHAPES = [
    (2, 512, 8, 64, 64),
    (1, 2048, 8, 64, 64),
    (1, 4096, 16, 64, 64),
    (2, 2048, 4, 128, 128),
    (1, 1024, 4, 256, 256),
]


def _time(fn, samples=20):
    import torch

    deadline = time.perf_counter() + WARMUP_S
    count = 0
    while count < 5 or time.perf_counter() < deadline:
        fn()
        count += 1
    torch.gcu.synchronize()
    begin = torch.gcu.Event(enable_timing=True)
    end = torch.gcu.Event(enable_timing=True)
    values = []
    for _ in range(samples):
        begin.record()
        fn()
        end.record()
        torch.gcu.synchronize()
        values.append(begin.elapsed_time(end))
    ordered = sorted(values)
    return {
        "samples_ms": [round(x, 6) for x in values],
        "median_ms": round(statistics.median(values), 6),
        "min_ms": round(ordered[0], 6),
        "p25_ms": round(ordered[len(ordered) // 4], 6),
        "max_ms": round(ordered[-1], 6),
        "warmup_iters": count,
    }


def _make(B, T, H, K, V, dtype):
    import torch

    torch.manual_seed(42)
    q = (torch.randn(B, T, H, K) / math.sqrt(K)).to(dtype).to(DEVICE)
    k = (torch.randn(B, T, H, K) / math.sqrt(K)).to(dtype).to(DEVICE)
    v = torch.randn(B, T, H, V).to(dtype).to(DEVICE)
    g = (-torch.rand(B, T, H, K) * 0.1).to(dtype).to(DEVICE)
    b = torch.rand(B, T, H, K).to(dtype).to(DEVICE)
    w = torch.rand(B, T, H, V).to(dtype).to(DEVICE)
    h0 = (torch.randn(B, H, K, V) * 0.01).to(DEVICE)
    return q, k, v, g, b, w, h0


def _tracer():
    from triton.runtime.jit import JITFunction

    seen = []
    original = JITFunction.run

    def traced(self, *args, **kwargs):
        grid = kwargs.get("grid")
        seen.append({
            "kernel": getattr(self.fn, "__name__", str(self.fn)),
            "grid": list(grid) if isinstance(grid, (tuple, list)) else repr(grid),
            "BK": kwargs.get("BK"),
            "BV": kwargs.get("BV"),
            "BT": kwargs.get("BT"),
        })
        return original(self, *args, **kwargs)

    JITFunction.run = traced
    return seen


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    args = parser.parse_args(argv)

    import torch

    from flag_attn import chunk_gdn2
    impl = importlib.import_module(
        "flag_attn.runtime.backend._enflame.FLA.gdn2.chunk_gdn2")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": 1, "task": "GDN2-003", "kind": "profile_meta",
        "hostname": platform.node(), "torch": torch.__version__,
        "device": DEVICE, "device_count": torch.gcu.device_count(),
        "current_device": torch.gcu.current_device(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "tile_adapt": impl._S60_TILES_ADAPT,
        "dtype": args.dtype,
        "loadavg": Path("/proc/loadavg").read_text().strip(),
        "started_at": time.time(),
    }
    out.write_text(json.dumps(meta, sort_keys=True) + "\n", encoding="utf-8")

    tracer = _tracer()
    for B, T, H, K, V in SHAPES:
        scale = K ** -0.5
        args7 = _make(B, T, H, K, V, dtype)

        def full():
            with torch.inference_mode():
                return chunk_gdn2(*args7[:6], scale=scale, initial_state=args7[6],
                                  output_final_state=True, chunk_size=CHUNK_SIZE)

        if hasattr(torch.gcu, "reset_peak_memory_stats"):
            torch.gcu.reset_peak_memory_stats()
        base_alloc = torch.gcu.memory_allocated()

        tracer.clear()
        t0 = time.perf_counter()
        o, ht = full()
        torch.gcu.synchronize()
        cold_s = time.perf_counter() - t0
        cold_launches = [dict(x) for x in tracer]

        with torch.inference_mode():
            w_wy, u_wy, qg, kg, Aqk, Akk, g_cumsum = impl.chunk_gdn2_fwd_intra_infer(
                q=args7[0], k=args7[1], v=args7[2], g=args7[3], b=args7[4],
                w_gate=args7[5], scale=scale, chunk_size=CHUNK_SIZE)

        def k1():
            with torch.inference_mode():
                return impl.chunk_gdn2_fwd_intra_infer(
                    q=args7[0], k=args7[1], v=args7[2], g=args7[3], b=args7[4],
                    w_gate=args7[5], scale=scale, chunk_size=CHUNK_SIZE)

        def k2():
            with torch.inference_mode():
                return impl.chunk_gdn2_fwd_h_o_infer(
                    kg=kg, w_wy=w_wy, u_wy=u_wy, gk=g_cumsum, qg=qg, Aqk=Aqk,
                    scale=scale, initial_state=args7[6], output_final_state=True,
                    chunk_size=CHUNK_SIZE)

        record = {
            "schema": 1, "task": "GDN2-003", "kind": "profile",
            "shape": [B, T, H, K, V], "tiles": {
                "K": impl._s60_tile(K, impl._S60_INTRA_BK),
                "V_intra": impl._s60_tile(V, impl._S60_INTRA_BV),
                "V_state": impl._s60_tile(V, impl._S60_STATE_BV),
                "state_BK": (impl._S60_STATE_BK_SMALL if K <= 64 else
                             impl._S60_STATE_BK_MEDIUM if K <= 128 else
                             impl._S60_STATE_BK_LARGE),
            },
            "cold_compile_s": round(cold_s, 6),
            "cold_launches": cold_launches,
            "full": _time(full),
            "k1_intra": _time(k1),
            "k2_state_output": _time(k2),
            "intermediate_mib": {
                name: round(tensor.numel() * tensor.element_size() / 2**20, 4)
                for name, tensor in {
                    "w_wy": w_wy, "u_wy": u_wy, "qg": qg, "kg": kg,
                    "Aqk": Aqk, "Akk": Akk, "gate_cumsum": g_cumsum,
                }.items()
            },
            "finite": [bool(torch.isfinite(o).all().item()),
                       bool(torch.isfinite(ht).all().item())],
            "peak_alloc_mib": round(
                (torch.gcu.max_memory_allocated() - base_alloc) / 2**20, 4),
            "peak_total_alloc_mib": round(torch.gcu.max_memory_allocated() / 2**20, 4),
            "loadavg": Path("/proc/loadavg").read_text().strip(),
        }
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps({k: record[k] for k in
                          ("shape", "tiles", "full", "k1_intra", "k2_state_output")}))
        del args7, o, ht, w_wy, u_wy, qg, kg, Aqk, Akk, g_cumsum
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
