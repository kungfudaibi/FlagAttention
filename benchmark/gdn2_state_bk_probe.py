#!/usr/bin/env python3
"""GDN2-004 analysis: is the K2 three-tier state BK the right rule?

``chunk_gdn2_fwd_h_o_infer`` picks BK = 64 (K<=64), 128 (K<=128), 256
(else).  For K in (128, 256] that issues a 256-lane dot of which only K lanes
carry data, and K=96 gets a 128-lane dot with 32 masked lanes.  This probe
compares the shipped rule with BK forced to 128, alternating the two inside
one process over several rounds so clock drift and host load hit both sides.

Unlike the V-tile split, BK *is* an accumulation axis (the readout sums over
K), so the probe also reports whether the outputs stay bit-identical.  No
source edit: the module constants are patched in memory and restored.

Usage: TOPS_VISIBLE_DEVICES=0 taskset -c 0-31 /flagos/bin/python \
    benchmark/gdn2_state_bk_probe.py --out agent-results/GDN2-004-state-bk.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

SEED = 42
CHUNK = 64
ROUNDS = 3
SHAPES = [(2, 512, 4, 96, 96), (2, 512, 4, 160, 160), (2, 512, 4, 192, 192),
          (1, 1024, 4, 256, 256), (2, 512, 4, 96, 160), (2, 512, 4, 160, 96)]


def _make(B, T, H, K, V, dtype, device):
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


def _time(fn, samples=15, warm=5, seconds=1.5):
    import torch
    deadline = time.perf_counter() + seconds
    n = 0
    while n < warm or time.perf_counter() < deadline:
        fn()
        n += 1
    torch.gcu.synchronize()
    begin = torch.gcu.Event(enable_timing=True)
    end = torch.gcu.Event(enable_timing=True)
    vals = []
    for _ in range(samples):
        begin.record(); fn(); end.record(); torch.gcu.synchronize()
        vals.append(begin.elapsed_time(end))
    o = sorted(vals)
    return {"median_ms": round(statistics.median(vals), 6),
            "min_ms": round(o[0], 6), "max_ms": round(o[-1], 6),
            "samples_ms": [round(x, 6) for x in vals], "iters": n}


def _digest(t):
    import torch
    raw = t.detach().to("cpu").contiguous()
    if raw.dtype == torch.bfloat16:
        raw = raw.view(torch.int16)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def _tracer():
    from triton.runtime.jit import JITFunction
    seen = []
    orig = JITFunction.run
    def traced(self, *args, **kwargs):
        seen.append({"kernel": getattr(self.fn, "__name__", str(self.fn)),
                     "BK": kwargs.get("BK"), "BV": kwargs.get("BV"),
                     "BT": kwargs.get("BT")})
        return orig(self, *args, **kwargs)
    JITFunction.run = traced
    return seen


def _metrics(ref, act):
    e = ref.double().flatten().cpu()
    a = act.double().flatten().cpu()
    d = (e - a).abs()
    rms = d.square().mean().sqrt().item()
    return {"max_abs": d.max().item(),
            "rms_rel": rms / (e.square().mean().sqrt().item() + 1e-12),
            "ref_rms": e.square().mean().sqrt().item(),
            "ref_max": e.abs().max().item()}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--rounds", type=int, default=ROUNDS)
    args = p.parse_args(argv)

    import torch
    sys.path.insert(0, str(Path.cwd() / "src"))
    sys.path.insert(0, str(Path.cwd() / "tests" / "flag_attn"))
    from gdn2_backward_ref import gdn2_forward
    from flag_attn import chunk_gdn2
    impl = importlib.import_module(
        "flag_attn.runtime.backend._enflame.FLA.gdn2.chunk_gdn2")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    out = Path(args.out)
    tracer = _tracer()
    for B, T, H, K, V in SHAPES:
        a = _make(B, T, H, K, V, dtype, "gcu:0")
        scale = K ** -0.5
        ref_o, ref_ht = gdn2_forward(*[x.cpu() for x in a[:6]], scale, a[6].cpu())

        def call():
            with torch.inference_mode():
                return chunk_gdn2(*a[:6], scale=scale, initial_state=a[6],
                                  output_final_state=True, chunk_size=CHUNK)

        o0, ht0 = call()
        torch.gcu.synchronize()
        with torch.inference_mode():
            w_wy, u_wy, qg, kg, Aqk, Akk, gk = impl.chunk_gdn2_fwd_intra_infer(
                q=a[0], k=a[1], v=a[2], g=a[3], b=a[4], w_gate=a[5],
                scale=scale, chunk_size=CHUNK)

        def k2():
            with torch.inference_mode():
                return impl.chunk_gdn2_fwd_h_o_infer(
                    kg=kg, w_wy=w_wy, u_wy=u_wy, gk=gk, qg=qg, Aqk=Aqk,
                    scale=scale, initial_state=a[6], output_final_state=True,
                    chunk_size=CHUNK)

        old = (impl._S60_STATE_BK_SMALL, impl._S60_STATE_BK_MEDIUM,
               impl._S60_STATE_BK_LARGE)
        rec = {"schema": 1, "task": "GDN2-004", "kind": "state_bk_probe",
               "shape": [B, T, H, K, V], "dtype": args.dtype,
               "shipped_rule": {"K<=64": old[0], "K<=128": old[1], "else": old[2]},
               "variants": {}}
        results = {}
        try:
            for rnd in range(1, args.rounds + 1):
                # Alternate which variant goes first, so neither always pays the
                # first-call compile/clock cost of the round.
                order = ("tiered", "forced128") if rnd % 2 else ("forced128", "tiered")
                for name in order:
                    consts = None if name == "tiered" else (128, 128, 128)
                    if consts is None:
                        impl._S60_STATE_BK_SMALL, impl._S60_STATE_BK_MEDIUM, \
                            impl._S60_STATE_BK_LARGE = old
                    else:
                        impl._S60_STATE_BK_SMALL, impl._S60_STATE_BK_MEDIUM, \
                            impl._S60_STATE_BK_LARGE = consts
                    tracer.clear()
                    o, ht = call()
                    torch.gcu.synchronize()
                    cfg = sorted({(x["BK"], x["BV"]) for x in tracer
                                  if x["kernel"] == "_chunk_gdn2_fwd_h_o_infer_kernel"})
                    entry = rec["variants"].setdefault(
                        name, {"state_bk_cfg": [list(x) for x in cfg], "rounds": []})
                    entry["rounds"].append({
                        "round": rnd,
                        "k2": _time(k2),
                        "full": _time(call),
                        "o": _metrics(ref_o, o),
                        "ht": _metrics(ref_ht, ht),
                        "o_sha256": _digest(o), "ht_sha256": _digest(ht),
                        "finite": [bool(torch.isfinite(o).all()),
                                   bool(torch.isfinite(ht).all())],
                    })
                results[rnd] = (o, ht)
        finally:
            impl._S60_STATE_BK_SMALL, impl._S60_STATE_BK_MEDIUM, \
                impl._S60_STATE_BK_LARGE = old
        for name, entry in rec["variants"].items():
            k2s = [x["k2"]["median_ms"] for x in entry["rounds"]]
            fulls = [x["full"]["median_ms"] for x in entry["rounds"]]
            entry["k2_median_ms"] = round(statistics.median(k2s), 6)
            entry["full_median_ms"] = round(statistics.median(fulls), 6)
            entry["k2_rounds_ms"] = k2s
        t = rec["variants"]["tiered"]
        f = rec["variants"]["forced128"]
        rec["forced128_speedup_k2"] = round(
            statistics.median(t["k2_rounds_ms"]) / statistics.median(f["k2_rounds_ms"]), 4)
        rec["bit_identical"] = bool(
            {x["o_sha256"] for x in t["rounds"]} == {x["o_sha256"] for x in f["rounds"]}
            and {x["ht_sha256"] for x in t["rounds"]} == {x["ht_sha256"] for x in f["rounds"]})
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        print(json.dumps({"shape": rec["shape"],
                          "tiered_bk": t["state_bk_cfg"], "forced_bk": f["state_bk_cfg"],
                          "k2_tiered": t["k2_rounds_ms"], "k2_forced": f["k2_rounds_ms"],
                          "speedup": rec["forced128_speedup_k2"],
                          "bit_identical": rec["bit_identical"]}))
        del a, o0, ht0, w_wy, u_wy, qg, kg, Aqk, Akk, gk, results
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
