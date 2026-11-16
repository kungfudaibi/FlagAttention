#!/usr/bin/env python3
"""GDN2-004: what K/V does the fused S60 forward actually accept?

The task asks for the real support range of K != V and non-power-of-two K/V,
not just the shapes the tests happen to use.  This walks the boundary: K or V
at 63/64/65/95/96/97/127/128/129/160/192/255/256/257/320/384/512, dictating
one irregular shape per step, and records success, the launched BK/BV tiles,
finiteness and the error against the CPU FP64 recurrence.  A failure is kept
with its exception so an unsupported extent is visible rather than hidden.

Usage: TOPS_VISIBLE_DEVICES=0 taskset -c 0-31 /flagos/bin/python \
    benchmark/gdn2_support_probe.py --out agent-results/GDN2-004-support.jsonl
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import traceback
from pathlib import Path

SEED = 42
CHUNK = 64
CASES = [
    # (B, T, H, K, V, note)
    (2, 65, 1, 63, 63, "below tile min"),
    (2, 65, 1, 64, 65, "K tile min, V just above"),
    (2, 65, 1, 65, 64, "K just above, V tile min"),
    (2, 65, 1, 95, 96, "non-pow2 pair"),
    (2, 65, 1, 96, 97, "non-pow2 pair"),
    (2, 130, 1, 127, 128, "K just below tile"),
    (2, 130, 1, 128, 129, "V just above tile"),
    (2, 130, 1, 160, 160, "K>128 tier to 256"),
    (2, 130, 1, 192, 96, "K>V irregular"),
    (2, 130, 1, 96, 192, "V>K irregular"),
    (1, 64, 1, 255, 256, "just under 256"),
    (1, 64, 1, 256, 255, "just under 256, V<K"),
    (1, 65, 1, 257, 256, "K just above 256"),
    (1, 65, 1, 256, 257, "V just above 256"),
    (1, 65, 1, 320, 320, "K,V above 256"),
    (1, 65, 1, 384, 96, "K=384 V=96"),
    (1, 64, 1, 96, 384, "K=96 V=384"),
    (1, 64, 1, 512, 512, "K,V=512"),
    (1, 1, 1, 96, 96, "T=1 non-pow2"),
    (1, 129, 1, 96, 96, "T just above 2 chunks"),
]


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


def _tracer():
    from triton.runtime.jit import JITFunction
    seen = []
    orig = JITFunction.run
    def traced(self, *args, **kwargs):
        seen.append({"kernel": getattr(self.fn, "__name__", str(self.fn)),
                     "BK": kwargs.get("BK"), "BV": kwargs.get("BV"),
                     "BT": kwargs.get("BT"),
                     "grid": kwargs.get("grid") if not callable(kwargs.get("grid")) else "callable"})
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
    p.add_argument("--dtype", default="float16")
    args = p.parse_args(argv)

    import torch
    sys.path.insert(0, str(Path.cwd() / "src"))
    sys.path.insert(0, str(Path.cwd() / "tests" / "flag_attn"))
    from gdn2_backward_ref import gdn2_forward
    from flag_attn import chunk_gdn2

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    out = Path(args.out)
    tracer = _tracer()
    for B, T, H, K, V, note in CASES:
        rec = {"schema": 1, "task": "GDN2-004", "kind": "support_probe",
               "shape": [B, T, H, K, V], "note": note, "dtype": args.dtype}
        try:
            a = _make(B, T, H, K, V, dtype, "gcu:0")
            tracer.clear()
            with torch.inference_mode():
                o, ht = chunk_gdn2(*a[:6], scale=K ** -0.5, initial_state=a[6],
                                   output_final_state=True, chunk_size=CHUNK)
            torch.gcu.synchronize()
            ref_o, ref_ht = gdn2_forward(*[x.cpu() for x in a[:6]],
                                         K ** -0.5, a[6].cpu())
            rec.update({
                "ok": True,
                "o_shape": list(o.shape), "ht_shape": list(ht.shape),
                "finite": [bool(torch.isfinite(o).all()),
                           bool(torch.isfinite(ht).all())],
                "o": _metrics(ref_o, o), "ht": _metrics(ref_ht, ht),
                "launches": [{"kernel": x["kernel"], "BK": x["BK"],
                              "BV": x["BV"], "BT": x["BT"]} for x in tracer],
            })
        except Exception as exc:  # keep the failure, do not hide it
            rec.update({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-1200:]})
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        print(json.dumps({"shape": rec["shape"], "ok": rec.get("ok"),
                          "err": rec.get("error"),
                          "tiles": sorted({(x["kernel"][-24:], x["BK"], x["BV"])
                                           for x in rec.get("launches", [])})}))
        if rec.get("ok"):
            del a, o, ht, ref_o, ref_ht
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
