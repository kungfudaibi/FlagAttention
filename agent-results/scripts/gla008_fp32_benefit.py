"""GLA-008 Q3: does the FP32 IEEE-dot path buy anything on this device?

Two independent questions, measured separately so neither can hide behind the
other:

1. Accuracy.  The FP32 parallel scan carries the state recurrence in IEEE FP32
   (``_chunk_bwd_state_scan_fp32_2d_group_kernel`` / FP32 ``local_state``), while
   the fallback may accumulate in a lower precision.  Compare both against an
   FP64 recurrence on the *same* FP32 inputs.
2. Speed.  FP32 elementwise/tile work is 2x the bytes of BF16, and the GCU dot
   unit has no faster FP32 mode, so the parallel path may simply be slower.  A
   speed loss with no accuracy gain is a reason NOT to dispatch FP32.

``TRITON_F32_DEFAULT`` is recorded with every row: ``ieee`` is the honest setting
for an IEEE-dot claim, and ``tf32`` shows what the hardware would do otherwise.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "agent-results/scripts/gla008_case.py"
RAW = ROOT / "agent-results/raw"
PY = "/flagos/bin/python"
CPU = "128-159"


def env(f32_default):
    e = dict(os.environ)
    e["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}:/opt/flagtree"
    for k in ("FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP16_STATE",
              "FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP32_STATE",
              "FLAGGEMS_ENFLAME_GLA_PARALLEL_STATE",
              "FLAGGEMS_ENFLAME_GLA_PARALLEL_BWD_STATE"):
        e.pop(k, None)
    e["TRITON_F32_DEFAULT"] = f32_default
    e.setdefault("OMP_NUM_THREADS", "16")
    return e


def run(args, f32_default="ieee", timeout=1800):
    cmd = ["taskset", "-c", CPU, PY, str(CASE)] + args
    proc = subprocess.run(cmd, cwd=ROOT, env=env(f32_default), text=True,
                          capture_output=True, timeout=timeout)
    rec = None
    for ln in proc.stdout.splitlines():
        if ln.startswith("GLA008_JSON "):
            rec = json.loads(ln[len("GLA008_JSON "):])
    if rec is None:
        raise SystemExit(f"no JSON from {args}: {proc.stderr[-1500:]}")
    if proc.returncode != 0 or rec.get("exit") != 0:
        raise SystemExit(f"case failed {args}: {rec.get('error')} "
                         f"{proc.stderr[-800:]}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RAW / "gla008_fp32_benefit.jsonl"))
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--shapes", default="32x64,64x64,96x128,256x64")
    ap.add_argument("--dtypes", default="fp32,bf16")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    shapes = [tuple(int(x) for x in s.split("x")) for s in args.shapes.split(",")]
    for (K, V) in shapes:
        for dtype in args.dtypes.split(","):
            for f32d in ("ieee", "tf32"):
                ref = Path(tempfile.gettempdir()) / f"gla008_q3_{K}x{V}_{dtype}.pt"
                common = ["--B", "2", "--T", "256", "--H", "2", "--K", str(K),
                          "--V", str(V), "--dtype", dtype, "--gln", "1.0",
                          "--exp-fp16", "1", "--exp-fp32", "1",
                          "--samples", str(args.samples), "--warmup", "2",
                          "--mem"]
                fb = run(common + ["--force-fallback", "--save", str(ref),
                                   "--oracle", "--tag", "q3_fb"], f32d)
                par = run(common + ["--compare", str(ref), "--oracle",
                                    "--tag", "q3_par"], f32d)
                row = {
                    "K": K, "V": V, "dtype": dtype, "f32_default": f32d,
                    "fallback_ms": fb["timing"]["full_fwd_bwd"]["median"],
                    "parallel_ms": par["timing"]["full_fwd_bwd"]["median"],
                    "fallback_samples": fb["timing"]["full_fwd_bwd"]["ms"],
                    "parallel_samples": par["timing"]["full_fwd_bwd"]["ms"],
                    "fb_fwd_h_ms": fb["timing"]["fwd_h"]["median"],
                    "par_fwd_h_ms": par["timing"]["fwd_h"]["median"],
                    "fb_bwd_dh_ms": fb["timing"]["bwd_dh_fp32"]["median"],
                    "par_bwd_dh_ms": par["timing"]["bwd_dh_fp32"]["median"],
                    "ratio": par["timing"]["full_fwd_bwd"]["median"]
                    / fb["timing"]["full_fwd_bwd"]["median"],
                    "fb_oracle_grade": fb.get("oracle_quant_grade"),
                    "par_oracle_grade": par.get("oracle_quant_grade"),
                    "path_grade": par.get("grade"),
                    "nan_shared": par.get("nan_shared_names"),
                    "fb_mem_MiB": round((fb.get("full_call_memory") or {})
                                        .get("peak_delta", 0) / 1048576, 2),
                    "par_mem_MiB": round((par.get("full_call_memory") or {})
                                         .get("peak_delta", 0) / 1048576, 2),
                    "local_state_bytes": sorted(
                        {x["bytes"] for x in par.get("local_state_observed") or []}),
                    "parallel_route": [par.get("parallel_fwd_h"),
                                       par.get("parallel_bwd_dh")],
                }
                # Relative accuracy vs the FP64 recurrence, on identical inputs.
                fb_o = {r["name"]: r["rel_rms"] for r in
                        fb.get("oracle_quant") or []}
                par_o = {r["name"]: r["rel_rms"] for r in
                         par.get("oracle_quant") or []}
                row["oracle_rel_fallback"] = fb_o
                row["oracle_rel_parallel"] = par_o
                if fb_o and par_o and not any(
                        v != v for v in list(fb_o.values()) + list(par_o.values())):
                    common_keys = set(fb_o) & set(par_o)
                    row["mean_oracle_rel_fallback"] = statistics.mean(
                        fb_o[k] for k in common_keys)
                    row["mean_oracle_rel_parallel"] = statistics.mean(
                        par_o[k] for k in common_keys)
                    row["accuracy_ratio_par_over_fb"] = (
                        row["mean_oracle_rel_parallel"]
                        / row["mean_oracle_rel_fallback"])
                rows.append(row)
                with open(out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"{K}x{V} {dtype} {f32d}: fb {row['fallback_ms']:.3f} "
                      f"par {row['parallel_ms']:.3f} ratio {row['ratio']:.2f} "
                      f"oracle fb={row['fb_oracle_grade']} "
                      f"par={row['par_oracle_grade']} "
                      f"acc_ratio={row.get('accuracy_ratio_par_over_fb')}",
                      flush=True)
                try:
                    os.unlink(ref)
                except OSError:
                    pass
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
