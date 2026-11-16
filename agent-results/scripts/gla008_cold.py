"""GLA-008 cold-compile measurement with an isolated Triton cache.

The shared ``~/.triton/cache`` is warm from GLA-006, so "cold" numbers taken with
it are just warm-cache launches (~0.2 ms).  This script points ``TRITON_CACHE_DIR``
at an empty directory so the first launch really does generate PTX/cubin, and
measures:

  * cold stage     : first ``fwd_h`` / ``bwd_dh`` launch in the process
  * cold whole call: first full ``chunk_gla`` forward+backward
  * warm stage     : the same calls after warm-up, for the compile share

Each row is one fresh process, so no compile can leak between rows.  The cache
directory is removed afterwards and is excluded from git.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "agent-results/scripts/gla008_case.py"
RAW = ROOT / "agent-results/raw"
PY = "/flagos/bin/python"
CPU = "128-159"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RAW / "gla008_cold.jsonl"))
    ap.add_argument("--cache-root", default=str(RAW / "gla008_cold_cache"))
    ap.add_argument("--shapes", default="32x64,64x64,96x128,256x64")
    ap.add_argument("--dtypes", default="fp16,bf16,fp32")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    cache_root = Path(args.cache_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for shape in args.shapes.split(","):
        K, V = (int(x) for x in shape.split("x"))
        for dtype in args.dtypes.split(","):
            cache = cache_root / f"{shape}_{dtype}"
            if cache.exists():
                shutil.rmtree(cache)
            cache.mkdir(parents=True)
            env = dict(os.environ)
            env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}:/opt/flagtree"
            env["TRITON_CACHE_DIR"] = str(cache)
            env["TRITON_F32_DEFAULT"] = "ieee"
            for k in ("FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP16_STATE",
                      "FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP32_STATE"):
                env.pop(k, None)
            env["OMP_NUM_THREADS"] = "16"
            cmd = ["taskset", "-c", CPU, PY, str(CASE),
                   "--B", "2", "--T", "256", "--H", "2", "--K", str(K),
                   "--V", str(V), "--dtype", dtype, "--gln", "1.0",
                   "--exp-fp16", "1", "--exp-fp32", "1",
                   "--samples", "5", "--warmup", "2", "--tag", "cold",
                   "--mem"]
            proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True,
                                  capture_output=True, timeout=3600)
            rec = None
            for ln in proc.stdout.splitlines():
                if ln.startswith("GLA008_JSON "):
                    rec = json.loads(ln[len("GLA008_JSON "):])
            if rec is None or rec.get("exit") != 0:
                row = {"K": K, "V": V, "dtype": dtype, "ok": False,
                       "error": (rec or {}).get("error") or proc.stderr[-1500:]}
                rows.append(row)
                with open(out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"{shape} {dtype}: FAILED {row['error'][:200]}", flush=True)
                continue
            n_files = sum(1 for _ in cache.rglob("*") if _.is_file())
            size = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file())
            row = {
                "K": K, "V": V, "dtype": dtype, "ok": True,
                "cache_files": n_files, "cache_bytes": size,
                "cold_fwd_h_ms": rec.get("cold_fwd_h_ms"),
                "cold_bwd_dh_ms": rec.get("cold_bwd_dh_ms"),
                "cold_full_fwd_bwd_ms": rec.get("cold_full_fwd_bwd_ms"),
                "warm_fwd_h_ms": (rec.get("timing") or {}).get("fwd_h", {}).get("median"),
                "warm_bwd_dh_ms": (rec.get("timing") or {}).get("bwd_dh_fp32", {}).get("median"),
                "warm_full_fwd_bwd_ms": (rec.get("timing") or {}).get("full_fwd_bwd", {}).get("median"),
                "parallel_fwd_h": rec.get("parallel_fwd_h"),
                "parallel_bwd_dh": rec.get("parallel_bwd_dh"),
                "device_ok": rec.get("device_ok"),
                "local_state_dtypes": rec.get("local_state_dtypes"),
            }
            rows.append(row)
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{shape} {dtype}: cold fwd_h {row['cold_fwd_h_ms']:.1f} ms, "
                  f"cold bwd {row['cold_bwd_dh_ms']:.1f} ms, "
                  f"cold full {row['cold_full_fwd_bwd_ms']:.1f} ms, "
                  f"warm full {row['warm_full_fwd_bwd_ms']:.2f} ms, "
                  f"cache {n_files} files / {size / 1e6:.1f} MB", flush=True)
            if not args.keep:
                shutil.rmtree(cache)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
