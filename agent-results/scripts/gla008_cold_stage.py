"""GLA-008 true cold stage compile.

``gla008_cold.py`` measures the full ``chunk_gla`` call first and the stage
kernels afterwards, so its ``cold_fwd_h_ms``/``cold_bwd_dh_ms`` fields are
actually post-full-call warm-cache launches (~0.2 ms) -- a harness artifact,
not a compile time.  This script measures the *first* launch of each stage in
its own fresh process with its own empty ``TRITON_CACHE_DIR``, so the number is
the real Python-trace + Triton compile + cubin load cost.
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "agent-results/raw"
PY = "/flagos/bin/python"
CPU = "128-159"

CODE = r"""
import json, os, time
import torch, torch.nn.functional as F
torch.gcu.set_device(4)
import flag_attn.runtime.backend._enflame.FLA.chunk_h as C
B, T, H, K, V = 2, 256, 2, {K}, {V}
dt = {DT}
dev = torch.device("gcu:4")
k = torch.rand(B, T, H, K, dtype=dt, device=dev)
v = torch.rand(B, T, H, V, dtype=dt, device=dev)
q = torch.rand(B, T, H, K, dtype=dt, device=dev)
g = F.logsigmoid(torch.rand(B, T, H, K, dtype=dt, device=dev))
gk = torch.cumsum(g, dim=1) * 1.4426950216
h0 = torch.rand(B, H, K, V, dtype=torch.float32, device=dev)
do = torch.randn(B, T, H, V, dtype=dt, device=dev)
dht = torch.randn(B, H, K, V, dtype=torch.float32, device=dev)
if "{STAGE}" == "fwd_h":
    fn = lambda: C.chunk_fwd_h(k=k, v=v, g=None, gk=gk, gv=None, h0=h0,
                               output_final_state=True, cu_seqlens=None,
                               chunk_size=64, states_in_fp32=False,
                               state_v_first=False)
else:
    fn = lambda: C.chunk_bwd_dh(q=q, k=k, v=v, g=None, gk=gk, gv=None, do=do,
                                h0=h0, dht=dht, scale=K ** -0.5,
                                cu_seqlens=None, chunk_size=64,
                                states_in_fp32=True, state_v_first=False)
t0 = time.perf_counter()
fn()
torch.gcu.synchronize()
print("GLA008_COLD_STAGE " + json.dumps({{
    "stage": "{STAGE}", "K": K, "V": V, "dtype": "{DTNAME}",
    "cold_ms": (time.perf_counter() - t0) * 1000,
    "device": int(torch.gcu.current_device()),
    "stream": str(torch.gcu.current_stream(4).device),
}}))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RAW / "gla008_cold_stage.jsonl"))
    ap.add_argument("--cache-root", default=str(RAW / "gla008_cold_stage_cache"))
    ap.add_argument("--shapes", default="32x64,96x128")
    ap.add_argument("--dtypes", default="fp16,bf16,fp32")
    args = ap.parse_args()
    out = Path(args.out)
    cache_root = Path(args.cache_root)
    out.write_text("")
    rows = []
    dmap = {"fp16": "torch.float16", "bf16": "torch.bfloat16",
            "fp32": "torch.float32"}
    for shape in args.shapes.split(","):
        K, V = (int(x) for x in shape.split("x"))
        for dtype in args.dtypes.split(","):
            for stage in ("fwd_h", "bwd_dh"):
                cache = cache_root / f"{shape}_{dtype}_{stage}"
                if cache.exists():
                    shutil.rmtree(cache)
                cache.mkdir(parents=True)
                env = dict(os.environ)
                env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}:/opt/flagtree"
                env["TRITON_CACHE_DIR"] = str(cache)
                env["TRITON_F32_DEFAULT"] = "ieee"
                env["OMP_NUM_THREADS"] = "16"
                env.pop("FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP16_STATE", None)
                env.pop("FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP32_STATE", None)
                code = CODE.format(K=K, V=V, DT=dmap[dtype], DTNAME=dtype,
                                   STAGE=stage)
                proc = subprocess.run(["taskset", "-c", CPU, PY, "-c", code],
                                      cwd=ROOT, env=env, text=True,
                                      capture_output=True, timeout=1800)
                rec = None
                for ln in proc.stdout.splitlines():
                    if ln.startswith("GLA008_COLD_STAGE "):
                        rec = json.loads(ln[len("GLA008_COLD_STAGE "):])
                n_files = sum(1 for p in cache.rglob("*") if p.is_file())
                if rec is None:
                    rec = {"stage": stage, "K": K, "V": V, "dtype": dtype,
                           "ok": False, "rc": proc.returncode,
                           "error": proc.stderr[-1500:]}
                else:
                    rec["ok"] = (proc.returncode == 0
                                 and rec["device"] == 4
                                 and rec["stream"] == "gcu:4")
                    rec["cache_files"] = n_files
                rows.append(rec)
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{shape} {dtype} {stage}: "
                      f"{rec.get('cold_ms', float('nan'))/1000:.2f} s "
                      f"({n_files} cache files) ok={rec['ok']}", flush=True)
                shutil.rmtree(cache)
    if any(r.get("stage") == "fwd_h" and not r.get("ok") for r in rows):
        raise SystemExit("GLA008_FAIL cold stage fwd_h")
    if any(r.get("stage") == "bwd_dh" and not r.get("ok") for r in rows):
        raise SystemExit("GLA008_FAIL cold stage bwd_dh")
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
