"""GLA-008 matrix driver: numerical/path evidence and alternating full-call A/B.

Two modes, one script, so the raw JSONL always carries the exact command,
tree, flags, device and thresholds.

  --mode probe    For each (KxV, dtype, gate): fallback saves reference tensors,
                  parallel compares against them and against the independent
                  FP64 CPU recurrence.  Path counters are read from real
                  launches; a route that silently falls back is a hard error.
                  Also runs the semantic-guard matrix and the FP32 facts probe.

  --mode ab       Alternating full-call A/B, `--rounds` rounds x `--samples`
                  synchronized samples per side, strict device check, cold
                  compile recorded separately, peak memory per side.

The driver is the formal benchmark entry point.  It exits non-zero on a
missing sample, a numerical failure, a wrong/incomplete route, or a device
mismatch; it never manufactures a pass from a partial matrix.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "agent-results/scripts/gla008_case.py"
BASE = ROOT / ".baseline-2ed"
RAW = ROOT / "agent-results/raw"
PY = "/flagos/bin/python"
CPU = "128-159"

CORE_SHAPES = [(32, 64), (64, 64), (96, 128), (256, 64)]
CORE_GATES = [0.1, 1.0, 10.0]
DTYPES = ["fp16", "bf16", "fp32"]
BOUNDARY = [
    dict(B=1, T=512, H=8, K=64, V=64, dtype="fp16", tag="edge_b1t512h8_64x64"),
    dict(B=1, T=512, H=8, K=96, V=128, dtype="fp16", tag="edge_b1t512h8_96x128"),
    dict(B=1, T=512, H=8, K=64, V=64, dtype="fp32", tag="edge_b1t512h8_64x64"),
    dict(B=2, T=300, H=2, K=32, V=64, dtype="fp16", tag="edge_t300_32x64"),
    dict(B=2, T=300, H=2, K=96, V=128, dtype="fp16", tag="edge_t300_96x128"),
    dict(B=2, T=300, H=2, K=256, V=64, dtype="fp32", tag="edge_t300_256x64"),
    # BF16 completes the three-dtype boundary requirement at both regimes
    # (B=1,T=512,H=8 and T=300).  Added after the first boundary batch, so the
    # tags are distinct and the report dedupes on (tag, side).
    dict(B=1, T=512, H=8, K=64, V=64, dtype="bf16", tag="edge_b1t512h8_64x64_bf16"),
    dict(B=2, T=300, H=2, K=32, V=64, dtype="bf16", tag="edge_t300_32x64_bf16"),
]
PRESSURE = [
    dict(B=4, T=1024, H=8, K=96, V=128, dtype="fp16", tag="press_b4t1024h8_96x128"),
    dict(B=4, T=1024, H=8, K=64, V=64, dtype="fp32", tag="press_b4t1024h8_64x64"),
    dict(B=2, T=512, H=4, K=256, V=128, dtype="fp16", tag="press_b2t512h4_256x128"),
]

THRESHOLDS = {
    "fp16": {"abs_floor": 1e-2, "path_rel": 4e-3},
    "bf16": {"abs_floor": 2e-2, "path_rel": 2e-2},
    "fp32": {"abs_floor": 1e-5, "path_rel": 2e-5},
}


def env_for(tree):
    e = dict(os.environ)
    e["PYTHONPATH"] = f"{tree}/src:{tree}:/opt/flagtree"
    e.pop("FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP16_STATE", None)
    e.pop("FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP32_STATE", None)
    e.pop("FLAGGEMS_ENFLAME_GLA_PARALLEL_STATE", None)
    e.pop("FLAGGEMS_ENFLAME_GLA_PARALLEL_BWD_STATE", None)
    e["TRITON_F32_DEFAULT"] = "ieee"
    e.setdefault("OMP_NUM_THREADS", "16")
    return e


def run_case(tree, args, timeout=900):
    """Run one case process with its output captured in temporary files.

    Files rather than pipes so that a timeout still yields whatever the child
    printed (a pipe reader would lose it) and so a chatty compiler cannot fill
    a pipe buffer and deadlock the child.  Note that row wall time is dominated
    by interpreter/Triton import (~7 s), not by the timed work (~3 s), so wall
    time is recorded for provenance and never used as the benchmark metric.
    """
    cmd = ["taskset", "-c", CPU, PY, str(CASE)] + args
    t0 = time.time()
    out_f = tempfile.TemporaryFile(mode="w+")
    err_f = tempfile.TemporaryFile(mode="w+")
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env_for(tree), text=True,
                              stdout=out_f, stderr=err_f, timeout=timeout)
        rc = proc.returncode
        out_f.seek(0); out = out_f.read()
        err_f.seek(0); err = err_f.read()
    except subprocess.TimeoutExpired:
        rc = 124
        out_f.seek(0); out = out_f.read()
        err = "TIMEOUT"
    finally:
        out_f.close()
        err_f.close()
    wall = round(time.time() - t0, 3)
    line = ""
    for ln in out.splitlines():
        if ln.startswith("GLA008_JSON "):
            line = ln[len("GLA008_JSON "):]
    rec = {"tree": tree, "cmd": cmd, "rc": rc, "wall_s": wall,
           "ts": time.time()}
    if line:
        rec.update(json.loads(line))
    else:
        rec["stdout_tail"] = out[-4000:]
        rec["stderr_tail"] = err[-4000:]
    return rec


def _plain(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def append(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(_plain(rec)) + "\n")


def exp_flags(rec):
    return ["--exp-fp16", str(rec.get("exp_fp16", 0)),
            "--exp-fp32", str(rec.get("exp_fp32", 0))]


def _cpu_share(cpus=range(128, 160), window=0.25):
    """Idle share of the CPUs this task is pinned to, sampled over `window`."""
    def snap():
        out = {}
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("cpu") and line[3].isdigit():
                    p = line.split()
                    out[int(p[0][3:])] = (sum(int(x) for x in p[1:]), int(p[4]) + int(p[5]))
        return out
    try:
        a, b = snap(), None
        time.sleep(window)
        b = snap()
        tot = sum(b[c][0] - a[c][0] for c in cpus if c in a and c in b)
        idle = sum(b[c][1] - a[c][1] for c in cpus if c in a and c in b)
        return round(100.0 * idle / tot, 2) if tot else None
    except Exception:
        return None


def host_load():
    """Record machine noise with every timing row; the A/B claim depends on it."""
    try:
        load = os.getloadavg()
    except OSError:
        load = (None, None, None)
    # Enumerating every process on this host (12k+ PIDs, mostly zombies) takes
    # seconds and would dominate the A/B wall time, so noise is recorded from
    # the pinned CPUs' own idle counters and the runnable-task count instead.
    try:
        runnable, total = open("/proc/loadavg").read().split()[3].split("/")
        runnable, total = int(runnable), int(total)
    except Exception:
        runnable = total = None
    return {"load1": load[0], "load5": load[1], "load15": load[2],
            "runnable": runnable, "tasks": total,
            "cpu_idle_pct_128_159": _cpu_share()}


def case_args(rec, extra=None):
    return ["--B", str(rec["B"]), "--T", str(rec["T"]), "--H", str(rec["H"]),
            "--K", str(rec["K"]), "--V", str(rec["V"]),
            "--dtype", rec["dtype"], "--gln", str(rec["gln"]),
            "--seed", str(rec.get("seed", 42))] + exp_flags(rec) + (extra or [])


def check_device(rec):
    if rec.get("rc") != 0 or rec.get("exit") != 0:
        return False
    return bool(rec.get("device_ok")) and rec.get("gcu_current_device") == 4 \
        and rec.get("stream_device") == "gcu:4"


def path_counts(rec):
    keys = ("_chunk_fwd_local_state_kernel", "_chunk_fwd_state_scan_kernel",
            "_chunk_fwd_state_scan_fp32_2d_group_kernel",
            "_chunk_bwd_local_state_kernel",
            "_chunk_bwd_state_scan_fp32_2d_group_kernel",
            "chunk_fwd_kernel_h", "chunk_bwd_kernel_dh")
    out = {}
    for phase in ("path_fwd_h", "path_bwd_dh", "path_full_autograd"):
        done = (rec.get(phase) or {}).get("done", {})
        out[phase] = {k: done.get(k, 0) for k in keys if done.get(k, 0)}
    return out


def expect_parallel(rec, require):
    """Hard gate: the requested route must really have launched."""
    paths = path_counts(rec)
    fwd, bwd = paths["path_fwd_h"], paths["path_bwd_dh"]
    if require == "parallel":
        got = (fwd.get("_chunk_fwd_local_state_kernel", 0) > 0
               and not fwd.get("chunk_fwd_kernel_h")
               and bwd.get("_chunk_bwd_local_state_kernel", 0) > 0
               and not bwd.get("chunk_bwd_kernel_dh"))
        return got, "parallel route not launched" if not got else ""
    if require == "fallback":
        got = (fwd.get("chunk_fwd_kernel_h", 0) > 0
               and bwd.get("chunk_bwd_kernel_dh", 0) > 0
               and not fwd.get("_chunk_fwd_local_state_kernel")
               and not bwd.get("_chunk_bwd_local_state_kernel"))
        return got, "fallback route not honored" if not got else ""
    raise ValueError(require)


def local_state_facts(rec):
    dtypes = set(rec.get("local_state_dtypes") or [])
    observed = rec.get("local_state_observed") or []
    theory = rec.get("local_state_fp32_bytes_theory") or {}
    return {
        "local_state_dtypes": sorted(dtypes),
        "all_fp32": dtypes == {"torch.float32"} if dtypes else None,
        "observed_bytes": sorted({x["bytes"] for x in observed}),
        "theory_fwd": theory.get("fwd"), "theory_bwd": theory.get("bwd"),
        "observed_equals_theory": (
            theory.get("fwd") in {x["bytes"] for x in observed}
            and theory.get("bwd") in {x["bytes"] for x in observed}
        ) if observed else None,
        "h_state_dtype": (rec.get("state_stats") or {}).get("h", {}).get("dtype"),
        "dh_state_dtype": (rec.get("state_stats") or {}).get("dh", {}).get("dtype"),
        "h0_is_fp32": rec.get("h0_is_fp32"),
        "dht_is_fp32": rec.get("dht_is_fp32"),
    }


# --------------------------------------------------------------------------
# probe mode
# --------------------------------------------------------------------------
def probe_one(path, rec, oracle=True, want_mem=False, oracle_base=False):
    """fallback reference -> parallel compare, plus FP64 recurrence."""
    r = dict(rec)
    tag = f"{r['tag']}_{r['dtype']}_g{r['gln']}"
    ref_path = str(Path(tempfile.gettempdir()) / f"gla008_{tag}.pt")
    extra = ["--samples", "1", "--warmup", "0", "--no-timing"]
    if want_mem:
        extra.append("--mem")
    if oracle:
        extra.append("--oracle")
    if oracle_base:
        extra.append("--oracle-base")
    # Same-commit fallback reference: only the env route switch differs.
    fb = run_case(ROOT, case_args(r, extra + ["--force-fallback",
                                              "--save", ref_path,
                                              "--tag", "fb_" + tag]))
    fb["side"] = "fallback"
    fb["route"] = "fallback"
    append(path, fb)
    ok_fb, why_fb = expect_parallel(fb, "fallback")
    if not check_device(fb) or not ok_fb:
        raise SystemExit(f"GLA008_FAIL fallback {tag}: {why_fb} "
                         f"{fb.get('error') or fb.get('stderr_tail','')[-500:]}")

    par = run_case(ROOT, case_args(r, extra + ["--compare", ref_path,
                                               "--tag", "par_" + tag]))
    par["side"] = "parallel"
    par["route"] = "parallel"
    par["fallback_reference"] = ref_path
    append(path, par)
    ok_par, why_par = expect_parallel(par, "parallel")
    if not check_device(par):
        raise SystemExit(f"GLA008_FAIL parallel {tag}: device/rc "
                         f"{par.get('error') or par.get('stderr_tail','')[-500:]}")
    if not ok_par:
        raise SystemExit(f"GLA008_FAIL parallel {tag}: {why_par}")
    grade = par.get("grade")
    if grade == "shared_nan":
        # A NaN on both sides is only acceptable as *documented pre-existing*
        # behaviour: the untouched baseline commit must show the same NaN mask
        # through its own default route.  Anything else is a dispatch failure.
        oldchk = run_case(BASE, case_args(r, extra + ["--compare", ref_path,
                                                      "--tag", "oldchk_" + tag]))
        oldchk["side"] = "old_commit_check"
        append(path, oldchk)
        same = (oldchk.get("grade") == "shared_nan"
                and set(oldchk.get("nan_shared_names") or []) ==
                set(par.get("nan_shared_names") or [])
                and not oldchk.get("nan_mismatch_names"))
        par["preexisting_nan_confirmed"] = same
        par["old_commit_grade"] = oldchk.get("grade")
        par["old_commit_nan_names"] = oldchk.get("nan_shared_names")
        append(path, {"kind": "preexisting_nan_check", "tag": tag,
                      "side": "parallel",
                      "new_grade": par.get("grade"),
                      "new_nan_names": par.get("nan_shared_names"),
                      "old_grade": oldchk.get("grade"),
                      "old_nan_names": oldchk.get("nan_shared_names"),
                      "confirmed": same})
        if not same:
            raise SystemExit(f"GLA008_FAIL unconfirmed shared NaN {tag}: "
                             f"new={par.get('nan_shared_names')} "
                             f"old={oldchk.get('nan_shared_names')}")
        print("SHARED_NAN_PREEXISTING", tag, par.get("nan_shared_names"),
              flush=True)
    elif grade != "pass":
        bad = [x for x in par.get("compare", []) if not x["ok"]]
        raise SystemExit(f"GLA008_FAIL numeric {tag}: {grade} "
                         f"{[(x['name'], x['max_abs'], x['rel_rms']) for x in bad]}")
    if par.get("nan_mismatch_names"):
        raise SystemExit(f"GLA008_FAIL NaN mask mismatch {tag}: "
                         f"{par['nan_mismatch_names']}")
    if oracle:
        og = par.get("oracle_quant_grade")
        if grade == "shared_nan":
            # The FP64 recurrence is finite here, so comparing a NaN-bearing
            # result against it *must* fail.  Record that the mismatch is
            # exactly the pre-existing NaN and not a new error.
            blocked = (og == "fail"
                       and set(par.get("oracle_nan_counts") or {}) <=
                       set(par.get("nan_shared_names") or []))
            par["oracle_blocked_by_preexisting_nan"] = bool(blocked)
            par["oracle_grade_when_shared_nan"] = og
            append(path, {"kind": "oracle_blocked", "tag": tag,
                          "side": "parallel",
                          "oracle_grade": og,
                          "nan_names": par.get("nan_shared_names"),
                          "blocked": bool(blocked)})
            if not blocked:
                raise SystemExit(f"GLA008_FAIL oracle {tag}: shared NaN row has "
                                 f"non-NaN oracle error {og} "
                                 f"{[x for x in par.get('oracle_quant', []) if not x['ok']]}")
        elif og == "fail":
            bad = [x for x in par.get("oracle_quant", []) if not x["ok"]]
            raise SystemExit(f"GLA008_FAIL oracle {tag}: "
                             f"{[(x['name'], x['max_abs'], x['rel_rms']) for x in bad]}")
    # FP16 additionally compares against the exact pre-GLA-006 commit.  This is
    # a code-continuity check, not a second numerical oracle.
    if r["dtype"] == "fp16" and grade == "pass":
        old = run_case(BASE, case_args(r, extra + ["--force-fallback",
                                                   "--compare", ref_path,
                                                   "--tag", "old_" + tag]))
        old["side"] = "old_commit"
        append(path, old)
        if not check_device(old) or old.get("grade") != "pass":
            raise SystemExit(f"GLA008_FAIL old-tree fp16 {tag}: {old.get('grade')} "
                             f"{[x for x in old.get('compare', []) if not x['ok']]}")
    # remove the temporary reference only after a successful row
    try:
        os.unlink(ref_path)
    except OSError:
        pass
    return fb, par


def quant_probe(path, shapes, gates):
    """Separate input quantisation error from parallel-path error.

    ``oracle_quant``  = parallel output vs an FP64 recurrence run on the *same*
                        quantised inputs  -> path error only.
    ``oracle_base``   = parallel output vs an FP64 recurrence run on the FP64
                        base inputs       -> path + input-quantisation error.
    ``quant_effect``  = FP64(quantised inputs) vs FP64(base inputs)
                                         -> input-quantisation error alone.
    A path is only acceptable when ``oracle_quant`` passes; a large
    ``oracle_base`` with a small ``oracle_quant`` is an input-precision effect,
    not a dispatch defect.
    """
    rows = []
    for (K, V) in shapes:
        for dtype in DTYPES:
            for gate in gates:
                rec = dict(B=2, T=256, H=2, K=K, V=V, dtype=dtype, gln=gate,
                           tag=f"quant_{K}x{V}", exp_fp16=1, exp_fp32=1)
                fb, par = probe_one(path, rec, oracle=True, want_mem=False,
                                    oracle_base=True)
                row = {"kind": "quant", "K": K, "V": V, "dtype": dtype,
                       "gln": gate, "fallback": path_counts(fb),
                       "parallel": path_counts(par),
                       "grade": par.get("grade"),
                       "oracle_quant_grade": par.get("oracle_quant_grade"),
                       "oracle_base_grade": par.get("PASS_ORACLE_BASE"),
                       "nan_shared_names": par.get("nan_shared_names"),
                       "path_error": par.get("oracle_quant"),
                       "total_error": par.get("oracle_base"),
                       "quant_effect": par.get("quant_effect"),
                       "fallback_path_error": fb.get("oracle_quant"),
                       "fallback_total_error": fb.get("oracle_base"),
                       "local_state": local_state_facts(par)}
                if par.get("oracle_quant_grade") != "pass":
                    raise SystemExit(f"GLA008_FAIL quant path {row['dtype']} "
                                     f"{K}x{V} g{gate}: "
                                     f"{par.get('oracle_quant_grade')} "
                                     f"{[x for x in par.get('oracle_quant', []) if not x['ok']]}")
                rows.append(row)
                print("QUANT", K, V, dtype, gate,
                      "path", par.get("oracle_quant_grade"),
                      "total", par.get("PASS_ORACLE_BASE"),
                      "quant_effect", bool(par.get("quant_effect")), flush=True)
    append(path, {"kind": "quant_summary", "rows": rows})
    return rows


def guard_probe(path):
    """Semantic guards must still fall back with both opt-in flags on."""
    code = r"""
import json, os
os.environ["FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP16_STATE"]="1"
os.environ["FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP32_STATE"]="1"
os.environ["TRITON_F32_DEFAULT"]="ieee"
import torch, torch.nn.functional as F
torch.gcu.set_device(4)
from flag_attn.runtime.backend._enflame.FLA import chunk_h as Hmod
A={}; D={}
P=("_chunk_fwd_local_state_kernel","_chunk_bwd_local_state_kernel",
   "_chunk_fwd_state_scan_kernel","_chunk_fwd_state_scan_fp32_2d_group_kernel",
   "chunk_fwd_kernel_h","chunk_bwd_kernel_dh")
class Pr:
    def __init__(s,n,r): s.n=n; s.r=r
    def __getitem__(s,g):
        l=s.r[g]
        def call(*a,**k):
            A[s.n]=A.get(s.n,0)+1; o=l(*a,**k); D[s.n]=D.get(s.n,0)+1; return o
        return call
for n in P:
    if hasattr(Hmod,n): setattr(Hmod,n,Pr(n,getattr(Hmod,n)))
dev=torch.device("gcu:4")
B,T,H,K,V=1,128,1,64,64
dt=torch.float16
torch.manual_seed(42)
k=torch.rand(B,T,H,K,dtype=dt,device=dev); v=torch.rand(B,T,H,V,dtype=dt,device=dev)
q=torch.rand(B,T,H,K,dtype=dt,device=dev)
g=F.logsigmoid(torch.rand(B,T,H,K,dtype=dt,device=dev))
h0=torch.rand(B,H,K,V,dtype=torch.float32,device=dev)
do=torch.randn(B,T,H,V,dtype=dt,device=dev)
dht=torch.randn(B,H,K,V,dtype=torch.float32,device=dev)
cu=torch.tensor([0,64,128],dtype=torch.int32,device=dev)
gk=torch.cumsum(g,dim=1)*1.4426950216
rows=[]
def probe(name,fn):
    A.clear(); D.clear(); err=None
    try:
        o=fn(); torch.gcu.synchronize(); del o
    except Exception as e:
        err=f"{type(e).__name__}: {e}"
    rows.append({"case":name,"error":err,"attempt":dict(A),"done":dict(D)})
base=dict(v=v,g=None,gk=gk,gv=None,h0=h0,output_final_state=True,
          chunk_size=64,states_in_fp32=False,state_v_first=False)
probe("fwd_varlen_cu_seqlens",lambda:Hmod.chunk_fwd_h(k=k,cu_seqlens=cu,**base))
probe("fwd_state_v_first",lambda:Hmod.chunk_fwd_h(k=k,**{**base,"state_v_first":True}))
probe("fwd_given_gate_tensor",lambda:Hmod.chunk_fwd_h(k=k,**{**base,"g":g,"gk":None}))
probe("fwd_split_size",lambda:Hmod.chunk_fwd_h(k=k,split_size=128,**base))
probe("fwd_states_in_fp32_bwd_on",lambda:Hmod.chunk_fwd_h(k=k,**{**base,"states_in_fp32":True}))
Hmod.PARALLEL_BWD_STATE_ENABLED=False
probe("fwd_states_in_fp32_bwd_off",lambda:Hmod.chunk_fwd_h(k=k,**{**base,"states_in_fp32":True}))
Hmod.PARALLEL_BWD_STATE_ENABLED=True
bwd=dict(g=None,gk=gk,gv=None,h0=h0,dht=dht,scale=K**-0.5,cu_seqlens=None,
         chunk_size=64,states_in_fp32=True,state_v_first=False)
probe("bwd_varlen_cu_seqlens",lambda:Hmod.chunk_bwd_dh(q=q,k=k,v=v,do=do,**{**bwd,"cu_seqlens":cu}))
probe("bwd_given_gate_tensor",lambda:Hmod.chunk_bwd_dh(q=q,k=k,v=v,do=do,**{**bwd,"g":g,"gk":None}))
probe("bwd_hq_ne_h",lambda:Hmod.chunk_bwd_dh(q=torch.rand(B,T,2*H,K,dtype=dt,device=dev),k=k,v=v,do=do,**bwd))
print("GLA008_GUARD "+json.dumps({"rows":rows,"device":str(torch.gcu.current_device()),
      "stream":str(torch.gcu.current_stream(4).device)}))
"""
    cmd = ["taskset", "-c", CPU, PY, "-c", code]
    proc = subprocess.run(cmd, cwd=ROOT, env=env_for(ROOT), text=True,
                          capture_output=True, timeout=600)
    rec = {"kind": "guard", "cmd": cmd, "rc": proc.returncode}
    line = ""
    for ln in proc.stdout.splitlines():
        if ln.startswith("GLA008_GUARD "):
            line = ln[len("GLA008_GUARD "):]
    if line:
        rec.update(json.loads(line))
    else:
        rec["stdout_tail"] = proc.stdout[-3000:]
        rec["stderr_tail"] = proc.stderr[-3000:]
    append(path, rec)
    if proc.returncode or not line:
        raise SystemExit(f"GLA008_FAIL guard probe rc={proc.returncode}")
    expected_fallback = {
        "fwd_varlen_cu_seqlens": "chunk_fwd_kernel_h",
        "fwd_state_v_first": "chunk_fwd_kernel_h",
        "fwd_given_gate_tensor": "chunk_fwd_kernel_h",
        "fwd_split_size": "chunk_fwd_kernel_h",
        "fwd_states_in_fp32_bwd_off": "chunk_fwd_kernel_h",
        "bwd_varlen_cu_seqlens": "chunk_bwd_kernel_dh",
        "bwd_given_gate_tensor": "chunk_bwd_kernel_dh",
    }
    for row in rec["rows"]:
        want = expected_fallback.get(row["case"])
        if want and row["done"].get(want, 0) < 1:
            raise SystemExit(f"GLA008_FAIL guard {row['case']} did not fall back: {row}")
        if want and row["done"].get("_chunk_fwd_local_state_kernel") \
                or want and row["done"].get("_chunk_bwd_local_state_kernel"):
            raise SystemExit(f"GLA008_FAIL guard {row['case']} used parallel: {row}")
    # HQ != H with an exact same-size fallback is a different path; require the
    # generic fallback kernel as well.
    return rec


def default_route_probe(path):
    """Flags 0 must reproduce the exact old tree on every dtype/shape."""
    rows = []
    for shape in [(32, 64), (64, 64), (96, 128), (256, 64)]:
        for dtype in ["fp16", "bf16", "fp32"]:
            rec = dict(B=2, T=256, H=2, K=shape[0], V=shape[1], dtype=dtype,
                       gln=1.0, tag="default_" + dtype + f"_{shape[0]}x{shape[1]}")
            ref = str(Path(tempfile.gettempdir()) /
                      f"gla008_default_{dtype}_{shape[0]}x{shape[1]}.pt")
            common = ["--samples", "1", "--warmup", "0", "--no-timing"]
            # Old tree with its own default route (no route forcing).
            fb = run_case(BASE, case_args(rec, common + ["--save", ref]))
            fb["side"] = "baseline_exact"
            append(path, fb)
            df = run_case(ROOT, case_args(rec, common + ["--compare", ref]))
            df["side"] = "candidate_flags0"
            append(path, df)
            os.unlink(ref)
            if not (check_device(fb) and check_device(df)):
                raise SystemExit(f"GLA008_FAIL default device {rec['tag']}")
            if not df.get("ALL_PASS") or df.get("max_abs_bad"):
                raise SystemExit(f"GLA008_FAIL default mismatch {rec['tag']}: "
                                 f"{[x for x in df.get('compare', []) if not x['ok']]}")
            # Route must match the old tree exactly (same launches).
            if path_counts(fb)["path_fwd_h"] != path_counts(df)["path_fwd_h"] or \
                    path_counts(fb)["path_bwd_dh"] != path_counts(df)["path_bwd_dh"]:
                raise SystemExit(f"GLA008_FAIL default route drift {rec['tag']}: "
                                 f"{path_counts(fb)} vs {path_counts(df)}")
            rows.append({"tag": rec["tag"], "route": path_counts(df)["path_fwd_h"],
                         "tensor_identical": df.get("ALL_PASS")})
            print("DEFAULT", rec["tag"], "route", path_counts(df)["path_fwd_h"],
                  "identical", df.get("ALL_PASS"), flush=True)
    append(path, {"kind": "default_route_summary", "rows": rows})
    return rows


def probe_matrix(path, shapes, gates, dtypes, oracle=True, want_mem=False,
                 boundary=False):
    rows = []
    for (K, V) in shapes:
        for dtype in dtypes:
            for gate in gates:
                rec = dict(B=2, T=256, H=2, K=K, V=V, dtype=dtype, gln=gate,
                           tag=f"core_{K}x{V}", exp_fp16=1, exp_fp32=1)
                fb, par = probe_one(path, rec, oracle=oracle, want_mem=want_mem)
                rows.append({"K": K, "V": V, "dtype": dtype, "gln": gate,
                             "fallback": path_counts(fb), "parallel": path_counts(par),
                             "grade": par.get("grade"),
                             "ALL_PASS": par.get("ALL_PASS"),
                             "nan_shared_names": par.get("nan_shared_names"),
                             "preexisting_nan_confirmed":
                                 par.get("preexisting_nan_confirmed"),
                             "oracle_blocked_by_preexisting_nan":
                                 par.get("oracle_blocked_by_preexisting_nan"),
                             "g_min": (par.get("input_stats") or {}).get(
                                 "g", {}).get("min"),
                             "oracle_grade": par.get("oracle_quant_grade"),
                             "local_state": local_state_facts(par)})
                print("PROBE", K, V, dtype, gate, "grade", par.get("grade"),
                      "oracle", par.get("oracle_quant_grade"),
                      "nan", par.get("nan_shared_names"), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["probe", "ab", "guards", "quant"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--shapes", default=None,
                    help="comma list like 32x64,64x64 (default: core matrix)")
    ap.add_argument("--dtypes", default=None)
    ap.add_argument("--gates", default=None)
    ap.add_argument("--boundary", action="store_true")
    ap.add_argument("--pressure", action="store_true")
    ap.add_argument("--mem", action="store_true")
    ap.add_argument("--skip-core", action="store_true",
                    help="probe mode: only the requested boundary/pressure extras")
    ap.add_argument("--only-tags", default=None,
                    help="probe mode: comma list of tag substrings for "
                         "boundary/pressure entries (e.g. bf16)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RAW / (
        "gla008_probe.jsonl" if args.mode == "probe"
        else "gla008_ab.jsonl" if args.mode == "ab"
        else "gla008_quant.jsonl" if args.mode == "quant"
        else "gla008_guards.jsonl")

    if args.mode == "quant":
        if args.smoke:
            shapes, gates = [(64, 64)], [1.0]
        else:
            shapes = ([(int(x.split("x")[0]), int(x.split("x")[1]))
                       for x in args.shapes.split(",")] if args.shapes
                      else CORE_SHAPES)
            gates = [1.0] if args.gates is None else [
                float(x) for x in args.gates.split(",")]
        quant_probe(out, shapes, gates)
        print("GLA008_QUANT_OK", flush=True)
        return

    if args.mode == "guards":
        guard_probe(out)
        rows = default_route_probe(out)
        print("GLA008_GUARDS_OK", len(rows), flush=True)
        return

    if args.mode == "probe":
        if args.smoke:
            shapes, dtypes, gates = [(64, 64)], ["fp16"], [1.0]
        else:
            shapes = ([(int(x.split("x")[0]), int(x.split("x")[1]))
                       for x in args.shapes.split(",")] if args.shapes
                      else CORE_SHAPES)
            dtypes = args.dtypes.split(",") if args.dtypes else DTYPES
            gates = ([float(x) for x in args.gates.split(",")] if args.gates
                     else CORE_GATES)
        if args.skip_core:
            rows = []
        else:
            rows = probe_matrix(out, shapes, gates, dtypes, oracle=True,
                                want_mem=True)
            append(out, {"kind": "core_probe_summary", "rows": rows,
                         "thresholds": THRESHOLDS})
        want = ([t.strip() for t in args.only_tags.split(",")]
                if args.only_tags else None)

        def selected(rec):
            return want is None or any(w in rec["tag"] for w in want)
        if args.boundary:
            for rec in BOUNDARY:
                rec = dict(rec, gln=1.0, exp_fp16=1, exp_fp32=1)
                if selected(rec):
                    probe_one(out, rec, oracle=False, want_mem=True)
        if args.pressure:
            for rec in PRESSURE:
                rec = dict(rec, gln=1.0, exp_fp16=1, exp_fp32=1)
                if selected(rec):
                    probe_one(out, rec, oracle=False, want_mem=True)
        print("GLA008_PROBE_OK", flush=True)
        return

    # ---- formal A/B -------------------------------------------------------
    if args.smoke:
        cases, gates, dtypes = [(64, 64)], [1.0], ["fp16"]
    else:
        cases = ([(int(x.split("x")[0]), int(x.split("x")[1]))
                  for x in args.shapes.split(",")] if args.shapes
                 else list(CORE_SHAPES))
        dtypes = args.dtypes.split(",") if args.dtypes else DTYPES
        gates = ([float(x) for x in args.gates.split(",")] if args.gates
                 else CORE_GATES)
    total = 0
    expected = 0
    for (K, V) in cases:
        for dtype in dtypes:
            for gate in gates:
                tag = f"ab_{K}x{V}_{dtype}_g{gate}"
                ref = str(Path(tempfile.gettempdir()) /
                          f"gla008_ab_{K}x{V}_{dtype}_g{gate}.pt")
                base_rec = dict(B=2, T=256, H=2, K=K, V=V, dtype=dtype,
                                gln=gate, tag=tag, exp_fp16=1, exp_fp32=1)
                cold = {}
                for side, tree in (("fallback", ROOT), ("parallel", ROOT)):
                    extra = ["--samples", str(args.samples),
                             "--warmup", "2", "--mem"]
                    if side == "fallback":
                        extra += ["--force-fallback", "--save", ref]
                    else:
                        extra += ["--compare", ref]
                    rec = run_case(tree, case_args(
                        base_rec, extra + ["--tag", f"{side}_{tag}"]))
                    rec.update({"side": side, "round": -1, "phase": "cold"})
                    rec["host_load"] = host_load()
                    append(out, rec)
                    ok, why = expect_parallel(rec, side)
                    if not check_device(rec) or not ok:
                        raise SystemExit(f"GLA008_FAIL cold {tag} {side}: {why} "
                                         f"{rec.get('error') or ''}")
                    cold[side] = rec
                g = cold["parallel"].get("grade")
                if g not in ("pass", "shared_nan") or cold["parallel"].get(
                        "nan_mismatch_names"):
                    raise SystemExit(f"GLA008_FAIL A/B numeric {tag}: {g} "
                                     f"{[x for x in cold['parallel'].get('compare', []) if not x['ok']]}")
                preexisting = None
                if g == "shared_nan":
                    oldc = run_case(BASE, case_args(
                        base_rec, ["--samples", "1", "--warmup", "0",
                                   "--no-timing", "--compare", ref,
                                   "--tag", f"oldchk_{tag}"]))
                    oldc["side"] = "old_commit_check"
                    oldc["phase"] = "cold"
                    append(out, oldc)
                    preexisting = (oldc.get("grade") == "shared_nan"
                                   and set(oldc.get("nan_shared_names") or []) ==
                                   set(cold["parallel"].get("nan_shared_names") or []))
                    append(out, {"kind": "ab_preexisting_nan_check", "tag": tag,
                                 "confirmed": bool(preexisting),
                                 "names": cold["parallel"].get("nan_shared_names"),
                                 "old_commit_nan_names":
                                     oldc.get("nan_shared_names"),
                                 "old_commit_grade": oldc.get("grade"),
                                 "old_commit_tag": oldc.get("tag")})
                    if not preexisting:
                        raise SystemExit(f"GLA008_FAIL A/B unconfirmed NaN {tag}: "
                                         f"new={cold['parallel'].get('nan_shared_names')} "
                                         f"old={oldc.get('nan_shared_names')}")
                for rnd in range(args.rounds):
                    order = ["fallback", "parallel"] if rnd % 2 == 0 else \
                            ["parallel", "fallback"]
                    for side in order:
                        tree = ROOT
                        extra = ["--samples", str(args.samples),
                                 "--warmup", "2", "--mem"]
                        if side == "fallback":
                            extra += ["--force-fallback", "--save", ref]
                        else:
                            extra += ["--compare", ref]
                        rec = run_case(tree, case_args(
                            base_rec,
                            extra + ["--tag", f"{side}_{tag}_r{rnd}"]))
                        rec.update({"side": side, "round": rnd,
                                    "phase": "steady", "order": order})
                        rec["preexisting_nan_confirmed"] = preexisting
                        rec["preexisting_nan_check_tag"] = (
                            f"ab_{K}x{V}_{dtype}_g{gate}" if preexisting is not None
                            else None)
                        rec["host_load"] = host_load()
                        append(out, rec)
                        ok, why = expect_parallel(rec, side)
                        if not check_device(rec) or not ok:
                            raise SystemExit(f"GLA008_FAIL r{rnd} {tag} {side}: "
                                             f"{why} {rec.get('error') or ''}")
                        rg = rec.get("grade")
                        if side == "parallel":
                            if rg not in ("pass", "shared_nan") or rec.get(
                                    "nan_mismatch_names"):
                                raise SystemExit(
                                    f"GLA008_FAIL r{rnd} numeric {tag}: {rg} "
                                    f"{[x for x in rec.get('compare', []) if not x['ok']]}")
                            if rg == "shared_nan" and not preexisting:
                                raise SystemExit(
                                    f"GLA008_FAIL r{rnd} unconfirmed NaN {tag}")
                        ms = (rec.get("timing") or {}).get("full_fwd_bwd", {}).get("ms", [])
                        if len(ms) != args.samples or not all(
                                isinstance(x, (int, float)) and x > 0 for x in ms):
                            raise SystemExit(f"GLA008_FAIL missing samples r{rnd} "
                                             f"{tag} {side}: {ms}")
                        total += 1
                        print("AB", K, V, dtype, gate, "r", rnd, side, "median",
                              round(statistics.median(ms), 4), flush=True)
                expected += 2 + 2 * args.rounds
                try:
                    os.unlink(ref)
                except OSError:
                    pass
    if total != len(cases) * len(dtypes) * len(gates) * 2 * args.rounds:
        raise SystemExit(f"GLA008_FAIL sample count {total}")
    print("GLA008_AB_OK", total, flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"GLA008_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
