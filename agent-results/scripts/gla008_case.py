"""GLA-008 single-case probe: same FP64 base, dtype quantisation, real route.

One process = one (tree, dtype, route, shape, gate).  The route is forced from
the command line so a fallback row can never be mistaken for a parallel one:

  --force-fallback   sets FLAGGEMS_ENFLAME_GLA_PARALLEL_STATE=0 and
                     FLAGGEMS_ENFLAME_GLA_PARALLEL_BWD_STATE=0
  default            uses the experiment opt-ins requested on the command line
                     (FP16/FP32) or the production default (BF16, whitelist)

All kernel objects are wrapped so actual ``[grid]`` launches are counted and
the launched ``local_state`` buffer dtype/bytes are recorded.  Inputs are
generated once in FP64 on CPU from a fixed seed and then quantised per dtype,
so every dtype sees the same underlying values.

The independent FP64 CPU recurrence (``--oracle``) is a plain step-by-step
reference; it is not the old fallback implementation.  ``--oracle-base`` runs
the same recurrence on the unquantised FP64 base, which separates input
representation error from kernel/path error.

Only GCU 4 (PCI 63:00.0) is used; ``torch.gcu.set_device(4)`` runs before the
FlagAttention import and every tensor/stream device is checked.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import triton

p = argparse.ArgumentParser()
p.add_argument("--B", type=int, default=2)
p.add_argument("--T", type=int, default=256)
p.add_argument("--H", type=int, default=2)
p.add_argument("--K", type=int, default=64)
p.add_argument("--V", type=int, default=64)
p.add_argument("--gln", type=float, default=1.0)
p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
p.add_argument("--seed", type=int, default=42)
p.add_argument("--exp-fp16", type=int, default=0)
p.add_argument("--exp-fp32", type=int, default=0)
p.add_argument("--tag", default="run")
p.add_argument("--save", default=None)
p.add_argument("--compare", default=None)
p.add_argument("--oracle", action="store_true")
p.add_argument("--oracle-base", action="store_true")
p.add_argument("--samples", type=int, default=5)
p.add_argument("--warmup", type=int, default=2)
p.add_argument("--no-timing", action="store_true")
p.add_argument("--mem", action="store_true")
p.add_argument("--force-fallback", action="store_true")
p.add_argument("--f32-default", default="ieee", choices=["ieee", "tf32", "unset"])
a = p.parse_args()

os.environ["FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP16_STATE"] = str(a.exp_fp16)
os.environ["FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP32_STATE"] = str(a.exp_fp32)
if a.f32_default != "unset":
    os.environ["TRITON_F32_DEFAULT"] = a.f32_default
if a.force_fallback:
    os.environ["FLAGGEMS_ENFLAME_GLA_PARALLEL_STATE"] = "0"
    os.environ["FLAGGEMS_ENFLAME_GLA_PARALLEL_BWD_STATE"] = "0"

torch.gcu.set_device(4)

from flag_attn import chunk_gla  # noqa: E402
import importlib  # noqa: E402

C = importlib.import_module("flag_attn.runtime.backend._enflame.FLA.gla.chunk_gla")
Hmod = importlib.import_module("flag_attn.runtime.backend._enflame.FLA.chunk_h")

DEV = torch.device("gcu:4")
B, T, H, K, V = a.B, a.T, a.H, a.K, a.V
DT = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]

# Declared before the run and also written into every record.  A tensor passes
# when it is finite, its relative RMS error is within the dtype threshold, and
# its absolute error is within max(floor, 10 * threshold * reference_absmax).
THRESHOLDS = {
    "fp16": {"path_rel": 4e-3, "oracle_rel": 6e-3, "abs_floor": 1e-2},
    "bf16": {"path_rel": 2e-2, "oracle_rel": 3e-2, "abs_floor": 2e-2},
    "fp32": {"path_rel": 2e-5, "oracle_rel": 3e-5, "abs_floor": 1e-5},
}
TH = THRESHOLDS[a.dtype]

ATTEMPT, DONE = {}, {}
LOCAL_STATE = []
PROBE_NAMES = (
    "_chunk_fwd_local_state_kernel", "_chunk_bwd_local_state_kernel",
    "_chunk_fwd_state_scan_kernel", "_chunk_fwd_state_scan_fp32_2d_group_kernel",
    "_chunk_bwd_state_scan_kernel", "_chunk_bwd_state_scan_fp32_2d_group_kernel",
    "chunk_fwd_kernel_h", "chunk_bwd_kernel_dh",
)


class _Probe:
    def __init__(self, name, real):
        self.name = name
        self.real = real

    def __getitem__(self, grid):
        launcher = self.real[grid]

        def call(*args, **kwargs):
            ATTEMPT[self.name] = ATTEMPT.get(self.name, 0) + 1
            state = kwargs.get("local_state")
            if state is not None:
                LOCAL_STATE.append({
                    "kernel": self.name,
                    "dtype": str(state.dtype),
                    "element_size": state.element_size(),
                    "numel": state.numel(),
                    "bytes": state.numel() * state.element_size(),
                    "shape": list(state.shape),
                    "device": str(state.device),
                })
            out = launcher(*args, **kwargs)
            DONE[self.name] = DONE.get(self.name, 0) + 1
            return out

        return call


for _n in PROBE_NAMES:
    if hasattr(Hmod, _n):
        setattr(Hmod, _n, _Probe(_n, getattr(Hmod, _n)))

result = {
    "tag": a.tag, "shape": [K, V], "B": B, "T": T, "H": H,
    "dtype": a.dtype, "gln": a.gln, "seed": a.seed,
    "exp_fp16": a.exp_fp16, "exp_fp32": a.exp_fp32,
    "force_fallback": bool(a.force_fallback),
    "pythonpath_head": os.environ.get("PYTHONPATH", "").split(":")[0],
    "chunk_h_file": Hmod.__file__,
    "device": "gcu:4",
    "gcu_current_device": None,
    "stream_device": None,
    "tensor_devices_ok": None,
    "device_ok": None,
    "triton_version": triton.__version__,
    "triton_f32_default": os.environ.get("TRITON_F32_DEFAULT", "<unset>"),
    "thresholds": TH,
    "exit": 0,
}


def _grads():
    return dict(o=o, ht=ht, dq=q.grad, dk=k.grad, dv=v.grad, dg=g.grad,
                dh0=h0.grad)


def _tensor_stat(t):
    return {
        "dtype": str(t.dtype),
        "shape": list(t.shape),
        "numel": t.numel(),
        "bytes": t.numel() * t.element_size(),
    }


def _compare(cur, ref, kind):
    """NaN-aware comparison.

    A NaN is never "close to" anything, so the masks are compared first.  If the
    NaN patterns differ the row fails outright; if they agree the finite entries
    are compared and the row is reported as ``nan_shared``.  This keeps a
    pre-existing shared-kernel overflow from being confused with a dispatch
    difference, and keeps a dispatch difference from hiding behind NaNs.
    """
    rel_thr = TH["path_rel"] if kind == "path" else TH["oracle_rel"]
    rows = []
    for name in cur:
        if name not in ref:
            continue
        x = cur[name].detach().float().cpu()
        y = ref[name].detach().float().cpu()
        nan_x = torch.isnan(x)
        nan_y = torch.isnan(y)
        nan_equal = bool(torch.equal(nan_x, nan_y))
        n_nan = int(nan_x.sum())
        diff = (x - y).abs()
        max_abs = diff.max().item() if diff.numel() else 0.0
        rel = ((x - y).square().mean().sqrt()
               / x.square().mean().sqrt().clamp_min(1e-8)).item()
        if nan_equal and n_nan:
            keep = ~nan_x
            if bool(keep.any()):
                xf, yf = x[keep], y[keep]
                diff = (xf - yf).abs()
                max_abs = diff.max().item()
                rel = ((xf - yf).square().mean().sqrt()
                       / xf.square().mean().sqrt().clamp_min(1e-8)).item()
            else:
                max_abs, rel = 0.0, 0.0
        flat = int(diff.argmax().item()) if diff.numel() else 0
        idx = []
        rem = flat
        for dim in reversed(x.shape):
            idx.append(rem % dim)
            rem //= dim
        idx.reverse()
        finite_cur = bool(torch.isfinite(x).all())
        finite_ref = bool(torch.isfinite(y).all())
        ref_absmax = y.abs().max().item() if y.numel() else 0.0
        abs_cap = max(TH["abs_floor"], 10.0 * rel_thr * ref_absmax)
        ok = (finite_cur and finite_ref and rel <= rel_thr
              and max_abs <= abs_cap)
        rows.append({
            "name": name, "kind": kind,
            "max_abs": max_abs, "rel_rms": rel,
            "nan_shared": bool(nan_equal and n_nan > 0),
            "nan_count_cur": n_nan, "nan_count_ref": int(nan_y.sum()),
            "nan_mask_equal": nan_equal,
            "finite_fraction_cur": float(torch.isfinite(x).float().mean()),
            "finite_fraction_ref": float(torch.isfinite(y).float().mean()),
            "nan_inf": not (finite_cur and finite_ref),
            "ref_absmax": ref_absmax, "abs_cap": abs_cap,
            "max_abs_index": idx,
            "ref_shape": list(y.shape), "cur_shape": list(x.shape),
            "cur_dtype": str(cur[name].dtype), "ref_dtype": str(ref[name].dtype),
            "ok": ok,
        })
    return rows


def _oracle(q64, k64, v64, g64, h0_64, do64, dht64, need_h=False):
    qr = q64.clone().requires_grad_()
    kr = k64.clone().requires_grad_()
    vr = v64.clone().requires_grad_()
    gr = g64.clone().requires_grad_()
    hr = h0_64.clone().requires_grad_()
    state = hr
    outs = []
    chunk_states = []
    with torch.enable_grad():
        for i in range(T):
            if need_h and i % 64 == 0:
                chunk_states.append(state)
            state = state * gr[:, i].exp().unsqueeze(-1)
            state = state + kr[:, i].unsqueeze(-1) * vr[:, i].unsqueeze(-2)
            outs.append(torch.einsum("bhk,bhkv->bhv", qr[:, i] * (K ** -0.5),
                                     state))
        o_ref = torch.stack(outs, dim=1)
        ht_ref = state
        loss = (o_ref * do64).sum() + (ht_ref * dht64).sum()
        loss.backward()
    out = dict(o=o_ref.detach(), ht=ht_ref.detach(), dq=qr.grad, dk=kr.grad,
               dv=vr.grad, dg=gr.grad, dh0=hr.grad)
    if need_h:
        out["h"] = torch.stack(chunk_states, dim=1)
    return out


def _grade(rows):
    """Overall verdict for a row set.

    ``pass``           every tensor finite and within threshold;
    ``shared_nan``     NaN masks identical on both sides with finite entries
                       within threshold (a shared-kernel defect, not dispatch);
    ``fail``           anything else, including differing NaN masks.
    """
    if not rows:
        return "fail"
    bad = [r for r in rows if not r["ok"]]
    if not bad:
        return "pass"
    if all(r["nan_shared"] and r["nan_mask_equal"] for r in bad) and all(
            r["rel_rms"] <= (TH["path_rel"] if r["kind"] == "path"
                             else TH["oracle_rel"]) for r in bad):
        return "shared_nan"
    return "fail"


t_start = time.perf_counter()
try:
    # ---- same FP64 base, quantised per dtype -----------------------------
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    q64 = torch.rand(B, T, H, K, dtype=torch.float64, generator=gen)
    k64 = torch.rand(B, T, H, K, dtype=torch.float64, generator=gen)
    v64 = torch.rand(B, T, H, V, dtype=torch.float64, generator=gen)
    glog64 = torch.rand(B, T, H, K, dtype=torch.float64, generator=gen)
    g64 = F.logsigmoid(glog64) / a.gln
    do64 = torch.randn(B, T, H, V, dtype=torch.float64, generator=gen)
    h064 = torch.rand(B, H, K, V, dtype=torch.float64, generator=gen)
    dht64 = torch.randn(B, H, K, V, dtype=torch.float64, generator=gen)
    base_blob = b"".join(t.contiguous().numpy().tobytes() for t in
                         (q64, k64, v64, g64, do64, h064, dht64))
    result["base_sha256"] = hashlib.sha256(base_blob).hexdigest()

    q = q64.to(DT).to(DEV).requires_grad_()
    k = k64.to(DT).to(DEV).requires_grad_()
    v = v64.to(DT).to(DEV).requires_grad_()
    g = g64.to(DT).to(DEV).requires_grad_()
    h0 = h064.to(torch.float32).to(DEV).requires_grad_()
    do = do64.to(DT).to(DEV)
    dht = dht64.to(torch.float32).to(DEV)

    tensors = dict(q=q, k=k, v=v, g=g, h0=h0, do=do, dht=dht)
    result["h0_is_fp32"] = (h0.dtype == torch.float32)
    result["dht_is_fp32"] = (dht.dtype == torch.float32)
    result["input_stats"] = {}
    for name, t in tensors.items():
        result["input_stats"][name] = {
            "dtype": str(t.dtype), "min": float(t.min()), "max": float(t.max()),
            "absmax": float(t.abs().max()), "finite": bool(torch.isfinite(t).all()),
        }
    result["gcu_current_device"] = int(torch.gcu.current_device())
    stream = torch.gcu.current_stream(4)
    result["stream_device"] = str(stream.device)
    result["tensor_devices_ok"] = all(t.device == DEV for t in tensors.values())
    result["device_ok"] = (result["gcu_current_device"] == 4
                           and stream.device == DEV
                           and result["tensor_devices_ok"])
    if not result["device_ok"]:
        raise RuntimeError(f"device mismatch: {result}")

    scale = K ** -0.5

    def full_fwd_bwd():
        for t in (q, k, v, g, h0):
            t.grad = None
        with torch.enable_grad():
            o, ht = chunk_gla(q=q, k=k, v=v, g=g, scale=scale,
                              initial_state=h0, output_final_state=True,
                              cu_seqlens=None)
            loss = (o * do).sum() + (ht * dht).sum()
            loss.backward()
        return o, ht

    t0 = time.perf_counter()
    o, ht = full_fwd_bwd()
    torch.gcu.synchronize()
    result["cold_full_fwd_bwd_ms"] = (time.perf_counter() - t0) * 1000

    grads = _grads()
    result["grad_finite"] = {n: bool(torch.isfinite(t).all())
                             for n, t in grads.items()}
    result["all_grads_finite"] = all(result["grad_finite"].values())
    result["grad_stats"] = {n: _tensor_stat(t) for n, t in grads.items()}

    with torch.no_grad():
        qd, kd, vd, gd, h0d = (t.detach() for t in (q, k, v, g, h0))
        g_cumsum, A, h_st, ht_st, o_st = C.chunk_gla_fwd(
            q=qd, k=kd, v=vd, g=gd, g_cumsum=None, scale=scale,
            initial_state=h0d, output_final_state=True, cu_seqlens=None,
            chunk_size=64, chunk_indices=None)
        torch.gcu.synchronize()

        def fwd_h():
            return C.chunk_fwd_h(k=kd, v=vd, g=None, gk=g_cumsum, gv=None,
                                 h0=h0d, output_final_state=True,
                                 cu_seqlens=None, chunk_size=64,
                                 states_in_fp32=False, state_v_first=False)

        def bwd_dh_fp32():
            return C.chunk_bwd_dh(q=qd, k=kd, v=vd, g=None, gk=g_cumsum,
                                  gv=None, do=do, h0=h0d, dht=dht,
                                  scale=scale, cu_seqlens=None, chunk_size=64,
                                  states_in_fp32=True, state_v_first=False)

        def whole_fwd():
            return C.chunk_gla_fwd(q=qd, k=kd, v=vd, g=gd, g_cumsum=None,
                                   scale=scale, initial_state=h0d,
                                   output_final_state=True, cu_seqlens=None,
                                   chunk_size=64, chunk_indices=None)

        def probe(fn):
            ATTEMPT.clear(); DONE.clear()
            out = fn()
            torch.gcu.synchronize()
            return out, {"attempt": dict(ATTEMPT), "done": dict(DONE)}

        # True cold stage calls (first launch includes Triton compilation).
        t0 = time.perf_counter(); h_st, ht_st = fwd_h(); torch.gcu.synchronize()
        result["cold_fwd_h_ms"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter(); dh_st, dh0_st = bwd_dh_fp32()
        torch.gcu.synchronize()
        result["cold_bwd_dh_ms"] = (time.perf_counter() - t0) * 1000

        (h_st, ht_st), fwd_path = probe(fwd_h)
        (dh_st, dh0_st), bwd_path = probe(bwd_dh_fp32)
        _, fwd_all_path = probe(whole_fwd)

        result["path_fwd_h"] = fwd_path
        result["path_bwd_dh"] = bwd_path
        result["path_whole_fwd"] = fwd_all_path
        result["parallel_fwd_h"] = fwd_path["done"].get(
            "_chunk_fwd_local_state_kernel", 0) > 0
        result["parallel_bwd_dh"] = bwd_path["done"].get(
            "_chunk_bwd_local_state_kernel", 0) > 0
        result["fallback_fwd_h"] = fwd_path["done"].get("chunk_fwd_kernel_h", 0)
        result["fallback_bwd_dh"] = bwd_path["done"].get(
            "chunk_bwd_kernel_dh", 0)
        result["local_state_observed"] = list(LOCAL_STATE)
        result["local_state_dtypes"] = sorted({x["dtype"] for x in LOCAL_STATE})
        result["local_state_bytes_observed"] = sorted(
            {x["bytes"] for x in LOCAL_STATE})
        result["local_state_fp32_bytes_theory"] = {
            "fwd": B * ((T + 63) // 64) * H * K * V * 4,
            "bwd": B * ((T + 63) // 64) * H * K * V * 4,
        }
        result["state_stats"] = {
            "h": _tensor_stat(h_st), "ht_state": _tensor_stat(ht_st),
            "dh": _tensor_stat(dh_st), "dh0_state": _tensor_stat(dh0_st),
        }
        stage_vals = dict(h=h_st, ht_state=ht_st, dh=dh_st, dh0_state=dh0_st)

        if a.mem:
            mem = {}

            def measure(fn):
                torch.gcu.synchronize()
                torch.gcu.reset_peak_memory_stats()
                before = torch.gcu.memory_allocated()
                out = fn()
                torch.gcu.synchronize()
                peak = torch.gcu.max_memory_allocated()
                after = torch.gcu.memory_allocated()
                reserved = torch.gcu.memory_reserved()
                del out
                return {"before": before, "peak": peak, "after": after,
                        "peak_delta": peak - before, "reserved_after": reserved}

            mem["fwd_h"] = measure(fwd_h)
            mem["bwd_dh_fp32"] = measure(bwd_dh_fp32)
            mem["whole_fwd"] = measure(whole_fwd)
            result["memory"] = mem
            result["memory_reserved"] = torch.gcu.memory_reserved()

    if a.mem:
        def measure_full():
            torch.gcu.synchronize()
            torch.gcu.reset_peak_memory_stats()
            before = torch.gcu.memory_allocated()
            with torch.enable_grad():
                o, ht = full_fwd_bwd()
            torch.gcu.synchronize()
            peak = torch.gcu.max_memory_allocated()
            after = torch.gcu.memory_allocated()
            del o, ht
            return {"before": before, "peak": peak, "after": after,
                    "peak_delta": peak - before,
                    "reserved_after": torch.gcu.memory_reserved()}

        result["full_call_memory"] = measure_full()

        if not a.no_timing:
            stages = {"fwd_h": fwd_h, "bwd_dh_fp32": bwd_dh_fp32,
                      "whole_fwd": whole_fwd}
            for _ in range(a.warmup):
                for f in stages.values():
                    f()
            torch.gcu.synchronize()
            timing = {}
            for name, f in stages.items():
                ts = []
                for _ in range(a.samples):
                    torch.gcu.synchronize()
                    t0 = time.perf_counter()
                    f()
                    torch.gcu.synchronize()
                    ts.append((time.perf_counter() - t0) * 1000)
                s = sorted(ts)
                timing[name] = {"ms": ts, "median": s[len(s) // 2], "min": s[0]}
            result["timing"] = timing

    if not a.no_timing:
        for _ in range(a.warmup):
            full_fwd_bwd()
        torch.gcu.synchronize()
        ts = []
        for _ in range(a.samples):
            torch.gcu.synchronize()
            t0 = time.perf_counter()
            full_fwd_bwd()
            torch.gcu.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        s = sorted(ts)
        result.setdefault("timing", {})["full_fwd_bwd"] = {
            "ms": ts, "median": s[len(s) // 2], "min": s[0]}

        ATTEMPT.clear(); DONE.clear(); LOCAL_STATE.clear()
        o2, ht2 = full_fwd_bwd()
        torch.gcu.synchronize()
        result["path_full_autograd"] = {"attempt": dict(ATTEMPT),
                                       "done": dict(DONE)}
        result["local_state_autograd"] = list(LOCAL_STATE)
        del o2, ht2

    # Final gradient set used for references / oracle comparisons.
    grads = _grads()
    current = {n: t.detach().float().cpu() for n, t in grads.items()}
    current.update({n: t.detach().float().cpu() for n, t in stage_vals.items()})

    if a.save:
        torch.save(current, a.save)
        result["saved"] = a.save

    if a.oracle:
        t0 = time.perf_counter()
        ref_q = _oracle(
            q64.to(DT).double(), k64.to(DT).double(), v64.to(DT).double(),
            g64.to(DT).double(), h064.to(torch.float32).double(),
            do64.to(DT).double(), dht64.to(torch.float32).double(),
            need_h=True)
        result["oracle_quant_s"] = time.perf_counter() - t0
        rows = _compare(current, ref_q, "oracle")
        result["oracle_quant"] = rows
        result["oracle_quant_grade"] = _grade(rows)
        result["oracle_nan_shared"] = all(r["nan_shared"] for r in rows)
        result["oracle_nan_counts"] = {r["name"]: r["nan_count_cur"] for r in rows
                                       if r["nan_count_cur"]}
        result["PASS_ORACLE_QUANT"] = all(r["ok"] for r in rows)
        if a.oracle_base:
            t0 = time.perf_counter()
            ref_b = _oracle(q64, k64, v64, g64, h064, do64, dht64,
                            need_h=True)
            result["oracle_base_s"] = time.perf_counter() - t0
            qrows = _compare(ref_q, ref_b, "oracle")
            result["quant_effect"] = qrows
            result["PASS_QUANT_MONOTONIC"] = None
            rows_b = _compare(current, ref_b, "oracle")
            result["oracle_base"] = rows_b
            result["PASS_ORACLE_BASE"] = all(r["ok"] for r in rows_b)

    if a.compare:
        ref = torch.load(a.compare, map_location="cpu")
        rows = _compare(current, ref, "path")
        result["compare"] = rows
        result["grade"] = _grade(rows)
        result["nan_shared_names"] = sorted(
            {r["name"] for r in rows if r["nan_shared"]})
        result["nan_mismatch_names"] = sorted(
            {r["name"] for r in rows
             if r["nan_count_cur"] != r["nan_count_ref"]})
        result["ALL_PASS"] = all(r["ok"] for r in rows)
        result["PASS_PATH"] = result["ALL_PASS"]
        missing = [n for n in ref if n not in current]
        result["missing_tensors"] = missing
        if missing:
            result["ALL_PASS"] = False
except Exception as exc:
    import traceback
    result["exit"] = 1
    result["error"] = f"{type(exc).__name__}: {exc}"
    result["traceback"] = traceback.format_exc()[-6000:]
    result["path_attempt_at_error"] = dict(ATTEMPT)
    result["path_done_at_error"] = dict(DONE)

result["wall_s"] = time.perf_counter() - t_start
print("GLA008_JSON " + json.dumps(result))
sys.exit(result["exit"])
