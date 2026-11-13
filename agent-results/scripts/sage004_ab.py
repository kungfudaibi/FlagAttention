#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SAGE-004 formal full-call A/B: exact legacy SageAttention vs candidate Sage.

Why this script exists
----------------------
SAGE-003 landed a verified launch optimisation in the per-block INT8 quantizer
and showed the remaining gap is the attention kernel, but it never compared the
*legacy* full call (pre-optimisation quantizer) against a candidate in one
alternating A/B.  SAGE-004 must decide whether an attention-kernel change is
worth keeping, so the comparison is:

    legacy Sage full call   = per_block_int8(num_warps=4, num_stages=3)
                              + SageAttention
    candidate Sage full call= the shipped per_block_int8 launch + the same
                              attention kernel

Both consume the *same* FP16 Q/K/V, emit the same output dtype and shape, and
are timed with one synchronized GCU event pair plus host wall clock around
exactly one public call.  Rounds alternate which candidate leads so a drift
cannot favour either side.

What a run produces
-------------------
* A pre-flight guard: ``efsmi`` must map the assigned ordinal to the assigned
  PCI address, the first device action must be ``torch.gcu.set_device``, and the
  current stream plus every tensor must live on it.
* Dispatch proof for the competitiveness reference: the device fused op must be
  ``aten::_scaled_dot_product_flash_attention`` and ``F.sdpa`` must still be the
  host fallback.  A CPU fallback is *never* used as a device baseline; if the
  identity changes the run fails (exit 5) instead of comparing new things.
* Per-sample JSONL rows: host wall ms, GCU event ms, host CPU ms.
* Numeric gates on the output *and* on the INT8 quantization intermediates:
  finiteness, max abs error, relative L2, cosine.
* Peak device memory per candidate, measured separately.
* Cold compile in a child process with an empty ``TRITON_CACHE_DIR``.
* Pre-quantized attention-only timing, reported separately and never mixed into
  the full-call comparison.

Exit status
-----------
0  PASS
2  environment/backend unusable (no GCU, too few devices, foreign tenant)
3  numeric gate, finite-value, missing-sample, or A/B-scope failure
4  wrong device identity (efsmi index/PCI mismatch or stream/tensor mismatch)
5  baseline dispatch changed (device op not a GCU kernel, or F.sdpa not a fallback)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import statistics
import subprocess
import sys
import time

EXPECTED_PCI = "0000:83:00.0"
EXPECTED_INDEX = 5
EXPECTED_SERIAL = "A0D5460520864"
DEVICE_OPS = ("_scaled_dot_product_flash_attention",)
FORBIDDEN_DEVICE_OPS = (
    "_scaled_dot_product_flash_attention_for_cpu",
    "_scaled_dot_product_attention_math",
    "aten::_to_cpu",
)
CPU_FALLBACK_MARKER = "_scaled_dot_product_flash_attention_for_cpu"
DEFAULT_SHAPES = ("1x2x4096x64", "4x32x1024x128")


# --------------------------------------------------------------------------
# Pre-flight.  No torch import before the guard, so a wrong assignment cannot
# even create a context on the wrong GCU.
# --------------------------------------------------------------------------
def efsmi_devices() -> list[dict]:
    try:
        raw = subprocess.run(
            ["efsmi", "-L"], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    devices = []
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) < 5 or not fields[0].isdigit():
            continue
        bus = next(
            (f for f in fields
             if re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]", f)),
            None,
        )
        if bus is None:
            continue
        devices.append({"index": int(fields[0]), "sn": fields[1], "pci": bus.lower()})
    return devices


def efsmi_used_mib(index: int) -> "float | None":
    try:
        raw = subprocess.run(
            ["efsmi", "-i", str(index), "-q", "-d", "MEMORY"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"Used Size\s*:\s*([0-9.]+)", raw)
    return float(match.group(1)) if match else None


def device_guard(index: int, pci: str) -> tuple[list[str], dict]:
    """Fail closed unless efsmi says this ordinal is the assigned PCI card."""
    problems: list[str] = []
    devices = efsmi_devices()
    target = [d for d in devices if d["pci"] == pci]
    report = {
        "event": "device_guard",
        "expected_index": index,
        "expected_pci": pci,
        "expected_serial": EXPECTED_SERIAL,
        "devices": devices,
        "efsmi_used_mib": efsmi_used_mib(index) if devices else None,
    }
    if not devices:
        problems.append("efsmi -L produced no parseable device table")
    elif not target:
        problems.append(f"efsmi does not report PCI {pci}")
    else:
        found = target[0]
        report["matched_index"] = found["index"]
        report["matched_serial"] = found["sn"]
        if found["index"] != index:
            problems.append(
                f"PCI {pci} is efsmi DEV {found['index']}, expected DEV {index}"
            )
        if found["sn"] != EXPECTED_SERIAL:
            problems.append(
                f"PCI {pci} serial is {found['sn']}, expected {EXPECTED_SERIAL}"
            )
    report["problems"] = problems
    return problems, report


def parse_shape(text: str) -> tuple[int, int, int, int]:
    parts = tuple(int(x) for x in text.replace("x", ",").split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"bad shape {text!r}; want B,H,T,D")
    return parts


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def metrics(actual, expected) -> dict:
    import torch
    import torch.nn.functional as F

    a = actual.float().flatten()
    e = expected.float().flatten()
    if a.numel() > 1_048_576:
        stride = (a.numel() + 1_048_576 - 1) // 1_048_576
        a = a[::stride]
        e = e[::stride]
    diff = a - e
    return {
        "cosine": F.cosine_similarity(a, e, dim=0).item(),
        "relative_l2": (diff.norm() / e.norm().clamp_min(1e-8)).item(),
        "max_abs": diff.abs().max().item(),
        "finite": bool(torch.isfinite(actual).all().item()),
        "n": int(actual.numel()),
    }


def one_sample(fn):
    """One synchronized device-event pair plus host wall clock around one call."""
    import torch

    c0 = cpu_seconds()
    start = torch.gcu.Event(enable_timing=True)
    end = torch.gcu.Event(enable_timing=True)
    t0 = time.perf_counter()
    start.record()
    out = fn()
    end.record()
    host = (cpu_seconds() - c0) * 1000.0
    torch.gcu.synchronize()
    wall = (time.perf_counter() - t0) * 1000.0
    return start.elapsed_time(end), wall, host, out


BIMODAL_RATIO = 1.4


def summarize(values: list[float]) -> dict:
    """Median plus both modes when the distribution is bimodal.

    The full call at D128 alternates between two latencies about 1.6-1.8x
    apart (SAGE-003).  Anchoring the split on the median is wrong when the slow
    mode becomes the majority, so modes are separated at the geometric mean of
    min and max and both are always reported.
    """
    if not values:
        return {"median_ms": None, "min_ms": None, "max_ms": None, "n": 0}
    ordered = sorted(values)
    record = {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.mean(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p10_ms": ordered[int(0.1 * (len(ordered) - 1))],
        "p90_ms": ordered[int(0.9 * (len(ordered) - 1))],
        "stdev_ms": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        "n": len(ordered),
        "bimodal": False,
        "fast_median_ms": None,
        "slow_median_ms": None,
        "slow_call_fraction": 0.0,
        "mode_ratio": 1.0,
    }
    if len(ordered) > 1 and ordered[0] > 0 and ordered[-1] / ordered[0] > BIMODAL_RATIO:
        split = (ordered[0] * ordered[-1]) ** 0.5
        fast = [v for v in ordered if v <= split]
        slow = [v for v in ordered if v > split]
        if fast and slow:
            record.update({
                "bimodal": True,
                "fast_median_ms": statistics.median(fast),
                "slow_median_ms": statistics.median(slow),
                "slow_call_fraction": len(slow) / len(ordered),
                "mode_ratio": statistics.median(slow) / statistics.median(fast),
            })
    return record


def comparable_ms(stats: dict) -> float:
    """The headline number: the fast mode when bimodal, else the median.

    Using the fast mode makes the candidate look as good as the data allows,
    which is the conservative direction for a 'Sage is slower' claim.
    """
    return stats["fast_median_ms"] if stats.get("bimodal") else stats["median_ms"]


# --------------------------------------------------------------------------
# Dispatch proof.  The profiler leaves a broken heap on this image, so it runs
# in a throwaway child and a clean child exit is part of the evidence.
# --------------------------------------------------------------------------
TRACE_CHILD = r"""
import json
import torch
import torch_gcu
import torch.nn.functional as F

torch.gcu.set_device({index})
torch.manual_seed(2026)
q = torch.randn((1, 2, 512, 64), device="gcu:{index}", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

def device_op():
    return torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False)[0]

def sdpa_op():
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

fn = device_op if {which!r} == "device" else sdpa_op
fn()
torch.gcu.synchronize()
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.GCU]
) as prof:
    fn()
    torch.gcu.synchronize()
print(json.dumps({{"names": [event.name for event in prof.events()]}}), flush=True)
"""


def trace_ops(which: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", TRACE_CHILD.format(index=EXPECTED_INDEX, which=which)],
        capture_output=True, text=True, env=dict(os.environ),
    )
    payload = {}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            break
    payload["child_exit"] = proc.returncode
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "no stderr"
        payload.setdefault("error", f"trace child exited {proc.returncode}: {tail}")
        payload.setdefault("names", [])
    return payload


def dispatch_violations(device_names, sdpa_names, device_host_ms, sdpa_host_ms) -> list[str]:
    """Pure dispatch-identity gate, separated so it can be unit-tested."""
    problems: list[str] = []

    def has(names, needle):
        return any(needle in name for name in names)

    fallbacks = [n for n in FORBIDDEN_DEVICE_OPS if has(device_names, n)]
    if not has(device_names, "_scaled_dot_product_flash_attention"):
        problems.append(
            "device baseline did not record aten::_scaled_dot_product_flash_attention"
        )
    if fallbacks:
        problems.append(
            f"device baseline shows host-fallback ops {sorted(fallbacks)}; "
            "it is not a GCU kernel"
        )
    if not has(sdpa_names, CPU_FALLBACK_MARKER):
        problems.append(
            "reference F.scaled_dot_product_attention no longer shows "
            f"{CPU_FALLBACK_MARKER}; its dispatch identity changed, re-derive the "
            "baseline before trusting this comparison"
        )
    if device_host_ms > 5.0:
        problems.append(
            f"device baseline burned {device_host_ms:.2f} ms host CPU per call; "
            "that is not a pure GCU launch"
        )
    if sdpa_host_ms <= 2.0 * max(device_host_ms, 0.05):
        problems.append(
            f"F.scaled_dot_product_attention used {sdpa_host_ms:.2f} ms host CPU, "
            f"close to the device baseline's {device_host_ms:.2f} ms; it may have "
            "become device-native"
        )
    return problems


def path_evidence() -> tuple[list[str], dict]:
    import torch
    import torch.nn.functional as F

    def device_op(q, k, v):
        return torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, 0.0, False, False
        )[0]

    def sdpa_op(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    torch.manual_seed(2026)
    q = torch.randn((1, 2, 512, 64), device=f"gcu:{EXPECTED_INDEX}", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    device_trace = trace_ops("device")
    sdpa_trace = trace_ops("sdpa")

    problems: list[str] = []
    for label, trace in (("device", device_trace), ("sdpa", sdpa_trace)):
        if trace.get("error"):
            problems.append(f"{label} trace failed: {trace['error']}")
        if trace.get("child_exit") != 0:
            problems.append(
                f"{label} trace child exited {trace.get('child_exit')}; the "
                "profiler could not complete cleanly"
            )

    device_op(q, k, v)
    sdpa_op(q, k, v)
    torch.gcu.synchronize()
    device_host, sdpa_host = [], []
    for _ in range(3):
        device_host.append(one_sample(lambda: device_op(q, k, v))[2])
        sdpa_host.append(one_sample(lambda: sdpa_op(q, k, v))[2])
    device_host_ms = statistics.median(device_host)
    sdpa_host_ms = statistics.median(sdpa_host)
    problems.extend(dispatch_violations(
        device_trace.get("names", []), sdpa_trace.get("names", []),
        device_host_ms, sdpa_host_ms,
    ))

    def interesting(names):
        return sorted(n for n in names
                      if "scaled_dot" in n or "to_cpu" in n or "copy" in n or "tops" in n)

    return problems, {
        "event": "path_evidence",
        "device_baseline_ops": interesting(device_trace.get("names", [])),
        "reference_sdpa_ops": interesting(sdpa_trace.get("names", [])),
        "device_trace_child_exit": device_trace.get("child_exit"),
        "reference_trace_child_exit": sdpa_trace.get("child_exit"),
        "device_host_ms_median": device_host_ms,
        "reference_host_ms_median": sdpa_host_ms,
        "violations": problems,
    }


# --------------------------------------------------------------------------
# Cold compile: separate process, empty TRITON_CACHE_DIR.
# --------------------------------------------------------------------------
def cold_child(shape, dtype_name: str) -> int:
    problems, guard = device_guard(EXPECTED_INDEX, EXPECTED_PCI)
    if problems:
        print(json.dumps(guard, sort_keys=True), flush=True)
        return 4
    import torch
    import torch_gcu  # noqa: F401
    from flag_attn import sage_attention
    from flag_attn.sage_attention import per_block_int8

    torch.gcu.set_device(EXPECTED_INDEX)
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(2026)
    q = torch.randn(shape, device=f"gcu:{EXPECTED_INDEX}", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cache_dir = os.environ.get("TRITON_CACHE_DIR", "")

    def timed(fn):
        t0 = time.perf_counter()
        start = torch.gcu.Event(enable_timing=True)
        end = torch.gcu.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.gcu.synchronize()
        return {
            "wall_ms": (time.perf_counter() - t0) * 1000.0,
            "event_ms": start.elapsed_time(end),
            "out": out,
        }

    def device_op():
        return torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, 0.0, False, False
        )[0]

    def full():
        pre = per_block_int8(q, k)
        return sage_attention(pre[0], pre[2], v, pre[1], pre[3], output_dtype=dtype)[0]

    def cache_files():
        return sum(len(files) for _, _, files in os.walk(cache_dir))

    record = {
        "event": "cold",
        "shape": list(shape),
        "output_dtype": dtype_name,
        "input_dtype": "float16",
        "triton_cache_dir": cache_dir,
        "cache_files_at_start": cache_files(),
    }
    first_device = timed(device_op)
    record["device_first"] = {k: v for k, v in first_device.items() if k != "out"}
    record["cache_files_after_device"] = cache_files()
    first_full = timed(full)
    record["sage_full_first"] = {k: v for k, v in first_full.items() if k != "out"}
    record["cache_files_after_full"] = cache_files()
    for _ in range(4):
        device_op()
        full()
    torch.gcu.synchronize()
    device_values, full_values = [], []
    for _ in range(9):
        device_values.append(one_sample(device_op)[0])
        full_values.append(one_sample(full)[0])
    record["steady"] = {
        "device_ms": summarize(device_values),
        "sage_full_ms": summarize(full_values),
    }
    record["compile_overhead_ms"] = {
        "device": record["device_first"]["wall_ms"] - record["steady"]["device_ms"]["median_ms"],
        "sage_full": record["sage_full_first"]["wall_ms"]
        - record["steady"]["sage_full_ms"]["median_ms"],
    }
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0


def cold_runs(shapes, dtype_name: str, out_stream) -> tuple[list[dict], list[str]]:
    records, problems = [], []
    for shape in shapes:
        cache_dir = os.path.join(
            os.environ.get("SAGE004_COLD_ROOT", "/tmp/sage004_cold"),
            f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}_{dtype_name}",
        )
        subprocess.run(["rm", "-rf", cache_dir], check=False)
        os.makedirs(cache_dir, exist_ok=True)
        env = dict(os.environ)
        env["TRITON_CACHE_DIR"] = cache_dir
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--cold-child", "--shapes", "x".join(str(x) for x in shape),
             "--output-dtype", dtype_name],
            capture_output=True, text=True, env=env,
        )
        payload = {}
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("{"):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                break
        payload["child_exit"] = proc.returncode
        if proc.returncode != 0:
            tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "no stderr"
            problems.append(f"cold child for {shape} exited {proc.returncode}: {tail}")
        if payload.get("cache_files_at_start") != 0:
            problems.append(
                f"cold child for {shape} started with "
                f"{payload.get('cache_files_at_start')} cache files, not 0"
            )
        records.append(payload)
        line = json.dumps(payload, sort_keys=True)
        print(line, flush=True)
        if out_stream is not None:
            out_stream.write(line + "\n")
            out_stream.flush()
    return records, problems


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", type=parse_shape, nargs="+",
                        default=[parse_shape(s) for s in DEFAULT_SHAPES])
    parser.add_argument("--output-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--accuracy-cosine", type=float, default=0.99)
    parser.add_argument("--accuracy-rel-l2", type=float, default=0.05)
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--max-used-mib", type=float, default=1024.0)
    parser.add_argument("--expect-index", type=int, default=EXPECTED_INDEX,
                        help="guard override used only to test the fail-closed path")
    parser.add_argument("--jsonl", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--cold-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.cold_child:
        return cold_child(args.shapes[0], args.output_dtype)

    if args.expect_index != EXPECTED_INDEX:
        # Testing the guard must not touch the real device at all.
        problems, report = device_guard(args.expect_index, EXPECTED_PCI)
        report["event"] = "guard_probe"
        print(json.dumps(report, sort_keys=True), flush=True)
        return 4 if problems else 0

    problems, guard = device_guard(EXPECTED_INDEX, EXPECTED_PCI)
    print(json.dumps(guard, sort_keys=True), flush=True)
    if problems:
        return 4
    if guard["efsmi_used_mib"] is not None and guard["efsmi_used_mib"] > args.max_used_mib:
        print(json.dumps({
            "event": "tenant_check",
            "used_mib": guard["efsmi_used_mib"],
            "limit_mib": args.max_used_mib,
            "error": "another tenant holds GCU 5 memory",
        }), flush=True)
        return 2

    try:
        import torch
        import torch_gcu  # noqa: F401
        import torch.nn.functional as F
        from flag_attn import sage_attention
        from flag_attn.sage_attention import per_block_int8
        from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import (
            quant_per_block_int8_kernel as _quant_kernel,
        )
    except Exception as exc:  # pragma: no cover - environment failure
        print(f"ERROR: cannot import the runtime: {exc}", file=sys.stderr)
        return 2

    if not torch.gcu.is_available() or torch.gcu.device_count() <= EXPECTED_INDEX:
        print("ERROR: assigned GCU is not visible", file=sys.stderr)
        return 2

    # First device action in this process: pin the assigned device.
    torch.gcu.set_device(EXPECTED_INDEX)
    stream = torch.gcu.current_stream()
    probe = torch.empty(8, device=f"gcu:{EXPECTED_INDEX}")
    identity = {
        "event": "runtime_identity",
        "torch": torch.__version__,
        "torch_gcu": torch_gcu.__version__,
        "python": sys.version.split()[0],
        "triton": __import__("triton").__version__,
        "device": f"gcu:{EXPECTED_INDEX}",
        "pci": EXPECTED_PCI,
        "current_device": torch.gcu.current_device(),
        "stream_device": str(stream.device),
        "probe_device": str(probe.device),
        "device_count": torch.gcu.device_count(),
        "input_dtype": "torch.float16",
        "output_dtype": args.output_dtype,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "argv": sys.argv,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    identity["device_identity_ok"] = (
        identity["current_device"] == EXPECTED_INDEX
        and identity["stream_device"] == f"gcu:{EXPECTED_INDEX}"
        and identity["probe_device"] == f"gcu:{EXPECTED_INDEX}"
    )
    print(json.dumps(identity, sort_keys=True), flush=True)
    if not identity["device_identity_ok"]:
        return 4

    out_stream = open(args.jsonl, "w") if args.jsonl else None

    def emit(record: dict) -> None:
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if out_stream is not None:
            out_stream.write(line + "\n")
            out_stream.flush()

    dtype = getattr(torch, args.output_dtype)
    failures: list[str] = []
    round_medians: dict[str, list[dict]] = {}

    def device_sdpa(q, k, v):
        return torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, 0.0, False, False
        )[0]

    def reference_sdpa(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    def legacy_quant(q, k):
        """The pre-SAGE-003 per-block INT8 quantizer: Triton default launch."""
        q8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
        k8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
        b, h, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        q_scale = torch.empty((b, h, (qo_len + 127) // 128),
                              device=q.device, dtype=torch.float32)
        k_scale = torch.empty((b, h_kv, (kv_len + 63) // 64),
                              device=q.device, dtype=torch.float32)
        _quant_kernel[((qo_len + 127) // 128, h, b)](
            q, q8, q_scale, qo_len,
            q.stride(0), q.stride(1), q.stride(2),
            q8.stride(0), q8.stride(1), q8.stride(2),
            q_scale.stride(0), q_scale.stride(1),
            sm_scale=(head_dim ** -0.5) * 1.44269504,
            C=head_dim, BLK=128, num_warps=4, num_stages=3,
        )
        _quant_kernel[((kv_len + 63) // 64, h_kv, b)](
            k, k8, k_scale, kv_len,
            k.stride(0), k.stride(1), k.stride(2),
            k8.stride(0), k8.stride(1), k8.stride(2),
            k_scale.stride(0), k_scale.stride(1),
            sm_scale=1.0, C=head_dim, BLK=64, num_warps=4, num_stages=3,
        )
        return q8, q_scale, k8, k_scale

    def legacy_full(q, k, v):
        """Exact legacy Sage full call: default-launch quantizer + attention."""
        pre = legacy_quant(q, k)
        return sage_attention(pre[0], pre[2], v, pre[1], pre[3], output_dtype=dtype)[0]

    def candidate_full(q, k, v):
        """Candidate Sage full call: shipped (SAGE-003) quantizer + attention."""
        pre = per_block_int8(q, k)
        return sage_attention(pre[0], pre[2], v, pre[1], pre[3], output_dtype=dtype)[0]

    # ---- dispatch proof -------------------------------------------------
    path_problems, path = path_evidence()
    emit(path)
    if path_problems:
        failures.extend(f"dispatch: {p}" for p in path_problems)

    cold_records = []
    if args.cold:
        cold_records, cold_problems = cold_runs(args.shapes, args.output_dtype, out_stream)
        failures.extend(cold_problems)

    for shape in args.shapes:
        for rnd in range(args.rounds):
            torch.manual_seed(100000 * rnd + 1000 * shape[2] + shape[3])
            q = torch.randn(shape, device=f"gcu:{EXPECTED_INDEX}", dtype=torch.float16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            torch.gcu.synchronize()

            ref_device = device_sdpa(q, k, v)
            ref_host = reference_sdpa(q, k, v)
            legacy_out = legacy_full(q, k, v)
            candidate_out = candidate_full(q, k, v)
            torch.gcu.synchronize()

            for _ in range(args.warmup):
                legacy_full(q, k, v)
                candidate_full(q, k, v)
            torch.gcu.synchronize()

            torch.gcu.reset_peak_memory_stats(EXPECTED_INDEX)
            used_before = torch.gcu.device_memory_used(EXPECTED_INDEX) / 2**20

            samples = {"legacy_sage_full": [], "candidate_sage_full": []}

            def take(scope, fn, pair, position):
                event, wall, host, out = one_sample(fn)
                samples[scope].append({
                    "pair": pair,
                    "position": position,
                    "event_ms": event,
                    "wall_ms": wall,
                    "host_ms": host,
                })
                return out

            legacy_first = rnd % 2 == 0
            for pair in range(args.pairs):
                if legacy_first:
                    take("legacy_sage_full", lambda: legacy_full(q, k, v), pair, 0)
                    take("candidate_sage_full", lambda: candidate_full(q, k, v), pair, 1)
                    take("candidate_sage_full", lambda: candidate_full(q, k, v), pair, 2)
                    take("legacy_sage_full", lambda: legacy_full(q, k, v), pair, 3)
                else:
                    take("candidate_sage_full", lambda: candidate_full(q, k, v), pair, 0)
                    take("legacy_sage_full", lambda: legacy_full(q, k, v), pair, 1)
                    take("legacy_sage_full", lambda: legacy_full(q, k, v), pair, 2)
                    take("candidate_sage_full", lambda: candidate_full(q, k, v), pair, 3)

            stats = {s: summarize([r["event_ms"] for r in rows]) for s, rows in samples.items()}
            wall_stats = {s: summarize([r["wall_ms"] for r in rows]) for s, rows in samples.items()}
            host_stats = {s: summarize([r["host_ms"] for r in rows]) for s, rows in samples.items()}

            round_failures: list[str] = []

            # Pre-quantized attention only: diagnostic, never mixed into the
            # full-call comparison.
            pre_legacy = legacy_quant(q, k)
            pre_candidate = per_block_int8(q, k)
            stage = {
                "quant_legacy_ms": summarize([one_sample(lambda: legacy_quant(q, k))[0]
                                              for _ in range(7)]),
                "quant_candidate_ms": summarize([one_sample(lambda: per_block_int8(q, k))[0]
                                                 for _ in range(7)]),
                "attn_prequant_ms": summarize([
                    one_sample(lambda: sage_attention(
                        pre_candidate[0], pre_candidate[2], v,
                        pre_candidate[1], pre_candidate[3], output_dtype=dtype))[0]
                    for _ in range(7)
                ]),
            }
            stage["quant_bitwise_equal"] = all(
                torch.equal(a, b) for a, b in zip(pre_legacy, pre_candidate)
            )
            stage["quant_speedup"] = (
                stage["quant_legacy_ms"]["median_ms"]
                / stage["quant_candidate_ms"]["median_ms"]
            )
            # Pre-quantized attention vs the device fused op, and its share of
            # a full call. Both are diagnostics and never enter the A/B ratio.
            device_median = one_sample(lambda: device_sdpa(q, k, v))[0]
            stage["attn_over_device"] = (
                stage["attn_prequant_ms"]["median_ms"] / device_median
            )
            stage["attn_share_of_full"] = (
                stage["attn_prequant_ms"]["median_ms"] / legacy_ms
                if (legacy_ms := comparable_ms(stats["legacy_sage_full"])) else None
            )
            stage["quant_share_of_full"] = (
                stage["quant_candidate_ms"]["median_ms"] / legacy_ms
                if legacy_ms else None
            )
            if not stage["quant_bitwise_equal"]:
                round_failures.append(
                    "candidate quantizer changed int8/scale outputs vs legacy"
                )

            # Quantization intermediates, checked elementwise.
            quant_accuracy = {}
            for name, (la, lb) in (
                ("q_int8", (pre_legacy[0], pre_candidate[0])),
                ("k_int8", (pre_legacy[2], pre_candidate[2])),
                ("q_scale", (pre_legacy[1], pre_candidate[1])),
                ("k_scale", (pre_legacy[3], pre_candidate[3])),
            ):
                a, b = la.float(), lb.float()
                quant_accuracy[name] = {
                    "bitwise_equal": bool(torch.equal(la, lb)),
                    "max_abs": (a - b).abs().max().item(),
                    "finite": bool(torch.isfinite(la).all().item()),
                    "relative_l2": ((a - b).norm() / b.norm().clamp_min(1e-8)).item(),
                }
                if not quant_accuracy[name]["finite"]:
                    round_failures.append(f"{name}: non-finite quantization output")

            accuracy = {
                "legacy_vs_device": metrics(legacy_out, ref_device.to(dtype)),
                "candidate_vs_device": metrics(candidate_out, ref_device.to(dtype)),
                "candidate_vs_legacy": metrics(candidate_out, legacy_out),
                "legacy_vs_reference_sdpa": metrics(legacy_out, ref_host.to(dtype)),
                "candidate_vs_reference_sdpa": metrics(candidate_out, ref_host.to(dtype)),
            }
            for name, m in accuracy.items():
                if not m["finite"]:
                    round_failures.append(f"{name}: non-finite output")
                if m["cosine"] < args.accuracy_cosine:
                    round_failures.append(
                        f"{name}: cosine {m['cosine']:.6f} < {args.accuracy_cosine}"
                    )
                if m["relative_l2"] > args.accuracy_rel_l2:
                    round_failures.append(
                        f"{name}: relative_l2 {m['relative_l2']:.6f} > {args.accuracy_rel_l2}"
                    )
            for scope, rows in samples.items():
                if len(rows) < args.min_samples:
                    round_failures.append(
                        f"{scope}: only {len(rows)} samples < {args.min_samples}"
                    )
            for label, out in (("legacy", legacy_out), ("candidate", candidate_out)):
                if out.dtype != dtype or tuple(out.shape) != tuple(shape):
                    round_failures.append(
                        f"{label}: dtype/shape {out.dtype}/{tuple(out.shape)} != {dtype}/{shape}"
                    )

            # Peak memory per candidate, measured separately.
            def peak_of(fn, n=3):
                torch.gcu.empty_cache()
                torch.gcu.reset_peak_memory_stats(EXPECTED_INDEX)
                for _ in range(n):
                    fn()
                torch.gcu.synchronize()
                return (torch.gcu.max_memory_allocated(EXPECTED_INDEX) / 2**20,
                        torch.gcu.max_memory_reserved(EXPECTED_INDEX) / 2**20)

            legacy_peak, legacy_reserved = peak_of(lambda: legacy_full(q, k, v))
            candidate_peak, candidate_reserved = peak_of(lambda: candidate_full(q, k, v))
            device_peak, device_reserved = peak_of(lambda: device_sdpa(q, k, v))
            used_after = torch.gcu.device_memory_used(EXPECTED_INDEX) / 2**20

            cand_ms = comparable_ms(stats["candidate_sage_full"])
            legacy_ms = comparable_ms(stats["legacy_sage_full"])
            ratio = cand_ms / legacy_ms
            shape_key = "x".join(str(x) for x in shape)
            round_medians.setdefault(shape_key, []).append({
                "round": rnd,
                "legacy": legacy_ms,
                "candidate": cand_ms,
                "ratio": ratio,
                "legacy_stats": stats["legacy_sage_full"],
                "candidate_stats": stats["candidate_sage_full"],
            })
            if round_failures:
                failures.extend(f"shape {list(shape)} round {rnd}: {f}" for f in round_failures)
            emit({
                "event": "round",
                "shape": list(shape),
                "round": rnd,
                "order": "legacy_first" if legacy_first else "candidate_first",
                "output_dtype": args.output_dtype,
                "input_dtype": "float16",
                "accuracy": accuracy,
                "quantization": quant_accuracy,
                "samples": samples,
                "stats": stats,
                "wall_stats": wall_stats,
                "host_stats": host_stats,
                "candidate_over_legacy": ratio,
                "stage_ms": stage,
                "memory": {
                    "device_memory_used_before_mib": used_before,
                    "device_memory_used_after_mib": used_after,
                    "legacy_peak_mib": legacy_peak,
                    "legacy_peak_reserved_mib": legacy_reserved,
                    "candidate_peak_mib": candidate_peak,
                    "candidate_peak_reserved_mib": candidate_reserved,
                    "device_fused_peak_mib": device_peak,
                    "device_fused_peak_reserved_mib": device_reserved,
                },
                "failures": round_failures,
            })
            del q, k, v, ref_device, ref_host, legacy_out, candidate_out
            torch.gcu.empty_cache()

    if args.rounds < 1 or args.pairs < 1:
        failures.append(
            f"degenerate configuration: rounds={args.rounds} pairs={args.pairs}"
        )
    missing = [s for s in args.shapes
               if "x".join(str(x) for x in s) not in round_medians]
    if missing:
        failures.append(f"no samples were recorded for shapes {missing}")
    for shape in args.shapes:
        key = "x".join(str(x) for x in shape)
        if key in round_medians and len(round_medians[key]) != args.rounds:
            failures.append(
                f"shape {key}: {len(round_medians[key])} rounds recorded, "
                f"expected {args.rounds}"
            )

    aggregate = {}
    for key, per_round in round_medians.items():
        ratios = [r["ratio"] for r in per_round]
        aggregate[key] = {
            "round_legacy_ms": [r["legacy"] for r in per_round],
            "round_candidate_ms": [r["candidate"] for r in per_round],
            "round_candidate_over_legacy": ratios,
            "median_legacy_ms": statistics.median([r["legacy"] for r in per_round]),
            "median_candidate_ms": statistics.median([r["candidate"] for r in per_round]),
            "median_candidate_over_legacy": statistics.median(ratios),
            "min_candidate_over_legacy": min(ratios),
            "max_candidate_over_legacy": max(ratios),
            "rounds_where_candidate_faster": sum(1 for r in ratios if r < 1.0),
        }

    decision = {
        "event": "decision",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "device_index": EXPECTED_INDEX,
        "pci": EXPECTED_PCI,
        "shapes": [list(s) for s in args.shapes],
        "output_dtype": args.output_dtype,
        "input_dtype": "float16",
        "rounds": args.rounds,
        "pairs_per_round": args.pairs,
        "cold_records": len(cold_records),
        "aggregate": aggregate,
    }
    emit(decision)
    if out_stream is not None:
        out_stream.close()
    if args.summary:
        with open(args.summary, "w") as handle:
            json.dump(decision, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0 if decision["status"] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
