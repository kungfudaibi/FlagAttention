#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SAGE-003 device-fused SDPA versus Enflame SageAttention alternating A/B.

Why this script exists
----------------------
SAGE-002 compared the Sage full call against two baselines and showed that on
this image ``F.scaled_dot_product_attention`` never reaches the GCU: it
dispatches to ``aten::_scaled_dot_product_flash_attention_for_cpu`` and copies
Q/K/V over PCIe.  That makes it a host fallback, not a device baseline.  This
script therefore *proves* the dispatch path first and only then compares
against ``aten::_scaled_dot_product_flash_attention``, the GCU-native fused
op.

What a run produces
-------------------
* A pre-flight guard: efsmi must map the assigned index to the assigned PCI
  address, ``torch.gcu.set_device`` must be the first device action, and the
  current stream plus every tensor must live on that device.
* Path evidence from the torch profiler plus host-CPU time per call: the
  device baseline must show the GCU op and no host copies, while
  ``F.scaled_dot_product_attention`` must show the CPU fallback.  If either
  identity changes the run fails instead of silently comparing new things.
* Per-shape, per-output-dtype numeric gates: cosine >= 0.99, relative L2 <=
  0.05, all values finite, plus max abs recorded.
* Cold-compile cost in a child process with a fresh ``TRITON_CACHE_DIR``, and
  the steady-state median in the same child.
* Three alternating A/B rounds on identical inputs: full Sage call versus the
  device fused op, one synchronized event pair per call, raw per-sample rows.
* Peak and current device memory, quantization-only and attention-only stage
  timings to attribute the gap, and a PASS/FAIL decision.

Exit status
-----------
0   PASS
2   environment/backend unusable (no GCU, device count, foreign tenant)
3   numeric gate, finite-value, or missing-sample failure
4   wrong device identity (efsmi index/PCI mismatch or stream mismatch)
5   baseline dispatch changed (device baseline fell back to CPU, or the
    reference ``F.scaled_dot_product_attention`` stopped being a fallback)
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import re
import statistics
import subprocess
import sys
import time

EXPECTED_PCI = "0000:83:00.0"
EXPECTED_INDEX = 5
FORBIDDEN_DEVICE_OPS = (
    "_scaled_dot_product_flash_attention_for_cpu",
    "_scaled_dot_product_attention_math",
    "aten::_to_cpu",
)
CPU_FALLBACK_MARKER = "_scaled_dot_product_flash_attention_for_cpu"


# --------------------------------------------------------------------------
# Pre-flight: no torch import before the device guard so a wrong assignment
# cannot even create a context on the wrong GCU.
# --------------------------------------------------------------------------
def efsmi_devices() -> list[dict]:
    """Parse the efsmi device table into [{index, sn, pci}]."""
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
            (
                f
                for f in fields
                if re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]", f)
            ),
            None,
        )
        if bus is None:
            continue
        devices.append({"index": int(fields[0]), "sn": fields[1], "pci": bus.lower()})
    return devices


def efsmi_field(index: int, section: str, label: str) -> "float | None":
    try:
        raw = subprocess.run(
            ["efsmi", "-i", str(index), "-q", "-d", section],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(rf"{label}\s*:\s*([0-9.]+)", raw)
    return float(match.group(1)) if match else None


def device_guard(index: int, pci: str) -> tuple[list[str], dict]:
    """Fail closed unless efsmi says this index is the assigned PCI device."""
    problems: list[str] = []
    devices = efsmi_devices()
    target = [d for d in devices if d["pci"] == pci]
    if not devices:
        problems.append("efsmi -L produced no parseable device table")
    elif not target:
        problems.append(f"efsmi does not report PCI {pci}; devices={devices}")
    else:
        found = target[0]
        if found["index"] != index:
            problems.append(
                f"PCI {pci} is efsmi DEV {found['index']}, but the task assigns "
                f"DEV {index}"
            )
        if found["sn"] != "A0D5460520864":
            problems.append(
                f"PCI {pci} serial {found['sn']} is not the assigned card "
                "A0D5460520864"
            )
    return problems, {
        "event": "device_guard",
        "assigned_index": index,
        "assigned_pci": pci,
        "devices": devices,
        "efsmi_used_mib": efsmi_field(index, "MEMORY", "Used Size"),
        "efsmi_total_mib": efsmi_field(index, "MEMORY", "Total Size"),
        "efsmi_gcu_usage_pct": efsmi_field(index, "USAGE", "GCU Usage"),
        "violations": problems,
    }


def parse_shape(text: str) -> tuple[int, int, int, int]:
    parts = tuple(int(x) for x in text.replace("x", ",").split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"bad shape {text!r}; want B,H,T,D")
    return parts


def cpu_seconds() -> float:
    """Process CPU time with microsecond resolution.

    ``resource.getrusage`` is not jiffy-quantised like ``/proc/self/stat``, so a
    sub-millisecond device launch is still distinguishable from a PCIe round
    trip.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def metrics(actual, expected) -> dict:
    import torch
    import torch.nn.functional as F

    a = actual.float().flatten()
    e = expected.float().flatten()
    if a.numel() > 1_048_576:
        # Deterministic stride subsample keeps the gate cheap on the largest
        # shapes without changing what is being measured.
        stride = (a.numel() + 1_048_576 - 1) // 1_048_576
        a = a[::stride]
        e = e[::stride]
    diff = a - e
    return {
        "cosine": F.cosine_similarity(a, e, dim=0).item(),
        "relative_l2": (diff.norm() / e.norm().clamp_min(1e-8)).item(),
        "max_abs": diff.abs().max().item(),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def one_sample(fn):
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
"""A max/min latency ratio above this is treated as two modes, not one tail."""


def summarize(values: list[float]) -> dict:
    """Median plus an honest description of the distribution's shape.

    The full Sage call at large shapes is *bimodal*: it alternates between two
    latencies about 1.8x apart (see the report).  A threshold anchored on the
    median is wrong for that data -- when the slow mode becomes the majority
    the median moves onto it and the fast calls stop counting as outliers, so
    a ``slow_call_fraction`` built that way can read 0.0 on a distribution that
    is half slow calls.  The split is therefore anchored on the extremes:
    modes are separated at the geometric mean of min and max, and both mode
    centres and both fractions are reported.  ``slow_call_fraction`` keeps its
    meaning (fraction of calls in the slow mode) but no longer depends on
    which mode happens to hold the median.
    """
    if not values:
        return {"median_ms": None, "min_ms": None, "max_ms": None, "n": 0}
    ordered = sorted(values)
    median = statistics.median(ordered)
    low, high = ordered[0], ordered[-1]
    record = {
        "median_ms": median,
        "mean_ms": statistics.mean(ordered),
        "min_ms": low,
        "max_ms": high,
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
    if len(ordered) > 1 and low > 0 and high / low > BIMODAL_RATIO:
        split = (low * high) ** 0.5
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


def trace_ops(which: str) -> dict:
    """Return the profiler event names recorded for one call.

    This image's profiler leaves a broken heap behind (the process aborts in
    ``malloc_consolidate`` at exit after a GCU trace), so tracing runs in a
    throwaway child process.  A clean interpreter exit there is part of the
    evidence: if the profiler itself starts failing the parent sees it.
    """
    code = TRACE_CHILD.format(index=EXPECTED_INDEX, which=which)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env=dict(os.environ),
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


def dispatch_violations(
    device_names, sdpa_names, device_host_ms: float, sdpa_host_ms: float
) -> list[str]:
    """Pure dispatch-identity gate, kept separate so it can be unit-tested.

    Returns a list of violations; empty means the comparison still means what
    the report says it means: the baseline is a GCU kernel and the reference
    is the host fallback.
    """
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


TRACE_CHILD = r"""
import json, sys
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

if {which!r} == "device":
    fn = device_op
else:
    fn = sdpa_op

fn()
torch.gcu.synchronize()
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.GCU]
) as prof:
    fn()
    torch.gcu.synchronize()
names = [event.name for event in prof.events()]
print(json.dumps({{"names": names}}), flush=True)
"""


def path_evidence() -> tuple[list[str], dict]:
    """Prove the device baseline is a GCU kernel and F.sdpa is a host fallback."""
    import torch
    import torch.nn.functional as F

    problems: list[str] = []

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
    device_names = device_trace["names"]
    sdpa_names = sdpa_trace["names"]

    def has(names, needle):
        return any(needle in name for name in names)

    for label, trace in (("device", device_trace), ("sdpa", sdpa_trace)):
        if trace.get("error"):
            problems.append(f"{label} trace failed: {trace['error']}")
        if trace.get("child_exit") != 0:
            problems.append(
                f"{label} trace child exited {trace.get('child_exit')}; the "
                "profiler could not complete cleanly"
            )

    device_fallback = [n for n in FORBIDDEN_DEVICE_OPS if has(device_names, n)]
    sdpa_fallback = has(sdpa_names, CPU_FALLBACK_MARKER)

    # Host CPU time per call corroborates the trace: a device kernel launches in
    # well under a millisecond, a PCIe round trip costs tens of milliseconds.
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
        device_names, sdpa_names, device_host_ms, sdpa_host_ms
    ))
    return problems, {
        "event": "path_evidence",
        "device_baseline_ops": sorted(
            name
            for name in device_names
            if "scaled_dot" in name or "to_cpu" in name or "copy" in name or "tops" in name
        ),
        "reference_sdpa_ops": sorted(
            name
            for name in sdpa_names
            if "scaled_dot" in name or "to_cpu" in name or "copy" in name or "tops" in name
        ),
        "device_trace_child_exit": device_trace.get("child_exit"),
        "reference_trace_child_exit": sdpa_trace.get("child_exit"),
        "device_baseline_has_gcu_kernel": has(
            device_names, "_scaled_dot_product_flash_attention"
        ) and not device_fallback,
        "reference_is_cpu_fallback": sdpa_fallback,
        "device_host_ms_median": device_host_ms,
        "reference_host_ms_median": sdpa_host_ms,
        "violations": problems,
    }


# --------------------------------------------------------------------------
# Cold-compile child: fresh process, fresh Triton cache directory.
# --------------------------------------------------------------------------
def cold_child(shape, dtype_name: str, scope: str) -> int:
    """Measure first-call cost in a pristine Triton cache directory.

    ``scope`` is either ``full`` (device baseline then the Sage full call, so
    the full call pays both kernel compilations) or ``stages`` (quantizer then
    attention separately, so each stage's compile cost is attributable).
    """
    problems, guard = device_guard(EXPECTED_INDEX, EXPECTED_PCI)
    if problems:
        print(json.dumps(guard, sort_keys=True), flush=True)
        return 4
    import torch
    import torch_gcu  # noqa: F401
    from flag_attn import sage_attention
    from flag_attn.sage_attention import per_block_int8

    torch.gcu.set_device(EXPECTED_INDEX)
    # The Enflame kernel takes FP16 Q/K/V and can emit either FP16 or BF16; it
    # cannot mix a BF16 V with its FP16 probability tile, so the input dtype is
    # fixed and only the output dtype is varied.
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(2026)
    q = torch.randn(shape, device=f"gcu:{EXPECTED_INDEX}", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cache_dir = os.environ.get("TRITON_CACHE_DIR", "")

    def timed(fn):
        c0 = cpu_seconds()
        t0 = time.perf_counter()
        start = torch.gcu.Event(enable_timing=True)
        end = torch.gcu.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        host = (cpu_seconds() - c0) * 1000.0
        torch.gcu.synchronize()
        return {
            "wall_ms": (time.perf_counter() - t0) * 1000.0,
            "event_ms": start.elapsed_time(end),
            "host_ms": host,
            "out": out,
        }

    def device_op():
        return torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, 0.0, False, False
        )[0]

    def quant_only():
        return per_block_int8(q, k)

    def attn_only(pre):
        return sage_attention(pre[0], pre[2], v, pre[1], pre[3], output_dtype=dtype)[0]

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
        "scope": scope,
        "triton_cache_dir": cache_dir,
        "cache_files_at_start": cache_files(),
    }

    if scope == "full":
        first_device = timed(device_op)
        record["device_first"] = {k: v for k, v in first_device.items() if k != "out"}
        record["cache_files_after_device"] = cache_files()
        first_full = timed(full)
        record["full_first"] = {k: v for k, v in first_full.items() if k != "out"}
        record["cache_files_after_full"] = cache_files()
        for _ in range(4):
            device_op()
            full()
        torch.gcu.synchronize()
        device_values, full_values = [], []
        for _ in range(9):
            device_values.append(one_sample(device_op)[0])
            full_values.append(one_sample(full)[0])
        steady = {
            "device_ms": summarize(device_values),
            "full_ms": summarize(full_values),
        }
    else:
        first_quant = timed(quant_only)
        record["quant_first"] = {k: v for k, v in first_quant.items() if k != "out"}
        hold = first_quant["out"]
        first_attn = timed(lambda: attn_only(hold))
        record["attn_first"] = {k: v for k, v in first_attn.items() if k != "out"}
        record["cache_files_after_stages"] = cache_files()
        for _ in range(4):
            quant_only()
            attn_only(hold)
        torch.gcu.synchronize()
        quant_values, attn_values = [], []
        for _ in range(9):
            quant_values.append(one_sample(quant_only)[0])
            attn_values.append(one_sample(lambda: attn_only(hold))[0])
        steady = {
            "quant_ms": summarize(quant_values),
            "attn_ms": summarize(attn_values),
        }
        # What the Sage call would cost if the attention kernel alone decided
        # it: the sum of the two stages measured independently.
        steady["stage_sum_ms"] = summarize(
            [q + a for q, a in zip(quant_values, attn_values)]
        )

    record["steady"] = steady
    record["compile_overhead_ms"] = {
        key.replace("_first", ""): record[key]["wall_ms"] - steady[key.replace("_first", "_ms")]["median_ms"]
        for key in list(record)
        if key.endswith("_first")
    }
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0


def cold_runs(shapes, dtype_name: str, out_stream) -> tuple[list[dict], list[str]]:
    """Re-exec this script per shape with a pristine Triton cache directory."""
    records, problems = [], []
    for shape in shapes:
        for scope in ("full", "stages"):
            cache_dir = (
                f"/tmp/sage003_cold_{int(time.time())}_{scope}_{shape[2]}x{shape[3]}"
            )
            env = dict(os.environ)
            env["TRITON_CACHE_DIR"] = cache_dir
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--cold-child",
                "--cold-scope",
                scope,
                "--shapes",
                "x".join(str(x) for x in shape),
                "--output-dtype",
                dtype_name,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            payload = None
            for line in reversed(proc.stdout.splitlines()):
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if candidate.get("event") == "cold":
                    payload = candidate
                    break
            if payload is None:
                problems.append(
                    f"cold {scope} run for {shape} failed (exit {proc.returncode}): "
                    f"{proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'no output'}"
                )
                continue
            payload["child_exit"] = proc.returncode
            if proc.returncode != 0:
                # The measurement is only usable from a clean process exit: a
                # nonzero status here means the run aborted during teardown or
                # crashed after printing, and the numbers are suspect.
                problems.append(
                    f"cold {scope} run for {shape} exited {proc.returncode} "
                    "after emitting a record; treat its numbers as invalid"
                )
            empty = payload.get("cache_files_at_start") == 0
            payload["cache_was_empty"] = empty
            if not empty:
                problems.append(
                    f"cold {scope} run for {shape} did not start from an empty "
                    f"Triton cache ({payload.get('cache_files_at_start')} files)"
                )
            records.append(payload)
            if out_stream is not None:
                out_stream.write(json.dumps(payload, sort_keys=True) + "\n")
                out_stream.flush()
    return records, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=parse_shape, nargs="+",
                        default=[(1, 2, 4096, 64), (4, 32, 1024, 128)])
    parser.add_argument("--output-dtype", choices=("float16", "bfloat16"), default="float16",
                        help="output dtype; Q/K/V inputs stay float16 because the "
                             "Enflame kernel cannot mix bfloat16 V with its float16 "
                             "probability tile")
    parser.add_argument("--input-dtype", choices=("float16",), default="float16",
                        help="kept explicit so the input dtype is part of the record")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--accuracy-cosine", type=float, default=0.99)
    parser.add_argument("--accuracy-rel-l2", type=float, default=0.05)
    parser.add_argument("--cold", action="store_true", help="also run fresh-cache cold-compile children")
    parser.add_argument("--max-used-mib", type=float, default=1024.0)
    parser.add_argument("--expect-index", type=int, default=EXPECTED_INDEX,
                        help="guard override used only to test the fail-closed path")
    parser.add_argument("--jsonl", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--cold-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cold-scope", choices=("full", "stages"), default="full",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.cold_child:
        return cold_child(args.shapes[0], args.output_dtype, args.cold_scope)

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

    def full_sage(q, k, v):
        q8, q_scale, k8, k_scale = per_block_int8(q, k)
        return sage_attention(q8, k8, v, q_scale, k_scale, output_dtype=dtype)[0]

    from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import (
        quant_per_block_int8_kernel as _quant_kernel,
    )

    def legacy_quant(q, k):
        """The pre-SAGE-003 launch (Triton's default 4 warps / 3 stages).

        Kept in the harness so the optimization's gain and its bitwise
        equivalence are both re-measured on every run instead of being taken
        from the report.
        """
        q8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
        k8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
        b, h, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        q_scale = torch.empty((b, h, (qo_len + 127) // 128), device=q.device, dtype=torch.float32)
        k_scale = torch.empty((b, h_kv, (kv_len + 63) // 64), device=q.device, dtype=torch.float32)
        _quant_kernel[((qo_len + 127) // 128, h, b)](
            q, q8, q_scale, qo_len,
            q.stride(0), q.stride(1), q.stride(2),
            q8.stride(0), q8.stride(1), q8.stride(2),
            q_scale.stride(0), q_scale.stride(1),
            sm_scale=(head_dim**-0.5) * 1.44269504,
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
            sage_out = full_sage(q, k, v)
            torch.gcu.synchronize()

            for _ in range(args.warmup):
                device_sdpa(q, k, v)
                full_sage(q, k, v)
            torch.gcu.synchronize()

            torch.gcu.reset_peak_memory_stats(EXPECTED_INDEX)
            used_before = torch.gcu.device_memory_used(EXPECTED_INDEX) / 2**20

            samples = {"device_fused_sdpa": [], "sage_full": []}

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

            # Three rounds alternate the leading scope so a monotonic drift
            # cannot favour either candidate.
            device_first = rnd % 2 == 0
            for pair in range(args.pairs):
                if device_first:
                    take("device_fused_sdpa", lambda: device_sdpa(q, k, v), pair, 0)
                    take("sage_full", lambda: full_sage(q, k, v), pair, 1)
                    take("sage_full", lambda: full_sage(q, k, v), pair, 2)
                    take("device_fused_sdpa", lambda: device_sdpa(q, k, v), pair, 3)
                else:
                    take("sage_full", lambda: full_sage(q, k, v), pair, 0)
                    take("device_fused_sdpa", lambda: device_sdpa(q, k, v), pair, 1)
                    take("device_fused_sdpa", lambda: device_sdpa(q, k, v), pair, 2)
                    take("sage_full", lambda: full_sage(q, k, v), pair, 3)

            stats = {
                scope: summarize([r["event_ms"] for r in rows])
                for scope, rows in samples.items()
            }
            host_stats = {
                scope: summarize([r["host_ms"] for r in rows])
                for scope, rows in samples.items()
            }

            round_failures: list[str] = []

            # Stage attribution after the A/B block (diagnostic, not the gate).
            pre = per_block_int8(q, k)
            stage = {
                "quant_tuned_ms": summarize([
                    one_sample(lambda: per_block_int8(q, k))[0] for _ in range(7)
                ]),
                "quant_default_launch_ms": summarize([
                    one_sample(lambda: legacy_quant(q, k))[0] for _ in range(7)
                ]),
                "attn_only_ms": summarize([
                    one_sample(lambda: sage_attention(
                        pre[0], pre[2], v, pre[1], pre[3], output_dtype=dtype
                    ))[0] for _ in range(7)
                ]),
            }
            legacy = legacy_quant(q, k)
            stage["quant_outputs_bitwise_equal"] = all(
                torch.equal(a, b) for a, b in zip(legacy, pre)
            )
            if not stage["quant_outputs_bitwise_equal"]:
                round_failures.append(
                    "quant launch-config change altered int8/scale outputs"
                )
            stage["quant_speedup_vs_default_launch"] = (
                stage["quant_default_launch_ms"]["median_ms"]
                / stage["quant_tuned_ms"]["median_ms"]
            )
            stage["quant_share_of_sage"] = (
                stage["quant_tuned_ms"]["median_ms"]
                / stats["sage_full"]["median_ms"]
            )
            stage["attn_share_of_sage"] = (
                stage["attn_only_ms"]["median_ms"]
                / stats["sage_full"]["median_ms"]
            )
            stage["attn_over_device"] = (
                stage["attn_only_ms"]["median_ms"]
                / stats["device_fused_sdpa"]["median_ms"]
            )
            # Headroom: the best ratio Sage could reach if quantization were
            # free, and the best it could reach if the attention kernel matched
            # the device op. Either bound alone cannot close the gap.
            stage["floor_if_quant_free"] = stage["attn_over_device"]
            stage["floor_if_attn_matched_device"] = (
                stage["quant_tuned_ms"]["median_ms"]
                / stats["device_fused_sdpa"]["median_ms"]
            )
            stage["stage_sum_over_full"] = (
                (stage["quant_tuned_ms"]["median_ms"] + stage["attn_only_ms"]["median_ms"])
                / stats["sage_full"]["median_ms"]
            )

            # Peak memory per candidate, measured separately so the Sage peak
            # is not hidden by the baseline's temporary buffers.
            def peak_of(fn, n=3):
                torch.gcu.empty_cache()
                torch.gcu.reset_peak_memory_stats(EXPECTED_INDEX)
                for _ in range(n):
                    fn()
                torch.gcu.synchronize()
                peak = torch.gcu.max_memory_allocated(EXPECTED_INDEX) / 2**20
                reserved = torch.gcu.max_memory_reserved(EXPECTED_INDEX) / 2**20
                return peak, reserved

            device_peak, device_reserved = peak_of(lambda: device_sdpa(q, k, v))
            sage_peak, sage_reserved = peak_of(lambda: full_sage(q, k, v))

            used_after = torch.gcu.device_memory_used(EXPECTED_INDEX) / 2**20
            accuracy = {
                # Compare in the output dtype so BF16 runs do not charge the
                # kernel for the reference's FP16->BF16 rounding, and keep the
                # raw mixed-dtype distance for the record.
                "sage_vs_device_fused": metrics(sage_out, ref_device.to(dtype)),
                "sage_vs_reference_sdpa": metrics(sage_out, ref_host.to(dtype)),
                "sage_vs_device_fused_raw": metrics(sage_out, ref_device),
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
                        f"{name}: relative_l2 {m['relative_l2']:.6f} > "
                        f"{args.accuracy_rel_l2}"
                    )
            for scope, rows in samples.items():
                if len(rows) < args.min_samples:
                    round_failures.append(
                        f"{scope}: only {len(rows)} samples < {args.min_samples}"
                    )
            if sage_out.dtype != dtype or tuple(sage_out.shape) != tuple(shape):
                round_failures.append(
                    f"sage_full: dtype/shape {sage_out.dtype}/{tuple(sage_out.shape)} "
                    f"!= {dtype}/{shape}"
                )
            if tuple(ref_device.shape) != tuple(shape):
                round_failures.append(
                    f"device baseline shape {tuple(ref_device.shape)} != {shape}"
                )

            ratio = stats["sage_full"]["median_ms"] / stats["device_fused_sdpa"]["median_ms"]
            shape_key = "x".join(str(x) for x in shape)
            round_medians.setdefault(shape_key, []).append({
                "round": rnd,
                "device": stats["device_fused_sdpa"]["median_ms"],
                "sage": stats["sage_full"]["median_ms"],
                "ratio": ratio,
            })
            if round_failures:
                failures.extend(
                    f"shape {list(shape)} round {rnd}: {f}" for f in round_failures
                )
            emit({
                "event": "round",
                "shape": list(shape),
                "round": rnd,
                "order": "device_first" if device_first else "sage_first",
                "accuracy": accuracy,
                "samples": samples,
                "stats": stats,
                "host_stats": host_stats,
                "sage_over_device": ratio,
                "device_over_sage": 1.0 / ratio,
                "stage_ms": stage,
                "memory": {
                    "device_memory_used_before_mib": used_before,
                    "device_memory_used_after_mib": used_after,
                    "device_peak_mib": device_peak,
                    "device_peak_reserved_mib": device_reserved,
                    "sage_peak_mib": sage_peak,
                    "sage_peak_reserved_mib": sage_reserved,
                    "sage_over_device_peak": sage_peak / device_peak if device_peak else None,
                },
                "failures": round_failures,
            })
            del q, k, v, ref_device, ref_host, sage_out, pre
            torch.gcu.empty_cache()

    # A run that measured nothing must not report PASS.
    if args.rounds < 1 or args.pairs < 1:
        failures.append(
            f"degenerate configuration: rounds={args.rounds} pairs={args.pairs}; "
            "at least one of each is required"
        )
    missing = [
        "x".join(str(x) for x in shape)
        for shape in args.shapes
        if "x".join(str(x) for x in shape) not in round_medians
    ]
    if missing:
        failures.append(f"no samples were recorded for shapes {missing}")
    for shape in args.shapes:
        key = "x".join(str(x) for x in shape)
        if key in round_medians and len(round_medians[key]) != args.rounds:
            failures.append(
                f"shape {key}: {len(round_medians[key])} rounds recorded, "
                f"expected {args.rounds}"
            )

    # Aggregate medians per shape across the three rounds.
    aggregate = {}
    for key, per_round in round_medians.items():
        aggregate[key] = {
            "round_device_ms": [r["device"] for r in per_round],
            "round_sage_ms": [r["sage"] for r in per_round],
            "round_sage_over_device": [r["ratio"] for r in per_round],
            "median_device_ms": statistics.median([r["device"] for r in per_round]),
            "median_sage_ms": statistics.median([r["sage"] for r in per_round]),
        }
        aggregate[key]["median_sage_over_device"] = (
            aggregate[key]["median_sage_ms"] / aggregate[key]["median_device_ms"]
        )
    decision = {
        "event": "decision",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "device_index": EXPECTED_INDEX,
        "pci": EXPECTED_PCI,
        "shapes": [list(s) for s in args.shapes],
        "output_dtype": args.output_dtype,
        "input_dtype": args.input_dtype,
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
