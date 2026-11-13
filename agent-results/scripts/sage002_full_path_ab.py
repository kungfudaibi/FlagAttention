#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SAGE-002 exact-baseline versus candidate full-call alternating A/B.

Purpose
-------
Decide whether the Enflame per-block INT8 SageAttention full call (quantize +
attention, the public ``sage_attention(per_block_int8(q, k), v)`` path) is a
net win over the baseline the repository actually uses, i.e.
``F.scaled_dot_product_attention``.  On this Enflame image that dispatcher
entry falls back to the CPU math kernel and copies tensors over PCIe, so the
script also measures the GCU-native fused op
``torch.ops.aten._scaled_dot_product_flash_attention`` as a control: a real
speedup has to be reported against both, otherwise the "win" is only against a
host fallback.

Method
------
* One process, one device (container GCU 1, PCI 0000:83:00.0), pinned CPUs.
* Every measurement is a synchronized device-event pair around exactly one
  public call, so host queueing cannot hide inside the number.
* Alternating A/B in ABBA order per pair: baseline, candidate, candidate,
  baseline.  Each raw sample is stored with its position, round and pair so a
  drift-sensitive reader can re-derive the ordering effect.
* Fresh inputs per round from a fixed seed; the *same* tensors feed baseline
  and candidate inside a pair.
* Baseline and candidate outputs are compared after every round (cosine,
  relative L2, max abs) and both are checked finite with the expected dtype
  and shape.
* Device memory is sampled before/after each round.

Exit status
-----------
0 only when every validity check passes: GCU 1 identity, quiet-host gate (when
requested), dtype/shape/finiteness, accuracy gate, no baseline fallback
surprise, and at least ``--min-samples`` usable samples per scope.  Any failure
returns 3 and is listed in the JSON summary so a caller cannot mistake a
broken run for a slow one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

import torch_gcu  # noqa: F401  (registers the gcu backend)

from flag_attn import sage_attention
from flag_attn.sage_attention import per_block_int8


EXPECTED_PCI = "0000:83:00.0"


def parse_shape(text: str) -> tuple[int, int, int, int]:
    parts = tuple(int(x) for x in text.replace("x", ",").split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"bad shape {text!r}; want B,H,T,D")
    return parts


def cpu_seconds() -> float:
    fields = open("/proc/self/stat").read().split()
    return (int(fields[13]) + int(fields[14])) / os.sysconf("SC_CLK_TCK")


def efsmi_devices() -> list[dict]:
    """Return [{index, pci, sn}] parsed from the efsmi device table.

    Sample row:
      1      A0D5460520864  N/A  1ea0:2a22  0000:83:00.0  128-255  TS9B37120204
    efsmi is the only reliable PCI-to-index mapping on this host; PyTorch
    device ordinals do not correspond to the physical slots on their own.
    """
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
        bus = next((f for f in fields if re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]", f)), None)
        if bus is None:
            continue
        devices.append({
            "index": int(fields[0]),
            "sn": fields[1],
            "pci": bus.lower(),
        })
    return devices


def efsmi_used_mib(index: int) -> float | None:
    """Device memory in use according to efsmi, independent of our context."""
    try:
        raw = subprocess.run(
            ["efsmi", "-i", str(index), "-q", "-d", "MEMORY"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"Used Size\s*:\s*(\d+)", raw)
    return float(match.group(1)) if match else None


def gcu_node_holders() -> list[dict]:
    """Processes with a GCU device node open; recorded, not itself a failure."""
    others = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        fd_dir = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                link = os.readlink(f"{fd_dir}/{fd}")
            except OSError:
                continue
            if "/dev/gcu" in link or "/dev/tops" in link:
                try:
                    name = open(f"/proc/{pid}/comm").read().strip()
                except OSError:
                    name = "?"
                others.append({"pid": int(pid), "comm": name, "fd": link})
                break
    return others


def quiet_violations(
    max_load: float, allowed_pci: str, max_used_mib: float, boot_used_mib
) -> tuple[list[str], dict]:
    """Fail closed unless GCU 1 itself looks idle before a formal run.

    The assigned device is identified by PCI address via efsmi, and foreign
    occupancy is efsmi's GCU 1 Used Size sampled before this process created a
    GCU context (our own context costs ~81-109 MiB and is not a tenant).
    Processes that merely hold a device node are reported for the record: an
    idle context can keep a node open without consuming GCU 1 memory, so their
    presence alone is not evidence of a competing tenant.
    """
    problems = []
    load1 = os.getloadavg()[0]
    if load1 > max_load:
        problems.append(f"1-minute load average {load1:.2f} exceeds {max_load:.2f}")
    devices = efsmi_devices()
    target = [d for d in devices if d["pci"] == allowed_pci]
    if not target:
        problems.append(
            f"efsmi does not report PCI {allowed_pci}; devices={devices}"
        )
    elif target[0]["index"] != 1:
        problems.append(
            f"PCI {allowed_pci} is efsmi DEV {target[0]['index']}, expected DEV 1"
        )
    if boot_used_mib is None:
        problems.append("efsmi could not report GCU 1 memory usage at startup")
    elif boot_used_mib > max_used_mib:
        problems.append(
            f"GCU 1 held {boot_used_mib:.1f} MiB before this process started "
            f"(> {max_used_mib:.1f} MiB); another tenant is resident"
        )
    return problems, {
        "load1": load1,
        "cpu_count": os.cpu_count(),
        "max_load": max_load,
        "devices": devices,
        "gcu1_foreign_mib_at_startup": boot_used_mib,
        "gcu1_used_mib_now": efsmi_used_mib(1),
        "gcu1_context_mib": torch.gcu.device_memory_used(1) / 2**20,
        "node_holders": gcu_node_holders(),
    }


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
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
    }


def one_sample(fn):
    """Return one (event_ms, wall_ms, host_ms, output) synchronized sample.

    ``host_ms`` is process CPU time consumed between the start event and the
    end-event enqueue, i.e. launch overhead only.  It is sampled *before*
    ``synchronize`` because a synchronizing wait spins on the CPU and would
    otherwise be charged to the kernel.
    """
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


def summarize(values: list[float]) -> dict:
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "stdev_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        type=parse_shape,
        nargs="+",
        default=[(1, 2, 4096, 64)],
        help="B,H,T,D; default is the longest SAGE-001 provisional shape",
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--pairs", type=int, default=10, help="ABBA pairs per scope per round"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--baseline",
        choices=("exact", "device", "both"),
        default="both",
        help="exact = F.scaled_dot_product_attention (repository baseline); "
        "device = aten fused op (GCU-native control)",
    )
    parser.add_argument("--output-dtype", choices=("float16", "bfloat16"),
                        default="float16")
    parser.add_argument("--accuracy-cosine", type=float, default=0.99)
    parser.add_argument("--accuracy-rel-l2", type=float, default=0.05)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument(
        "--require-quiet",
        action="store_true",
        help="fail unless load average and GCU holders look idle",
    )
    parser.add_argument(
        "--max-load", type=float, default=0.0,
        help="1-minute load average ceiling with --require-quiet; 0 derives "
             "0.25 * cpu_count, because a shared 256-CPU host is idle for one "
             "job well above a load of 1",
    )
    parser.add_argument(
        "--max-used-mib", type=float, default=64.0,
        help="efsmi-reported GCU 1 Used Size ceiling with --require-quiet; "
             "our own process context is excluded because it is bootstrapped "
             "before the gate runs",
    )
    parser.add_argument("--tag", default="ab")
    parser.add_argument("--out", default="")
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    # Sample foreign occupancy before any GCU context exists in this process;
    # afterwards efsmi also counts our own context and cannot separate tenants.
    boot_used_mib = efsmi_used_mib(1) if args.require_quiet else None

    if not torch.gcu.is_available():
        print("ERROR: no GCU backend", file=sys.stderr)
        return 2
    if torch.gcu.device_count() < 2:
        print("ERROR: GCU 1 unavailable; refusing to run on GCU 0", file=sys.stderr)
        return 2
    torch.gcu.set_device(1)
    dtype = getattr(torch, args.output_dtype)

    stream = open(args.out, "w") if args.out else None

    def emit(record: dict) -> None:
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if stream is not None:
            stream.write(line + "\n")
            stream.flush()

    failures: list[str] = []
    if args.require_quiet:
        max_load = args.max_load if args.max_load > 0 else 0.25 * (os.cpu_count() or 1)
        problems, gate = quiet_violations(
            max_load, EXPECTED_PCI, args.max_used_mib, boot_used_mib
        )
        if problems:
            failures.extend(f"quiet gate: {p}" for p in problems)
        gate.update({"event": "quiet_gate", "required": True,
                     "violations": problems})
        emit(gate)

    emit({
        "event": "meta",
        "tag": args.tag,
        "device": "gcu:1",
        "pci": EXPECTED_PCI,
        "scope": "full_call_vs_exact_baseline",
        "baseline_mode": args.baseline,
        "output_dtype": str(dtype),
        "rounds": args.rounds,
        "pairs_per_round": args.pairs,
        "warmup": args.warmup,
        "ordering": "ABBA",
        "accuracy_threshold": {
            "cosine_min": args.accuracy_cosine,
            "relative_l2_max": args.accuracy_rel_l2,
        },
        "torch": torch.__version__,
        "torch_gcu": torch_gcu.__version__,
        "triton": __import__("triton").__version__,
        "python": sys.version.split()[0],
        "argv": sys.argv,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    def exact_sdpa(q, k, v):
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )

    def device_sdpa(q, k, v):
        return torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, 0.0, False, False
        )[0]

    def candidate(q, k, v):
        q8, q_scale, k8, k_scale = per_block_int8(q, k)
        return sage_attention(
            q8, k8, v, q_scale, k_scale, output_dtype=dtype
        )[0]

    baselines = []
    if args.baseline in ("exact", "both"):
        baselines.append(("exact_sdpa", exact_sdpa))
    if args.baseline in ("device", "both"):
        baselines.append(("device_sdpa", device_sdpa))

    for shape in args.shapes:
        for rnd in range(args.rounds):
            torch.manual_seed(100000 * rnd + 1000 * shape[2] + shape[3])
            q = torch.randn(shape, device="gcu:1", dtype=dtype)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            torch.gcu.synchronize()

            # Reference outputs for the accuracy gate (not timed).
            refs = {name: fn(q, k, v) for name, fn in baselines}
            torch.gcu.synchronize()

            for _, fn in baselines:
                for _ in range(args.warmup):
                    fn(q, k, v)
            for _ in range(args.warmup):
                candidate(q, k, v)
            torch.gcu.synchronize()

            torch.gcu.reset_peak_memory_stats(1)
            used_before = torch.gcu.device_memory_used(1) / 2**20
            free_before = torch.gcu.mem_get_info(1)[0] / 2**20

            samples = {"candidate": []}
            for name, _ in baselines:
                samples[name] = []

            def take(scope: str, fn, pair: int, position: int):
                event, wall, host, out = one_sample(fn)
                samples[scope].append({
                    "pair": pair,
                    "position": position,
                    "event_ms": event,
                    "wall_ms": wall,
                    "host_ms": host,
                })
                return out

            for pair in range(args.pairs):
                # ABBA: order effects cancel when a reader groups positions.
                for name, fn in baselines:
                    take(name, lambda fn=fn: fn(q, k, v), pair, 0)
                cand_a = take("candidate", lambda: candidate(q, k, v), pair, 1)
                cand_b = take("candidate", lambda: candidate(q, k, v), pair, 2)
                for name, fn in reversed(baselines):
                    take(name, lambda fn=fn: fn(q, k, v), pair, 3)

            used_after = torch.gcu.device_memory_used(1) / 2**20
            free_after = torch.gcu.mem_get_info(1)[0] / 2**20

            accuracy = {}
            round_failures = []
            for name, ref in refs.items():
                m = metrics(cand_a, ref)
                accuracy[f"candidate_vs_{name}"] = m
                if not m["finite"]:
                    round_failures.append(f"{name}: candidate output non-finite")
                if m["cosine"] < args.accuracy_cosine:
                    round_failures.append(
                        f"{name}: cosine {m['cosine']:.6f} < {args.accuracy_cosine}"
                    )
                if m["relative_l2"] > args.accuracy_rel_l2:
                    round_failures.append(
                        f"{name}: relative_l2 {m['relative_l2']:.6f} > "
                        f"{args.accuracy_rel_l2}"
                    )
                if "exact_sdpa" in refs and name != "exact_sdpa":
                    cross = metrics(refs[name], refs["exact_sdpa"])
                    accuracy[f"{name}_vs_exact_sdpa"] = cross
            accuracy["candidate_repeat"] = metrics(cand_b, cand_a)
            if not accuracy["candidate_repeat"]["finite"]:
                round_failures.append("candidate output non-finite on repeat")

            for scope, out in (("candidate", cand_a),) + tuple(
                (n, refs[n]) for n, _ in baselines
            ):
                if out.dtype != dtype:
                    round_failures.append(
                        f"{scope}: dtype {out.dtype} != {dtype}"
                    )
                if tuple(out.shape) != tuple(shape):
                    round_failures.append(
                        f"{scope}: shape {tuple(out.shape)} != {shape}"
                    )
                if not torch.isfinite(out).all().item():
                    round_failures.append(f"{scope}: non-finite output")
            for scope, rows in samples.items():
                if len(rows) < args.min_samples:
                    round_failures.append(
                        f"{scope}: only {len(rows)} samples < {args.min_samples}"
                    )
            # The exact baseline must not silently become a GCU kernel: if it
            # suddenly drops to device speed the comparison changed meaning.
            exact_rows = samples.get("exact_sdpa")
            if exact_rows and args.baseline == "both":
                exact_median = statistics.median(r["event_ms"] for r in exact_rows)
                device_median = statistics.median(
                    r["event_ms"] for r in samples["device_sdpa"]
                )
                if exact_median < 2.0 * device_median:
                    round_failures.append(
                        "exact_sdpa is no longer the CPU-fallback path "
                        f"({exact_median:.3f} ms vs device {device_median:.3f} ms); "
                        "baseline identity changed, re-measure the comparison"
                    )

            stats = {
                scope: summarize([r["event_ms"] for r in rows])
                for scope, rows in samples.items()
            }
            host_stats = {
                scope: summarize([r["host_ms"] for r in rows])
                for scope, rows in samples.items()
            }
            ratio = {}
            for name, _ in baselines:
                ratio[name] = (
                    stats["candidate"]["median_ms"] / stats[name]["median_ms"]
                )

            if round_failures:
                failures.extend(
                    f"shape {list(shape)} round {rnd}: {f}" for f in round_failures
                )
            emit({
                "event": "round",
                "shape": list(shape),
                "round": rnd,
                "accuracy": accuracy,
                "samples": samples,
                "stats": stats,
                "host_stats": host_stats,
                "speedup_vs": {k: 1.0 / v for k, v in ratio.items()},
                "candidate_over_baseline": ratio,
                "memory": {
                    "device_memory_used_before_mib": used_before,
                    "device_memory_used_after_mib": used_after,
                    "free_before_mib": free_before,
                    "free_after_mib": free_after,
                    "max_allocated_mib": torch.gcu.max_memory_allocated(1) / 2**20,
                    "max_reserved_mib": torch.gcu.max_memory_reserved(1) / 2**20,
                },
                "failures": round_failures,
            })
            del q, k, v, refs, cand_a, cand_b

    status = "PASS" if not failures else "FAIL"
    decision = {"event": "decision", "status": status, "failures": failures}
    emit(decision)
    if stream is not None:
        stream.close()
    if args.summary:
        with open(args.summary, "w") as handle:
            json.dump(decision, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
