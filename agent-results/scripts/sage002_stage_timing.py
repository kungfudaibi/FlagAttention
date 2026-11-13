#!/usr/bin/env python3
"""SAGE-002 stage breakdown on container GCU 1 (PCI 0000:83:00.0).

Scopes, all on the same FP16 inputs and the same FP16 output dtype:

  quant      Q/K per-block INT8 quantization (per_block_int8)
  attention  SageAttention kernel on pre-quantized Q/K
  full       quantization + attention (the public full call)
  exact_sdpa F.scaled_dot_product_attention, the repository baseline
  device_sdpa torch.ops.aten._scaled_dot_product_flash_attention (control)

Every sample is a synchronized device-event pair; CPU time is sampled from
/proc/self/stat so a host fallback (the repository SDPA baseline copies to CPU
in this image) is visible rather than silently mixed into device time.
Three rounds and every raw sample are emitted as JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

import torch_gcu  # noqa: F401

from flag_attn import sage_attention
from flag_attn.sage_attention import per_block_int8


def parse_shape(text: str) -> tuple[int, int, int, int]:
    parts = tuple(int(x) for x in text.replace("x", ",").split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"bad shape {text!r}; want B,H,T,D")
    return parts


def cpu_seconds() -> float:
    fields = open("/proc/self/stat").read().split()
    return (int(fields[13]) + int(fields[14])) / os.sysconf("SC_CLK_TCK")


def make_inputs(shape, dtype, seed=42):
    torch.manual_seed(seed)
    q = torch.randn(shape, device="gcu:1", dtype=dtype)
    k = torch.randn(shape, device="gcu:1", dtype=dtype)
    v = torch.randn(shape, device="gcu:1", dtype=dtype)
    return q, k, v


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


def timed(fn, samples: int, warmup: int):
    """Warm up, then return per-sample event/wall/host times and last output.

    ``host`` is process CPU time consumed before the synchronizing wait, i.e.
    launch cost.  Charging the spin-wait to the kernel would inflate CPU time
    for a fully device-resident path, so it is excluded on purpose.
    """
    for _ in range(warmup):
        fn()
    torch.gcu.synchronize()
    events, walls, hosts = [], [], []
    out = None
    for _ in range(samples):
        c0 = cpu_seconds()
        start = torch.gcu.Event(enable_timing=True)
        end = torch.gcu.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        out = fn()
        end.record()
        hosts.append((cpu_seconds() - c0) * 1000.0)
        torch.gcu.synchronize()
        walls.append((time.perf_counter() - t0) * 1000.0)
        events.append(start.elapsed_time(end))
    return {"event": events, "wall": walls, "host": hosts}, out


def first_call(fn):
    torch.gcu.synchronize()
    start = torch.gcu.Event(enable_timing=True)
    end = torch.gcu.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.gcu.synchronize()
    return start.elapsed_time(end), out


def summarize(samples: dict) -> dict:
    out = {}
    for name, values in samples.items():
        out[name] = {
            "median_ms": statistics.median(values),
            "mean_ms": statistics.mean(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "stdev_ms": statistics.pstdev(values),
            "n": len(values),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=parse_shape, nargs="+",
                        default=[(1, 2, 512, 64), (1, 2, 1024, 64),
                                 (1, 2, 2048, 64), (1, 2, 4096, 64)])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output-dtype", choices=("float16", "bfloat16"),
                        default="float16")
    parser.add_argument("--tag", default="stage")
    parser.add_argument("--out", default="")
    parser.add_argument("--accuracy-cosine", type=float, default=0.99)
    parser.add_argument("--accuracy-rel-l2", type=float, default=0.05)
    args = parser.parse_args()

    if not torch.gcu.is_available() or torch.gcu.device_count() < 2:
        print("ERROR: GCU device 1 unavailable", file=sys.stderr)
        return 2
    torch.gcu.set_device(1)
    dtype = getattr(torch, args.output_dtype)
    stream = open(args.out, "w") if args.out else None

    def emit(record):
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if stream is not None:
            stream.write(line + "\n")
            stream.flush()

    emit({
        "event": "meta",
        "tag": args.tag,
        "device": "gcu:1",
        "pci": "0000:83:00.0",
        "output_dtype": str(dtype),
        "rounds": args.rounds,
        "samples_per_round": args.samples,
        "warmup": args.warmup,
        "accuracy_threshold": {
            "cosine_min": args.accuracy_cosine,
            "relative_l2_max": args.accuracy_rel_l2,
        },
        "torch": torch.__version__,
        "torch_gcu": torch_gcu.__version__,
        "triton": __import__("triton").__version__,
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    failures = []
    for shape in args.shapes:
        q, k, v = make_inputs(shape, dtype)
        exact_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False)
        device_out = torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, 0.0, False, False)[0]
        q8, qs, k8, ks = per_block_int8(q, k)
        attention_out, _ = sage_attention(q8, k8, v, qs, ks,
                                          output_dtype=dtype)
        fq8, fqs, fk8, fks = per_block_int8(q, k)
        full_out, _ = sage_attention(fq8, fk8, v, fqs, fks,
                                     output_dtype=dtype)
        torch.gcu.synchronize()

        accuracy = {
            "full_vs_exact_sdpa": metrics(full_out, exact_out),
            "full_vs_device_sdpa": metrics(full_out, device_out),
            "attention_vs_device_sdpa": metrics(attention_out, device_out),
            "exact_sdpa_vs_device_sdpa": metrics(exact_out, device_out),
            "full_vs_attention": metrics(full_out, attention_out),
        }
        shape_failures = []
        if full_out.dtype != dtype or exact_out.dtype != dtype:
            shape_failures.append(f"output dtype mismatch: {full_out.dtype}/{exact_out.dtype}")
        if full_out.shape != q.shape:
            shape_failures.append(f"output shape mismatch: {tuple(full_out.shape)}")
        for scope in ("full_vs_exact_sdpa", "full_vs_device_sdpa"):
            m = accuracy[scope]
            if not m["finite"]:
                shape_failures.append(f"{scope}: non-finite output")
            if m["cosine"] < args.accuracy_cosine:
                shape_failures.append(f"{scope}: cosine {m[cosine]:.6f} < {args.accuracy_cosine}")
            if m["relative_l2"] > args.accuracy_rel_l2:
                shape_failures.append(f"{scope}: relative_l2 {m[relative_l2]:.6f} > {args.accuracy_rel_l2}")
        if shape_failures:
            failures.extend(shape_failures)
            emit({"event": "accuracy_fail", "shape": list(shape),
                  "failures": shape_failures, "accuracy": accuracy})
            if stream is not None:
                stream.close()
            return 3

        def quant():
            return per_block_int8(q, k)

        def attention():
            return sage_attention(q8, k8, v, qs, ks, output_dtype=dtype)

        def full():
            a, b, c, d = per_block_int8(q, k)
            return sage_attention(a, c, v, b, d, output_dtype=dtype)

        def exact_sdpa():
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False)

        def device_sdpa():
            return torch.ops.aten._scaled_dot_product_flash_attention(
                q, k, v, 0.0, False, False)[0]

        scopes = (("quant", quant), ("attention", attention), ("full", full),
                  ("exact_sdpa", exact_sdpa), ("device_sdpa", device_sdpa))
        first_ms = {}
        for name, fn in scopes:
            elapsed, _ = first_call(fn)
            first_ms[name] = elapsed
        torch.gcu.synchronize()

        for rnd in range(args.rounds):
            torch.gcu.synchronize()
            torch.gcu.reset_peak_memory_stats(1)
            before_free = torch.gcu.mem_get_info(1)[0] / 2**20
            before_used = torch.gcu.device_memory_used(1) / 2**20
            all_samples = {}
            outputs = {}
            for name, fn in scopes:
                result, out = timed(fn, args.samples, args.warmup)
                all_samples[name] = result
                outputs[name] = out
            after_free = torch.gcu.mem_get_info(1)[0] / 2**20
            after_used = torch.gcu.device_memory_used(1) / 2**20
            stats = {name: summarize({k: s[k] for k in ("event", "wall", "host")})
                     for name, s in all_samples.items()}
            emit({
                "event": "round",
                "shape": list(shape),
                "round": rnd,
                "accuracy": accuracy,
                "first_call_ms": first_ms,
                "raw": {name: {k: s[k] for k in ("event", "wall", "host")}
                        for name, s in all_samples.items()},
                "stats": stats,
                "memory": {
                    "device_memory_used_before_mib": before_used,
                    "device_memory_used_after_mib": after_used,
                    "free_before_mib": before_free,
                    "free_after_mib": after_free,
                    "max_allocated_mib": torch.gcu.max_memory_allocated(1) / 2**20,
                    "max_reserved_mib": torch.gcu.max_memory_reserved(1) / 2**20,
                    "allocated_now_mib": torch.gcu.memory_allocated(1) / 2**20,
                },
            })
        del q, k, v, q8, k8, qs, ks, fq8, fk8, fqs, fks
        del exact_out, device_out, attention_out, full_out
    decision = ({"event": "decision", "status": "FAIL", "failures": failures}
                if failures else {"event": "decision", "status": "OK"})
    emit(decision)
    if stream is not None:
        stream.close()
    return 3 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
