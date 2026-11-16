#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SAGE-005 cold-compile and peak-memory probe.

Cold compile
    Each measurement runs in a *child* process with its own empty
    ``TRITON_CACHE_DIR`` and an empty ``_attn_fwd`` device cache, so the first
    call really does pay the whole lowering pipeline.  Timing the first call in
    the parent would reuse the on-disk cache and report a warm number.

Peak memory
    Device memory high-water mark around one call, measured with the allocator
    counters reset immediately before the call.  The measurement is per path, so
    the Sage full call, the attention-only kernel and the device fused op are
    never summed together.

Both are emitted as JSON lines on ``--jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

DEVICE_INDEX = 5
DEVICE_PCI = "0000:83:00.0"
DEVICE_SERIAL = "A0D5460520864"

CHILD = r'''
import json, sys, time
import torch, torch_gcu
torch.gcu.set_device({device})
sys.path.insert(0, {src!r})
from flag_attn.runtime.backend._enflame.sage_attention import attn_qk_int8_per_block as attn
from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8

shape = tuple({shape})
head_dim = shape[-1]
torch.manual_seed(11)
q = torch.randn(shape, device="gcu:5", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
torch.gcu.synchronize()
time.sleep(0.3)
start = time.perf_counter()
if {which!r} == "native":
    torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False)
else:
    qi, qs, ki, ks = per_block_int8(q, k, sm_scale=head_dim ** -0.5)
    attn.forward(qi, ki, v, qs, ks, output_dtype=torch.float16)
torch.gcu.synchronize()
elapsed_ms = (time.perf_counter() - start) * 1e3
print("COLD " + json.dumps({{"path": {which!r}, "shape": list(shape), "cold_ms": elapsed_ms}}))
'''


def device_guard():
    identity = ""
    for field in ("DEVICE", "PCIE"):
        identity += subprocess.run(
            ["efsmi", "-q", "-d", field, "-i", str(DEVICE_INDEX)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    bus = re.search(r"Bus\s*:\s*([0-9a-fA-F]+)", identity)
    dev = re.search(r"Dev\s*:\s*([0-9a-fA-F]+)", identity)
    func = re.search(r"Func\s*:\s*([0-9a-fA-F]+)", identity)
    if not (bus and dev and func):
        raise SystemExit(f"cannot read a PCI address for GCU {DEVICE_INDEX}")
    observed = f"0000:{int(bus.group(1), 16):02x}:{int(dev.group(1), 16):02x}.{int(func.group(1), 16)}"
    if observed != DEVICE_PCI or DEVICE_SERIAL not in identity:
        raise SystemExit(
            f"GCU {DEVICE_INDEX} is {observed}, not {DEVICE_PCI} ({DEVICE_SERIAL}); refusing to run."
        )


def make_sage_full_call():
    import torch

    from flag_attn.runtime.backend._enflame.sage_attention import (
        attn_qk_int8_per_block as attn,
    )
    from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import (
        per_block_int8,
    )

    def call(q, k, v, head_dim):
        qi, qs, ki, ks = per_block_int8(q, k, sm_scale=head_dim**-0.5)
        return attn.forward(qi, ki, v, qs, ks, output_dtype=torch.float16)

    return call


def peak_memory(torch, fn, repeats=5):
    """Peak device MiB attributable to one call, allocator counters reset first."""
    fn()
    torch.gcu.synchronize()
    peaks = []
    for _ in range(repeats):
        torch.gcu.empty_cache()
        torch.gcu.synchronize()
        torch.gcu.reset_peak_memory_stats()
        before = torch.gcu.memory_allocated()
        fn()
        torch.gcu.synchronize()
        peaks.append(torch.gcu.max_memory_allocated() - before)
        del before
    return statistics.median(peaks) / (1 << 20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--shapes", nargs="+", default=["1x2x4096x64", "4x32x1024x128"])
    parser.add_argument("--src", default=os.environ.get("SAGE005_SRC", "src"))
    parser.add_argument("--skip-cold", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    args = parser.parse_args()

    device_guard()
    src = os.path.abspath(args.src)
    handle = open(args.jsonl, "w")

    def emit(**fields):
        fields["device_index"] = DEVICE_INDEX
        fields["pci"] = DEVICE_PCI
        handle.write(json.dumps(fields, sort_keys=True) + "\n")
        handle.flush()
        print(json.dumps(fields, sort_keys=True), flush=True)

    shapes = [tuple(int(x) for x in spec.split("x")) for spec in args.shapes]

    if not args.skip_cold:
        for shape in shapes:
            for which in ("native", "sage"):
                cache = tempfile.mkdtemp(prefix=f"sage005_cold_{which}_")
                try:
                    env = dict(os.environ)
                    env["TRITON_CACHE_DIR"] = cache
                    env["PYTHONPATH"] = os.pathsep.join(
                        ["/opt/flagtree", src, env.get("PYTHONPATH", "")]
                    )
                    script = CHILD.format(
                        device=DEVICE_INDEX, src=src, shape=list(shape), which=which
                    )
                    path = os.path.join(cache, "child.py")
                    with open(path, "w") as child:
                        child.write(script)
                    out = subprocess.run(
                        [sys.executable, "-u", path],
                        capture_output=True,
                        text=True,
                        env=env,
                        check=False,
                    ).stdout
                    record = None
                    for line in out.splitlines():
                        if line.startswith("COLD "):
                            record = json.loads(line[5:])
                    if record is None:
                        emit(probe="cold", path=which, shape=list(shape), error="child produced no timing")
                    else:
                        emit(probe="cold", **record)
                finally:
                    shutil.rmtree(cache, ignore_errors=True)

    if not args.skip_memory:
        import torch
        import torch_gcu  # noqa: F401

        from flag_attn.runtime.backend._enflame.sage_attention import (
            attn_qk_int8_per_block as attn,
        )
        from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import (
            per_block_int8,
        )

        torch.gcu.set_device(DEVICE_INDEX)
        sage_full_call = make_sage_full_call()
        for shape in shapes:
            head_dim = shape[-1]
            torch.manual_seed(11)
            q = torch.randn(shape, device="gcu:5", dtype=torch.float16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            qi, qs, ki, ks = per_block_int8(q, k, sm_scale=head_dim**-0.5)
            torch.gcu.synchronize()

            paths = {
                "native_device_op": lambda: torch.ops.aten._scaled_dot_product_flash_attention(
                    q, k, v, 0.0, False, False
                ),
                "sage_attn_only": lambda: attn.forward(
                    qi, ki, v, qs, ks, output_dtype=torch.float16
                ),
                "sage_full": lambda: sage_full_call(q, k, v, head_dim),
            }
            for label, fn in paths.items():
                emit(
                    probe="memory",
                    path=label,
                    shape=list(shape),
                    peak_mib=peak_memory(torch, fn),
                )
            del q, k, v, qi, qs, ki, ks
            torch.gcu.empty_cache()

    handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
