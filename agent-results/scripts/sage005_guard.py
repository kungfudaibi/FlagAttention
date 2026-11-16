#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared GCU-5 guard, event timer and JSONL writer for the SAGE-005 rechecks.

The ordinal is container-local, so every run proves the PCI address *and* the
serial number before the first device action.  A mismatch is fatal; there is no
fallback to another card.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys

DEVICE_INDEX = 5
DEVICE_PCI = "0000:83:00.0"
DEVICE_SERIAL = "A0D5460520864"


def device_guard():
    import torch
    import torch_gcu  # noqa: F401

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
    torch.gcu.set_device(DEVICE_INDEX)
    if torch.gcu.current_device() != DEVICE_INDEX:
        raise SystemExit("torch.gcu.set_device did not take effect")
    return torch


def bench(torch, fn, warmup=3, iters=20):
    fn()
    torch.gcu.synchronize()
    for _ in range(warmup):
        fn()
    torch.gcu.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.gcu.Event(enable_timing=True)
        end = torch.gcu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.gcu.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), samples


def writer(path):
    handle = open(path, "w", encoding="utf-8")

    def emit(**fields):
        fields["device_index"] = DEVICE_INDEX
        fields["pci"] = DEVICE_PCI
        handle.write(json.dumps(fields, sort_keys=True) + "\n")
        handle.flush()
        summary = {k: v for k, v in fields.items() if not isinstance(v, list)}
        print(json.dumps(summary, sort_keys=True), flush=True)

    def close():
        handle.close()

    return emit, close


def repo_root():
    """Repository root: explicit override, else the checkout we are run from.

    The script is often staged outside the tree (the worker only exposes the
    clone over SSH), so cwd is authoritative and the file location is not.
    """
    from pathlib import Path

    override = os.environ.get("SAGE006_REPO")
    if override:
        return Path(override).resolve()
    return Path.cwd().resolve()


def kernel_path():
    return repo_root() / "src/flag_attn/runtime/backend/_enflame/sage_attention/attn_qk_int8_per_block.py"


def require_argv(script, expected):
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(f"usage: {script} JSONL", file=sys.stderr)
        raise SystemExit(2)
    return sys.argv[1]
