#!/usr/bin/env python
"""GDN2-003 preflight: check GCU0 (PCI 23:00.0) and reconcile efsmi vs PyTorch.

Records the visible device count, PCI address, CPU affinity, torch allocator
accounting and the efsmi device-memory counter before/after one warm forward.
Device memory reported by efsmi includes the GCU context and lazy modules, so it
is expected to be larger than ``torch.gcu.memory_allocated()``, which counts
only live tensor storage.

Usage::

    TOPS_VISIBLE_DEVICES=0 taskset -c 0-31 /flagos/bin/python \
        benchmark/gdn2_forward_preflight.py --out agent-results/GDN2-003-preflight.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import subprocess
import time
from pathlib import Path

DEVICE = "gcu:0"
EXPECTED_PCI = "23:00.0"


def efsmi_json():
    proc = subprocess.run(["efsmi", "-q", "-d", "PCIE", "--json-format"],
                          capture_output=True, text=True, timeout=30, check=False)
    return json.loads(proc.stdout)


def efsmi_device(index=0):
    proc = subprocess.run(["efsmi", "-q", "-i", str(index)],
                          capture_output=True, text=True, timeout=30, check=False)
    text = proc.stdout

    def grab(pattern):
        match = re.search(pattern, text)
        return match.group(1) if match else None

    return {
        "used_mib": grab(r"Used Size\s*:\s*(\d+)\s*MiB"),
        "reserved_mib": grab(r"Reserved Size\s*:\s*(\d+)\s*MiB"),
        "free_mib": grab(r"Free Size\s*:\s*(\d+)\s*MiB"),
        "total_mib": grab(r"Total Size\s*:\s*(\d+)\s*MiB"),
        "gcu_usage_pct": grab(r"GCU Usage\s*:\s*([\d.]+)\s*%"),
        "power_w": grab(r"Power Usage\s*:\s*(\d+)\s*W"),
        "temp_c": grab(r"GCU Temp\s*:\s*(\d+)"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import torch

    sys_path_root = Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0, str(sys_path_root / "src"))
    from flag_attn import chunk_gdn2  # noqa: E402
    import importlib
    impl = importlib.import_module(
        "flag_attn.runtime.backend._enflame.FLA.gdn2.chunk_gdn2")

    pcie = efsmi_json()
    dev0 = next(item for item in pcie["devices"] if str(item["id"]) == "0")
    info = dev0["PCIe Info"]
    pci = f"{info['Bus']}:{info['Dev']}.{info['Func']}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": 1,
        "task": "GDN2-003",
        "kind": "preflight",
        "hostname": platform.node(),
        "torch": torch.__version__,
        "toplevel_visible": os.environ.get("TOPS_VISIBLE_DEVICES"),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count": torch.gcu.device_count(),
        "current_device": torch.gcu.current_device(),
        "device_name": torch.gcu.get_device_name(0),
        "pci0": pci,
        "pci0_matches_expected": pci == EXPECTED_PCI,
        "link": {k: info["Link Info"][k] for k in ("Cur Link Speed", "Cur Link Width")},
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "tile_adapt": impl._S60_TILES_ADAPT,
        "impl_file": impl.__file__,
        "efsmi_before": efsmi_device(0),
        "torch_allocated_mib_before": torch.gcu.memory_allocated() / 2**20,
        "torch_reserved_mib_before": torch.gcu.memory_reserved() / 2**20,
        "started_at": time.time(),
    }

    if pci != EXPECTED_PCI or torch.gcu.device_count() != 1:
        record["ok"] = False
        out.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        return 2

    B, T, H, K, V = 2, 512, 8, 64, 64
    dtype = torch.bfloat16
    torch.manual_seed(42)
    q = (torch.randn(B, T, H, K) / math.sqrt(K)).to(dtype).to(DEVICE)
    k = (torch.randn(B, T, H, K) / math.sqrt(K)).to(dtype).to(DEVICE)
    v = torch.randn(B, T, H, V).to(dtype).to(DEVICE)
    g = (-torch.rand(B, T, H, K) * 0.1).to(dtype).to(DEVICE)
    b = torch.rand(B, T, H, K).to(dtype).to(DEVICE)
    w = torch.rand(B, T, H, V).to(dtype).to(DEVICE)
    h0 = (torch.randn(B, H, K, V) * 0.01).to(DEVICE)
    torch.gcu.reset_peak_memory_stats()
    allocated_before_call = torch.gcu.memory_allocated()
    with torch.inference_mode():
        o, ht = chunk_gdn2(q, k, v, g, b, w, scale=K ** -0.5, initial_state=h0,
                           output_final_state=True, chunk_size=64)
    torch.gcu.synchronize()
    record.update({
        "ok": True,
        "probe_shape": [B, T, H, K, V],
        "torch_allocated_mib_after": torch.gcu.memory_allocated() / 2**20,
        "torch_peak_alloc_mib": (torch.gcu.max_memory_allocated()
                                 - allocated_before_call) / 2**20,
        "torch_peak_total_alloc_mib": torch.gcu.max_memory_allocated() / 2**20,
        "torch_reserved_mib_after": torch.gcu.memory_reserved() / 2**20,
        "efsmi_after": efsmi_device(0),
        "finite_o": bool(torch.isfinite(o).all().item()),
        "finite_ht": bool(torch.isfinite(ht).all().item()),
        "ended_at": time.time(),
    })
    out.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
