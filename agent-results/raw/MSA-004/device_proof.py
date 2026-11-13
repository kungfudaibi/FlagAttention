"""MSA-004 device-binding proof for GCU 2 (PCI 34:00.0, CPU 64-95)."""

import os
import subprocess
import sys

import torch
import torch_gcu  # noqa: F401


def efsmi_dev2():
    out = subprocess.run(
        ["efsmi", "-i", "2", "-q", "-d", "MEMORY"], capture_output=True, text=True
    ).stdout
    keep = [line.strip() for line in out.splitlines() if "Used Size" in line or "Total Size" in line]
    return keep


def efsmi_list():
    out = subprocess.run(["efsmi", "-L"], capture_output=True, text=True).stdout
    return [line.strip() for line in out.splitlines() if line.startswith("2 ")]


print("device_count:", torch.gcu.device_count())
print("efsmi_dev2:", efsmi_list())
print("cpu_affinity:", sorted(os.sched_getaffinity(0)))
print("before:", efsmi_dev2())
torch.gcu.set_device(2)
print("current_device_after_set:", torch.gcu.current_device())
buf = torch.empty(2 * 1024**3 // 4, dtype=torch.float32, device="gcu:2")
buf.fill_(1.0)
torch.gcu.synchronize()
print("during_2gib:", efsmi_dev2())
print("buf_device:", buf.device, "nbytes:", buf.numel() * buf.element_size())
del buf
torch.gcu.empty_cache()
torch.gcu.synchronize()
print("after_free:", efsmi_dev2())
print("allocated_bytes_now:", torch.gcu.memory_allocated(2))
sys.exit(0)
