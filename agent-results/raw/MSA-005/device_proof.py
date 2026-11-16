"""MSA-005 device/stream binding proof for GCU 2 (PCI 34:00.0, CPU 64-95)."""

import os
import subprocess
import sys

import torch
import torch_gcu  # noqa: F401


def efsmi(args):
    return subprocess.run(["efsmi", *args], capture_output=True, text=True).stdout


def dev2_memory():
    out = efsmi(["-i", "2", "-q", "-d", "MEMORY"])
    return [line.strip() for line in out.splitlines() if "Used Size" in line or "Total Size" in line]


def all_devices():
    rows = []
    for line in efsmi(["-L"]).splitlines():
        fields = line.split()
        if fields and fields[0].isdigit() and len(fields) >= 6:
            rows.append((fields[0], fields[1], fields[4], fields[5]))
    return rows


def busy_rows():
    rows = {}
    for index in range(torch.gcu.device_count()):
        out = efsmi(["-i", str(index), "-q", "-d", "MEMORY"])
        used = [line.strip() for line in out.splitlines() if "Used Size" in line]
        rows[index] = used[0] if used else "?"
    return rows


print("device_count:", torch.gcu.device_count())
print("efsmi_devices:", all_devices())
print("cpu_affinity:", sorted(os.sched_getaffinity(0)))
torch.gcu.set_device(2)
stream = torch.gcu.current_stream()
print("current_device_after_set:", torch.gcu.current_device())
print("current_stream:", stream, "stream_device:", stream.device)
print("memory_before:", dev2_memory())
print("used_before:", busy_rows())
buf = torch.empty(2 * 1024**3 // 4, dtype=torch.float32, device="gcu:2")
buf.fill_(1.0)
torch.gcu.synchronize()
print("during_2gib:", dev2_memory())
print("used_during:", busy_rows())
print("buf_device:", buf.device, "nbytes:", buf.numel() * buf.element_size())
del buf
torch.gcu.empty_cache()
torch.gcu.synchronize()
print("after_free:", dev2_memory())
print("used_after:", busy_rows())
print("allocated_bytes_now:", torch.gcu.memory_allocated(2))
if torch.gcu.current_device() != 2 or getattr(stream.device, "index", None) != 2:
    print("DEVICE_IDENTITY=FAIL")
    sys.exit(1)
print("DEVICE_IDENTITY=PASS")
sys.exit(0)
