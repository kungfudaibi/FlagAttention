"""MSA-002 profiler probe: can this image attribute GCU device activity?"""

import sys

import torch
import torch_gcu  # noqa: F401

sys.path.insert(0, "/workspace/agent-workspaces/msa-stage-profile")
sys.path.insert(0, "/workspace/agent-workspaces/msa-stage-profile/src")
torch.gcu.set_device(1)

from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

x = torch.randn(1024, 1024, device="gcu:1", dtype=torch.bfloat16)
for _ in range(5):
    y = x * 2
torch.gcu.synchronize()

for acts in (
    [ProfilerActivity.CPU],
    [ProfilerActivity.GCU],
    [ProfilerActivity.CPU, ProfilerActivity.GCU],
    [ProfilerActivity.PrivateUse1],
):
    try:
        with profile(activities=acts, acc_events=True) as prof:
            with record_function("outer"):
                for _ in range(3):
                    y = x * 2
                torch.gcu.synchronize()
        ev = list(prof.events())
        print("ACTS", [a.name for a in acts], "events", len(ev))
        for e in ev[:8]:
            print("   ", e.name, "cpu_us", e.self_cpu_time_total, "dev_us", e.device_time_total)
        rows = prof.key_averages()
        print("   table rows", len(rows))
    except Exception as exc:  # noqa: BLE001
        print("ACTS", [a.name for a in acts], "FAILED", type(exc).__name__, exc)

# export chrome trace
try:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.GCU], acc_events=True) as prof:
        for _ in range(3):
            y = x * 2
        torch.gcu.synchronize()
    prof.export_chrome_trace("/tmp/msa002_trace.json")
    with open("/tmp/msa002_trace.json") as f:
        data = f.read()
    print("trace bytes", len(data), "has_kernel", '"kernel"' in data, "has_Memcpy" , "Memcpy" in data)
    print("trace head", data[:300].replace("\n", " "))
except Exception as exc:  # noqa: BLE001
    print("TRACE_FAILED", type(exc).__name__, exc)
