"""Record the environment and the device identity for GLA-008.

Writes one JSON object (not JSONL: it is a single snapshot) with the runtime
versions, the container-GCU to PCI mapping read from ``efsmi``, the device the
process is actually bound to, and the CPU affinity the benchmark is pinned to.

The point is to make the isolation claim checkable from the evidence alone:
``container_device == 4``, ``pci == "63:00.0"`` and ``cpu_affinity`` a subset of
``128-159``.  All three are asserted here, so a wrong device fails the script
instead of quietly producing numbers on the wrong card.
"""
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "agent-results/raw/gla008_env.json"
DEVICE = 4
WANT_PCI = "63:00.0"
PREFIX = "128-159"


def efsmi_devices():
    try:
        txt = subprocess.run(["efsmi"], text=True, capture_output=True,
                             timeout=60).stdout
    except Exception as exc:                       # pragma: no cover
        return {}, f"efsmi failed: {exc}"
    devs, cur = {}, None
    for line in txt.splitlines():
        m = re.match(r"\|\s*(\d+)\s+(\S+)\s*\|", line)
        if m:
            cur = int(m.group(1))
            blob = line
            if "MiB" in line:            # single-line cards (some efsmi builds)
                blob = line
            mm = re.search(r"(\d{2}:[0-9a-f]{2}\.[0-9])", line)
            devs[cur] = {"name": m.group(2),
                         "pci": mm.group(1) if mm else None}
            continue
        if cur is not None:
            mm = re.search(r"(\d{2}:[0-9a-f]{2}\.[0-9])", line)
            if mm and not devs[cur].get("pci"):
                devs[cur]["pci"] = mm.group(1)
            m2 = re.search(r"(\d+)MiB", line)
            if m2 and "mem_mib" not in devs[cur]:
                devs[cur]["mem_mib"] = int(m2.group(1))
    return devs, txt


def main():
    import torch
    import torch_gcu
    import triton

    torch.gcu.set_device(DEVICE)
    props = torch.gcu.get_device_properties(DEVICE)
    stream = torch.gcu.current_stream(DEVICE)
    t = torch.zeros(8, device=f"gcu:{DEVICE}")
    torch.gcu.synchronize(DEVICE)

    devs, raw = efsmi_devices()
    pci = (devs.get(DEVICE) or {}).get("pci")
    affinity = sorted(os.sched_getaffinity(0))
    rec = {
        "kind": "environment",
        "host": platform.node(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "cpu_model": next((ln.split(":", 1)[1].strip()
                           for ln in open("/proc/cpuinfo")
                           if ln.startswith("model name")), None),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_gcu": getattr(torch_gcu, "__version__", None),
        "triton": triton.__version__,
        "flagtree": os.environ.get("PYTHONPATH", ""),
        "container_device": DEVICE,
        "pci": pci,
        "expected_pci": WANT_PCI,
        "device_name": props.name,
        "device_cc": f"{props.major}.{props.minor}",
        "device_total_mem_mib": round(props.total_memory / 1048576, 1),
        "device_sm_count": getattr(props, "multi_processor_count", None),
        "visible_device_count": torch.gcu.device_count(),
        "current_device": torch.gcu.current_device(),
        "stream_device": str(stream.device),
        "tensor_device": str(t.device),
        "device_ok": (torch.gcu.current_device() == DEVICE
                      and str(t.device) == f"gcu:{DEVICE}"
                      and str(stream.device) == f"gcu:{DEVICE}"),
        "cpu_affinity": affinity,
        "cpu_affinity_ok": set(affinity) <= set(range(128, 160)),
        "efsmi": {str(k): v for k, v in devs.items()},
        "f32_default_env": os.environ.get("TRITON_F32_DEFAULT", "<unset>"),
        "experimental_flags": {
            k: os.environ.get(k)
            for k in ("FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP16_STATE",
                      "FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP32_STATE",
                      "FLAGGEMS_ENFLAME_GLA_PARALLEL_STATE",
                      "FLAGGEMS_ENFLAME_GLA_PARALLEL_BWD_STATE")},
    }
    if not rec["device_ok"]:
        print("GLA008_FAIL device binding", rec["current_device"],
              rec["stream_device"], rec["tensor_device"], file=sys.stderr)
        return 1
    if pci != WANT_PCI:
        print(f"GLA008_FAIL pci {pci} != {WANT_PCI}", file=sys.stderr)
        return 1
    if not rec["cpu_affinity_ok"]:
        print("GLA008_FAIL cpu affinity", affinity[:4], file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in
                      ("python", "torch", "torch_gcu", "triton", "pci",
                       "device_cc", "device_total_mem_mib", "stream_device",
                       "cpu_affinity_ok", "device_ok")}))
    print(f"GLA008_ENV_OK {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
