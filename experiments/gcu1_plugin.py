"""pytest plugin: pin every test allocation to container GCU 1 (host GCU 5)."""
import torch
import torch_gcu  # noqa: F401


def pytest_configure(config):
    torch.gcu.set_device(1)
    assert torch.gcu.current_device() == 1, torch.gcu.current_device()
    print(f"[p004] pinned pytest to GCU {torch.gcu.current_device()}", flush=True)
