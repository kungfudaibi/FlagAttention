"""MSA-002 launch-cost probe: isolate host dispatch from device work."""

import sys
import time

import torch
import torch_gcu  # noqa: F401
import triton
import triton.language as tl

sys.path.insert(0, "/workspace/agent-workspaces/msa-stage-profile")
sys.path.insert(0, "/workspace/agent-workspaces/msa-stage-profile/src")
torch.gcu.set_device(1)

print("device", torch.gcu.current_device(), flush=True)


@triton.jit
def _noop(x_ptr):
    pid = tl.program_id(0)
    tl.store(x_ptr + pid, tl.load(x_ptr + pid) + 1.0)


x = torch.zeros(8, device="gcu:1", dtype=torch.float32)
_noop[(1,)](x)
torch.gcu.synchronize()
print("noop compiled", x[0].item(), flush=True)

y = torch.empty(1024 * 1024, device="gcu:1", dtype=torch.bfloat16)
src = torch.randn(1024 * 1024, device="gcu:1", dtype=torch.bfloat16)


def timeit(label, fn, n=50):
    fn()
    torch.gcu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t1 = time.perf_counter()
    torch.gcu.synchronize()
    t2 = time.perf_counter()
    a = torch.gcu.Event(enable_timing=True)
    b = torch.gcu.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.gcu.synchronize()
    ev = a.elapsed_time(b)
    print(
        "%s n=%d host_enqueue_us=%.1f sync_us=%.1f event_total_ms=%.3f event_per_us=%.1f"
        % (label, n, (t1 - t0) * 1e6 / n, (t2 - t1) * 1e6, ev, ev * 1000 / n),
        flush=True,
    )


timeit("noop_triton_1cta", lambda: _noop[(1,)](x), 50)
timeit("torch_copy_1MB", lambda: y.copy_(src), 50)
timeit("torch_mul_1MB", lambda: torch.mul(src, 2, out=y), 50)
timeit("torch_add_scalar_8", lambda: x.add_(1.0), 50)
timeit("torch.empty_1MB", lambda: torch.empty(1024 * 1024, device="gcu:1", dtype=torch.bfloat16), 50)
timeit("event_record_only", lambda: torch.gcu.Event(enable_timing=True).record(), 50)
timeit("sync_only", lambda: torch.gcu.synchronize(), 50)
