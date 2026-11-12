"""Semantic-guard checks: the opt-in dtype flags must not bypass route guards.

Each case calls chunk_fwd_h / chunk_bwd_dh directly with the experiment flag on
and records which kernels actually launched.  A guard-honouring implementation
falls back to chunk_fwd_kernel_h / chunk_bwd_kernel_dh.
"""
import json
import os

import torch
import torch.nn.functional as F

os.environ["FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP16_STATE"] = "1"
os.environ["FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP32_STATE"] = "1"
os.environ["TRITON_F32_DEFAULT"] = "ieee"

import triton  # noqa: E402
from flag_attn.runtime.backend._enflame.FLA import chunk_h as Hmod  # noqa: E402

ATTEMPT, DONE = {}, {}
PROBES = ("_chunk_fwd_local_state_kernel", "_chunk_bwd_local_state_kernel",
          "_chunk_fwd_state_scan_kernel", "_chunk_bwd_state_scan_fp32_2d_group_kernel",
          "chunk_fwd_kernel_h", "chunk_bwd_kernel_dh")


class _Probe:
    def __init__(self, name, real):
        self.name, self.real = name, real

    def __getitem__(self, grid):
        launcher = self.real[grid]

        def call(*args, **kwargs):
            ATTEMPT[self.name] = ATTEMPT.get(self.name, 0) + 1
            out = launcher(*args, **kwargs)
            DONE[self.name] = DONE.get(self.name, 0) + 1
            return out

        return call


for _n in PROBES:
    if hasattr(Hmod, _n):
        setattr(Hmod, _n, _Probe(_n, getattr(Hmod, _n)))

dev = torch.device("gcu:0")
B, T, H, K, V = 1, 128, 1, 64, 64
dt = torch.float16
torch.manual_seed(42)
k = torch.rand(B, T, H, K, dtype=dt, device=dev)
v = torch.rand(B, T, H, V, dtype=dt, device=dev)
q = torch.rand(B, T, H, K, dtype=dt, device=dev)
g = F.logsigmoid(torch.rand(B, T, H, K, dtype=dt, device=dev))
h0 = torch.rand(B, H, K, V, dtype=torch.float32, device=dev)
do = torch.randn(B, T, H, V, dtype=dt, device=dev)
dht = torch.randn(B, H, K, V, dtype=torch.float32, device=dev)
cu = torch.tensor([0, 64, 128], dtype=torch.int32, device=dev)
gk = torch.cumsum(g, dim=1) * 1.4426950216

rows = []


def probe(name, fn):
    ATTEMPT.clear(); DONE.clear()
    err = None
    try:
        out = fn()
        torch.gcu.synchronize()
        del out
    except Exception as exc:  # record, do not hide
        err = f"{type(exc).__name__}: {exc}"
    rows.append({"case": name, "error": err,
                 "attempt": dict(ATTEMPT), "done": dict(DONE)})
    print(name, "error=", err, "done=", dict(DONE), flush=True)


base_kw = dict(v=v, g=None, gk=gk, gv=None, h0=h0, output_final_state=True,
               chunk_size=64, states_in_fp32=False, state_v_first=False)
probe("fwd_varlen_cu_seqlens",
      lambda: Hmod.chunk_fwd_h(k=k, cu_seqlens=cu, **base_kw))
probe("fwd_state_v_first",
      lambda: Hmod.chunk_fwd_h(k=k, **{**base_kw, "state_v_first": True}))
probe("fwd_given_gate_tensor",
      lambda: Hmod.chunk_fwd_h(k=k, **{**base_kw, "g": g, "gk": None}))
probe("fwd_states_in_fp32_bwd_parallel_on",
      lambda: Hmod.chunk_fwd_h(k=k, **{**base_kw, "states_in_fp32": True}))
Hmod.PARALLEL_BWD_STATE_ENABLED = False
probe("fwd_states_in_fp32_bwd_parallel_off",
      lambda: Hmod.chunk_fwd_h(k=k, **{**base_kw, "states_in_fp32": True}))
Hmod.PARALLEL_BWD_STATE_ENABLED = True
probe("fwd_split_size",
      lambda: Hmod.chunk_fwd_h(k=k, split_size=128, **base_kw))

bwd_kw = dict(g=None, gk=gk, gv=None, h0=h0, dht=dht, scale=K ** -0.5,
              cu_seqlens=None, chunk_size=64, states_in_fp32=True,
              state_v_first=False)
probe("bwd_varlen_cu_seqlens",
      lambda: Hmod.chunk_bwd_dh(q=q, k=k, v=v, do=do,
                                **{**bwd_kw, "cu_seqlens": cu}))
probe("bwd_given_gate_tensor",
      lambda: Hmod.chunk_bwd_dh(q=q, k=k, v=v, do=do,
                                **{**bwd_kw, "g": g, "gk": None}))
print("GLA006_GUARD " + json.dumps(rows))
