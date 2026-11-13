# Copyright 2026 FlagOS Contributors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regression coverage for the opt-in PARALLAX_STATIC_D64 launch table.

The static table only replaces the autotune sweep for the exact measured
shape B2/T256/H2/HQ8/D64/G4; everything else must keep the old dispatch.  The
tests below therefore check three things separately:

* the gate's shape/dtype/window/varlen decisions (cheap, no compilation);
* numerics for output plus all four gradients on the gated target shape,
  together with the exact launched configs and a zero autotune count;
* the T63 short-kernel fallback, which must stay untouched by the switch.

The real autotuned path for a nearby shape costs ~6 minutes of cold compile
per dtype, so it is exercised by agent-results/scripts/parallax007_ab.py
rather than by the default pytest run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from flag_attn import parallel_parallax
from flag_attn.runtime.backend._enflame.FLA.parallax import parallel as parallax_parallel

pytestmark = pytest.mark.parallel_parallax

DTYPES = (torch.float16, torch.bfloat16)
TOLERANCE = {torch.float16: 0.005, torch.bfloat16: 0.02}
STATIC = {
    torch.float16: {"fwd": (128, 1, 2), "dqr": (128, 1, 2), "dkv": (64, 1, 1)},
    torch.bfloat16: {"fwd": (128, 1, 2), "dqr": (64, 1, 1), "dkv": (64, 1, 2)},
}
TARGET_KERNELS = {
    "parallel_parallax_fwd_kernel": "fwd",
    "parallel_parallax_bwd_kernel_dqr_fused": "dqr",
    "parallel_parallax_bwd_kernel_dkv_grouped": "dkv",
}
SHORT_KERNELS = {
    "parallel_parallax_fwd_kernel_short",
    "parallel_parallax_bwd_kernel_short",
}


def _device() -> torch.device:
    return torch.device("gcu")


def _target_meta():
    return dict(B=2, T=256, H=2, HQ=8, K=64)


def _gate(meta, dtype=torch.float16, *, window=-1, cu_seqlens=None):
    q = torch.empty(1, dtype=dtype, device=_device())
    return parallax_parallel._static_d64_config(
        q,
        meta["B"],
        meta["T"],
        meta["H"],
        meta["HQ"],
        meta["K"],
        cu_seqlens,
        window,
    )


def _make_inputs(B, T, H, HQ, D, dtype, *, seed=42):
    torch.manual_seed(seed)
    q, r = (
        torch.randn(B, T, HQ, D, dtype=dtype, device=_device()).requires_grad_()
        for _ in range(2)
    )
    k, v = (
        torch.randn(B, T, H, D, dtype=dtype, device=_device()).requires_grad_()
        for _ in range(2)
    )
    torch.manual_seed(seed)
    grad_output = torch.randn(B, T, HQ, D, dtype=dtype, device=_device())
    return (q, r, k, v), grad_output


class _LaunchRecorder:
    """Record launches and autotune benchmarks for one test."""

    def __init__(self):
        self.launches = []
        self.autotune_configs = 0

    def __enter__(self):
        import triton.runtime.autotuner as autotuner_mod
        import triton.runtime.jit as jit_mod

        self._jit = jit_mod
        self._autotuner = autotuner_mod
        self._run = jit_mod.JITFunction.run
        self._bench = autotuner_mod.Autotuner._bench
        recorder = self

        def run(self_fn, *args, grid, warmup, **kwargs):
            if not warmup:
                recorder.launches.append(
                    {
                        "kernel": getattr(getattr(self_fn, "fn", self_fn), "__name__", ""),
                        "grid": None if callable(grid) else list(grid),
                        "BS": kwargs.get("BS"),
                        "num_warps": kwargs.get("num_warps"),
                        "num_stages": kwargs.get("num_stages"),
                    }
                )
            return recorder._run(self_fn, *args, grid=grid, warmup=warmup, **kwargs)

        def bench(self_fn, *args, config, **meta):
            recorder.autotune_configs += 1
            return recorder._bench(self_fn, *args, config=config, **meta)

        jit_mod.JITFunction.run = run
        autotuner_mod.Autotuner._bench = bench
        return self

    def __exit__(self, *exc_info):
        self._jit.JITFunction.run = self._run
        self._autotuner.Autotuner._bench = self._bench
        return False


def _reference_parallax(*inputs, scale, window_size):
    import importlib.util
    import sys

    path = Path(__file__).with_name("test_parallax.py")
    name = "_p007_reference_parallax"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module._reference_parallax(
        *inputs, scale=scale, window_size=window_size
    )


def _assert_all_close(actual, expected, tolerance, name):
    assert actual.shape == expected.shape, f"{name}: shape {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"{name}: dtype {actual.dtype} != {expected.dtype}"
    assert torch.isfinite(actual).all(), f"{name}: non-finite actual"
    delta = (actual.float() - expected.float()).abs()
    ratio = (delta.square().mean().sqrt() / expected.float().square().mean().sqrt().clamp_min(1e-8)).item()
    assert delta.max().item() <= 1e-6 or ratio < tolerance, (
        f"{name}: max_abs={delta.max().item():.6f} relative_rms={ratio:.6f} "
        f"tolerance={tolerance}"
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=("float16", "bfloat16"))
def test_static_gate_accepts_target_shape(monkeypatch, dtype):
    monkeypatch.setenv("PARALLAX_STATIC_D64", "1")
    assert parallax_parallel._static_d64_enabled() is True
    isinstance_result = _gate(_target_meta(), dtype)
    assert isinstance_result == STATIC[dtype]
    # The table is shared mutable state: the gate must hand back the same
    # per-role tuples that the source documents.
    for role, value in STATIC[dtype].items():
        assert parallax_parallel._STATIC_D64_CONFIGS[dtype][role] == value


@pytest.mark.parametrize(
    "delta",
    (
        {"B": 1},
        {"B": 3},
        {"T": 255},
        {"T": 512},
        {"H": 1},
        {"HQ": 4},
        {"HQ": 16},
        {"K": 128},
        {"K": 32},
    ),
    ids=lambda d: "-".join(f"{k}{v}" for k, v in d.items()),
)
def test_static_gate_rejects_nearby_shapes(monkeypatch, delta):
    monkeypatch.setenv("PARALLAX_STATIC_D64", "1")
    meta = _target_meta()
    meta.update(delta)
    assert _gate(meta) is None


def test_static_gate_rejects_window_varlen_and_other_dtypes(monkeypatch):
    monkeypatch.setenv("PARALLAX_STATIC_D64", "1")
    meta = _target_meta()
    assert _gate(meta, window=64) is None
    assert _gate(meta, window=0) is None
    cu_seqlens = torch.tensor([0, 128, 256], dtype=torch.int32, device=_device())
    assert _gate(meta, cu_seqlens=cu_seqlens) is None
    # float32 is never a valid attention input, but the gate must also refuse
    # to index the table with it rather than raising KeyError.
    assert _gate(meta, torch.float32) is None


def test_static_gate_defaults_off(monkeypatch):
    monkeypatch.delenv("PARALLAX_STATIC_D64", raising=False)
    assert parallax_parallel._static_d64_enabled() is False
    assert _gate(_target_meta()) is None
    for value in ("", "0", "false", "no"):
        monkeypatch.setenv("PARALLAX_STATIC_D64", value)
        assert _gate(_target_meta()) is None
    for value in ("1", "true", "yes"):
        monkeypatch.setenv("PARALLAX_STATIC_D64", value)
        assert _gate(_target_meta()) == STATIC[torch.float16]


@pytest.mark.parametrize("dtype", DTYPES, ids=("float16", "bfloat16"))
def test_static_target_output_and_all_gradients(monkeypatch, dtype):
    monkeypatch.setenv("PARALLAX_STATIC_D64", "1")
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    inputs, grad_output = _make_inputs(2, 256, 2, 8, 64, dtype)

    reference_inputs = tuple(t.detach().clone().requires_grad_() for t in inputs)
    expected = _reference_parallax(
        *reference_inputs, scale=64**-0.5, window_size=None
    )
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad_output)

    with _LaunchRecorder() as recorder:
        actual = parallel_parallax(*inputs, scale=64**-0.5, window_size=None)
        actual_grads = torch.autograd.grad(actual, inputs, grad_output)
        torch.gcu.synchronize()

    assert recorder.autotune_configs == 0, (
        f"static target must not autotune, saw {recorder.autotune_configs} configs"
    )
    assert {l["kernel"] for l in recorder.launches} == set(TARGET_KERNELS)
    for launch in recorder.launches:
        role = TARGET_KERNELS[launch["kernel"]]
        assert (launch["BS"], launch["num_warps"], launch["num_stages"]) == STATIC[dtype][role]
        assert launch["grid"] is not None
    grids = {l["kernel"]: l["grid"] for l in recorder.launches}
    assert grids["parallel_parallax_fwd_kernel"] == [4, 16]
    assert grids["parallel_parallax_bwd_kernel_dqr_fused"] == [4, 16]
    assert grids["parallel_parallax_bwd_kernel_dkv_grouped"] == [4, 4]

    tolerance = TOLERANCE[dtype]
    _assert_all_close(actual, expected, tolerance, "o")
    for name, got, want in zip(("dq", "dr", "dk", "dv"), actual_grads, expected_grads):
        _assert_all_close(got, want, tolerance, name)
    assert all(torch.isfinite(g).all() for g in actual_grads)


def test_static_switch_leaves_t63_short_fallback(monkeypatch):
    monkeypatch.setenv("PARALLAX_STATIC_D64", "1")
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    inputs, grad_output = _make_inputs(1, 63, 1, 1, 64, torch.float16)
    with _LaunchRecorder() as recorder:
        actual = parallel_parallax(*inputs, scale=64**-0.5, window_size=None)
        actual_grads = torch.autograd.grad(actual, inputs, grad_output)
        torch.gcu.synchronize()
    assert recorder.autotune_configs == 0
    assert {l["kernel"] for l in recorder.launches} == SHORT_KERNELS
    assert all(l["grid"] == [1, 1] for l in recorder.launches)

    reference_inputs = tuple(t.detach().clone().requires_grad_() for t in inputs)
    expected = _reference_parallax(
        *reference_inputs, scale=64**-0.5, window_size=None
    )
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad_output)
    _assert_all_close(actual, expected, TOLERANCE[torch.float16], "o")
    for name, got, want in zip(("dq", "dr", "dk", "dv"), actual_grads, expected_grads):
        _assert_all_close(got, want, TOLERANCE[torch.float16], name)


def test_static_target_rejects_float32(monkeypatch):
    monkeypatch.setenv("PARALLAX_STATIC_D64", "1")
    inputs = tuple(
        torch.randn((2, 256, 8, 64), dtype=torch.float32, device=_device())
        for _ in range(4)
    )
    with pytest.raises(TypeError, match="requires bf16 or fp16"):
        parallel_parallax(*inputs)
