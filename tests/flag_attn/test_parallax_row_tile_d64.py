# Copyright 2026 FlagOS Contributors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regression coverage for the opt-in PARALLAX_ROW_TILE_D64 launch shape.

PARALLAX-004..007 tuned only BS / num_warps / num_stages for the dense
B2/T256/H2/HQ8/D64/G4 shape; the query-row tile BT was pinned to 64.  This
experiment widens BT and is gated exactly like the static table, so the tests
check the gate, the launched (BS, BT, warps, stages) values, numerics for the
output plus all four gradients, and that the neighbouring shapes still take
the untouched autotuned route.

The nearby-shape autotuned path costs minutes of cold compile, so the
end-to-end identity check (flag on vs off, plus FP32-reference agreement for
the output and all four gradients) lives in
agent-results/scripts/parallax008_nearby.py rather than in the default pytest
run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from flag_attn import parallel_parallax
from flag_attn.runtime.backend._enflame.FLA.parallax import parallel as parallax_parallel

pytestmark = pytest.mark.parallel_parallax

DTYPES = (torch.float16, torch.bfloat16)
TOLERANCE = {torch.float16: 0.005, torch.bfloat16: 0.02}
ROW_TILE = {
    torch.float16: {"fwd": (128, 256, 1, 1), "dqr": (128, 128, 1, 1), "dkv": (64, 256, 1, 1)},
    torch.bfloat16: {"fwd": (128, 256, 1, 1), "dqr": (128, 128, 1, 1), "dkv": (64, 256, 1, 1)},
}
TARGET_KERNELS = {
    "parallel_parallax_fwd_kernel": "fwd",
    "parallel_parallax_bwd_kernel_dqr_fused": "dqr",
    "parallel_parallax_bwd_kernel_dkv_grouped": "dkv",
}


def _device() -> torch.device:
    return torch.device("gcu")


def _target_meta():
    return dict(B=2, T=256, H=2, HQ=8, K=64)


def _gate(meta, dtype=torch.float16, *, window=-1, cu_seqlens=None):
    q = torch.empty(1, dtype=dtype, device=_device())
    return parallax_parallel._row_tile_d64_config(
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
    """Record non-warmup launches and count autotune benchmarks."""

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
                        "BT": kwargs.get("BT"),
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
    path = Path(__file__).with_name("test_parallax.py")
    name = "_p008_reference_parallax"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module._reference_parallax(*inputs, scale=scale, window_size=window_size)


def _assert_all_close(actual, expected, tolerance, name):
    assert actual.shape == expected.shape, f"{name}: shape {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"{name}: dtype {actual.dtype} != {expected.dtype}"
    assert torch.isfinite(actual).all(), f"{name}: non-finite actual"
    delta = (actual.float() - expected.float()).abs()
    ratio = (delta.square().mean().sqrt() / expected.float().square().mean().sqrt().clamp_min(1e-8)).item()
    assert delta.max().item() <= 1e-6 or ratio < tolerance, (
        f"{name}: max_abs={delta.max().item():.6f} relative_rms={ratio:.6f} tolerance={tolerance}"
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=("float16", "bfloat16"))
def test_row_tile_gate_accepts_target_shape(monkeypatch, dtype):
    monkeypatch.setenv("PARALLAX_ROW_TILE_D64", "1")
    assert parallax_parallel._row_tile_d64_enabled() is True
    assert _gate(_target_meta(), dtype) == ROW_TILE[dtype]
    for role, value in ROW_TILE[dtype].items():
        assert parallax_parallel._ROW_TILE_D64_CONFIGS[dtype][role] == value


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
        {"K": 32},
        {"K": 128},
    ),
)
def test_row_tile_gate_rejects_other_shapes(monkeypatch, delta):
    monkeypatch.setenv("PARALLAX_ROW_TILE_D64", "1")
    meta = _target_meta() | delta
    assert _gate(meta, torch.float16) is None


@pytest.mark.parametrize("window", (0, 16, 64))
def test_row_tile_gate_rejects_window(monkeypatch, window):
    monkeypatch.setenv("PARALLAX_ROW_TILE_D64", "1")
    assert _gate(_target_meta(), torch.float16, window=window) is None


def test_row_tile_gate_rejects_varlen(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROW_TILE_D64", "1")
    cu_seqlens = torch.tensor([0, 256], dtype=torch.int32, device=_device())
    assert _gate(_target_meta(), torch.float16, cu_seqlens=cu_seqlens) is None


def test_row_tile_gate_rejects_float32(monkeypatch):
    monkeypatch.setenv("PARALLAX_ROW_TILE_D64", "1")
    assert _gate(_target_meta(), torch.float32) is None


@pytest.mark.parametrize("value", ("", "0", "false", "no", "off"))
def test_row_tile_gate_disabled(monkeypatch, value):
    monkeypatch.setenv("PARALLAX_ROW_TILE_D64", value)
    assert parallax_parallel._row_tile_d64_enabled() is False
    assert _gate(_target_meta(), torch.float16) is None


@pytest.mark.parametrize("value", ("1", "true", "yes"))
def test_row_tile_gate_enabled_values(monkeypatch, value):
    monkeypatch.setenv("PARALLAX_ROW_TILE_D64", value)
    assert parallax_parallel._row_tile_d64_enabled() is True


@pytest.mark.parametrize("dtype", DTYPES, ids=("float16", "bfloat16"))
def test_row_tile_target_numerics_and_launches(monkeypatch, dtype):
    monkeypatch.setenv("PARALLAX_ROW_TILE_D64", "1")
    inputs, grad_output = _make_inputs(2, 256, 2, 8, 64, dtype)
    with _LaunchRecorder() as recorder:
        actual = parallel_parallax(*inputs, scale=64**-0.5)
        actual_grads = torch.autograd.grad(actual, inputs, grad_output)
    torch.gcu.synchronize()

    seen = {TARGET_KERNELS[l["kernel"]]: l for l in recorder.launches if l["kernel"] in TARGET_KERNELS}
    assert set(seen) == {"fwd", "dqr", "dkv"}, recorder.launches
    assert recorder.autotune_configs == 0
    for role, (bs, bt, warps, stages) in ROW_TILE[dtype].items():
        launch = seen[role]
        assert (launch["BS"], launch["BT"], launch["num_warps"], launch["num_stages"]) == (
            bs,
            bt,
            warps,
            stages,
        ), (role, launch)
        assert launch["grid"] == [256 // bt if role != "dkv" else 256 // bs, 16 if role != "dkv" else 4]

    reference_inputs = tuple(t.detach().clone().requires_grad_() for t in inputs)
    expected = _reference_parallax(*reference_inputs, scale=64**-0.5, window_size=None)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad_output)
    _assert_all_close(actual, expected, TOLERANCE[dtype], "o")
    for name, got, want in zip(("dq", "dr", "dk", "dv"), actual_grads, expected_grads):
        _assert_all_close(got, want, TOLERANCE[dtype], name)


@pytest.mark.parametrize(
    "meta",
    (
        {"B": 1, "T": 256, "H": 2, "HQ": 8, "K": 64},   # nearby batch
        {"B": 1, "T": 63, "H": 1, "HQ": 1, "K": 64},    # short fallback
        {"B": 2, "T": 256, "H": 2, "HQ": 8, "K": 64, "window": 64},  # sliding window
    ),
)
def test_row_tile_nearby_shapes_are_not_gated(monkeypatch, meta):
    """Neighbouring shapes must keep the untouched generic dispatch.

    Numerics for these shapes are validated end-to-end by
    agent-results/scripts/parallax008_nearby.py: B1/T256, B2/T128 and B1/T63
    are bit-identical with the flag on and off, for both dtypes.  The
    autotuned nearby path costs minutes of cold compile per dtype, which does
    not belong in the default pytest run.
    """
    monkeypatch.setenv("PARALLAX_ROW_TILE_D64", "1")
    meta = dict(meta)
    window = meta.pop("window", -1)
    assert _gate(meta, torch.float16, window=window) is None


def test_row_tile_does_not_enable_static_table(monkeypatch):
    """The two switches are independent: row tile alone leaves STATIC off."""
    monkeypatch.delenv("PARALLAX_STATIC_D64", raising=False)
    monkeypatch.setenv("PARALLAX_ROW_TILE_D64", "1")
    assert parallax_parallel._static_d64_enabled() is False
    assert parallax_parallel._static_d64_config(
        torch.empty(1, dtype=torch.float16, device=_device()), 2, 256, 2, 8, 64, None, -1
    ) is None
