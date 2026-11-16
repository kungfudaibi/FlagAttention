# Copyright 2026 FlagOS Contributors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The three opt-in PARALLAX switches must never mix within one call.

`PARALLAX_STATIC_D64` (PARALLAX-007) and `PARALLAX_ROW_TILE_D64` (PARALLAX-008)
both fix launch shapes for the same target, and `PARALLAX_FIXED_D64` replaces
the backward kernels outright.  Precedence is STATIC > ROW_TILE, and FIXED
suppresses ROW_TILE.  Without this, a machine with more than one of them
exported launched FWD from one table and the backward pass from another, which
is a silent mixed-shape call rather than an obvious error.

These are asserted on the real launch tuples, not by re-deriving the dispatch
condition, so reordering the dispatch fails the test instead of being mirrored.
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

TARGET_KERNELS = {
    "parallel_parallax_fwd_kernel": "fwd",
    "parallel_parallax_bwd_kernel_dqr_fused": "dqr",
    "parallel_parallax_bwd_kernel_dkv_grouped": "dkv",
}
DTYPES = (torch.float16, torch.bfloat16)
FLAGS = ("PARALLAX_STATIC_D64", "PARALLAX_ROW_TILE_D64", "PARALLAX_FIXED_D64")


@pytest.fixture(scope="module")
def src():
    """Reuse the launch recorder and input builder from the row-tile tests."""
    path = Path(__file__).with_name("test_parallax_row_tile_d64.py")
    name = "_p008_precedence_source"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def _record(src, dtype):
    inputs, grad_output = src._make_inputs(2, 256, 2, 8, 64, dtype)
    with src._LaunchRecorder() as recorder:
        out = parallel_parallax(*inputs, scale=64**-0.5)
        torch.autograd.grad(out, inputs, grad_output)
    torch.gcu.synchronize()
    return recorder.launches


def _row_tile_launches(launches):
    seen = {}
    for launch in launches:
        role = TARGET_KERNELS.get(launch["kernel"])
        if role is not None:
            seen[role] = launch
    return seen


@pytest.mark.parametrize("dtype", DTYPES, ids=("float16", "bfloat16"))
def test_static_d64_wins_over_row_tile(monkeypatch, src, dtype):
    """Both flags on: every stage must come from the STATIC table, none from ROW."""
    monkeypatch.setenv("PARALLAX_ROW_TILE_D64", "1")
    monkeypatch.setenv("PARALLAX_STATIC_D64", "1")
    monkeypatch.delenv("PARALLAX_FIXED_D64", raising=False)

    seen = _row_tile_launches(_record(src, dtype))
    assert set(seen) == {"fwd", "dqr", "dkv"}, seen

    static = parallax_parallel._STATIC_D64_CONFIGS[dtype]
    row_tile = parallax_parallel._ROW_TILE_D64_CONFIGS[dtype]
    assert static["fwd"] != row_tile["fwd"][:3], "fixture assumption: tables must differ"

    for role in ("fwd", "dqr", "dkv"):
        bs, warps, stages = static[role]
        launch = seen[role]
        assert (launch["BS"], launch["num_warps"], launch["num_stages"]) == (bs, warps, stages), (
            role,
            launch,
            static[role],
        )
        # Only the row-tile path moves BT off its _BLOCK_T default.
        assert launch["BT"] == parallax_parallel._BLOCK_T, (role, launch)
        assert launch["BT"] != row_tile[role][1], (role, launch, row_tile[role])


@pytest.mark.parametrize("dtype", DTYPES, ids=("float16", "bfloat16"))
def test_row_tile_applies_when_static_is_off(monkeypatch, src, dtype):
    """Control for the test above: without STATIC the row-tile shapes are used."""
    monkeypatch.setenv("PARALLAX_ROW_TILE_D64", "1")
    monkeypatch.delenv("PARALLAX_STATIC_D64", raising=False)
    monkeypatch.delenv("PARALLAX_FIXED_D64", raising=False)

    seen = _row_tile_launches(_record(src, dtype))
    assert set(seen) == {"fwd", "dqr", "dkv"}, seen
    row_tile = parallax_parallel._ROW_TILE_D64_CONFIGS[dtype]
    for role, (bs, bt, warps, stages) in row_tile.items():
        launch = seen[role]
        assert (launch["BS"], launch["BT"], launch["num_warps"], launch["num_stages"]) == (
            bs,
            bt,
            warps,
            stages,
        ), (role, launch)


@pytest.mark.parametrize("dtype", DTYPES, ids=("float16", "bfloat16"))
def test_fixed_d64_suppresses_row_tile(monkeypatch, src, dtype):
    """FIXED replaces the backward kernels, so row-tile must stand down there."""
    monkeypatch.setenv("PARALLAX_ROW_TILE_D64", "1")
    monkeypatch.setenv("PARALLAX_FIXED_D64", "1")
    monkeypatch.delenv("PARALLAX_STATIC_D64", raising=False)

    launches = _record(src, dtype)
    row_tile = parallax_parallel._ROW_TILE_D64_CONFIGS[dtype]

    # FWD is owned by row tile only when FIXED is absent; with FIXED enabled the
    # forward launch must not carry a row-tile BT.
    fwd = _row_tile_launches(launches)["fwd"]
    assert fwd["BT"] != row_tile["fwd"][1], fwd
    assert fwd["BT"] == parallax_parallel._BLOCK_T, fwd

    # The backward pass uses the dense-aligned kernels, not the autotuned pair.
    kernels = {launch["kernel"] for launch in launches}
    assert "parallel_parallax_bwd_kernel_dqr_dense_aligned" in kernels, kernels
    assert "parallel_parallax_bwd_kernel_dkv_dense_aligned" in kernels, kernels


def test_row_tile_alone_does_not_activate_other_switches(monkeypatch):
    """ROW_TILE must not switch on either older path."""
    for flag in FLAGS:
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv("PARALLAX_ROW_TILE_D64", "1")
    assert parallax_parallel._row_tile_d64_enabled() is True
    assert parallax_parallel._static_d64_enabled() is False
    assert parallax_parallel._fixed_d64_enabled() is False
