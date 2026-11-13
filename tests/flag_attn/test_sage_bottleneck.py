# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SAGE-002/SAGE-003 correctness and dispatch checks for the Enflame Sage path.

The tests are deliberately independent of the benchmark harness: they compare
the public full call against PyTorch SDPA on the same FP16 inputs and the same
output dtype, check that the full call is exactly the pre-quantized attention
call on the same quantized tensors, and exercise the non-default kernel
configuration fallbacks. The final test verifies that the operator registry
selects the Enflame override while the generic implementation remains the
fallback for other vendors.
"""

import os
import subprocess
import sys
import time

import pytest
import torch
import torch.nn.functional as F

from flag_attn import sage_attention
from flag_attn.sage_attention import per_block_int8
from flag_attn.runtime.backend import get_backend_name, resolve_operator
from flag_attn.testing import backend as test_backend


pytestmark = [
    pytest.mark.skipif(
        not test_backend.supports_operator("sage_attention"),
        reason=test_backend.skip_reason("sage_attention"),
    ),
]

IS_ENFLAME = get_backend_name() == "enflame"
# SAGE-002 is assigned container GCU 1 (PCI 83:00.0). Tensor device strings are
# explicit so the test never touches GCU 0 by accident; the override exists for
# running the same checks on another assigned device.
DEVICE = os.environ.get(
    "SAGE002_TEST_DEVICE", "gcu:5" if IS_ENFLAME else test_backend.device
)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AB_SCRIPT = os.path.join(REPO_ROOT, "agent-results", "scripts", "sage003_ab.py")


@pytest.fixture(autouse=True)
def _pin_enflame_device():
    if not IS_ENFLAME:
        yield
        return
    if DEVICE != "gcu:0" and torch.gcu.max_memory_allocated(0) > 0:
        # This image aborts (topsMemcpyAsync topsErrorInvalidValue) when one
        # process holds contexts on both GCUs, so a mixed-device process is
        # skipped rather than crashed; run this file on its own for coverage.
        pytest.skip(
            "another GCU device already has a context in this process; "
            "this image cannot mix GCU 0 and GCU 1 in one process"
        )
    index = int(DEVICE.split(":")[1]) if ":" in DEVICE else 0
    previous = torch.gcu.current_device()
    torch.gcu.set_device(index)
    try:
        yield
    finally:
        torch.gcu.set_device(previous)


def _inputs(seq_len, dtype=torch.float16, batch=1, heads=2, head_dim=64):
    torch.manual_seed(2026)
    shape = (batch, heads, seq_len, head_dim)
    q = torch.randn(shape, device=DEVICE, dtype=dtype)
    k = torch.randn(shape, device=DEVICE, dtype=dtype)
    v = torch.randn(shape, device=DEVICE, dtype=dtype)
    return q, k, v


def _metrics(actual, expected):
    a = actual.float().flatten()
    e = expected.float().flatten()
    diff = a - e
    cosine = F.cosine_similarity(a, e, dim=0).item()
    relative_l2 = (diff.norm() / e.norm().clamp_min(1e-8)).item()
    return cosine, relative_l2, diff.abs().max().item()


@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [128, 512])
def test_full_path_matches_sdpa_same_dtype(seq_len, output_dtype):
    q, k, v = _inputs(seq_len)
    q8, q_scale, k8, k_scale = per_block_int8(q, k)
    actual, lse = sage_attention(
        q8, k8, v, q_scale, k_scale, output_dtype=output_dtype
    )
    expected = F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=False
    ).to(output_dtype)

    assert actual.dtype == output_dtype
    assert actual.shape == q.shape
    assert torch.isfinite(actual).all().item()
    # return_lse=False keeps the historical empty placeholder instead of None.
    assert lse.numel() == 0

    cosine, relative_l2, max_abs = _metrics(actual, expected)
    assert cosine >= 0.99, f"cosine {cosine:.6f} below 0.99"
    assert relative_l2 <= 0.05, f"relative_l2 {relative_l2:.6f} above 0.05"
    # FP16 output rounding is far looser than this bound; keep a raw check so a
    # silently broken kernel cannot pass through the aggregate gate alone.
    assert max_abs <= 0.05, f"max_abs {max_abs:.6f} above 0.05"


def test_return_lse_shape_and_dtype():
    q, k, v = _inputs(384)
    q8, q_scale, k8, k_scale = per_block_int8(q, k)
    out, lse = sage_attention(
        q8, k8, v, q_scale, k_scale, return_lse=True
    )
    assert out.shape == q.shape
    assert lse.shape == (1, 2, 384)
    assert lse.dtype == torch.float32
    assert torch.isfinite(lse).all().item()


def test_full_call_is_exactly_prequantized_attention_on_same_tensors():
    q, k, v = _inputs(512)
    q8, q_scale, k8, k_scale = per_block_int8(q, k)
    pure, _ = sage_attention(q8, k8, v, q_scale, k_scale)
    fq8, fq_scale, fk8, fk_scale = per_block_int8(q, k)
    full, _ = sage_attention(fq8, fk8, v, fq_scale, fk_scale)

    torch.testing.assert_close(fq8, q8, rtol=0, atol=0)
    torch.testing.assert_close(fq_scale, q_scale, rtol=0, atol=0)
    torch.testing.assert_close(fk8, k8, rtol=0, atol=0)
    torch.testing.assert_close(fk_scale, k_scale, rtol=0, atol=0)
    # Identical quantized tensors must give bitwise identical attention output,
    # which isolates the attention stage from the quantization stage.
    assert torch.equal(pure, full)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"predequantize_k": False},
        {"tile_major_predequant_k": False},
        {"qk_scale_before_dot": False},
        {"prefetch_v": False, "prefetch_k_next": False},
    ],
)
def test_forward_configuration_fallback_paths(kwargs):
    q, k, v = _inputs(256)
    q8, q_scale, k8, k_scale = per_block_int8(q, k)
    default, _ = sage_attention(q8, k8, v, q_scale, k_scale)
    fallback, _ = sage_attention(q8, k8, v, q_scale, k_scale, **kwargs)
    expected = F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=False
    )

    assert fallback.dtype == torch.float16
    assert fallback.shape == q.shape
    assert torch.isfinite(fallback).all().item()
    cosine, relative_l2, max_abs = _metrics(fallback, expected)
    assert cosine >= 0.99, f"{kwargs}: cosine {cosine:.6f} below 0.99"
    assert relative_l2 <= 0.05, f"{kwargs}: relative_l2 {relative_l2:.6f} above 0.05"
    assert max_abs <= 0.05, f"{kwargs}: max_abs {max_abs:.6f} above 0.05"
    # A fallback must stay the same computation, not silently change semantics.
    cosine_vs_default, _, _ = _metrics(fallback, default)
    assert cosine_vs_default >= 0.9999, (
        f"{kwargs}: fallback diverges from default, cosine {cosine_vs_default:.6f}"
    )


def test_per_block_int8_shapes_determinism_and_scale_positivity():
    q, k, _ = _inputs(300, batch=2, heads=4)
    q8, q_scale, k8, k_scale = per_block_int8(q, k)
    fq8, fq_scale, fk8, fk_scale = per_block_int8(q, k)

    assert q8.dtype == torch.int8
    assert k8.dtype == torch.int8
    assert q8.shape == q.shape
    assert k8.shape == k.shape
    assert q_scale.shape == (2, 4, (300 + 127) // 128)
    assert k_scale.shape == (2, 4, (300 + 63) // 64)
    assert torch.equal(q8, fq8)
    assert torch.equal(k8, fk8)
    assert torch.equal(q_scale, fq_scale)
    assert torch.equal(k_scale, fk_scale)
    assert torch.isfinite(q_scale).all().item()
    assert torch.isfinite(k_scale).all().item()
    assert (q_scale > 0).all().item()
    assert (k_scale > 0).all().item()


@pytest.mark.skipif(not IS_ENFLAME, reason="registry override check is Enflame-specific")
def test_registry_selects_enflame_override_and_keeps_generic_fallback():
    generic_module = "flag_attn.sage_attention.attn_qk_int8_per_block"
    enflame = resolve_operator(
        "sage_attention", generic_module, "forward", vendor="enflame"
    )
    generic = resolve_operator(
        "sage_attention", generic_module, "forward", vendor="cpu"
    )

    assert callable(enflame)
    assert callable(generic)
    assert enflame is not generic
    assert enflame.__module__.startswith("flag_attn.runtime.backend._enflame.")
    assert generic.__module__ == generic_module

    generic_quant = resolve_operator(
        "per_block_int8",
        "flag_attn.sage_attention.quant_per_block",
        "per_block_int8",
        vendor="cpu",
    )
    enflame_quant = resolve_operator(
        "sage_attention_per_block_int8",
        "flag_attn.sage_attention.quant_per_block",
        "per_block_int8",
        vendor="enflame",
    )
    assert enflame_quant.__module__.startswith(
        "flag_attn.runtime.backend._enflame."
    )
    assert generic_quant.__module__ == "flag_attn.sage_attention.quant_per_block"


# ---------------------------------------------------------------------------
# SAGE-003: device identity, real device-fused baseline, and the quantizer
# launch-config optimization.
# ---------------------------------------------------------------------------
def _device_fused_sdpa(q, k, v):
    """The GCU-native fused attention op, not the F.sdpa host fallback."""
    return torch.ops.aten._scaled_dot_product_flash_attention(
        q, k, v, 0.0, False, False
    )[0]


@pytest.mark.skipif(not IS_ENFLAME, reason="device identity check is Enflame-specific")
def test_tensors_and_stream_live_on_the_assigned_device():
    index = int(DEVICE.split(":")[1])
    q, k, v = _inputs(128)
    q8, q_scale, k8, k_scale = per_block_int8(q, k)
    out, lse = sage_attention(q8, k8, v, q_scale, k_scale, return_lse=True)

    assert torch.gcu.current_device() == index
    assert str(torch.gcu.current_stream().device) == DEVICE
    for name, tensor in (
        ("q", q), ("k", k), ("v", v), ("q8", q8), ("k8", k8),
        ("q_scale", q_scale), ("k_scale", k_scale), ("out", out), ("lse", lse),
    ):
        assert str(tensor.device) == DEVICE, f"{name} lives on {tensor.device}"


@pytest.mark.skipif(not IS_ENFLAME, reason="device-fused baseline is Enflame-specific")
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim,seq_len", [(64, 512), (64, 2048), (128, 1024)])
def test_matches_device_fused_sdpa_not_only_host_fallback(head_dim, seq_len, output_dtype):
    q, k, v = _inputs(seq_len, head_dim=head_dim)
    q8, q_scale, k8, k_scale = per_block_int8(q, k)
    actual, _ = sage_attention(
        q8, k8, v, q_scale, k_scale, output_dtype=output_dtype
    )
    expected = _device_fused_sdpa(q, k, v).to(output_dtype)

    # Prove the baseline really ran on the GCU before trusting the comparison.
    assert str(expected.device) == DEVICE
    assert torch.isfinite(expected).all().item()
    assert torch.isfinite(actual).all().item()

    cosine, relative_l2, max_abs = _metrics(actual, expected)
    assert cosine >= 0.99, f"D{head_dim} T{seq_len}: cosine {cosine:.6f} below 0.99"
    assert relative_l2 <= 0.05, (
        f"D{head_dim} T{seq_len}: relative_l2 {relative_l2:.6f} above 0.05"
    )
    assert max_abs <= 0.05, f"D{head_dim} T{seq_len}: max_abs {max_abs:.6f} above 0.05"


def _quant_with_launch(q, k, num_warps, num_stages):
    """Reference launch of the same quantizer with an explicit launch shape."""
    from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import (
        quant_per_block_int8_kernel,
    )

    q8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    b, h_qo, qo_len, head_dim = q.shape
    _, h_kv, kv_len, _ = k.shape
    q_scale = torch.empty(
        (b, h_qo, (qo_len + 127) // 128), device=q.device, dtype=torch.float32
    )
    k_scale = torch.empty(
        (b, h_kv, (kv_len + 63) // 64), device=q.device, dtype=torch.float32
    )
    quant_per_block_int8_kernel[((qo_len + 127) // 128, h_qo, b)](
        q, q8, q_scale, qo_len,
        q.stride(0), q.stride(1), q.stride(2),
        q8.stride(0), q8.stride(1), q8.stride(2),
        q_scale.stride(0), q_scale.stride(1),
        sm_scale=(head_dim**-0.5) * 1.44269504,
        C=head_dim, BLK=128, num_warps=num_warps, num_stages=num_stages,
    )
    quant_per_block_int8_kernel[((kv_len + 63) // 64, h_kv, b)](
        k, k8, k_scale, kv_len,
        k.stride(0), k.stride(1), k.stride(2),
        k8.stride(0), k8.stride(1), k8.stride(2),
        k_scale.stride(0), k_scale.stride(1),
        sm_scale=1.0, C=head_dim, BLK=64, num_warps=num_warps, num_stages=num_stages,
    )
    return q8, q_scale, k8, k_scale


@pytest.mark.parametrize(
    "batch,heads,seq_len,head_dim",
    [(1, 2, 300, 64), (1, 2, 257, 128), (2, 3, 128, 64), (1, 4, 129, 128)],
)
def test_quant_launch_config_is_bitwise_equivalent(batch, heads, seq_len, head_dim):
    """The SAGE-003 launch change must not perturb a single int8 or scale value.

    Partial trailing blocks are included on purpose: the mask path is where a
    different warp count would most plausibly change the reduction result.
    """
    q, k, _ = _inputs(seq_len, batch=batch, heads=heads, head_dim=head_dim)
    tuned = per_block_int8(q, k)
    default = _quant_with_launch(q, k, num_warps=4, num_stages=3)
    for name, a, b in zip(("q_int8", "q_scale", "k_int8", "k_scale"), tuned, default):
        assert torch.equal(a, b), f"{name} differs between launch configurations"
        assert str(a.device) == DEVICE


@pytest.mark.skipif(not IS_ENFLAME, reason="harness device guard is Enflame-specific")
def test_harness_fails_closed_on_wrong_device():
    """A wrong device assignment must exit nonzero without touching the GCU."""
    proc = subprocess.run(
        [sys.executable, AB_SCRIPT, "--expect-index", "6"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
    )
    assert proc.returncode == 4, (
        f"expected exit 4 for a wrong device, got {proc.returncode}\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )
    assert '"violations"' in proc.stdout
    assert "is efsmi DEV 5, but the task assigns DEV 6" in proc.stdout


@pytest.mark.skipif(not IS_ENFLAME, reason="harness run is Enflame-specific")
def test_harness_fails_on_missing_samples():
    """Too few samples must exit nonzero rather than report a fast run."""
    proc = subprocess.run(
        [
            sys.executable, AB_SCRIPT,
            "--shapes", "1x2x128x64",
            "--rounds", "1", "--pairs", "1", "--warmup", "1",
            "--min-samples", "99",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=1200,
    )
    assert proc.returncode == 3, (
        f"expected exit 3 for a missing-sample run, got {proc.returncode}\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )
    assert "only 2 samples" in proc.stdout


def test_summary_reports_bimodality_regardless_of_which_mode_dominates():
    """The slow-call fraction must not collapse when slow calls become the median.

    The full Sage call at large shapes alternates between two latencies ~1.8x
    apart. A threshold anchored on the median reports 0.0 slow calls for a
    distribution that is 60% slow calls, once the slow mode takes the median.
    """
    sys.path.insert(0, os.path.join(REPO_ROOT, "agent-results", "scripts"))
    from sage003_ab import summarize

    # Slow-mode majority: the exact per-sample data recorded for 4x32x2048x128.
    mixed = [30.96, 17.2, 30.88, 30.88, 31.08, 30.87, 30.87, 18.09, 17.18, 18.04]
    stats = summarize(mixed)
    assert stats["bimodal"] is True
    assert stats["slow_call_fraction"] == 0.6
    assert 17.0 < stats["fast_median_ms"] < 18.5
    assert 30.5 < stats["slow_median_ms"] < 31.5
    assert 1.6 < stats["mode_ratio"] < 1.9

    # The mirror case: the same two modes, but the fast one dominates.
    flipped = [17.2, 17.3, 17.2, 17.3, 17.2, 18.0, 17.3, 17.2, 17.3, 30.9]
    flipped_stats = summarize(flipped)
    assert flipped_stats["bimodal"] is True
    assert flipped_stats["slow_call_fraction"] == 0.1
    assert flipped_stats["median_ms"] < 18.0

    # A genuinely unimodal run must not be reported as two modes.
    steady = summarize([14.28, 14.29, 14.35, 14.36, 14.37, 14.29, 14.28, 14.36])
    assert steady["bimodal"] is False
    assert steady["slow_call_fraction"] == 0.0


@pytest.mark.skipif(not IS_ENFLAME, reason="device timing is Enflame-specific")
def test_full_call_latency_swing_is_device_side_not_counter_noise():
    """A large swing in the full call must be visible to both timing methods.

    If the swing were host counter noise it would not reproduce in wall clock.
    This pins the observation the report relies on: whatever spread the device
    shows, elapsed-time and wall-clock must agree on it, and when two modes do
    appear they are the same ~1.8x pair rather than a smeared tail.
    """
    sys.path.insert(0, os.path.join(REPO_ROOT, "agent-results", "scripts"))
    from sage003_ab import summarize

    torch.manual_seed(2026)
    shape = (4, 32, 1024, 128)
    q = torch.randn(shape, device=DEVICE, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def full():
        q8, q_scale, k8, k_scale = per_block_int8(q, k)
        return sage_attention(
            q8, k8, v, q_scale, k_scale, output_dtype=torch.float16
        )[0]

    for _ in range(5):
        full()
    torch.gcu.synchronize()

    event_ms, wall_ms = [], []
    for _ in range(12):
        start = torch.gcu.Event(enable_timing=True)
        end = torch.gcu.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        full()
        end.record()
        torch.gcu.synchronize()
        wall_ms.append((time.perf_counter() - wall_start) * 1e3)
        event_ms.append(start.elapsed_time(end))

    event_spread = max(event_ms) / min(event_ms)
    wall_spread = max(wall_ms) / min(wall_ms)
    assert abs(event_spread - wall_spread) < 0.35, (
        f"event spread {event_spread:.2f}x and wall spread {wall_spread:.2f}x "
        "disagree; the latency swing is not a stable device-side property"
    )
    for name, values in (("event", event_ms), ("wall", wall_ms)):
        stats = summarize(values)
        if stats["bimodal"]:
            assert 1.5 < stats["mode_ratio"] < 2.4, (
                f"{name} modes are {stats['mode_ratio']:.2f}x apart"
            )
