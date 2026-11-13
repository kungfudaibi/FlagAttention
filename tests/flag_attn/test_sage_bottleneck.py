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

"""SAGE-002 correctness and dispatch checks for the Enflame Sage path.

The tests are deliberately independent of the benchmark harness: they compare
the public full call against PyTorch SDPA on the same FP16 inputs and the same
output dtype, check that the full call is exactly the pre-quantized attention
call on the same quantized tensors, and exercise the non-default kernel
configuration fallbacks. The final test verifies that the operator registry
selects the Enflame override while the generic implementation remains the
fallback for other vendors.
"""

import os

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
    "SAGE002_TEST_DEVICE", "gcu:1" if IS_ENFLAME else test_backend.device
)


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
