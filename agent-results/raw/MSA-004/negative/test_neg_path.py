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

"""Enflame prefill ``GROUP_SIZE_Q`` routing: reference numerics and real path.

The opt-in gate (``MSA002_GROUP16_EXPERIMENT``) widens the dense-pool
``GROUP_SIZE_Q=16`` branch from ``cdiv(total_q, 16) * num_kv_heads >= 64`` down
to ``>= 32`` at ``total_q >= 512``. These tests pin both sides of that band:

* shapes inside the band must run the grouped kernel with ``GROUP_SIZE_Q=16``
  and match the independent per-query PyTorch reference;
* shapes outside it (page-pool, top-k, head-count, query-length, total-length
  gates) must keep the shipped dispatch (``GROUP_SIZE_Q=8`` or the direct
  kernel) and still match the same reference.

Every case records the actual Triton launch. A missing or unexpected launch
fails the test instead of silently skipping it, and ``MSA004_REQUIRE_DEVICE=1``
turns the "not an enflame host" skip into a collection error so a CI run
cannot report success without having executed a single case.
"""

from __future__ import annotations

import contextlib
import os

import pytest
import torch
from triton.runtime.jit import JITFunction

from flag_attn import minimax_m3_sparse_attn, runtime
from flag_attn.minimax_sparse_attention import (
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
from flag_attn.testing import backend as test_backend
from tests.flag_attn import test_minimax_sparse_attention as refmod

GROUPED = "_gqa_sparse_fwd_grouped8"
DIRECT = "_gqa_sparse_fwd_direct"

# Pinned total of parametrised cases: a case silently dropped from the tables
# (or a collection error that hides half of them) must fail the run.
EXPECTED_CASES = 46
REQUIRE_RUN = os.environ.get("MSA004_REQUIRE_DEVICE") == "1"

if REQUIRE_RUN and runtime.device.vendor_name != "enflame":
    raise RuntimeError(
        "MSA004_REQUIRE_DEVICE=1 but vendor is "
        f"{runtime.device.vendor_name!r}; refusing to skip the GROUP_SIZE_Q routing tests"
    )
if REQUIRE_RUN and not test_backend.supports_operator("minimax_m3_sparse_attn"):
    raise RuntimeError(
        "MSA004_REQUIRE_DEVICE=1 but the backend does not support minimax_m3_sparse_attn: "
        f"{test_backend.skip_reason('minimax_m3_sparse_attn')}"
    )

pytestmark = [
    pytest.mark.minimax_m3_sparse_attn,
    pytest.mark.skipif(
        runtime.device.vendor_name != "enflame",
        reason="the GROUP_SIZE_Q routing under test is enflame-specific",
    ),
    pytest.mark.skipif(
        not test_backend.supports_operator("minimax_m3_sparse_attn"),
        reason=test_backend.skip_reason("minimax_m3_sparse_attn"),
    ),
]


def _select_device() -> int:
    """Bind to ``S60_TEST_DEVICE``, else keep the device the caller selected."""
    module = getattr(torch, runtime.device.name, None)
    configured = os.environ.get("S60_TEST_DEVICE")
    if configured is not None:
        index = int(configured)
    elif module is not None and hasattr(module, "current_device"):
        index = int(module.current_device())
    else:
        index = 0
    runtime.torch_device_fn.set_device(index)
    return index


@pytest.fixture(autouse=True)
def _device() -> int:
    return _select_device()


@contextlib.contextmanager
def _capture_launches():
    """Record real Triton launches (post-warmup) without changing arguments."""
    captured: list[tuple[str, object, object]] = []
    original = JITFunction.run

    def run(self, *args, **kwargs):
        if kwargs.get("warmup") is False:
            grid = kwargs.get("grid")
            captured.append(
                (
                    self.__name__,
                    tuple(grid) if isinstance(grid, (list, tuple)) else grid,
                    kwargs.get("GROUP_SIZE_Q"),
                )
            )
        return original(self, *args, **kwargs)

    JITFunction.run = run
    try:
        yield captured
    finally:
        JITFunction.run = original


def _attention_launch(captured) -> tuple[str, object, object]:
    found = [item for item in captured if "sparse_fwd" in item[0]]
    assert len(found) == 1, f"expected exactly one attention launch, captured {captured!r}"
    return found[0]


def _full_call(data, topk: int, output) -> tuple[torch.Tensor, torch.Tensor]:
    qlen = max(data.query_lens)
    kv = data.kv_cache.shape[1]
    scores = minimax_m3_index_score(
        data.idx_q,
        data.index_kv_cache,
        data.block_table,
        data.cu_q,
        data.seq_lens,
        data.prefix_lens,
        qlen,
        data.max_seq_len,
        kv,
    )
    topk_idx = minimax_m3_index_topk(scores, data.cu_q, data.prefix_lens, qlen, topk, 1, 2)
    minimax_m3_sparse_attn(
        data.q,
        data.kv_cache,
        topk_idx,
        data.block_table,
        data.cu_q,
        data.seq_lens,
        data.prefix_lens,
        qlen,
        kv,
        data.sm_scale,
        output,
    )
    return scores, topk_idx


def _run_variant(data, topk: int, enabled: bool):
    """Run one dispatch variant, returning its output and captured launch."""
    import flag_attn.runtime.backend._enflame.msa.sparse_attn as enflame_sparse

    enflame_sparse._MSA002_GROUP16_EXPERIMENT = bool(enabled)
    output = torch.empty_like(data.q)
    # Compile before capturing so only the measured dispatch is recorded.
    _full_call(data, topk, output)
    torch.gcu.synchronize()

    with _capture_launches() as captured:
        scores, topk_idx = _full_call(data, topk, output)
    torch.gcu.synchronize()
    kernel, grid, group_q = _attention_launch(captured)
    names = [item[0] for item in captured]
    assert "_index_block_score_kernel" in names, f"missing index-score launch: {captured!r}"
    assert any(name.startswith("_prefill_topk_") for name in names), (
        f"missing index-topk launch: {captured!r}"
    )
    return {
        "output": output,
        "scores": scores,
        "topk_idx": topk_idx,
        "kernel": kernel,
        "grid": grid,
        "group_q": group_q,
        "captured": list(captured),
    }


def _check_case(seqs, prefixes, kv, group, topk, shipped, candidate, case_id):
    data = refmod.make_data(seqs, kv, group, decode=False, prefix_lens=prefixes, mode="bf16")
    import flag_attn.runtime.backend._enflame.msa.sparse_attn as enflame_sparse

    try:
        ref_scores = refmod._ref_index_score(data)
        ref_topk = refmod._ref_topk(ref_scores, data, topk, 1, 2)
        ref_output = refmod._ref_sparse_attn(data, ref_topk)
        torch.gcu.synchronize()

        runs = {}
        for tag, enabled in (("shipped", False), ("candidate", True)):
            runs[tag] = _run_variant(data, topk, enabled)
        expected = {"shipped": shipped, "candidate": candidate}
        for tag, run in runs.items():
            want_kernel, want_group = expected[tag]
            assert run["kernel"] == want_kernel, (
                f"{case_id}/{tag}: kernel {run['kernel']} != {want_kernel}; "
                f"launches={run['captured']!r}"
            )
            assert run["group_q"] == want_group, (
                f"{case_id}/{tag}: GROUP_SIZE_Q {run['group_q']} != {want_group}; "
                f"launches={run['captured']!r}"
            )
            refmod._assert_topk_match(run["topk_idx"], ref_topk, data, topk, decode=False)
            refmod._assert_attention_match(run["output"], ref_output, data)
        refmod._assert_score_match(runs["shipped"]["scores"], ref_scores, data, decode=False)
        # Both dispatch variants consume the same inputs; their outputs must
        # agree far inside the BF16 tolerance rather than merely each pass.
        torch.testing.assert_close(
            runs["shipped"]["output"], runs["candidate"]["output"], atol=2e-2, rtol=2e-2
        )
    finally:
        enflame_sparse._MSA002_GROUP16_EXPERIMENT = False
        torch.gcu.empty_cache()


# (seq_lens, prefix_lens, num_kv_heads, group_size, topk, shipped, candidate, id)
# shipped/candidate = (expected attention kernel, expected GROUP_SIZE_Q);
# None marks the direct kernel, which takes no GROUP_SIZE_Q.
BAND_CASES = [
    ((4096,), (3584,), 1, 8, 16, (GROUPED, 8), (GROUPED, 8), "q512_s4096_g8_k16"),
    ((4096,), (3584,), 1, 1, 16, (GROUPED, 8), (GROUPED, 16), "q512_s4096_g1_k16"),
    ((4096,), (3584,), 1, 2, 16, (GROUPED, 8), (GROUPED, 16), "q512_s4096_g2_k16"),
    ((4096,), (3584,), 1, 4, 16, (GROUPED, 8), (GROUPED, 16), "q512_s4096_g4_k16"),
    ((4096,), (3496,), 1, 8, 16, (GROUPED, 8), (GROUPED, 16), "q600_s4096_g8_k16"),
    ((4096,), (3088,), 1, 8, 16, (GROUPED, 8), (GROUPED, 16), "q1008_s4096_g8_k16"),
    # Non-tile-aligned sequence lengths (pool 32 / 31 pages).
    ((4000,), (3488,), 1, 8, 16, (GROUPED, 8), (GROUPED, 16), "q512_s4000_g8_k16"),
    ((3968,), (3456,), 1, 8, 16, (GROUPED, 8), (GROUPED, 16), "q512_s3968_g8_k16"),
    ((2048,), (1536,), 1, 8, 16, (GROUPED, 8), (GROUPED, 16), "q512_s2048_g8_k16"),
    ((1024,), (512,), 1, 8, 16, (GROUPED, 8), (GROUPED, 16), "q512_s1024_g8_k16"),
    ((512,), (0,), 1, 8, 16, (GROUPED, 8), (GROUPED, 16), "q512_s512_g8_k16"),
    # Index boundaries: last page partially valid, prefix ends mid-page.
    ((513,), (1,), 1, 8, 16, (GROUPED, 8), (GROUPED, 16), "q512_s513_prefix1_g8_k16"),
    ((612,), (100,), 1, 8, 16, (GROUPED, 8), (GROUPED, 16), "q512_s612_prefix100_g8_k16"),
    # Ragged batches whose total query length only just reaches the gate.
    ((4096, 4096), (3616, 4064), 1, 8, 16, (GROUPED, 8), (GROUPED, 16), "ragged480_32_g8_k16"),
    ((4096, 4096), (3840, 3840), 1, 8, 16, (GROUPED, 8), (GROUPED, 16), "ragged256_256_g8_k16"),
    ((4096,) * 4, (3968,) * 4, 1, 8, 16, (GROUPED, 8), (GROUPED, 16), "ragged128x4_g8_k16"),
    ((4096, 4096), (3592, 3592), 1, 8, 16, (GROUPED, 8), (GROUPED, 16), "ragged504x2_g8_k16"),
    ((4096,) * 32, (4080,) * 32, 1, 8, 16, (GROUPED, 8), (GROUPED, 16), "batch32_q16_g8_k16"),
    ((4096,) * 16, (4064,) * 16, 1, 8, 16, (GROUPED, 8), (GROUPED, 16), "batch16_q32_g8_k16"),
    # total_q=516: 129 queries over four requests, nine valid pages.
    ((1024,) * 4, (895,) * 4, 1, 8, 16, (GROUPED, 8), (GROUPED, 16), "ragged129x4_s1024_k16"),
    # Smaller page pools, including the topk=4 and topk=8 branches.
    ((1024,), (512,), 1, 8, 4, (GROUPED, 8), (GROUPED, 16), "q512_s1024_g8_k4"),
    ((1024,), (512,), 1, 8, 8, (GROUPED, 8), (GROUPED, 16), "q512_s1024_g8_k8"),
    # total_q just past the 512 boundary (513, not a multiple of 16), and a
    # single-query request sharing the batch with one full-length request.
    ((4096,), (3583,), 1, 8, 16, (GROUPED, 8), (GROUPED, 16), "q513_s4096_g8_k16"),
    ((4096, 256), (3585, 255), 1, 8, 16, (GROUPED, 8), (GROUPED, 16), "ragged511_1_g8_k16"),
    # Eight requests with mixed 32..96 query lengths summing to exactly 512.
    (
        (4096,) * 8,
        (4000, 4064, 4016, 4048, 4032, 4032, 4016, 4048),
        1,
        8,
        16,
        (GROUPED, 8),
        (GROUPED, 16),
        "ragged_mixed512_g8_k16",
    ),
    # topk=1/2/3 with one page per request: the page-pool gate still allows
    # 2*topk pages, so these are inside the band rather than in fallback.
    ((128,) * 32, (112,) * 32, 1, 8, 1, (GROUPED, 8), (GROUPED, 16), "topk1_pad_q512_g8_k1"),
    ((128,) * 32, (112,) * 32, 1, 8, 2, (GROUPED, 8), (GROUPED, 16), "topk2_pad_q512_g8_k2"),
    ((768,) * 32, (752,) * 32, 1, 8, 3, (GROUPED, 8), (GROUPED, 16), "topk3_pad_q512_g8_k3"),
    # topk=4 is the one branch that also accepts a wider page pool (8 == 2*4);
    # seq_len 900 is not a multiple of the 128-token block.
    ((900,) * 32, (884,) * 32, 1, 8, 4, (GROUPED, 8), (GROUPED, 16), "q512_s900_pool8_topk4"),
]

FALLBACK_CASES = [
    # Page-pool gate: 33 pages > 2*topk=32 -> direct kernel before and after.
    ((4097,), (3585,), 1, 8, 16, (DIRECT, None), (DIRECT, None), "pool33_k16_direct"),
    # top-k gate: 32 > 16 -> direct kernel.
    ((4096,), (3584,), 1, 8, 32, (DIRECT, None), (DIRECT, None), "topk32_direct"),
    # Head-count gate: 16 query heads per KV head -> GROUP_SIZE_Q=4.
    ((4096,), (3584,), 1, 16, 16, (GROUPED, 4), (GROUPED, 4), "gqa16_group4"),
    # Already inside the shipped dense-pool gate -> untouched.
    ((4096,), (3584,), 2, 8, 16, (GROUPED, 16), (GROUPED, 16), "kv2_g8_already16"),
    ((4096,), (3584,), 4, 2, 16, (GROUPED, 16), (GROUPED, 16), "kv4_g2_already16"),
    ((4096,), (3584,), 8, 1, 16, (GROUPED, 16), (GROUPED, 16), "kv8_g1_already16"),
    # topk=1 with a 32-page pool -> direct kernel (pool > 2*topk).
    ((4096,), (3584,), 1, 8, 1, (DIRECT, None), (DIRECT, None), "topk1_direct"),
    # topk=2 stays grouped at GROUP_SIZE_Q=8: this shape is below the
    # total_q>=512 band. (topk=2 with a wide pool is deliberately not used as
    # a numeric oracle: init=1/local=2 already fill two slots, so the third
    # forced page is a real tie and any choice is valid.)
    ((256,), (0,), 1, 8, 2, (GROUPED, 8), (GROUPED, 8), "q256_s256_topk2_group8"),
    # topk=4 keeps the shipped 8-query grouped kernel, pool 32 > 2*topk.
    ((4096,), (3584,), 1, 8, 4, (GROUPED, 8), (GROUPED, 8), "topk4_pool32_group8"),
    # topk=8 with pool 32 > 16 -> direct kernel.
    ((4096,), (3584,), 1, 8, 8, (DIRECT, None), (DIRECT, None), "topk8_pool32_direct"),
    # Below the total_q>=512 gate (488 and 500).
    ((4096, 4096), (3796, 3908), 1, 8, 16, (GROUPED, 8), (GROUPED, 8), "total488_group8"),
    ((4096,), (3596,), 1, 8, 16, (GROUPED, 8), (GROUPED, 8), "total500_group8"),
    # max_query_len < 16: the grouped kernel keeps GROUP_SIZE_Q=8.
    ((4096,) * 64, (4088,) * 64, 1, 8, 16, (GROUPED, 8), (GROUPED, 8), "batch64_q8_group8"),
    ((4096,) * 34, (4081,) * 34, 1, 8, 16, (GROUPED, 8), (GROUPED, 8), "batch34_q15_group8"),
    # total_q>=512 but 16*batch=560 is not reached (525 queries over 35
    # requests of 15): the per-request minimum keeps the shipped path.
    ((4096,) * 35, (4081,) * 35, 1, 8, 16, (GROUPED, 8), (GROUPED, 8), "batch35_q15_total525"),
    # 1009 queries already clear the shipped >=64 program gate -> group 16
    # on both sides, so the experiment must not touch it.
    ((4096,), (3087,), 1, 8, 16, (GROUPED, 16), (GROUPED, 16), "q1009_already_group16"),
    # topk=3 with 7 pages > 2*topk=6 -> direct kernel on both sides.
    ((896,) * 32, (880,) * 32, 1, 8, 3, (DIRECT, None), (DIRECT, None), "topk3_pool7_direct"),
]


@pytest.mark.parametrize(
    "case",
    BAND_CASES,
    ids=[case[-1] for case in BAND_CASES],
)
def test_group16_band_reference_and_path(case) -> None:
    seqs, prefixes, kv, group, topk, shipped, candidate, case_id = case
    _check_case(seqs, prefixes, kv, group, topk, shipped, candidate, case_id)


def test_case_tables_are_complete() -> None:
    """Guard against a silently truncated parametrisation."""
    total = len(BAND_CASES) + len(FALLBACK_CASES)
    assert total == EXPECTED_CASES, f"expected {EXPECTED_CASES} cases, table has {total}"
    both = [case[-1] for case in BAND_CASES + FALLBACK_CASES]
    assert len(both) == len(set(both)), f"duplicate case ids: {sorted(both)}"


@pytest.mark.parametrize(
    "case",
    FALLBACK_CASES,
    ids=[case[-1] for case in FALLBACK_CASES],
)
def test_group16_fallbacks_keep_shipped_path(case) -> None:
    seqs, prefixes, kv, group, topk, shipped, candidate, case_id = case
    _check_case(seqs, prefixes, kv, group, topk, shipped, candidate, case_id)
