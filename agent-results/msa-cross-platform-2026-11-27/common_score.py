# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
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

# Extracted unchanged from the 810E Score kernel at FlagAttention d01b5dd1.
import triton
import triton.language as tl

@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _index_block_score_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens,  # [batch+1] query start offsets
    seq_lens,  # [batch] total K length
    prefix_lens,  # [batch] context length before this chunk's queries
    num_idx_heads,
    head_dim: tl.constexpr,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    NUM_KV_SPLITS: tl.constexpr = 1,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // num_idx_heads
    pid_h = pid_bh % num_idx_heads

    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return

    q_ptrs = tl.make_block_ptr(
        base=q_ptr + seq_start * stride_q_n + pid_h * stride_q_h,
        shape=(q_len, head_dim),
        strides=(stride_q_n, stride_q_d),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, head_dim),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0,), padding_option="zero")
    q_start = prefix_len + pid_q * BLOCK_SIZE_Q

    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q + prefix_len
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    # Block table row for this request.
    bt_row = block_table_ptr + pid_b * stride_bt_b
    # Causal window: only blocks up to the last query token's position.
    hi = min(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)
    # Pages have independent outputs: partition without a reduction/merge.
    for i in tl.range(
        tl.program_id(2) * BLOCK_SIZE_K, hi, BLOCK_SIZE_K * NUM_KV_SPLITS
    ):
        blk = i // BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = i + off_k
        # index-K for this page: [BLOCK_SIZE_D, BLOCK_SIZE_K] (transposed)
        # we don't need masked load for K, because KV cache ensures
        # allocation is multiple of BLOCK_SIZE_K.
        # for tokens beyond seqlen, they will be masked in qk later.
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[None, :] * stride_ik_pos
            + off_d[:, None] * stride_ik_d,
        )
        # FP8 MMA may round distinct block scores to the same value. Widening
        # FP8 inputs to FP16 is exact and preserves their ordering with FP32
        # accumulation, which matters when selecting the last top-k block.
        if q.dtype.is_fp8():
            qk = tl.dot(q.to(tl.float16), k.to(tl.float16))
        else:
            qk = tl.dot(q, k)
        # apply causal mask as needed
        if q_start < i + BLOCK_SIZE_K:
            qk = tl.where(off_q[:, None] >= pos[None, :], qk, float("-inf"))
        # one sparse block per K-tile -> max over the 128 positions
        score = tl.max(qk, axis=1)  # [BLOCK_SIZE_Q]
        s_ptrs = (
            score_ptr
            + pid_h * stride_s_h
            + (seq_start + pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
            * stride_s_n
            + blk * stride_s_k
        )
        q_store_mask = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < q_len
        tl.store(s_ptrs, score, mask=q_store_mask)
