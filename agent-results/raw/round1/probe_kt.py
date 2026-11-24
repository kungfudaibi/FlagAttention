#!/usr/bin/env python
"""Probe: replace tl.trans(K) with a transposed block load; measure + IR."""
import os, sys, json, statistics, subprocess, re
REPO = os.environ["MSA_REPO"]
sys.path.insert(0, os.path.join(REPO, "src")); sys.path.insert(0, REPO)
import torch, triton
import triton.language as tl

torch.gcu.set_device(5)

@triton.jit
def _direct_var(
    q_ptr, kv_cache_ptr, t_ptr, o_ptr, block_table_ptr,
    cu_seqlens_q, seq_lens, prefix_lens, num_kv_heads,
    gqa_group_size: tl.constexpr, head_dim: tl.constexpr, max_topk: tl.constexpr,
    num_q_loop: tl.constexpr, sm_scale: tl.constexpr,
    stride_qn: tl.constexpr, stride_qh: tl.constexpr, stride_qd: tl.constexpr,
    stride_kv_blk: tl.constexpr, stride_kv_h: tl.constexpr,
    stride_kv_pos: tl.constexpr, stride_kv_d: tl.constexpr,
    stride_th: tl.constexpr, stride_tn: tl.constexpr, stride_tk: tl.constexpr,
    stride_on: tl.constexpr, stride_oh: tl.constexpr, stride_od: tl.constexpr,
    stride_bt_b: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_QH: tl.constexpr,
    partial_ptr=0, maximum_ptr=0, denom_ptr=0, count_ptr=0,
    GROUP_Q: tl.constexpr = 1, NUM_GROUPS: tl.constexpr = 1,
    STRIDE_PN: tl.constexpr = 0, STRIDE_PH: tl.constexpr = 0,
    STRIDE_PD: tl.constexpr = 0, STRIDE_SN: tl.constexpr = 0,
    USE_SHARED_PREFIX: tl.constexpr = False,
    K_LAYOUT: tl.constexpr = 0,
    LACC: tl.constexpr = 0,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_block_start = q_start
    q_block_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q * num_q_loop >= q_block_len:
        return
    real_q_loop = min(num_q_loop, q_block_len - pid_q * num_q_loop)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_n = tl.arange(0, BLOCK_SIZE_K)
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        t_ptr_j = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
        q_abs = prefix_len + pid_q_j * BLOCK_SIZE_Q
        valid_blocks = (q_abs + BLOCK_SIZE_K) // BLOCK_SIZE_K
        real_topk = tl.minimum(max_topk, valid_blocks)
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + (q_start + pid_q_j) * stride_qn + pid_kh * gqa_group_size * stride_qh,
            shape=(gqa_group_size, head_dim), strides=(stride_qh, stride_qd),
            offsets=(0, 0), block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D), order=(1, 0))
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
        m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
        if LACC == 0:
            l_acc = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_K), dtype=tl.float32)
        acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)
        shared_count = 0
        if USE_SHARED_PREFIX:
            shared_count = tl.load(count_ptr + (pid_b * num_kv_heads + pid_kh) * NUM_GROUPS + pid_q_j // GROUP_Q)
            hs = tl.arange(0, BLOCK_SIZE_QH)
            state_base = (q_start + pid_q_j) * STRIDE_SN + pid_kh * gqa_group_size
            m_i = tl.load(maximum_ptr + state_base + hs, mask=hs < gqa_group_size, other=float("-inf"))
            l_i = tl.load(denom_ptr + state_base + hs, mask=hs < gqa_group_size, other=0.0)
            pp = tl.make_block_ptr(
                partial_ptr + (q_start + pid_q_j) * STRIDE_PN + pid_kh * gqa_group_size * STRIDE_PH,
                (gqa_group_size, head_dim), (STRIDE_PH, STRIDE_PD), (0, 0),
                (BLOCK_SIZE_QH, BLOCK_SIZE_D), (1, 0))
            acc_o = tl.load(pp, boundary_check=(0, 1), padding_option="zero")
            if LACC == 0:
                l_acc = tl.where(off_n[None, :] == 0, l_i[:, None], 0.0)
            t_ptr_j += shared_count * stride_tk
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)
        q_qk = q
        for _ in range(shared_count, real_topk):
            blk = tl.load(t_ptr_j).to(tl.int32)
            t_ptr_j = t_ptr_j + stride_tk
            c = blk * BLOCK_SIZE_K
            page = tl.load(bt_row + blk).to(tl.int32)
            pos = c + off_n
            pos_mask = pos < seq_len
            k_base_ptr = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
            if K_LAYOUT == 0:
                k_ptrs = tl.make_block_ptr(
                    base=k_base_ptr, shape=(BLOCK_SIZE_K, head_dim),
                    strides=(stride_kv_pos, stride_kv_d), offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D), order=(1, 0))
                k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
                kt = tl.trans(k)
            else:
                kt_ptrs = tl.make_block_ptr(
                    base=k_base_ptr, shape=(head_dim, BLOCK_SIZE_K),
                    strides=(stride_kv_d, stride_kv_pos), offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K), order=(0, 1))
                kt = tl.load(kt_ptrs, boundary_check=(0, 1), padding_option="zero")
            qk_dot = tl.dot(q_qk, kt)
            is_full_causal = (c + BLOCK_SIZE_K) <= q_abs
            is_full_seq = (c + BLOCK_SIZE_K) <= seq_len
            qk = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_K), dtype=tl.float32)
            if not is_full_causal:
                qk += tl.where(q_abs >= pos[None, :], 0, float("-inf"))
            qk += qk_dot * sm_scale_log2e
            if not is_full_seq:
                qk += tl.where(pos_mask[None, :], 0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)
            acc_o = acc_o * alpha[:, None]
            if LACC == 0:
                l_acc = l_acc * alpha[:, None] + p
            else:
                l_i = l_i * alpha + tl.sum(p, axis=1)
            v_ptrs = tl.make_block_ptr(
                base=k_base_ptr + head_dim * stride_kv_d, shape=(BLOCK_SIZE_K, head_dim),
                strides=(stride_kv_pos, stride_kv_d), offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D), order=(1, 0))
            v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
            acc_o += tl.dot(p.to(v.dtype), v)
            m_i = m_ij
        if LACC == 0:
            l_i = tl.sum(l_acc, axis=1)
        inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
        acc_o = acc_o * inv_l[:, None]
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + (q_start + pid_q_j) * stride_on + pid_kh * gqa_group_size * stride_oh,
            shape=(gqa_group_size, head_dim), strides=(stride_oh, stride_od),
            offsets=(0, 0), block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D), order=(1, 0))
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))


def main():
    from benchmark import test_minimax_sparse_attention_benchmark as b
    from flag_attn.runtime.backend._enflame.msa import sparse_attn as sa
    shape = b.PREFILL_SHAPES[6]; batch, qlen, seq, kvh, nh = shape
    gen = torch.Generator(device="cpu").manual_seed(6)
    data = b.make_data(batch, seq, kvh, nh, torch.device("gcu"), "bf16", decode=False,
                       decode_qlen=1, query_len=qlen, randomize_pages=True, generator=gen)
    scores = b.minimax_m3_index_score(data.idx_q, data.index_kv_cache, data.block_table,
                                      data.cu_q, data.seq_lens, data.prefix_lens, qlen, seq, kvh)
    tk = b.minimax_m3_index_topk(scores, data.cu_q, data.prefix_lens, qlen, 16, 1, 2)
    torch.gcu.synchronize()
    out = {k: torch.empty_like(data.q) for k in ("stock", "kt", "kt_lacc", "stock_lacc")}
    b._call_sparse(b.minimax_m3_sparse_attn, data, tk, out["stock"], decode=False,
                   max_query_len=qlen, num_kv_heads=kvh, decode_qlen=1)
    torch.gcu.synchronize()

    def run(name, K_LAYOUT, LACC, capture=False):
        partial = torch.empty(data.q.shape, device=data.q.device, dtype=torch.float32)
        maximum = torch.empty(data.q.shape[:2], device=data.q.device, dtype=torch.float32)
        denom = torch.empty_like(maximum)
        counts = torch.empty((batch, kvh, triton.cdiv(qlen, 16)), device=data.q.device, dtype=torch.int32)
        sa._gqa_sparse_common_prefix[(triton.cdiv(qlen, 16), kvh, batch)](
            data.q, data.kv_cache, tk, data.block_table, data.cu_q, data.seq_lens, data.prefix_lens,
            partial, maximum, denom, counts, 8, 128, 16, data.sm_scale, *data.q.stride(),
            *data.kv_cache.stride(), *tk.stride(), data.block_table.stride(0), *partial.stride(),
            maximum.stride(0), kvh, triton.cdiv(qlen, 16), 16, 8, 128, num_warps=1, num_stages=1)
        h = _direct_var[(qlen, kvh, batch)](
            data.q, data.kv_cache, tk, out[name], data.block_table, data.cu_q, data.seq_lens,
            data.prefix_lens, kvh, 8, 128, 16, 1, data.sm_scale,
            *data.q.stride(), *data.kv_cache.stride(), *tk.stride(), *out[name].stride(),
            data.block_table.stride(0), BLOCK_SIZE_Q=1, BLOCK_SIZE_K=128, BLOCK_SIZE_D=128,
            BLOCK_SIZE_H=8, BLOCK_SIZE_QH=8, partial_ptr=partial, maximum_ptr=maximum,
            denom_ptr=denom, count_ptr=counts, GROUP_Q=16, NUM_GROUPS=triton.cdiv(qlen, 16),
            STRIDE_PN=partial.stride(0), STRIDE_PH=partial.stride(1), STRIDE_PD=partial.stride(2),
            STRIDE_SN=maximum.stride(0), USE_SHARED_PREFIX=True, K_LAYOUT=K_LAYOUT, LACC=LACC,
            num_warps=1, num_stages=1)
        if capture:
            return h
        return None
    for name, kl, la in (("kt", 1, 0), ("stock_lacc", 0, 1), ("kt_lacc", 1, 1)):
        h = run(name, kl, la, capture=True)
        torch.gcu.synchronize()
        md = h.metadata
        print(name, "shared", md.shared, "num_warps", md.num_warps, "spills", getattr(h, "n_spills", None))
        d = (out[name].float() - out["stock"].float()).abs()
        print("  diff vs stock max", d.max().item(), "nonzero", int((d > 0).sum()), "n", d.numel())

    def bench(fns, rounds=7, warmup=3):
        order = list(fns)
        for _ in range(warmup):
            for n in order: fns[n]()
        torch.gcu.synchronize()
        s = {n: [] for n in order}
        for r in range(rounds):
            rot = order[r % len(order):] + order[:r % len(order)]
            for n in rot:
                a = torch.gcu.Event(enable_timing=True); c = torch.gcu.Event(enable_timing=True)
                a.record(); fns[n](); c.record(); c.synchronize()
                s[n].append(a.elapsed_time(c))
        return s
    fns = {"stock": lambda: b._call_sparse(b.minimax_m3_sparse_attn, data, tk, out["stock"], decode=False,
                                           max_query_len=qlen, num_kv_heads=kvh, decode_qlen=1)}
    for name, kl, la in (("kt", 1, 0), ("stock_lacc", 0, 1), ("kt_lacc", 1, 1)):
        fns[name] = (lambda kl=kl, la=la, name=name: run(name, kl, la))
    s = bench(fns)
    for n, v in s.items():
        print(n, "median", round(statistics.median(v), 4), "samples", [round(x, 4) for x in v])

if __name__ == "__main__":
    main()
