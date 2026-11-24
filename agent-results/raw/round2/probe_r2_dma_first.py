#!/usr/bin/env python
"""Probe 8 (round 2): hoist the K/V DMA to the top of the page iteration and
software-pipeline the dependent scalar gathers (topk slot -> block table -> page).

Variants are pure reorderings: same values, same order of accumulation.
"""
import os, sys, statistics, json
REPO = os.environ["MSA_REPO"]
sys.path.insert(0, os.path.join(REPO, "src")); sys.path.insert(0, REPO)
import torch, triton
import triton.language as tl
torch.gcu.set_device(5)

@triton.jit
def _k(
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
    MODE: tl.constexpr = 0,  # 0 stock-order, 1 early-V only, 2 early-V + DMA-first + scalar pipeline
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
        t_base = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
        q_abs = prefix_len + pid_q_j * BLOCK_SIZE_Q
        valid_blocks = (q_abs + BLOCK_SIZE_K) // BLOCK_SIZE_K
        real_topk = tl.minimum(max_topk, valid_blocks)
        q = tl.load(tl.make_block_ptr(
            base=q_ptr + (q_start + pid_q_j) * stride_qn + pid_kh * gqa_group_size * stride_qh,
            shape=(gqa_group_size, head_dim), strides=(stride_qh, stride_qd),
            offsets=(0, 0), block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D), order=(1, 0)),
            boundary_check=(0, 1), padding_option="zero")
        m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
        l_acc = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_K), dtype=tl.float32)
        acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)
        shared_count = 0
        if USE_SHARED_PREFIX:
            shared_count = tl.load(count_ptr + (pid_b * num_kv_heads + pid_kh) * NUM_GROUPS + pid_q_j // GROUP_Q)
            hs = tl.arange(0, BLOCK_SIZE_QH)
            state_base = (q_start + pid_q_j) * STRIDE_SN + pid_kh * gqa_group_size
            m_i = tl.load(maximum_ptr + state_base + hs, mask=hs < gqa_group_size, other=float("-inf"))
            l_i = tl.load(denom_ptr + state_base + hs, mask=hs < gqa_group_size, other=0.0)
            acc_o = tl.load(tl.make_block_ptr(
                partial_ptr + (q_start + pid_q_j) * STRIDE_PN + pid_kh * gqa_group_size * STRIDE_PH,
                (gqa_group_size, head_dim), (STRIDE_PH, STRIDE_PD), (0, 0),
                (BLOCK_SIZE_QH, BLOCK_SIZE_D), (1, 0)),
                boundary_check=(0, 1), padding_option="zero")
            l_acc = tl.where(off_n[None, :] == 0, l_i[:, None], 0.0)
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)
        blk = tl.load(t_base + shared_count * stride_tk).to(tl.int32)
        page = tl.load(bt_row + blk).to(tl.int32)
        for s in range(shared_count, real_topk):
            c = blk * BLOCK_SIZE_K
            k_base = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
            kp = tl.make_block_ptr(base=k_base, shape=(BLOCK_SIZE_K, head_dim),
                strides=(stride_kv_pos, stride_kv_d), offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D), order=(1, 0))
            vp = tl.make_block_ptr(base=k_base + head_dim * stride_kv_d,
                shape=(BLOCK_SIZE_K, head_dim), strides=(stride_kv_pos, stride_kv_d),
                offsets=(0, 0), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D), order=(1, 0))
            if MODE >= 2:
                # both DMAs issued at the top of the iteration, before any
                # integer/mask arithmetic and before the QK convgen
                k = tl.load(kp, boundary_check=(0, 1), padding_option="zero")
                v = tl.load(vp, boundary_check=(0, 1), padding_option="zero")
                nblk = tl.load(t_base + tl.minimum(s + 1, real_topk) * stride_tk).to(tl.int32)
                npage = tl.load(bt_row + nblk).to(tl.int32)
            else:
                k = tl.load(kp, boundary_check=(0, 1), padding_option="zero")
                nblk = 0
                npage = 0
            if MODE == 1:
                v = tl.load(vp, boundary_check=(0, 1), padding_option="zero")
            qk_dot = tl.dot(q, tl.trans(k))
            pos = c + off_n
            qk = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_K), dtype=tl.float32)
            if not ((c + BLOCK_SIZE_K) <= q_abs):
                qk += tl.where(q_abs >= pos[None, :], 0, float("-inf"))
            qk += qk_dot * sm_scale_log2e
            if not ((c + BLOCK_SIZE_K) <= seq_len):
                qk += tl.where((pos < seq_len)[None, :], 0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)
            acc_o = acc_o * alpha[:, None]
            l_acc = l_acc * alpha[:, None] + p
            if MODE == 0:
                v = tl.load(vp, boundary_check=(0, 1), padding_option="zero")
            acc_o += tl.dot(p.to(v.dtype), v)
            m_i = m_ij
            if MODE >= 2:
                blk = nblk
                page = npage
        l_i = tl.sum(l_acc, axis=1)
        inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
        acc_o = acc_o * inv_l[:, None]
        tl.store(tl.make_block_ptr(
            base=o_ptr + (q_start + pid_q_j) * stride_on + pid_kh * gqa_group_size * stride_oh,
            shape=(gqa_group_size, head_dim), strides=(stride_oh, stride_od),
            offsets=(0, 0), block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D), order=(1, 0)),
            acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))


def main():
    from benchmark import test_minimax_sparse_attention_benchmark as b
    from flag_attn.runtime.backend._enflame.msa import sparse_attn as sa
    batch, qlen, seq, kvh, nh = b.PREFILL_SHAPES[6]
    gen = torch.Generator(device="cpu").manual_seed(6)
    data = b.make_data(batch, seq, kvh, nh, torch.device("gcu"), "bf16", decode=False,
                       decode_qlen=1, query_len=qlen, randomize_pages=True, generator=gen)
    tk = b.minimax_m3_index_topk(b.minimax_m3_index_score(
        data.idx_q, data.index_kv_cache, data.block_table, data.cu_q, data.seq_lens,
        data.prefix_lens, qlen, seq, kvh), data.cu_q, data.prefix_lens, qlen, 16, 1, 2)
    torch.gcu.synchronize()
    ref = torch.empty_like(data.q); out = torch.empty_like(data.q)
    b._call_sparse(b.minimax_m3_sparse_attn, data, tk, ref, decode=False,
                   max_query_len=qlen, num_kv_heads=kvh, decode_qlen=1)
    torch.gcu.synchronize()
    ngr = triton.cdiv(qlen, 16)
    def launch(mode):
        p = torch.empty(data.q.shape, device=data.q.device, dtype=torch.float32)
        mx = torch.empty(data.q.shape[:2], device=data.q.device, dtype=torch.float32)
        dn = torch.empty_like(mx)
        ct = torch.empty((batch, kvh, ngr), device=data.q.device, dtype=torch.int32)
        sa._gqa_sparse_common_prefix[(ngr, kvh, batch)](
            data.q, data.kv_cache, tk, data.block_table, data.cu_q, data.seq_lens, data.prefix_lens,
            p, mx, dn, ct, 8, 128, 16, data.sm_scale, *data.q.stride(), *data.kv_cache.stride(),
            *tk.stride(), data.block_table.stride(0), *p.stride(), mx.stride(0), kvh, ngr, 16, 8, 128,
            num_warps=1, num_stages=1)
        return _k[(qlen, kvh, batch)](
            data.q, data.kv_cache, tk, out, data.block_table, data.cu_q, data.seq_lens,
            data.prefix_lens, kvh, 8, 128, 16, 1, data.sm_scale,
            *data.q.stride(), *data.kv_cache.stride(), *tk.stride(), *out.stride(),
            data.block_table.stride(0), BLOCK_SIZE_Q=1, BLOCK_SIZE_K=128, BLOCK_SIZE_D=128,
            BLOCK_SIZE_H=8, BLOCK_SIZE_QH=8, partial_ptr=p, maximum_ptr=mx, denom_ptr=dn,
            count_ptr=ct, GROUP_Q=16, NUM_GROUPS=ngr, STRIDE_PN=p.stride(0), STRIDE_PH=p.stride(1),
            STRIDE_PD=p.stride(2), STRIDE_SN=mx.stride(0), USE_SHARED_PREFIX=True,
            MODE=mode, num_warps=1, num_stages=1)
    res = {}
    for name, mode in (("early_v", 1), ("dma_first_pipe", 2)):
        try:
            h = launch(mode); torch.gcu.synchronize()
            d = (out.float() - ref.float()).abs()
            res[name] = dict(ok=True, shared=h.metadata.shared, maxdiff=d.max().item(),
                             nonzero=int((d > 0).sum()))
            print(f"{name:16s} shared={h.metadata.shared} maxdiff={d.max().item():.6g} "
                  f"nz={int((d>0).sum())}")
        except Exception as e:
            res[name] = dict(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
            print(f"{name:16s} FAILED {type(e).__name__}: {str(e)[:200]}")
    fns = {"stock": lambda: b._call_sparse(b.minimax_m3_sparse_attn, data, tk, out, decode=False,
                                           max_query_len=qlen, num_kv_heads=kvh, decode_qlen=1)}
    for name, mode in (("early_v", 1), ("dma_first_pipe", 2)):
        if res[name]["ok"]:
            fns[name] = (lambda mode=mode: launch(mode))
    order = list(fns)
    for n in order:
        for _ in range(3): fns[n]()
    torch.gcu.synchronize()
    samples = {n: [] for n in order}
    for r in range(9):
        rot = order[r % len(order):] + order[:r % len(order)]
        for n in rot:
            a = torch.gcu.Event(enable_timing=True); c = torch.gcu.Event(enable_timing=True)
            a.record(); fns[n](); c.record(); c.synchronize(); samples[n].append(a.elapsed_time(c))
    for n in order:
        res.setdefault(n, {})["bench_median"] = statistics.median(samples[n])
        res[n]["bench_samples"] = samples[n]
        print(f"{n:16s} median {statistics.median(samples[n]):.4f} min {min(samples[n]):.4f}")
    json.dump(res, open("/tmp/probe_r2.json", "w"), indent=1)

if __name__ == "__main__":
    main()
