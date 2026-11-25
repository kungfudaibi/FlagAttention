#!/usr/bin/env python
"""MSA-010: K staging vs in-device layout vs M=8 QK cost on S60 (C200).

Fixed case: benchmark PREFILL_SHAPES[6] = (B, Q, KV, Hkv, Hq) = (1, 1024,
16384, 1, 8), D=128, TopK=16, 128-token pages, args.seed + shape_index,
upstream `make_data` (random physical page permutation) and the real
score -> topk chain.  Only 13 pages reach the direct kernel because the
shared-prefix path absorbs 3 (MSA-008/009 result, re-checked here).

Two entry points, both device-5 only:

  --verify     * stock chain + FP32 reference (semantics, row subset)
               * falsifiable causal-mask check on the stock route
               * probe contracts: every K-staging variant must reproduce the
                 stock output bit-exactly (same Q, same selected pages, same
                 page order); the pre-transposed probe must reproduce the
                 *same selected pages'* QK matrix (checked against a torch
                 float32 `q @ k.T`), not just the attention output.
               * launch/grid/constexpr capture + IR evidence per variant.

  --benchmark  * anchors: stock Attention-only / full chain, and the verified
                 FLAG_ATTN_MSA_EARLY_V_DMA=1 opt-in in both forms.
               * paired decomposition arms (same page order, same page loop,
                 same grid), >=3 interleaved rounds, every raw sample kept.
               * matched QK-only vs load/layout-only micro kernels (pre-
                 transposed K, identical loop/grid) to price the M=8 convgen
                 against the page read.
               * static IR census (ttir/ttir ops, gcuir page-loop op counts,
                 llir tops_* calls, n_regs/n_spills/shared) per variant.
               * optional profiler / efsmi diagnostics, with a positive
                 control, and an explicit "unavailable" record when dead.

Nothing here changes `tests/`, `benchmark/`, the default dispatch or any
public API: the experimental kernels and the host replica live in this file
and are only reachable through this script's own calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

SHAPE_INDEX = 6                 # PREFILL_SHAPES[6] == (1, 1024, 16384, 1, 8)
BYPASS_INDEX = 1                # PREFILL_SHAPES[1] == (1, 512, 4096, 1, 8)
SEED_OFFSET = 0                 # benchmark default: seed + shape_index
PREFIX_GROUP = 16
BF16_ATOL = BF16_RTOL = 2e-2
EARLY_V_ENV = "FLAG_ATTN_MSA_EARLY_V_DMA"
VARIANT_ORDER = ["stock", "copy", "k_pre", "k_pre_trans", "xp_k_pre", "xp_sub4",
                 "xp_sub13", "dia_a", "dia_b", "dia_c"]



# --------------------------------------------------------------------------
# shell / environment / occupancy gate
# --------------------------------------------------------------------------
def sh(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except Exception as exc:  # diagnostics only
        return f"<error {type(exc).__name__}: {exc}>"


def process_tree_pids():
    pids, cur = set(), os.getpid()
    while cur and cur not in pids:
        pids.add(cur)
        try:
            stat = open(f"/proc/{cur}/stat").read()
            cur = int(stat.rsplit(")", 1)[1].split()[1])
        except Exception:
            break
    return pids


def foreign_pids(pmon_text):
    mine, out = process_tree_pids(), []
    for line in pmon_text.splitlines():
        m = re.match(r"\s*\d+\s+(\d+)\s", line)
        if m and int(m.group(1)) not in mine:
            out.append((int(m.group(1)), line.strip()))
    return out


def occupancy_check(device_index):
    """Hard gate: refuse to run if any other process is on the declared card."""
    pmon = sh(f"efsmi pmon -i {device_index} -c 1")
    usage = sh(f"efsmi -i {device_index} -q -d USAGE")
    foreign = foreign_pids(pmon)
    m = re.search(r"GCU Usage\s*:\s*([\d.]+)", usage)
    return dict(pmon_raw=pmon, usage_raw=usage,
                gcu_usage_pct=float(m.group(1)) if m else None,
                own_pids=sorted(process_tree_pids()), foreign_pids=foreign,
                busy=bool(foreign))


def env_record(device_index):
    q = sh(f"efsmi -i {device_index} -q")
    pmon = sh(f"efsmi pmon -i {device_index} -c 1")
    props = torch.gcu.get_device_properties(device_index)
    bus = re.search(r"Bus\s*:\s*(\d+)", q)
    dev = re.search(r"Dev\s*:\s*(\d+)", q)
    fn = re.search(r"Func\s*:\s*(\d+)", q)
    pci = f"0000:{bus.group(1)}:{dev.group(1)}.{fn.group(1)}" if bus and dev and fn else None
    rec = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device_index": device_index,
        "torch_gcu_current_device": torch.gcu.current_device(),
        "device_name": torch.gcu.get_device_name(device_index),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": sys.version.split()[0],
        "pci": pci,
        "props": {"major": props.major, "minor": props.minor,
                  "total_memory_mb": round(props.total_memory / (1 << 20)),
                  "multi_processor_count": props.multi_processor_count,
                  "sip_count": getattr(props, "sip_count", None)},
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "env": {k: os.environ.get(k) for k in
                ("PYTHONPATH", "TRITON_CACHE_DIR", "TRITON_GCU_OP_TRACE", "TRITON_GCU_DEBUG")},
        "efsmi_pmon_before": pmon,
        "efsmi_query_raw": q,
        "own_pids": sorted(process_tree_pids()),
        "foreign_pids": foreign_pids(pmon),
    }
    rec["busy_before"] = bool(rec["foreign_pids"])
    return rec


def gate(device):
    """Fail closed unless this is GCU 5 at 0000:83:00.0 and the card is idle."""
    if device != 5:
        print(f"FATAL: this task is restricted to GCU 5 (got --device {device})")
        sys.exit(3)
    occ = occupancy_check(5)
    print(f"[gate] pmon foreign_pids={occ['foreign_pids']} usage={occ['gcu_usage_pct']}%")
    if occ["busy"] or (occ["gcu_usage_pct"] or 0.0) > 1.0:
        print("[gate] FATAL: device 5 is busy (external PID or non-zero usage). "
              "Wait for it to drain; nothing was measured.")
        print(occ["pmon_raw"])
        sys.exit(4)
    q = sh("efsmi -i 5 -q")
    bus = re.search(r"Bus\s*:\s*(\d+)", q)
    dev = re.search(r"Dev\s*:\s*(\d+)", q)
    fn = re.search(r"Func\s*:\s*(\d+)", q)
    pci = f"0000:{bus.group(1)}:{dev.group(1)}.{fn.group(1)}" if bus and dev and fn else None
    if pci != "0000:83:00.0":
        print(f"[gate] FATAL: device 5 reports PCI {pci}, expected 0000:83:00.0")
        sys.exit(4)
    return occ



# --------------------------------------------------------------------------
# diagnostic kernel copy of the shipped BF16 direct path
# --------------------------------------------------------------------------
# KSOURCE 0: shipped layout, K rows [BLOCK_SIZE_K, head_dim] (dim1 = d).
# KSOURCE 1: pre-transposed diagnostic table, tile [head_dim, BLOCK_SIZE_K]
#            (dim1 = token position) -- tl.dot(q, k) with no tl.trans.
# KSOURCE 2: shipped layout but `page = blk` (logical block id == physical
#            page), i.e. the page-table indirection is removed while the same
#            selected page data is still read through the normal KV DMA.
# KSOURCE 3: the SAME shipped cache, read through a transposed *view*
#            (dim0 = head_dim stride 1, dim1 = token position stride
#            STRIDE_KV_POS).  No second table, no tl.trans -- the DTE is asked
#            for the [head_dim, BLOCK_SIZE_K] tile directly.
@triton.jit
def _load_k(kv_cache_ptr, kt_ptr, page, pid_kh,
            STRIDE_KV_BLK: tl.constexpr, STRIDE_KV_H: tl.constexpr,
            STRIDE_KV_POS: tl.constexpr, STRIDE_KV_D: tl.constexpr,
            STRIDE_KT_BLK: tl.constexpr, STRIDE_KT_H: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
            HEAD_DIM: tl.constexpr, KSOURCE: tl.constexpr):
    if KSOURCE == 3:
        k = tl.load(
            tl.make_block_ptr(
                base=kv_cache_ptr + page * STRIDE_KV_BLK + pid_kh * STRIDE_KV_H,
                shape=(HEAD_DIM, BLOCK_SIZE_K),
                strides=(STRIDE_KV_D, STRIDE_KV_POS),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
                order=(0, 1),
            ), boundary_check=(0, 1), padding_option="zero")
        v_base = kv_cache_ptr + page * STRIDE_KV_BLK + pid_kh * STRIDE_KV_H + HEAD_DIM * STRIDE_KV_D
    elif KSOURCE == 1:
        k = tl.load(
            tl.make_block_ptr(
                base=kt_ptr + page * STRIDE_KT_BLK + pid_kh * STRIDE_KT_H,
                shape=(HEAD_DIM, BLOCK_SIZE_K),
                strides=(BLOCK_SIZE_K, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
                order=(1, 0),
            ), boundary_check=(0, 1), padding_option="zero")
        v_base = kv_cache_ptr + page * STRIDE_KV_BLK + pid_kh * STRIDE_KV_H + HEAD_DIM * STRIDE_KV_D
    else:
        k_base = kv_cache_ptr + page * STRIDE_KV_BLK + pid_kh * STRIDE_KV_H
        k = tl.load(
            tl.make_block_ptr(
                base=k_base,
                shape=(BLOCK_SIZE_K, HEAD_DIM),
                strides=(STRIDE_KV_POS, STRIDE_KV_D),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                order=(1, 0),
            ), boundary_check=(0, 1), padding_option="zero")
        v_base = k_base + HEAD_DIM * STRIDE_KV_D
    return k, v_base


@triton.jit
def _page_math(q, k, v_pre, v_base_ptr, acc_o, l_acc, m_i,
               blk, q_abs, seq_len, off_n,
               sm_scale_log2e: tl.constexpr,
               STRIDE_KV_POS: tl.constexpr, STRIDE_KV_D: tl.constexpr,
               BLOCK_SIZE_QH: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
               BLOCK_SIZE_D: tl.constexpr,
               MODE: tl.constexpr, K_IS_T: tl.constexpr,
               V_PRELOADED: tl.constexpr):
    c = blk * BLOCK_SIZE_K
    pos = c + off_n
    pos_mask = pos < seq_len
    if MODE == 1:
        # page read only (no transpose, no convgen): observable checksum
        s = tl.sum(k.to(tl.float32), axis=1)
        acc_o += s[None, :] * q
        return acc_o, l_acc, m_i
    if MODE == 2:
        # page read + explicit transpose, no convgen: prices the layout step
        kt = tl.trans(k)
        s = tl.sum(kt.to(tl.float32), axis=1)
        acc_o += s[None, :] * q
        return acc_o, l_acc, m_i
    if K_IS_T:
        qk_dot = tl.dot(q, k)
    else:
        qk_dot = tl.dot(q, tl.trans(k))
    qk = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_K), dtype=tl.float32)
    if not ((c + BLOCK_SIZE_K) <= q_abs):
        qk += tl.where(q_abs >= pos[None, :], 0, float("-inf"))
    qk += qk_dot * sm_scale_log2e
    if not ((c + BLOCK_SIZE_K) <= seq_len):
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
    if MODE == 3:
        # QK convgen only (+causal/seq mask), checksum instead of softmax
        acc_o += tl.sum(qk, axis=1)[:, None] * q
        return acc_o, l_acc, m_i
    m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
    p = tl.exp2(qk - m_ij[:, None])
    alpha = tl.exp2(m_i - m_ij)
    acc_o = acc_o * alpha[:, None]
    l_acc = l_acc * alpha[:, None] + p
    if MODE == 4:
        # QK + online softmax, no V DMA and no PV convgen
        acc_o += tl.sum(p, axis=1)[:, None] * q
        return acc_o, l_acc, m_ij
    if MODE == 5:
        # full online softmax; PV reads the *already loaded* K tile so the V
        # DMA + its DTE wait leave the chain (everything else stays)
        acc_o += tl.dot(p.to(k.dtype), k)
        return acc_o, l_acc, m_ij
    if V_PRELOADED:
        v = v_pre
    else:
        v = tl.load(
            tl.make_block_ptr(
                base=v_base_ptr,
                shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                strides=(STRIDE_KV_POS, STRIDE_KV_D),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                order=(1, 0),
            ), boundary_check=(0, 1), padding_option="zero")
    acc_o += tl.dot(p.to(v.dtype), v)
    return acc_o, l_acc, m_ij



@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _exp_direct(
    q_ptr, kv_cache_ptr, k_scale_ptr, v_scale_ptr, t_ptr, o_ptr, block_table_ptr, kt_ptr,
    cu_seqlens_q, cu_seqblocks_q, seq_lens, prefix_lens, num_kv_heads,
    gqa_group_size: tl.constexpr, head_dim: tl.constexpr, max_topk: tl.constexpr,
    num_q_loop: tl.constexpr, sm_scale: tl.constexpr,
    stride_qn: tl.constexpr, stride_qh: tl.constexpr, stride_qd: tl.constexpr,
    stride_kv_blk: tl.constexpr, stride_kv_h: tl.constexpr,
    stride_kv_pos: tl.constexpr, stride_kv_d: tl.constexpr,
    stride_ks_h: tl.constexpr, stride_ks_t: tl.constexpr,
    stride_vs_h: tl.constexpr, stride_vs_t: tl.constexpr,
    stride_th: tl.constexpr, stride_tn: tl.constexpr, stride_tk: tl.constexpr,
    stride_on: tl.constexpr, stride_oh: tl.constexpr, stride_od: tl.constexpr,
    stride_bt_b: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_QH: tl.constexpr,
    USE_FP8: tl.constexpr = False, KV_SCALE_MODE: tl.constexpr = 0,
    partial_ptr=0, maximum_ptr=0, denom_ptr=0, count_ptr=0,
    GROUP_Q: tl.constexpr = 1, NUM_GROUPS: tl.constexpr = 1,
    STRIDE_PN: tl.constexpr = 0, STRIDE_PH: tl.constexpr = 0,
    STRIDE_PD: tl.constexpr = 0, STRIDE_SN: tl.constexpr = 0,
    USE_SHARED_PREFIX: tl.constexpr = False,
    # ---- MSA-010 knobs (all default to the shipped BF16 configuration) ----
    KSOURCE: tl.constexpr = 0,       # 0 shipped rows, 1 pre-transposed table,
                                     # 2 shipped rows but page := logical blk
    K_IS_T: tl.constexpr = False,    # K tile arrives as [head_dim, BLOCK_SIZE_K]
    KT_STRIDE_BLK: tl.constexpr = 0, KT_STRIDE_H: tl.constexpr = 0,
    V_PRELOADED: tl.constexpr = False,
    MODE: tl.constexpr = 0,
    PAGE_SUB_FROM: tl.constexpr = 0,   # skip that many pages after the prefix
    PAGE_SUB_TO: tl.constexpr = 0,     # skip that many pages at the tail
    EARLY_V: tl.constexpr = False,
):
    """Copy of the shipped BF16 prefill direct body (sparse_attn.py:141).

    MODE 0 is the faithful full kernel.  The diagnostic MODEs are documented in
    _page_math; each is an *upper or lower bound probe*, never a candidate.
    """
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
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
            shape=(gqa_group_size, head_dim),
            strides=(stride_qh, stride_qd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
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
            pp = tl.make_block_ptr(
                partial_ptr + (q_start + pid_q_j) * STRIDE_PN + pid_kh * gqa_group_size * STRIDE_PH,
                (gqa_group_size, head_dim),
                (STRIDE_PH, STRIDE_PD),
                (0, 0),
                (BLOCK_SIZE_QH, BLOCK_SIZE_D),
                (1, 0),
            )
            acc_o = tl.load(pp, boundary_check=(0, 1), padding_option="zero")
            l_acc = tl.where(off_n[None, :] == 0, l_i[:, None], 0.0)
            t_ptr_j += shared_count * stride_tk
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)
        q_qk = q
        lo = shared_count + PAGE_SUB_FROM
        hi = real_topk - PAGE_SUB_TO
        for _p in range(lo, hi):
            blk = tl.load(t_ptr_j).to(tl.int32)
            t_ptr_j = t_ptr_j + stride_tk
            page = tl.load(bt_row + blk).to(tl.int32)
            if KSOURCE == 2:
                page = blk
            k, v_base_ptr = _load_k(
                kv_cache_ptr, kt_ptr, page, pid_kh,
                stride_kv_blk, stride_kv_h, stride_kv_pos, stride_kv_d,
                KT_STRIDE_BLK, KT_STRIDE_H, BLOCK_SIZE_K, BLOCK_SIZE_D,
                head_dim, KSOURCE)
            v_pre = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=q.dtype)
            if EARLY_V:
                v_pre = tl.load(
                    tl.make_block_ptr(
                        base=v_base_ptr, shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                        strides=(stride_kv_pos, stride_kv_d), offsets=(0, 0),
                        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D), order=(1, 0),
                    ), boundary_check=(0, 1), padding_option="zero")
            acc_o, l_acc, m_i = _page_math(
                q_qk, k, v_pre, v_base_ptr, acc_o, l_acc, m_i, blk, q_abs, seq_len, off_n,
                sm_scale_log2e, stride_kv_pos, stride_kv_d,
                BLOCK_SIZE_QH, BLOCK_SIZE_K, BLOCK_SIZE_D,
                MODE, K_IS_T, EARLY_V)
        l_i = tl.sum(l_acc, axis=1)
        inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
        acc_o = acc_o * inv_l[:, None]
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + (q_start + pid_q_j) * stride_on + pid_kh * gqa_group_size * stride_oh,
            shape=(gqa_group_size, head_dim), strides=(stride_oh, stride_od),
            offsets=(0, 0), block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D), order=(1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))



# --------------------------------------------------------------------------
# host replica of the shipped dispatch (sparse_attn.py:719 tail)
# --------------------------------------------------------------------------
def make_dispatch(sa, launch_log):
    def dispatch(q, kv_cache, topk_idx, block_table, cu_seqlens_q, seq_lens, prefix_lens,
                 max_query_len, num_kv_heads, sm_scale, output, kt=None,
                 enable_prefix=True, use_copy=False, mode=0, k_is_t=False,
                 ksource=0, v_preloaded=False, page_sub=(0, 0), early_v=False,
                 num_warps=1, num_stages=1, tag="unnamed"):
        total_q, num_heads, head_dim = q.shape
        batch = cu_seqlens_q.shape[0] - 1
        gqa_group_size = num_heads // num_kv_heads
        block_size_h = triton.next_power_of_2(gqa_group_size)
        topk = topk_idx.shape[-1]
        prefix_groups = triton.cdiv(max_query_len, PREFIX_GROUP)
        partial = maximum = denom = counts = output
        if enable_prefix:
            partial = torch.empty(q.shape, device=q.device, dtype=torch.float32)
            maximum = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
            denom = torch.empty_like(maximum)
            counts = torch.empty((batch, num_kv_heads, prefix_groups), device=q.device,
                                 dtype=torch.int32)
            launch_log.append({"kernel": "_gqa_sparse_common_prefix",
                               "grid": [prefix_groups, num_kv_heads, batch],
                               "num_warps": 1, "num_stages": 1, "tag": tag,
                               "constexpr": {"GROUP_Q": PREFIX_GROUP,
                                             "BLOCK_SIZE_H": block_size_h}})
            sa._gqa_sparse_common_prefix[(prefix_groups, num_kv_heads, batch)](
                q, kv_cache, topk_idx, block_table, cu_seqlens_q, seq_lens, prefix_lens,
                partial, maximum, denom, counts, gqa_group_size, head_dim, topk, sm_scale,
                *q.stride(), *kv_cache.stride(), *topk_idx.stride(), block_table.stride(0),
                *partial.stride(), maximum.stride(0), num_kv_heads, prefix_groups, PREFIX_GROUP,
                block_size_h, triton.next_power_of_2(head_dim), num_warps=1, num_stages=1)
        grid = (max_query_len, num_kv_heads, batch)
        if kt is None:
            kt = kv_cache
        kt_stride_blk, kt_stride_h = (kt.stride(0), kt.stride(1)) if ksource == 1 else (0, 0)
        kwargs = dict(BLOCK_SIZE_Q=1, BLOCK_SIZE_K=128,
                      BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
                      BLOCK_SIZE_H=block_size_h, BLOCK_SIZE_QH=block_size_h,
                      USE_FP8=False, KV_SCALE_MODE=0,
                      num_warps=num_warps, num_stages=num_stages,
                      partial_ptr=partial, maximum_ptr=maximum, denom_ptr=denom,
                      count_ptr=counts, GROUP_Q=PREFIX_GROUP, NUM_GROUPS=prefix_groups,
                      STRIDE_PN=partial.stride(0), STRIDE_PH=partial.stride(1),
                      STRIDE_PD=partial.stride(2), STRIDE_SN=maximum.stride(0),
                      USE_SHARED_PREFIX=enable_prefix)
        kernel = _exp_direct if use_copy else sa._gqa_sparse_fwd_direct
        meta = {"kernel": "_gqa_sparse_fwd_direct" if not use_copy else "_exp_direct",
                "grid": list(grid), "num_warps": num_warps, "num_stages": num_stages,
                "tag": tag, "constexpr": {k: v for k, v in kwargs.items()
                                          if isinstance(v, (int, bool))}}
        if use_copy:
            kwargs.update(KSOURCE=ksource, K_IS_T=k_is_t,
                          KT_STRIDE_BLK=kt_stride_blk, KT_STRIDE_H=kt_stride_h,
                          MODE=mode, PAGE_SUB_FROM=page_sub[0], PAGE_SUB_TO=page_sub[1],
                          EARLY_V=early_v)
            meta["constexpr"].update(KSOURCE=ksource, K_IS_T=k_is_t, MODE=mode,
                                     PAGE_SUB_FROM=page_sub[0], PAGE_SUB_TO=page_sub[1],
                                     EARLY_V=early_v, V_PRELOADED=v_preloaded)
        launch_log.append(meta)
        args = [q, kv_cache, output, output, topk_idx, output, block_table, cu_seqlens_q,
                cu_seqlens_q, seq_lens, prefix_lens, num_kv_heads, gqa_group_size, head_dim,
                topk, 1, sm_scale,
                q.stride(0), q.stride(1), q.stride(2),
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), kv_cache.stride(3),
                0, 0, 0, 0,
                topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
                output.stride(0), output.stride(1), output.stride(2), block_table.stride(0)]
        if use_copy:
            args.insert(7, kt)      # kt_ptr sits right after block_table_ptr
        ret = kernel[grid](*args, **kwargs)
        return dict(partial=partial, maximum=maximum, denom=denom, counts=counts,
                    grid=grid, compiled=ret, launch=meta)

    return dispatch


def build_case(bench, shape_index):
    batch, qlen, seq, kvh, nh = bench.PREFILL_SHAPES[shape_index]
    gen = torch.Generator(device="cpu").manual_seed(SEED_OFFSET + shape_index)
    data = bench.make_data(batch, seq, kvh, nh, torch.device("gcu"), "bf16", decode=False,
                           decode_qlen=1, query_len=qlen, randomize_pages=True, generator=gen)
    return data, (batch, qlen, seq, kvh, nh)


def score_topk(bench, data, qlen, seq, kvh, topk=16, init_blocks=1, local_blocks=2):
    scores = bench.minimax_m3_index_score(data.idx_q, data.index_kv_cache, data.block_table,
                                          data.cu_q, data.seq_lens, data.prefix_lens,
                                          qlen, seq, kvh)
    return bench.minimax_m3_index_topk(scores, data.cu_q, data.prefix_lens, qlen,
                                       topk, init_blocks, local_blocks)


def call_sparse(bench, data, tk, out, qlen, kvh):
    bench._call_sparse(bench.minimax_m3_sparse_attn, data, tk, out, decode=False,
                       max_query_len=qlen, num_kv_heads=kvh, decode_qlen=1)


def run_chain(bench, data, out, qlen, seq, kvh, dispatch=None, kt=None,
              early_v=False, k_is_t=False, ksource=0, pack=False):
    """Full chain: index score + topk + attention (the shipped call shape)."""
    tk = score_topk(bench, data, qlen, seq, kvh)
    if dispatch is None:
        call_sparse(bench, data, tk, out, qlen, kvh)
        return tk
    if pack:
        kt = data.kv_cache[..., :128].permute(0, 1, 3, 2).contiguous()
    dispatch(data.q, data.kv_cache, tk, data.block_table, data.cu_q, data.seq_lens,
             data.prefix_lens, qlen, kvh, data.sm_scale, out, kt=kt, use_copy=True,
             early_v=early_v, k_is_t=k_is_t, ksource=ksource, tag="chain")
    return tk


def with_env(value):
    class _E:
        def __enter__(self):
            self.old = os.environ.get(EARLY_V_ENV)
            if value is None:
                os.environ.pop(EARLY_V_ENV, None)
            else:
                os.environ[EARLY_V_ENV] = value
            return self

        def __exit__(self, *exc):
            if self.old is None:
                os.environ.pop(EARLY_V_ENV, None)
            else:
                os.environ[EARLY_V_ENV] = self.old
            return False
    return _E()


def diff_stats(a, b):
    """Elementwise diff, plus an explicitly computed cosine.

    torch's fused cosine_similarity returns ~0.9999 even for identical BF16
    tensors on this backend (it is the *fused* kernel, not a maths difference),
    so the cosine is computed here in float64 and only used as a coarse guard;
    the bit-exact comparison is the actual authority.
    """
    d = (a.float().cpu() - b.float().cpu()).abs()
    af, bf = a.float().cpu().reshape(-1).double(), b.float().cpu().reshape(-1).double()
    denom = (af.norm() * bf.norm()).item()
    cos = (af @ bf).item() / denom if denom else 0.0
    return dict(max_abs=d.max().item(), mean_abs=d.mean().item(),
                nonzero=int((d > 0).sum().item()), numel=d.numel(),
                cosine=cos, bit_identical=bool(torch.equal(a.cpu(), b.cpu())))


def summarize(values):
    return dict(n=len(values), samples=list(values),
                median=statistics.median(values), mean=statistics.fmean(values),
                min=min(values), max=max(values),
                stdev=statistics.stdev(values) if len(values) > 1 else 0.0)



# --------------------------------------------------------------------------
# torch float32 reference (same selected pages / same page order)
# --------------------------------------------------------------------------
def torch_reference(data, tk, rows, scale, seq_len, num_heads):
    q = data.q.float().cpu()
    kv = data.kv_cache.float().cpu()
    bt = data.block_table.cpu()
    tkc = tk.cpu()
    out = torch.zeros((len(rows), num_heads, q.shape[-1]), dtype=torch.float32)
    prefix_len = int(data.prefix_lens[0].item())
    for i, row in enumerate(rows):
        q_abs = prefix_len + row
        acc = torch.zeros((num_heads, q.shape[-1]), dtype=torch.float32)
        m = torch.full((num_heads,), float("-inf"))
        l = torch.zeros((num_heads,))
        # The reference walks every selected page in page order, exactly like
        # the stock kernel: the shared-prefix kernel seeds the first three
        # (measured count == 3) but the direct loop then continues from page 3;
        # together they cover the same 16 pages in the same order.
        for slot in range(tkc.shape[-1]):
            first = int(tkc[0, row, slot])
            if first < 0:
                continue
            page = int(bt[0, first])
            k = kv[page, 0, :, :128]
            v = kv[page, 0, :, 128:]
            pos = first * 128 + torch.arange(128)
            valid = (pos < seq_len) & (pos <= q_abs)
            scores = (q[row] @ k.T) * scale
            scores = scores.masked_fill(~valid[None, :], -float("inf"))
            m_new = torch.maximum(m, scores.max(dim=-1).values)
            safe = torch.where(m_new > -float("inf"), m_new, torch.zeros(()))
            p = torch.where(valid[None, :],
                            torch.exp2((scores - safe[:, None]) * 1.4426950409), torch.zeros(()))
            alpha = torch.where(m > float("-inf"),
                                torch.exp2((m - safe) * 1.4426950409), torch.zeros(()))
            l = l * alpha + p.sum(dim=-1)
            acc = acc * alpha[:, None] + p @ v
            m = m_new
        out[i] = acc / l[:, None]
    return out


def mask_check(bench, data, tk, clean, qlen, kvh, rows, poison=30000.0):
    """Falsifiable per-row causal-mask test on logical positions."""
    bt = data.block_table.cpu()
    blocks = sorted({int(x) for x in tk[0].reshape(-1).tolist() if int(x) >= 0})
    page_of = {blk: int(bt[0, blk]) for blk in blocks}
    prefix_len = int(data.prefix_lens[0].item())

    def run(edits):
        kv = data.kv_cache.clone()
        big = torch.tensor(poison, dtype=kv.dtype, device=kv.device)
        n = 0
        for page, offs in edits.items():
            if offs:
                kv[page, :, torch.tensor(sorted(offs), device=kv.device), :] = big
                n += len(offs)
        saved = data.kv_cache
        data.kv_cache = kv
        try:
            out = torch.empty_like(data.q)
            call_sparse(bench, data, tk, out, qlen, kvh)
        finally:
            data.kv_cache = saved
        return out, n

    res = {}
    for r in rows:
        q_abs = prefix_len + r
        fut, vis = {}, {}
        for blk in blocks:
            page, base = page_of[blk], blk * 128
            f = [o for o in range(128) if base + o > q_abs]
            v = [o for o in range(128) if base + o <= q_abs]
            if f:
                fut.setdefault(page, []).extend(f)
            if v:
                vis.setdefault(page, []).extend(v)
        dirty, n_f = run(fut)
        ch_f = int((dirty[r] != clean[r]).sum())
        ctrl, n_v = run(vis)
        ch_v = int((ctrl[r] != clean[r]).sum())
        res[f"row{r}"] = dict(q_abs=q_abs, future_positions=n_f, future_changed=ch_f,
                              control_positions=n_v, control_changed=ch_v,
                              enforced=(ch_f == 0 and ch_v > 0))
        print(f"[mask r={r}] q_abs={q_abs} future n={n_f} changed={ch_f} | "
              f"control n={n_v} changed={ch_v} -> "
              f"{'ENFORCED' if ch_f == 0 and ch_v > 0 else 'LEAK/INCONCLUSIVE'}")
    return res



# --------------------------------------------------------------------------
# micro kernels: QK matrix equivalence per real page, and local-reuse control
# --------------------------------------------------------------------------
@triton.jit
def _qk_page(q_ptr, kv_ptr, kt_ptr, o_ptr, page_ptr, ROW: tl.constexpr,
             STRIDE_QN: tl.constexpr, STRIDE_QH: tl.constexpr,
             STRIDE_KV_BLK: tl.constexpr, STRIDE_KV_POS: tl.constexpr,
             STRIDE_KV_D: tl.constexpr, STRIDE_KT_BLK: tl.constexpr,
             BLOCK_SIZE_QH: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
             BLOCK_SIZE_D: tl.constexpr, KSOURCE: tl.constexpr):
    """Dump the raw [BLOCK_SIZE_QH, BLOCK_SIZE_K] QK matrix for one real page."""
    i = tl.program_id(0)
    page = tl.load(page_ptr + i).to(tl.int32)
    q = tl.load(
        tl.make_block_ptr(
            base=q_ptr + ROW * STRIDE_QN, shape=(BLOCK_SIZE_QH, BLOCK_SIZE_D),
            strides=(STRIDE_QH, 1), offsets=(0, 0),
            block_shape=(BLOCK_SIZE_QH, BLOCK_SIZE_D), order=(1, 0),
        ), boundary_check=(0, 1), padding_option="zero")
    if KSOURCE == 1:
        k = tl.load(
            tl.make_block_ptr(
                base=kt_ptr + page * STRIDE_KT_BLK, shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
                strides=(BLOCK_SIZE_K, 1), offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K), order=(1, 0),
            ), boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(q, k)
    else:
        k = tl.load(
            tl.make_block_ptr(
                base=kv_ptr + page * STRIDE_KV_BLK, shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                strides=(STRIDE_KV_POS, STRIDE_KV_D), offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D), order=(1, 0),
            ), boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(q, tl.trans(k))
    hs = tl.arange(0, BLOCK_SIZE_QH)
    on = tl.arange(0, BLOCK_SIZE_K)
    tl.store(o_ptr + i * BLOCK_SIZE_QH * BLOCK_SIZE_K + hs[:, None] * BLOCK_SIZE_K + on[None, :], qk)


# REUSE 0: N distinct real pages, one K DMA per page (matches the shipped loop).
# REUSE 1: the same real page N times -> the *address* is hot, but every
#          iteration still issues its own global->local K DMA and wait.
# REUSE 2: the same real page is loaded once, outside the loop, and the N QK
#          convgens read that already-local tile -> N-1 K DMAs really removed.
@triton.jit
def _reuse_micro(q_ptr, kv_ptr, t_ptr, page_ptr, o_ptr, ROW: tl.constexpr,
                 STRIDE_QN: tl.constexpr, STRIDE_QH: tl.constexpr,
                 STRIDE_KV_BLK: tl.constexpr, STRIDE_KV_POS: tl.constexpr,
                 STRIDE_KV_D: tl.constexpr, STRIDE_TN: tl.constexpr,
                 STRIDE_TK: tl.constexpr, NPAGES: tl.constexpr,
                 BLOCK_SIZE_QH: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                 BLOCK_SIZE_D: tl.constexpr, REUSE: tl.constexpr):
    pid = tl.program_id(0)
    q = tl.load(
        tl.make_block_ptr(
            base=q_ptr + ROW * STRIDE_QN, shape=(BLOCK_SIZE_QH, BLOCK_SIZE_D),
            strides=(STRIDE_QH, 1), offsets=(0, 0),
            block_shape=(BLOCK_SIZE_QH, BLOCK_SIZE_D), order=(1, 0),
        ), boundary_check=(0, 1), padding_option="zero")
    acc = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_K), dtype=tl.float32)
    if REUSE == 2:
        k0 = tl.load(
            tl.make_block_ptr(
                base=kv_ptr + tl.load(page_ptr).to(tl.int32) * STRIDE_KV_BLK,
                shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                strides=(STRIDE_KV_POS, STRIDE_KV_D), offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D), order=(1, 0),
            ), boundary_check=(0, 1), padding_option="zero")
        for _ in range(NPAGES):
            acc += tl.dot(q, tl.trans(k0))
    else:
        for i in range(NPAGES):
            if REUSE == 1:
                page = tl.load(page_ptr).to(tl.int32)
            else:
                page = tl.load(page_ptr + i).to(tl.int32)
            k = tl.load(
                tl.make_block_ptr(
                    base=kv_ptr + page * STRIDE_KV_BLK,
                    shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                    strides=(STRIDE_KV_POS, STRIDE_KV_D), offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D), order=(1, 0),
                ), boundary_check=(0, 1), padding_option="zero")
            acc += tl.dot(q, tl.trans(k))
    hs = tl.arange(0, BLOCK_SIZE_QH)
    on = tl.arange(0, BLOCK_SIZE_K)
    base = pid * BLOCK_SIZE_QH * BLOCK_SIZE_K + hs[:, None] * BLOCK_SIZE_K + on[None, :]
    tl.store(o_ptr + base, acc)



# --------------------------------------------------------------------------
# timing helpers
# --------------------------------------------------------------------------
def bench_samples(fns, rounds, warmup=3):
    """Rotating alternation; one device-event sample per call, order recorded."""
    names = list(fns)
    for n in names:
        for _ in range(warmup):
            fns[n]()
    torch.gcu.synchronize()
    samples = {n: [] for n in names}
    orders = []
    for r in range(rounds):
        rot = names[r % len(names):] + names[:r % len(names)]
        orders.append(rot)
        for n in rot:
            a = torch.gcu.Event(enable_timing=True)
            b = torch.gcu.Event(enable_timing=True)
            a.record()
            fns[n]()
            b.record()
            b.synchronize()
            samples[n].append(a.elapsed_time(b))
    return samples, orders


def event_overhead(rounds=21):
    ts = []
    for _ in range(rounds):
        a = torch.gcu.Event(enable_timing=True)
        b = torch.gcu.Event(enable_timing=True)
        a.record()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return summarize(ts)


# --------------------------------------------------------------------------
# flat QK kernel: identical per-block work, controllable resident blocks
# --------------------------------------------------------------------------
@triton.jit
def _qk_only_flat(q_ptr, kv_ptr, t_ptr, row_ptr, off_ptr, o_ptr,
                  STRIDE_QN: tl.constexpr, STRIDE_QH: tl.constexpr,
                  STRIDE_KV_BLK: tl.constexpr, STRIDE_KV_POS: tl.constexpr,
                  STRIDE_KV_D: tl.constexpr, STRIDE_TN: tl.constexpr,
                  STRIDE_ON: tl.constexpr, NPAGES: tl.constexpr,
                  NPAGE_TABLE: tl.constexpr, BLOCK_SIZE_QH: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
                  KSOURCE: tl.constexpr):
    """Per page: topk load + page-table indirection + K DMA + transpose + QK.

    `NPAGES` selected pages of one row, exactly like one program of the shipped
    direct kernel, but with no softmax / no V / no PV.  `off_ptr` spreads blocks
    over the page table; the grid is padded so the resident block count and the
    per-block loop are identical across NPAGES=4 vs 13 and across K layouts.
    """
    pid = tl.program_id(0)
    row = tl.load(row_ptr + pid)
    off = tl.load(off_ptr + pid)
    q = tl.load(tl.make_block_ptr(base=q_ptr + row * STRIDE_QN,
                                  shape=(BLOCK_SIZE_QH, BLOCK_SIZE_D),
                                  strides=(STRIDE_QH, 1), offsets=(0, 0),
                                  block_shape=(BLOCK_SIZE_QH, BLOCK_SIZE_D), order=(1, 0)),
                boundary_check=(0, 1), padding_option="zero")
    acc = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_K), dtype=tl.float32)
    for i in range(NPAGES):
        blk = tl.load(t_ptr + row * STRIDE_TN + i).to(tl.int32)
        pg = (off + blk) % NPAGE_TABLE
        if KSOURCE == 1:
            k = tl.load(tl.make_block_ptr(base=kv_ptr + pg * STRIDE_KV_BLK,
                                          shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
                                          strides=(BLOCK_SIZE_K, 1), offsets=(0, 0),
                                          block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
                                          order=(1, 0)),
                        boundary_check=(0, 1), padding_option="zero")
            acc += tl.dot(q, k)
        else:
            k = tl.load(tl.make_block_ptr(base=kv_ptr + pg * STRIDE_KV_BLK,
                                          shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                                          strides=(STRIDE_KV_POS, STRIDE_KV_D), offsets=(0, 0),
                                          block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                                          order=(1, 0)),
                        boundary_check=(0, 1), padding_option="zero")
            acc += tl.dot(q, tl.trans(k))
    hs = tl.arange(0, BLOCK_SIZE_QH)
    on = tl.arange(0, BLOCK_SIZE_K)
    tl.store(o_ptr + pid * STRIDE_ON + hs[:, None] * BLOCK_SIZE_K + on[None, :], acc)


def run_micro(data, tk, kt, qlen, rounds=5):
    """Flat QK-only vs pre-transposed at NPAGES 4 and 13, plus local reuse."""
    import triton  # noqa: F401
    torch.gcu.synchronize()
    dev = data.q.device
    qh = 8
    n_prog = 1024
    row = torch.arange(n_prog, dtype=torch.int32, device=dev) % qlen
    off = torch.arange(n_prog, dtype=torch.int32, device=dev) * 4
    outbuf = torch.empty((n_prog, qh, 128), dtype=torch.float32, device=dev)
    res = {"n_programs": n_prog, "qk_flat": {}, "reuse": {}, "event_overhead": event_overhead()}
    fns = {}
    metas = {}
    for npages in (4, 13):
        for ksource, name in ((0, "shipped"), (1, "pretrans")):
            tag = f"np{npages}_{name}"
            def mk(npages=npages, ksource=ksource):
                def run():
                    return _qk_only_flat[(n_prog,)](
                        data.q, kt if ksource else data.kv_cache, tk, row, off, outbuf,
                        data.q.stride(0), data.q.stride(1),
                        data.kv_cache.stride(0), data.kv_cache.stride(2), data.kv_cache.stride(3),
                        tk.stride(1), outbuf.stride(0), npages, int(data.kv_cache.shape[0]),
                        qh, 128, 128, ksource, num_warps=1, num_stages=1)
                return run
            fns[tag] = mk()
            metas[tag] = dict(npages=npages, ksource=ksource)
    samples, orders = bench_samples(fns, rounds)
    comp = {}
    for tag, fn in fns.items():
        c = fn()
        torch.gcu.synchronize()
        comp[tag] = c
    for tag in fns:
        res["qk_flat"][tag] = dict(meta=metas[tag], timing=summarize(samples[tag]),
                                   ir=ir_brief(ir_census(comp[tag], tag)))
    res["qk_flat_orders"] = orders
    # marginal per page from the 4 -> 13 page slope (same resident block count)
    for name in ("shipped", "pretrans"):
        a = res["qk_flat"][f"np4_{name}"]["timing"]["median"]
        b = res["qk_flat"][f"np13_{name}"]["timing"]["median"]
        res["qk_flat"][f"slope_{name}_ms_per_page"] = (b - a) / 9.0
        res["qk_flat"][f"slope_{name}_us_per_page"] = (b - a) / 9.0 * 1000.0
    return res


def run_reuse(data, tk, kt, qlen, rounds=5):
    """Local-tile reuse control on the SAME real page (equivalence-checked).

    REUSE 0: 13 distinct real pages, one global->local K DMA each (the shipped
             page structure, QK only).
    REUSE 1: one real page addressed 13 times.  The *address* is hot; every
             iteration still issues its own DMA + wait -- this is the control
             that shows "hot address" is NOT "already local".
    REUSE 2: one real page loaded ONCE before the loop; 13 QK convgens read that
             same local tile, so 12 K DMAs really leave the chain.
    """
    dev = data.q.device
    qh = 8
    n_prog = 1024
    row = torch.arange(n_prog, dtype=torch.int32, device=dev) % qlen
    outbuf = torch.empty((n_prog, qh, 128), dtype=torch.float32, device=dev)
    bt = data.block_table[0]
    blocks = [int(x) for x in tk[0, 0].tolist() if int(x) >= 0][:4]
    pages = torch.tensor([int(bt[b]) for b in blocks], dtype=torch.int32, device=dev)
    res = {"pages": pages.tolist(), "blocks": blocks, "rows": int(row.max()) + 1}
    fns, comp = {}, {}
    for reuse, name in ((0, "distinct_pages"), (1, "hot_address"), (2, "local_tile")):
        def mk(reuse=reuse):
            def run():
                return _reuse_micro[(n_prog,)](
                    data.q, data.kv_cache, tk, pages, outbuf, 0,
                    data.q.stride(0), data.q.stride(1),
                    data.kv_cache.stride(0), data.kv_cache.stride(2), data.kv_cache.stride(3),
                    tk.stride(1), tk.stride(2), 13, qh, 128, 128, reuse,
                    num_warps=1, num_stages=1)
            return run
        fns[name] = mk()
    samples, orders = bench_samples(fns, rounds)
    for name, fn in fns.items():
        c = fn()
        torch.gcu.synchronize()
        comp[name] = c
    for name in fns:
        res[name] = dict(timing=summarize(samples[name]),
                         ir=ir_brief(ir_census(comp[name], name)))
    res["orders"] = orders
    # static per-program counts, from the compiled page loop
    def _dmareads(rec):
        c = rec.get("gcuir_page_loop", {}).get("counts", {})
        return c.get("memref.dma_start", 0), c.get("memref.dma_wait", 0)
    res["per_program_page_loop"] = {
        name: dict(dma_start=_dmareads(res[name]["ir"])[0],
                   dma_wait=_dmareads(res[name]["ir"])[1],
                   pages_per_program=13 if name != "local_tile" else 1)
        for name in ("distinct_pages", "hot_address", "local_tile")}
    res["actual_k_dma_issued_per_program"] = {
        "distinct_pages": 13, "hot_address": 13, "local_tile": 1}
    res["actual_k_dma_removed_vs_distinct"] = {
        "distinct_pages": 0, "hot_address": 0, "local_tile": 12}
    return res



# --------------------------------------------------------------------------
# static IR / compile-metadata census (from the CompiledKernel, not the cache)
# --------------------------------------------------------------------------
COUNT_RE = re.compile(r'"([A-Za-z_0-9]+\.[A-Za-z_0-9]+)"\s*[\\(\\{]')
TOPS_RE = re.compile(r"@(tops_[a-z0-9_]+)")


def _count_ops(text):
    out = {}
    for m in COUNT_RE.finditer(text):
        out[m.group(1)] = out.get(m.group(1), 0) + 1
    return out


def _count_tops(text):
    out = {}
    for m in TOPS_RE.finditer(text):
        out[m.group(1)] = out.get(m.group(1), 0) + 1
    return out


def _balanced_region(text, start):
    i = text.index("{", start)
    depth = 0
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1
    return text[start:]


def ir_census(compiled, tag):
    """Op census + compile metadata of one compiled probe kernel."""
    asm = getattr(compiled, "asm", {}) or {}
    md = getattr(compiled, "metadata", None)
    meta = {}
    for key in ("num_warps", "num_stages", "shared", "max_shared", "warp_size",
                "num_ctas", "arch", "vector_length", "hash", "name"):
        meta[key] = getattr(md, key, None) if md is not None else None
    rec = {"tag": tag,
           "hash": getattr(compiled, "hash", None) or meta.get("hash"),
           "n_regs": getattr(compiled, "n_regs", None),
           "n_spills": getattr(compiled, "n_spills", None),
           "metadata": meta,
           "asm_keys": sorted(asm.keys())}
    ttir = asm.get("ttir")
    if isinstance(ttir, bytes):
        ttir = ttir.decode("utf-8", "replace")
    gcuir = asm.get("gcuir")
    if isinstance(gcuir, bytes):
        gcuir = gcuir.decode("utf-8", "replace")
    llir = asm.get("llir")
    if isinstance(llir, bytes):
        llir = llir.decode("utf-8", "replace")
    if ttir:
        ops = {}
        for line in ttir.splitlines():
            m = re.search(r"=\s*([a-z_]+\.[a-z_0-9]+)\b", line)
            if m:
                ops[m.group(1)] = ops.get(m.group(1), 0) + 1
        rec["ttir_ops"] = dict(sorted(ops.items(), key=lambda kv: -kv[1]))
        rec["ttir_has_trans"] = "tt.trans" in ttir
    if gcuir:
        rec["gcuir_counts"] = dict(sorted(_count_ops(gcuir).items(), key=lambda kv: -kv[1]))
        starts = [m.start() for m in re.finditer(r'"scf\.for"\(', gcuir)]
        rec["gcuir_loop_count"] = len(starts)
        # the page loop is the LAST scf.for (the first ones are the K/V DMA
        # descriptor loops emitted for each block pointer)
        body = None
        for st in reversed(starts):
            cand = _balanced_region(gcuir, st)
            if "gcu.matmul" in cand:
                body = cand
                break
        if body is not None:
            cnt = _count_ops(body)
            rec["gcuir_page_loop"] = dict(lines=body.count("\n"),
                                          has_matmul=True,
                                          counts=dict(sorted(cnt.items(), key=lambda kv: -kv[1])))
            # ordered DMA / transpose sequence inside the page body
            seq = []
            for m in re.finditer(r'"(memref\.dma_start|memref\.dma_wait|memref_ext\.[a-z_]+|'
                                 r'gcu\.matmul|memref\.alloc)"', body):
                seq.append(m.group(1))
            rec["gcuir_page_loop"]["sequence"] = seq
            rec["gcuir_page_loop"]["sequence_compact"] = _compact_seq(seq)
    if llir:
        rec["llir_tops"] = dict(sorted(_count_tops(llir).items(), key=lambda kv: -kv[1]))
        rec["llir_allocas"] = len(re.findall(r"\balloca\b", llir))
        rec["shared_allocas"] = len(re.findall(r"alloca[^\n]*addrspace\(3\)", llir))
        rec["llir_convgen"] = {m.group(1): rec["llir_tops"].get(m.group(1), 0)
                               for m in re.finditer(r"@(tops_convgen_[a-z0-9_]+)", llir)}
    return rec


def _compact_seq(seq):
    out = []
    for op in seq:
        if out and out[-1][0] == op:
            out[-1][1] += 1
        else:
            out.append([op, 1])
    return [f"{op}x{n}" if n > 1 else op for op, n in out]


def ir_brief(rec):
    """A one-screen summary for the console."""
    out = {"tag": rec["tag"], "hash": (rec.get("hash") or "")[:12],
           "n_regs": rec.get("n_regs"), "n_spills": rec.get("n_spills"),
           "shared": rec["metadata"].get("shared"),
           "num_warps": rec["metadata"].get("num_warps"),
           "num_stages": rec["metadata"].get("num_stages")}
    g = rec.get("gcuir_page_loop", {}).get("counts", {})
    out["page_loop"] = {k: g.get(k, 0) for k in
                        ("memref.dma_start", "memref.dma_wait", "memref_ext.transpose_start",
                         "memref_ext.slice_start", "memref_ext.deslice_start",
                         "gcu.matmul", "memref.alloc")}
    t = rec.get("llir_tops", {})
    out["tops"] = {k: t.get(k, 0) for k in
                   ("tops_wait_dte", "tops_memcpy_async", "tops_transpose_async_2_bf16",
                    "tops_slice_async_2_bf16", "tops_deslice_async_2_bf16",
                    "tops_convgen_m8_n128_k128_bf16_fp32",
                    "tops_convgen_m8_n128_k128_bias_bf16_fp32")}
    out["page_seq"] = rec.get("gcuir_page_loop", {}).get("sequence_compact")
    out["convgen_all"] = rec.get("llir_convgen")
    out["ttir_has_trans"] = rec.get("ttir_has_trans")
    return out



# --------------------------------------------------------------------------
# --verify
# --------------------------------------------------------------------------
def run_verify(device, env):
    from benchmark import test_minimax_sparse_attention_benchmark as bench
    from flag_attn.runtime.backend._enflame.msa import sparse_attn as sa

    res = {"mode": "verify", "env": env, "checks": {}, "contracts": {},
           "torch": torch.__version__, "triton": triton.__version__}
    ok = True
    log = []
    dispatch = make_dispatch(sa, log)

    data, shape = build_case(bench, SHAPE_INDEX)
    batch, qlen, seq, kvh, nh = shape
    tk = score_topk(bench, data, qlen, seq, kvh)
    tk_ref = tk.clone()
    out_stock = torch.empty_like(data.q)
    call_sparse(bench, data, tk, out_stock, qlen, kvh)
    torch.gcu.synchronize()
    with with_env("1"):
        out_early = torch.empty_like(data.q)
        call_sparse(bench, data, tk, out_early, qlen, kvh)
    torch.gcu.synchronize()
    res["shape"] = dict(batch=batch, query_len=qlen, seq_len=seq, num_kv_heads=kvh,
                        num_heads=nh, head_dim=128, topk=int(tk.shape[-1]),
                        shape_index=SHAPE_INDEX, seed=SEED_OFFSET + SHAPE_INDEX,
                        pages_selected=int((tk[0] >= 0).sum(-1).max().item()))
    res["topk_stable"] = bool(torch.equal(tk, tk_ref))

    rows = sorted({0, 1, 2, 3, 4, 5, 6, 7, qlen // 4, qlen // 2, qlen - 24, qlen - 1})
    ref = torch_reference(data, tk, rows, data.sm_scale, seq, nh)
    d_stock = diff_stats(out_stock.cpu()[rows], ref)
    d_early = diff_stats(out_early.cpu()[rows], ref)
    d_pair = diff_stats(out_stock, out_early)
    print(f"[verify] stock vs fp32  max={d_stock['max_abs']:.3e} cos={d_stock['cosine']:.8f}")
    print(f"[verify] early vs fp32  max={d_early['max_abs']:.3e} cos={d_early['cosine']:.8f}")
    print(f"[verify] stock vs early max={d_pair['max_abs']:.3e} nz={d_pair['nonzero']}")
    res["checks"]["stock_vs_fp32"] = d_stock
    res["checks"]["early_vs_fp32"] = d_early
    res["checks"]["stock_vs_early"] = d_pair
    stock_ok = d_stock["max_abs"] <= 2.0 and d_stock["cosine"] > 0.999
    ok &= stock_ok and d_pair["bit_identical"]

    # ---------------- probe contracts on the K-staging variants ----------------
    kt = build_pretransposed_table(data)
    res["kt_table"] = dict(shape=list(kt.shape), dtype=str(kt.dtype),
                           stride=list(kt.stride()))
    out = torch.empty_like(data.q)
    copies = {}
    # Arms that must be BIT-EXACT: same Q, same selected pages, same page order.
    for tag, kwargs in (
        ("copy", dict()),
        ("k_pre", dict()),
        ("k_pre_trans", dict(k_is_t=True, ksource=1)),
        ("kview_trans", dict(k_is_t=True, ksource=3)),
    ):
        o = torch.empty_like(data.q)
        dispatch(data.q, data.kv_cache, tk, data.block_table, data.cu_q, data.seq_lens,
                 data.prefix_lens, qlen, kvh, data.sm_scale, o, kt=kt, use_copy=True,
                 tag=tag, **kwargs)
        torch.gcu.synchronize()
        copies[tag] = o
        d = diff_stats(o, out_stock)
        res["contracts"][tag] = {"kind": "equivalence", "vs_stock": d,
                                 "pass": bool(d["bit_identical"]),
                                 "exact": bool(d["bit_identical"])}
        print(f"[verify:{tag:12s}] vs stock max={d['max_abs']:.3e} "
              f"nz={d['nonzero']} exact={d['bit_identical']} -> "
              f"{'PASS' if d['bit_identical'] else 'FAIL'}")
        ok &= bool(d["bit_identical"])

    # Diagnostic probes that deliberately change the computation.  They must
    # produce a finite, non-degenerate, *observable* output (so nothing can be
    # dead-code-eliminated) and are expected to differ from stock.
    for tag, kwargs, expect in (
        ("k_pre_noptr", dict(ksource=2), "page-table indirection removed"),
        ("xp_sub4", dict(page_sub=(0, 9)), "4-page sub-chain"),
    ):
        o = torch.empty_like(data.q)
        meta = dispatch(data.q, data.kv_cache, tk, data.block_table, data.cu_q, data.seq_lens,
                        data.prefix_lens, qlen, kvh, data.sm_scale, o, kt=kt, use_copy=True,
                        tag=tag, **kwargs)
        torch.gcu.synchronize()
        d = diff_stats(o, out_stock)
        nz_out = int((o != 0).sum().item())
        live = bool(torch.isfinite(o).all().item()) and nz_out > 0.9 * o.numel()
        res["contracts"][tag] = {
            "kind": "diagnostic_probe", "intent": expect, "vs_stock": d,
            "output_nonzero_elems": nz_out, "output_numel": int(o.numel()),
            "live_output": live, "differs_from_stock": bool(d["nonzero"] > 0),
            "page_sub": list(kwargs.get("page_sub", (0, 0))),
            "pass": bool(live and d["nonzero"] > 0)}
        print(f"[probe:{tag:13s}] ({expect}) vs stock max={d['max_abs']:.3e} "
              f"nz={d['nonzero']} out_nonzero={nz_out}/{o.numel()} finite={live} -> "
              f"{'OK' if res['contracts'][tag]['pass'] else 'FAIL'}")
        ok &= res["contracts"][tag]["pass"]

    # A full-length probe (no page subtraction) must also be bit-exact: this is
    # the "same 13 pages, page table kept" arm used in the timing matrix.
    o = torch.empty_like(data.q)
    dispatch(data.q, data.kv_cache, tk, data.block_table, data.cu_q, data.seq_lens,
             data.prefix_lens, qlen, kvh, data.sm_scale, o, kt=kt, use_copy=True,
             tag="k_pre_full13")
    torch.gcu.synchronize()
    d = diff_stats(o, out_stock)
    res["contracts"]["k_pre_full13"] = {"kind": "equivalence", "vs_stock": d,
                                       "pass": bool(d["bit_identical"]),
                                       "exact": bool(d["bit_identical"])}
    print(f"[verify:k_pre_full13] vs stock max={d['max_abs']:.3e} exact={d['bit_identical']}")
    ok &= bool(d["bit_identical"])

    # copy_early_v is the MSA-009 scheduling candidate, reproduced in this copy
    o = torch.empty_like(data.q)
    variant_call(bench, data, tk, o, qlen, kvh, dispatch, kt, "copy_early_v", early_v=True)
    torch.gcu.synchronize()
    d = diff_stats(o, out_stock)
    res["contracts"]["copy_early_v"] = {"vs_stock": d, "pass": bool(d["bit_identical"]),
                                        "exact": bool(d["bit_identical"])}
    print(f"[verify:copy_early_v ] vs stock max={d['max_abs']:.3e} nz={d['nonzero']} "
          f"exact={d['bit_identical']}")
    ok &= bool(d["bit_identical"])

    # pre-transposed table must contain exactly the same K bytes as the cache
    kbytes_ok = pretransposed_bytes_match(data, kt)
    res["contracts"]["kt_bytes_equal_kv_k"] = bool(kbytes_ok)
    print(f"[verify] pre-transposed table equals kv_cache[..., :128] permuted: {kbytes_ok}")
    ok &= kbytes_ok

    # per-page QK matrix: shipped layout vs pre-transposed layout
    qk_eq = qk_page_equivalence(data, tk, kt, qlen)
    res["contracts"]["qk_page_equivalence"] = qk_eq
    print(f"[verify] QK per-page: pages={qk_eq['pages']} max_abs={qk_eq['max_abs']:.3e} "
          f"cos={qk_eq['cosine']:.9f} -> {'PASS' if qk_eq['pass_'] else 'FAIL'}")
    ok &= qk_eq["pass_"]

    # a deliberately wrong mapping must fail: proves the check is falsifiable
    bad = qk_page_equivalence(data, tk, kt.flip(0).contiguous(), qlen)
    res["contracts"]["qk_page_equivalence_negative_control"] = bad
    print(f"[verify] QK negative control (flipped table): max_abs={bad['max_abs']:.3e} "
          f"-> {'DETECTED' if not bad['pass_'] else 'NOT DETECTED (bad)'}")
    ok &= not bad["pass_"]

    # logical block id vs physical page: same data?  (documents the indirection)
    phys = data.block_table[0][tk[0, 0]]
    logical = tk[0, 0]
    res["page_indirection"] = dict(
        row0_blocks=logical.tolist(), row0_pages=phys.tolist(),
        identical=bool(torch.equal(phys.to(torch.int64), logical.to(torch.int64))),
        note="random physical page table (make_data randomize_pages=True)")
    print(f"[verify] page indirection: logical!=physical for "
          f"{int((phys != logical).sum())}/{len(logical)} pages")

    # ---------------- mask check on the stock route ----------------
    mask_rows = [0, 1, qlen // 2, qlen - 1]
    mask_res = mask_check(bench, data, tk, out_stock, qlen, kvh, mask_rows)
    res["checks"]["mask_stock"] = mask_res
    ok &= all(v["enforced"] for v in mask_res.values())

    # ---------------- launch + IR ----------------
    log.clear()
    full = dispatch(data.q, data.kv_cache, tk, data.block_table, data.cu_q, data.seq_lens,
                    data.prefix_lens, qlen, kvh, data.sm_scale, out, tag="full")
    res["launch"] = log[-2:]
    res["ir"] = [ir_census(full["compiled"], "full")]
    stock_log = []
    stock_dispatch = make_dispatch(sa, stock_log)
    stock_full = stock_dispatch(data.q, data.kv_cache, tk, data.block_table, data.cu_q,
                                data.seq_lens, data.prefix_lens, qlen, kvh, data.sm_scale,
                                out, use_copy=False, tag="stock")
    res["ir"].append(ir_census(stock_full["compiled"], "stock"))
    for tag, kw in (("k_pre_trans", dict(k_is_t=True, ksource=1)),
                    ("k_pre_noptr", dict(ksource=2))):
        l2 = []
        d2 = make_dispatch(sa, l2)
        r2 = d2(data.q, data.kv_cache, tk, data.block_table, data.cu_q, data.seq_lens,
                data.prefix_lens, qlen, kvh, data.sm_scale, out, kt=kt, use_copy=True,
                tag=tag, **kw)
        res["ir"].append(ir_census(r2["compiled"], tag))
    res["ir_brief"] = [ir_brief(r) for r in res["ir"]]
    for b in res["ir_brief"]:
        print("[ir]", json.dumps(b, sort_keys=True))

    res["pass"] = bool(ok)
    print(f"[verify] overall={'PASS' if ok else 'FAIL'}")
    return res


def build_pretransposed_table(data):
    """Diagnostic-only [page, h, d, pos] copy of the K half of the paged cache.

    Values are the *same* K bytes as `kv_cache[..., :128]`, only with the last
    two axes swapped (plus the matching element permutation).  This is not a
    production layout: it is built once on the device, off the timed path.
    """
    k = data.kv_cache[..., :128]                    # [pages, h, pos, d]
    return k.permute(0, 1, 3, 2).contiguous()       # [pages, h, d, pos]


def pretransposed_bytes_match(data, kt):
    a = data.kv_cache[..., :128].permute(0, 1, 3, 2).contiguous()
    return bool(torch.equal(a, kt))


def qk_page_equivalence(data, tk, kt, qlen, row=0):
    """QK of the *same selected pages* through both layouts, vs torch fp32."""
    import benchmark.test_minimax_sparse_attention_benchmark as bench  # noqa: F401
    bt = data.block_table[0]
    blocks = [int(x) for x in tk[0, row].tolist() if int(x) >= 0]
    pages = torch.tensor([int(bt[b]) for b in blocks], dtype=torch.int32, device=data.q.device)
    n = len(blocks)
    qh = 8
    o0 = torch.zeros((n, qh, 128), dtype=torch.float32, device=data.q.device)
    o1 = torch.zeros_like(o0)
    grid = (n,)
    _qk_page[grid](data.q, data.kv_cache, kt, o0, pages, row,
                   data.q.stride(0), data.q.stride(1),
                   data.kv_cache.stride(0), data.kv_cache.stride(2), data.kv_cache.stride(3),
                   kt.stride(0), qh, 128, 128, 0, num_warps=1, num_stages=1)
    _qk_page[grid](data.q, data.kv_cache, kt, o1, pages, row,
                   data.q.stride(0), data.q.stride(1),
                   data.kv_cache.stride(0), data.kv_cache.stride(2), data.kv_cache.stride(3),
                   kt.stride(0), qh, 128, 128, 1, num_warps=1, num_stages=1)
    torch.gcu.synchronize()
    qf = data.q[row].float().cpu()                  # [H, D]
    kf = data.kv_cache.float().cpu()
    ref = torch.stack([qf @ kf[int(p), 0, :, :128].T for p in pages.cpu()])
    d01 = diff_stats(o0, o1)
    d0r = diff_stats(o0, ref)
    d1r = diff_stats(o1, ref)
    return dict(pages=n, row=row, blocks=blocks,
                max_abs=float(max(d0r["max_abs"], d1r["max_abs"])),
                cosine=float(min(d0r["cosine"], d1r["cosine"])),
                layout_diff=d01,
                shipped_vs_fp32=d0r, pretransposed_vs_fp32=d1r,
                pass_=bool(d01["bit_identical"] and d0r["max_abs"] < 0.5
                           and d1r["max_abs"] < 0.5))



# --------------------------------------------------------------------------
# --benchmark
# --------------------------------------------------------------------------
def variant_call(bench, data, tk, out, qlen, kvh, dispatch, kt, tag, **kw):
    return dispatch(data.q, data.kv_cache, tk, data.block_table, data.cu_q, data.seq_lens,
                    data.prefix_lens, qlen, kvh, data.sm_scale, out, kt=kt,
                    use_copy=True, tag=tag, **kw)


def check_equivalence(data, tk, stock_out, cand_out, rows, qlen):
    """Full-layout equivalence vs stock: bit-exact, else a loud tolerance check."""
    d = diff_stats(cand_out, stock_out)
    sub = diff_stats(cand_out.cpu()[rows], stock_out.cpu()[rows])
    return dict(full=d, rows=sub, bit_identical=bool(d["bit_identical"]),
                pass_=bool(d["bit_identical"]
                           or (d["max_abs"] <= 2e-3 and d["cosine"] > 0.99999)))


def run_benchmark(device, env, rounds=3, out_path=None, with_extra=True,
                  sub_pages=(4, 13)):
    from benchmark import test_minimax_sparse_attention_benchmark as bench
    from flag_attn.runtime.backend._enflame.msa import sparse_attn as sa

    data, shape = build_case(bench, SHAPE_INDEX)
    batch, qlen, seq, kvh, nh = shape
    tk = score_topk(bench, data, qlen, seq, kvh)
    kt = build_pretransposed_table(data)
    torch.gcu.synchronize()
    out = {k: torch.empty_like(data.q) for k in ("stock", "parity", "xcheck", "chain")}
    log = []
    dispatch = make_dispatch(sa, log)
    res = {"mode": "benchmark", "env": env,
           "shape": dict(batch=batch, query_len=qlen, seq_len=seq, num_kv_heads=kvh,
                         num_heads=nh, head_dim=128, topk=int(tk.shape[-1]),
                         shape_index=SHAPE_INDEX, seed=SEED_OFFSET + SHAPE_INDEX),
           "rounds": rounds, "event_overhead": event_overhead()}

    # ---------------- validity gate (before any timing claim) ----------------
    call_sparse(bench, data, tk, out["stock"], qlen, kvh)
    call_sparse(bench, data, tk, out["parity"], qlen, kvh)
    with with_env("1"):
        call_sparse(bench, data, tk, out["xcheck"], qlen, kvh)
    torch.gcu.synchronize()
    rows = sorted({0, 1, 2, 3, 4, 5, 6, 7, qlen // 4, qlen // 2, qlen - 24, qlen - 1})
    ref = torch_reference(data, tk, rows, data.sm_scale, seq, nh)
    gate = {
        "stock_vs_fp32": diff_stats(out["stock"].cpu()[rows], ref),
        "stock_repeatable": diff_stats(out["stock"], out["parity"]),
        "early_v_vs_stock": diff_stats(out["xcheck"], out["stock"]),
        "early_v_vs_fp32": diff_stats(out["xcheck"].cpu()[rows], ref),
    }
    valid = (gate["stock_vs_fp32"]["max_abs"] <= 2.0
             and gate["stock_vs_fp32"]["cosine"] > 0.999
             and gate["stock_repeatable"]["bit_identical"]
             and gate["early_v_vs_stock"]["bit_identical"])
    res["gate"] = gate
    res["gate_pass"] = bool(valid)
    res["topk_digest"] = hashlib.sha256(tk.cpu().numpy().tobytes()).hexdigest()
    res["block_table_digest"] = hashlib.sha256(
        data.block_table.cpu().numpy().tobytes()).hexdigest()
    print(f"[gate] stock vs fp32 max={gate['stock_vs_fp32']['max_abs']:.3e} "
          f"cos={gate['stock_vs_fp32']['cosine']:.8f} | repeat={gate['stock_repeatable']['bit_identical']}"
          f" | early_v==stock={gate['early_v_vs_stock']['bit_identical']}")
    if not valid:
        print("[benchmark] GATE FAILED -- no timing is reported")
        res["pass"] = False
        return res

    # ---------------- equivalence of each diagnostic arm ----------------
    eq = {}
    outs = {}
    arms = [
        ("copy", dict()),
        ("copy_early_v", dict(early_v=True)),
        ("k_pre_trans", dict(k_is_t=True, ksource=1)),
        ("kview_trans", dict(k_is_t=True, ksource=3)),
        ("kview_trans_early_v", dict(k_is_t=True, ksource=3, early_v=True)),
        ("dia_a_kpre_early_v", dict(k_is_t=True, ksource=1, early_v=True)),
    ]
    for tag, kw in arms:
        o = torch.empty_like(data.q)
        variant_call(bench, data, tk, o, qlen, kvh, dispatch, kt, tag, **kw)
        torch.gcu.synchronize()
        outs[tag] = o
        eq[tag] = check_equivalence(data, tk, out["stock"], o, rows, qlen)
        print(f"[equiv:{tag:18s}] full max={eq[tag]['full']['max_abs']:.3e} "
              f"nz={eq[tag]['full']['nonzero']} exact={eq[tag]['bit_identical']} "
              f"-> {'PASS' if eq[tag]['pass_'] else 'FAIL'}")
    # ---- candidate path: prepack the K half once, then read it packed ----
    # `pack` is a pure permutation of the same K bytes (checked byte-for-byte),
    # so the only change is data movement.  Cost is paid once per cache update,
    # not per query row, hence it is timed separately and reported explicitly.
    def pack():
        return data.kv_cache[..., :128].permute(0, 1, 3, 2).contiguous()

    o = torch.empty_like(data.q)
    o2 = torch.empty_like(data.q)
    variant_call(bench, data, tk, o, qlen, kvh, dispatch, kt, "pack_plus_attn",
                 k_is_t=True, ksource=1)
    variant_call(bench, data, tk, o2, qlen, kvh, dispatch, kt, "pack_plus_attn_early_v",
                 k_is_t=True, ksource=1, early_v=True)
    torch.gcu.synchronize()
    eq["pack_plus_attn"] = check_equivalence(data, tk, out["stock"], o, rows, qlen)
    eq["pack_plus_attn_early_v"] = check_equivalence(data, tk, out["stock"], o2, rows, qlen)
    for tag in ("pack_plus_attn", "pack_plus_attn_early_v"):
        print(f"[equiv:{tag:18s}] full max={eq[tag]['full']['max_abs']:.3e} "
              f"exact={eq[tag]['bit_identical']} -> {'PASS' if eq[tag]['pass_'] else 'FAIL'}")
    # pack once, attend once (this is the real per-call shape: pack is amortised
    # over the query rows of one cache generation, so per-call it is paid once)
    for tag, ev in (("pack_once_then_attn", False), ("pack_once_then_attn_early_v", True)):
        pk = pack()
        o = torch.empty_like(data.q)
        variant_call(bench, data, tk, o, qlen, kvh, dispatch, pk, tag,
                     k_is_t=True, ksource=1, early_v=ev)
        torch.gcu.synchronize()
        eq[tag] = check_equivalence(data, tk, out["stock"], o, rows, qlen)
        print(f"[equiv:{tag:18s}] full max={eq[tag]['full']['max_abs']:.3e} "
              f"exact={eq[tag]['bit_identical']} -> "
              f"{'PASS' if eq[tag]['pass_'] else 'FAIL'}")
    res["equivalence"] = eq
    res["all_arms_equivalent"] = bool(all(v["pass_"] for v in eq.values()))
    # ablation arms are not equivalent by construction; they must stay LIVE
    # (non-zero, finite output) so nothing is dead-code-eliminated.
    live = {}
    probe_arms = [("m1_k_load_only", dict(mode=1)),
                  ("m2_k_load_trans", dict(mode=2)),
                  ("m3_qk_only", dict(mode=3)),
                  ("m4_qk_softmax", dict(mode=4)),
                  ("m5_pv_reuse_k", dict(mode=5)),
                  ("m1t_k_load_only", dict(mode=1, k_is_t=True, ksource=1)),
                  ("m3t_qk_only", dict(mode=3, k_is_t=True, ksource=1)),
                  ("m3v_qk_only", dict(mode=3, k_is_t=True, ksource=3))]
    for tag, kw in probe_arms:
        o = torch.empty_like(data.q)
        variant_call(bench, data, tk, o, qlen, kvh, dispatch, kt, tag, **kw)
        torch.gcu.synchronize()
        nz = int((o != 0).sum().item())
        fin = bool(torch.isfinite(o).all().item())
        d = diff_stats(o, out["stock"])
        live[tag] = dict(output_nonzero=nz, output_numel=int(o.numel()),
                         finite=fin, differs_from_stock=bool(d["nonzero"] > 0),
                         pass_=bool(fin and nz > 0.9 * o.numel() and d["nonzero"] > 0))
        print(f"[live:{tag:18s}] nonzero={nz}/{o.numel()} finite={fin} "
              f"differs={live[tag]['differs_from_stock']} -> "
              f"{'OK' if live[tag]['pass_'] else 'FAIL'}")
    res["ablation_live"] = live
    res["all_ablations_live"] = bool(all(v["pass_"] for v in live.values()))

    # ---------------- anchors + decomposition arms, interleaved ----------------
    fns = {}

    def stock_fn():
        call_sparse(bench, data, tk, out["stock"], qlen, kvh)
    fns["stock_attn"] = stock_fn
    fns["stock_chain"] = lambda: run_chain(bench, data, out["chain"], qlen, seq, kvh)

    def early_attn():
        with with_env("1"):
            call_sparse(bench, data, tk, out["xcheck"], qlen, kvh)

    def early_chain():
        with with_env("1"):
            run_chain(bench, data, out["chain"], qlen, seq, kvh)
    fns["early_v_attn"] = early_attn
    fns["early_v_chain"] = early_chain

    # full-chain anchors for the candidate path (same chain as the V-early one)
    fns["pack_chain"] = lambda: run_chain(bench, data, out["chain"], qlen, seq, kvh,
                                          dispatch=dispatch, kt=kt, k_is_t=True, ksource=1,
                                          pack=True)
    fns["pack_chain_early_v"] = lambda: run_chain(bench, data, out["chain"], qlen, seq, kvh,
                                                  dispatch=dispatch, kt=kt, k_is_t=True,
                                                  ksource=1, pack=True, early_v=True)
    o_diag = torch.empty_like(data.q)
    fns["copy"] = lambda: variant_call(bench, data, tk, o_diag, qlen, kvh, dispatch, kt, "copy")
    fns["k_pre"] = lambda: variant_call(bench, data, tk, o_diag, qlen, kvh, dispatch, kt,
                                        "k_pre")
    fns["k_pre_noptr"] = lambda: variant_call(bench, data, tk, o_diag, qlen, kvh, dispatch, kt,
                                              "k_pre_noptr", ksource=2)
    fns["k_pre_trans"] = lambda: variant_call(bench, data, tk, o_diag, qlen, kvh, dispatch, kt,
                                              "k_pre_trans", k_is_t=True, ksource=1)
    fns["kview_trans"] = lambda: variant_call(bench, data, tk, o_diag, qlen, kvh, dispatch, kt,
                                             "kview_trans", k_is_t=True, ksource=3)
    fns["kview_trans_early_v"] = lambda: variant_call(bench, data, tk, o_diag, qlen, kvh,
                                                      dispatch, kt, "kview_trans_early_v",
                                                      k_is_t=True, ksource=3, early_v=True)
    fns["pack_only"] = lambda: pack()
    # genuinely re-packs inside the timed region: the worst case where the
    # cache changed on every call
    fns["re_pack_every_call"] = lambda: variant_call(
        bench, data, tk, o_diag, qlen, kvh, dispatch, pack(), "re_pack_every_call",
        k_is_t=True, ksource=1)
    fns["re_pack_every_call_early_v"] = lambda: variant_call(
        bench, data, tk, o_diag, qlen, kvh, dispatch, pack(), "re_pack_every_call_early_v",
        k_is_t=True, ksource=1, early_v=True)
    fns["m3v_qk_only"] = lambda: variant_call(bench, data, tk, o_diag, qlen, kvh, dispatch,
                                              kt, "m3v_qk_only", mode=3,
                                              k_is_t=True, ksource=3)
    fns["dia_a_kpre_early_v"] = lambda: variant_call(bench, data, tk, o_diag, qlen, kvh,
                                                     dispatch, kt, "dia_a_kpre_early_v",
                                                     k_is_t=True, ksource=1, early_v=True)
    # matched ablations on the SAME compiled full kernel family: every arm keeps
    # the page loop and grid, and each writes an observable output.
    fns["m1_k_load_only"] = lambda: variant_call(bench, data, tk, o_diag, qlen, kvh, dispatch,
                                                 kt, "m1_k_load_only", mode=1)
    fns["m2_k_load_trans"] = lambda: variant_call(bench, data, tk, o_diag, qlen, kvh, dispatch,
                                                  kt, "m2_k_load_trans", mode=2)
    fns["m3_qk_only"] = lambda: variant_call(bench, data, tk, o_diag, qlen, kvh, dispatch,
                                             kt, "m3_qk_only", mode=3)
    fns["m4_qk_softmax"] = lambda: variant_call(bench, data, tk, o_diag, qlen, kvh, dispatch,
                                                kt, "m4_qk_softmax", mode=4)
    fns["m5_pv_reuse_k"] = lambda: variant_call(bench, data, tk, o_diag, qlen, kvh, dispatch,
                                                kt, "m5_pv_reuse_k", mode=5)
    # pre-transposed equivalents: same ablations with the layout fixed
    fns["m1t_k_load_only"] = lambda: variant_call(bench, data, tk, o_diag, qlen, kvh, dispatch,
                                                  kt, "m1t_k_load_only", mode=1,
                                                  k_is_t=True, ksource=1)
    fns["m3t_qk_only"] = lambda: variant_call(bench, data, tk, o_diag, qlen, kvh, dispatch,
                                              kt, "m3t_qk_only", mode=3,
                                              k_is_t=True, ksource=1)
    samples, orders = bench_samples(fns, rounds)
    res["samples"] = {k: summarize(v) for k, v in samples.items()}
    res["order"] = orders
    med = {k: v["median"] for k, v in res["samples"].items()}
    res["derived_ms"] = {
        "k_read_plus_dte": med["copy"] - med["k_pre_noptr"],
        "page_table_included_in_above": True,
        "transpose_plus_kernel_delta": med["k_pre"] - med["k_pre_trans"],
        "early_v_attn_saving": med["stock_attn"] - med["early_v_attn"],
        "early_v_chain_saving": med["stock_chain"] - med["early_v_chain"],
        "diag_a_vs_kpre_trans": med["k_pre_trans"] - med["dia_a_kpre_early_v"],
        "pack_cost": med["pack_only"],
        "k_pre_trans_saving_vs_stock": med["stock_attn"] - med["k_pre_trans"],
        "repack_every_call_saving_vs_stock": med["stock_attn"] - med["re_pack_every_call"],
        "repack_every_call_early_v_vs_stock": med["stock_attn"]
                                              - med["re_pack_every_call_early_v"],
        "amortised_pack_overhead_per_call_q1024": med["pack_only"],
        "pack_payback_calls": med["pack_only"] / max(med["stock_attn"] - med["k_pre_trans"], 1e-9),
        "pack_chain_saving_vs_stock_chain": med["stock_chain"] - med["pack_chain"],
        "pack_chain_early_v_vs_stock_chain": med["stock_chain"] - med["pack_chain_early_v"],
        "pack_chain_early_v_vs_early_v_chain": med["early_v_chain"] - med["pack_chain_early_v"],
    }

    # ---------------- page-count sub-chains (marginal cost, same shape) ----------------
    if with_extra:
        sub = {}
        sub_fns = {}
        # 13 pages, then 13-9 = 4 pages: same page order, same shape, so the
        # 4<->13 slope is a marginal per-page cost on the *same* compiled kernel.
        n_hi, n_lo = sub_pages[1], sub_pages[1] - 9
        for npages, drop in ((n_hi, 0), (n_lo, 9)):
            arm_kw = dict(copy=dict(), k_pre=dict(),
                          k_pre_trans=dict(k_is_t=True, ksource=1),
                          k_pre_noptr=dict(ksource=2))
            for arm, extra in arm_kw.items():
                def mk(arm=arm, extra=extra, npages=npages, drop=drop):
                    def run():
                        return variant_call(bench, data, tk, o_diag, qlen, kvh, dispatch, kt,
                                            f"{arm}_{npages}p", page_sub=(0, drop), **extra)
                    return run
                sub_fns[f"{arm}_{npages}p"] = mk()
        s2, o2 = bench_samples(sub_fns, rounds)
        res["subchain"] = {k: summarize(v) for k, v in s2.items()}
        res["subchain_order"] = o2
        res["subchain_pages"] = list(sub_pages)
        m = {k: v["median"] for k, v in res["subchain"].items()}
        res["marginal_ms_per_page"] = {}
        for arm in ("copy", "k_pre", "k_pre_trans", "k_pre_noptr"):
            res["marginal_ms_per_page"][arm] = (
                m[f"{arm}_{n_hi}p"] - m[f"{arm}_{n_lo}p"]) / 9.0

    # ---------------- micro kernels ----------------
    if with_extra:
        try:
            res["micro"] = run_micro(data, tk, kt, qlen, rounds=rounds)
        except Exception as exc:
            res["micro"] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[micro] FAILED: {res['micro']['error']}")
        try:
            res["reuse"] = run_reuse(data, tk, kt, qlen, rounds=rounds)
        except Exception as exc:
            res["reuse"] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[reuse] FAILED: {res['reuse']['error']}")
        res["reuse_equivalence"] = reuse_equivalence(data, tk, qlen)

    # ---------------- launch + IR for every variant ----------------
    ir_list = []
    for tag, kw in (("full_copy", dict()), ("k_pre", dict()),
                    ("k_pre_noptr", dict(ksource=2)),
                    ("k_pre_trans", dict(k_is_t=True, ksource=1)),
                    ("kview_trans", dict(k_is_t=True, ksource=3)),
                    ("dia_a_kpre_early_v", dict(k_is_t=True, ksource=1, early_v=True)),
                    ("dia_c_earlyv_ship", dict(early_v=True))):
        r = variant_call(bench, data, tk, o_diag, qlen, kvh, dispatch, kt, tag, **kw)
        torch.gcu.synchronize()
        ir_list.append(ir_census(r["compiled"], tag))
        ir_list[-1]["launch"] = r["launch"]
    stock_log = []
    stock_dispatch = make_dispatch(sa, stock_log)
    r = stock_dispatch(data.q, data.kv_cache, tk, data.block_table, data.cu_q, data.seq_lens,
                       data.prefix_lens, qlen, kvh, data.sm_scale, o_diag,
                       use_copy=False, tag="stock")
    torch.gcu.synchronize()
    ir_list.append(ir_census(r["compiled"], "stock"))
    ir_list[-1]["launch"] = r["launch"]
    res["ir"] = ir_list
    res["ir_brief"] = [ir_brief(x) for x in ir_list]
    res["launch_common_prefix"] = log[:1]
    for b in res["ir_brief"]:
        print("[ir]", json.dumps(b, sort_keys=True))

    # ---------------- optional profiler / counters ----------------
    if with_extra:
        res["counters"] = counter_probe()
        res["profiler"] = profiler_probe(bench, data, tk, out["stock"], qlen, kvh)

    res["pass"] = bool(res["gate_pass"] and res["all_arms_equivalent"])
    for name, s in res["samples"].items():
        print(f"[bench] {name:22s} n={s['n']} median={s['median']:.4f} min={s['min']:.4f} "
              f"max={s['max']:.4f} stdev={s['stdev']:.4f}")
    for k, v in res["derived_ms"].items():
        if isinstance(v, float):
            print(f"[derived] {k:28s} {v:+.4f} ms")
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"[bench] wrote {out_path}")
    return res


# --------------------------------------------------------------------------
# diagnostics that must never contaminate the timed path
# --------------------------------------------------------------------------
def counter_probe():
    """efsmi dmon, with a known-positive matrix control, interleaved with MSA."""
    out = {"gpm_metrics": "2,3,18,19", "samples": {}, "positive_control": None,
           "available": False}
    try:
        before = sh("efsmi dmon -i 5 --gpm-metrics 2,3,18,19 -w 0.2 -c 3")
        a = torch.randn(4096, 4096, device="gcu", dtype=torch.bfloat16)
        b = torch.randn(4096, 4096, device="gcu", dtype=torch.bfloat16)
        holder = {}

        def sampler():
            holder["txt"] = sh("efsmi dmon -i 5 --gpm-metrics 2,3,18,19 -w 0.2 -c 8")
        t = threading.Thread(target=sampler)
        t.start()
        time.sleep(0.1)
        for _ in range(40):
            torch.matmul(a, b)
        torch.gcu.synchronize()
        t.join()
        txt = holder.get("txt", "")
        vals = []
        for line in txt.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "5":
                try:
                    vals.extend(float(x) for x in parts[1:5])
                except ValueError:
                    pass
        out["idle_raw"] = before[-400:]
        out["positive_control"] = {"raw_tail": txt[-600:], "nonzero": sum(1 for v in vals if v),
                                   "n_values": len(vals), "max": max(vals) if vals else None}
        out["available"] = bool(out["positive_control"]["nonzero"])
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    print(f"[counters] available={out['available']} "
          f"positive_control={out.get('positive_control', {}).get('nonzero')}")
    return out


def profiler_probe(bench, data, tk, out, qlen, kvh):
    """torch.profiler PrivateUse1 activity: does it yield kernel events?"""
    rec = {"activities": [], "kernel_events": 0, "available": False}
    try:
        acts = [torch.profiler.ProfilerActivity.CPU]
        pu = getattr(torch.profiler.ProfilerActivity, "PrivateUse1", None)
        if pu is not None:
            acts.append(pu)
        rec["activities"] = [str(a) for a in acts]
        from torch.profiler import profile
        with profile(activities=acts) as prof:
            call_sparse(bench, data, tk, out, qlen, kvh)
            torch.gcu.synchronize()
        evts = prof.key_averages()
        rows = []
        for e in evts:
            rows.append({"key": e.key, "count": e.count,
                         "cuda_time": getattr(e, "self_device_time_total", None),
                         "cpu_time": e.self_cpu_time_total})
        rec["events"] = rows[:40]
        rec["kernel_events"] = sum(1 for r in rows if r["count"] and "gqa" in r["key"])
        rec["available"] = rec["kernel_events"] > 0
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    print(f"[profiler] available={rec['available']} kernel_events={rec['kernel_events']}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=5)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", type=str, default=None,
                    help="write JSON here (default: stdout only)")
    ap.add_argument("--parity", action="store_true",
                    help="deprecated alias kept off the public path")
    args = ap.parse_args()
    if not (args.verify or args.benchmark):
        ap.error("choose --verify and/or --benchmark")
    occ = gate(args.device)
    torch.gcu.set_device(5)
    if torch.gcu.current_device() != 5:
        print("FATAL: could not select device 5")
        sys.exit(4)
    env = env_record(5)
    env["occupancy_gate"] = {k: v for k, v in occ.items() if k != "pmon_raw"}
    print(f"[env] pci={env['pci']} affinity={env['cpu_affinity']} "
          f"triton={env['triton']} torch={env['torch']}")
    as_json = {}
    if args.verify:
        as_json["verify"] = run_verify(args.device, env)
    if args.benchmark:
        as_json["benchmark"] = run_benchmark(args.device, env, rounds=args.rounds,
                                             out_path=args.out)
    as_json["env"] = env
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(as_json, fh, indent=1)
        print(f"[main] wrote {args.out}")
    else:
        print("MSA010_JSON_BEGIN")
        print(json.dumps(as_json, indent=1))
        print("MSA010_JSON_END")
    ok = True
    if "verify" in as_json:
        ok &= bool(as_json["verify"].get("pass"))
    if "benchmark" in as_json:
        ok &= bool(as_json["benchmark"].get("pass"))
    print(f"[main] overall={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1



def reuse_equivalence(data, tk, qlen, n_prog=256):
    """REUSE 1 (hot address, 13 DMAs) vs REUSE 2 (one local tile, 1 DMA).

    Both consume the SAME page with the SAME number of QK convgens in the same
    order, so the only difference is where the K tile comes from.  Bit-identical
    output therefore proves the measured gap is the 12 removed DMAs and not a
    different computation.
    """
    dev = data.q.device
    row = torch.arange(n_prog, dtype=torch.int32, device=dev) % qlen
    bt = data.block_table[0]
    blocks = [int(x) for x in tk[0, 0].tolist() if int(x) >= 0][:4]
    pages = torch.tensor([int(bt[b]) for b in blocks], dtype=torch.int32, device=dev)
    outs = {}
    for reuse in (1, 2):
        o = torch.empty((n_prog, 8, 128), dtype=torch.float32, device=dev)
        _reuse_micro[(n_prog,)](
            data.q, data.kv_cache, tk, pages, o, 0,
            data.q.stride(0), data.q.stride(1),
            data.kv_cache.stride(0), data.kv_cache.stride(2), data.kv_cache.stride(3),
            tk.stride(1), tk.stride(2), 13, 8, 128, 128, reuse,
            num_warps=1, num_stages=1)
        torch.gcu.synchronize()
        outs[reuse] = o
    d = diff_stats(outs[2], outs[1])
    rec = {"pages": pages.tolist(), "blocks": blocks, "programs": n_prog,
           "hot_vs_local": d, "pass": bool(d["max_abs"] <= 1e-5),
           "bit_identical": bool(d["bit_identical"]),
           "interpretation": "identical math; only the K source differs"}
    print(f"[reuse-equiv] hot_address vs local_tile max={d['max_abs']:.3e} "
          f"nz={d['nonzero']} exact={d['bit_identical']} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}")
    return rec



if __name__ == "__main__":
    sys.exit(main())

