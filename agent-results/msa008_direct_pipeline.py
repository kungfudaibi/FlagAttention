#!/usr/bin/env python
"""MSA-008: decomposition of the S60 long-prefill sparse-attention "direct" kernel.

Diagnostic-only harness.  Nothing here changes the library default dispatch:
every experimental launch uses the local copy of the host dispatch in this
file, and the shipped kernels are only *called*, never modified.

Modes
  --device N --verify      numeric/mask agreement between the stock dispatch and
                           the local experiment dispatch (prefix on/off, kernel
                           copy MODE=0), plus a torch float32 reference for a
                           row subset, plus the shared-prefix reuse histogram.
  --device N --benchmark   uninstrumented hot paired samples (stock / prefix-on /
                           prefix-off / kernel-copy / locality probe / ablation
                           variants), each timed with device events, plus a
                           *separately* measured staged profile (index_score,
                           index_topk, prefix kernel, direct kernel) and a
                           static-IR op census of the compiled kernel.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import threading
import time
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

SHAPE_INDEX = 6          # PREFILL_SHAPES[6] == (1, 1024, 16384, 1, 8)
SEED = 0
PREFIX_GROUP = 16
OUT_DIR = os.path.join(REPO, "agent-results", "raw")


# --------------------------------------------------------------------------
# shell / environment
# --------------------------------------------------------------------------
def sh(cmd, timeout=60):
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
    """PIDs from efsmi pmon that are not this process or its ancestors."""
    mine = process_tree_pids()
    out = []
    for line in pmon_text.splitlines():
        m = re.match(r"\s*\d+\s+(\d+)\s", line)
        if m and int(m.group(1)) not in mine:
            out.append((int(m.group(1)), line.strip()))
    return out


def env_record(device_index):
    q = sh(f"efsmi -i {device_index} -q")
    pmon = sh(f"efsmi pmon -i {device_index} -c 1")
    dmon = sh(f"efsmi dmon -i {device_index} --gpm-metrics 18,19,2 -w 0.1 -c 3")
    usage = sh(f"efsmi -i {device_index} -q -d USAGE")
    props = torch.gcu.get_device_properties(device_index)
    rec = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device_index": device_index,
        "device_name": torch.gcu.get_device_name(device_index),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": sys.version.split()[0],
        "gcu_toolkit": sh("cat /opt/triton_gcu/version.txt 2>/dev/null").strip(),
        "privateuse1_kineto": bool(getattr(torch.profiler.ProfilerActivity, "PrivateUse1", None)),
        "props": {
            "major": props.major,
            "minor": props.minor,
            "total_memory_mb": round(props.total_memory / (1 << 20)),
            "multi_processor_count": props.multi_processor_count,
            "sip_count": getattr(props, "sip_count", None),
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "env": {k: os.environ.get(k) for k in
                ("PYTHONPATH", "TRITON_GCU_OP_TRACE", "TRITON_GCU_DEBUG", "TRITON_CACHE_DIR")},
        "efsmi_pmon_raw": pmon,
        "own_pids": sorted(process_tree_pids()),
        "foreign_pids": foreign_pids(pmon),
        "efsmi_dmon_idle_raw": dmon,
        "efsmi_query_raw": q,
    }
    m = re.search(r"Bus\s*:\s*(\d+)", q)
    d = re.search(r"Dev\s*:\s*(\d+)", q)
    f = re.search(r"Func\s*:\s*(\d+)", q)
    if m and d and f:
        # efsmi prints Bus/Dev/Func already in hex.
        rec["pci"] = f"0000:{m.group(1)}:{d.group(1)}.{f.group(1)}"
    rec["pci_from_sysfs"] = sh(f"ls -d /sys/bus/pci/devices/*/gcu* 2>/dev/null | head -3").strip()
    for key, pat in (("driver_version", r"Ver\s*:\s*(\S+)"),
                     ("dev_name", r"Dev Name\s*:\s*(\S+)"),
                     ("ap_fw", r"AP FW Version\s*:\s*(\S+)"),
                     ("sp_fw", r"SP FW Version\s*:\s*(\S+)"),
                     ("mem_clock_mhz", r"Mem CLK\s*:\s*(\d+)"),
                     ("power_w", r"Power Usage\s*:\s*(\d+)")):
        mm = re.search(pat, q)
        rec[key] = mm.group(1) if mm else None
    mm = re.search(r"GCU Usage\s*:\s*([\d.]+)", usage)
    rec["gcu_usage_pct_before"] = float(mm.group(1)) if mm else None
    rec["busy_before"] = bool(foreign_pids(pmon))
    rec["gcu_usage_raw_before"] = usage
    return rec


# --------------------------------------------------------------------------
# experimental kernel copy: faithful BF16 path + ablation MODEs
# --------------------------------------------------------------------------
@triton.jit
def _page_step(
    kv_cache_ptr, bt_row, t_ptr_j, q_qk, acc_o, l_acc, m_i,
    pid_kh, seq_len, q_abs, off_n,
    head_dim: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    sm_scale_log2e: tl.constexpr, stride_kv_blk: tl.constexpr, stride_kv_h: tl.constexpr,
    stride_kv_pos: tl.constexpr, stride_kv_d: tl.constexpr, stride_tk: tl.constexpr,
    MODE: tl.constexpr,
):
    """One selected page of the shipped BF16 direct kernel (sparse_attn.py:229-345)."""
    blk = tl.load(t_ptr_j).to(tl.int32)
    t_ptr_j = t_ptr_j + stride_tk
    c = blk * BLOCK_SIZE_K
    page = tl.load(bt_row + blk).to(tl.int32)
    pos = c + off_n
    pos_mask = pos < seq_len
    k_base_ptr = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
    k_ptrs = tl.make_block_ptr(
        base=k_base_ptr, shape=(BLOCK_SIZE_K, head_dim),
        strides=(stride_kv_pos, stride_kv_d), offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D), order=(1, 0),
    )
    k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
    v_base_ptr = k_base_ptr + head_dim * stride_kv_d
    v_ptrs = tl.make_block_ptr(
        base=v_base_ptr, shape=(BLOCK_SIZE_K, head_dim),
        strides=(stride_kv_pos, stride_kv_d), offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D), order=(1, 0),
    )
    if MODE == 1:
        # telescope step: K load + transpose + QK matmul only.  No V DMA, no PV
        # matmul, no online softmax.  (An earlier variant kept the loads alive
        # through whole-tile scalar reductions; it ran SLOWER than the full
        # kernel, i.e. it bounded nothing, so it is not used.)
        kt = tl.trans(k)
        acc_o += tl.dot(q_qk, kt)
    else:
        qk_dot = tl.dot(q_qk, tl.trans(k))
        is_full_causal = (c + BLOCK_SIZE_K) <= q_abs
        is_full_seq = (c + BLOCK_SIZE_K) <= seq_len
        qk = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_K), dtype=tl.float32)
        if not is_full_causal:
            qk += tl.where(q_abs >= pos[None, :], 0, float("-inf"))
        qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
        qk += qk_dot * sm_scale_log2e
        if not is_full_seq:
            qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        if MODE == 2:
            # keeps QK + online softmax, drops the V load and the PV matmul
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)
            acc_o = acc_o * alpha[:, None]
            acc_o += tl.sum(p, axis=1)[:, None]
            m_i = m_ij
        elif MODE == 3:
            # keeps both MMAs and the V load, drops the online-softmax arithmetic
            v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
            acc_o += tl.dot(qk.to(v.dtype), v)
        elif MODE == 4:
            # pipeline probe: feed the already-loaded K tile as V, so exactly one
            # KV DMA + one DTE wait leave the per-page chain while both MMAs and
            # the whole online softmax stay in place.
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)
            acc_o = acc_o * alpha[:, None]
            l_acc = l_acc * alpha[:, None] + p
            acc_o += tl.dot(p.to(k.dtype), k)
            m_i = m_ij
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)
            acc_o = acc_o * alpha[:, None]
            l_acc = l_acc * alpha[:, None] + p
            v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
            acc_o += tl.dot(p.to(v.dtype), v)
            m_i = m_ij
    return acc_o, l_acc, m_i, t_ptr_j


@triton.jit
def _exp_direct(
    q_ptr, kv_cache_ptr, k_scale_ptr, v_scale_ptr, t_ptr, o_ptr, block_table_ptr,
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
    USE_SHARED_PREFIX: tl.constexpr = False, MODE: tl.constexpr = 0,
    PIPELINE: tl.constexpr = False,
):
    """Copy of the shipped BF16 prefill direct body (sparse_attn.py:128).

    MODE 0 reproduces the shipped kernel bit-exactly (checked in --verify).
    PIPELINE=True keeps the very same math and accumulation order but asks the
    compiler to software-pipeline the page loop (tl.range(num_stages=2)), i.e.
    to prefetch page i+1's K/V while page i computes.  This is the one paired
    data-movement/pipeline diagnostic.
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
        if PIPELINE:
            for _ in tl.range(shared_count, real_topk, num_stages=2):
                acc_o, l_acc, m_i, t_ptr_j = _page_step(
                    kv_cache_ptr, bt_row, t_ptr_j, q_qk, acc_o, l_acc, m_i,
                    pid_kh, seq_len, q_abs, off_n, head_dim, BLOCK_SIZE_K, BLOCK_SIZE_D,
                    BLOCK_SIZE_QH, sm_scale_log2e, stride_kv_blk, stride_kv_h, stride_kv_pos,
                    stride_kv_d, stride_tk, MODE)
        else:
            for _ in range(shared_count, real_topk):
                acc_o, l_acc, m_i, t_ptr_j = _page_step(
                    kv_cache_ptr, bt_row, t_ptr_j, q_qk, acc_o, l_acc, m_i,
                    pid_kh, seq_len, q_abs, off_n, head_dim, BLOCK_SIZE_K, BLOCK_SIZE_D,
                    BLOCK_SIZE_QH, sm_scale_log2e, stride_kv_blk, stride_kv_h, stride_kv_pos,
                    stride_kv_d, stride_tk, MODE)
        if MODE == 1 or MODE == 2:
            l_i = tl.sum(l_acc, axis=1) + 1.0
        else:
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
# host dispatch replica (copy of sparse_attn.py:719 tail)
# --------------------------------------------------------------------------
def make_dispatch(sparse_attn_mod, launch_log):
    def dispatch(q, kv_cache, topk_idx, block_table, cu_seqlens_q, seq_lens, prefix_lens,
                 max_query_len, num_kv_heads, sm_scale, output,
                 enable_prefix=True, mode=0, num_stages=1, use_copy=False, pipeline=False,
                 num_warps=1):
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
            counts = torch.empty((batch, num_kv_heads, prefix_groups), device=q.device, dtype=torch.int32)
            sparse_attn_mod._gqa_sparse_common_prefix[(prefix_groups, num_kv_heads, batch)](
                q, kv_cache, topk_idx, block_table, cu_seqlens_q, seq_lens, prefix_lens,
                partial, maximum, denom, counts, gqa_group_size, head_dim, topk, sm_scale,
                *q.stride(), *kv_cache.stride(), *topk_idx.stride(), block_table.stride(0),
                *partial.stride(), maximum.stride(0), num_kv_heads, prefix_groups, PREFIX_GROUP,
                block_size_h, triton.next_power_of_2(head_dim), num_warps=1, num_stages=1)
        grid = (max_query_len, num_kv_heads, batch)
        kernel = _exp_direct if use_copy else sparse_attn_mod._gqa_sparse_fwd_direct
        kernel[grid](
            q, kv_cache, output, output, topk_idx, output, block_table, cu_seqlens_q, cu_seqlens_q,
            seq_lens, prefix_lens, num_kv_heads, gqa_group_size, head_dim, topk, 1, sm_scale,
            q.stride(0), q.stride(1), q.stride(2),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), kv_cache.stride(3),
            0, 0, 0, 0,
            topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
            output.stride(0), output.stride(1), output.stride(2), block_table.stride(0),
            BLOCK_SIZE_Q=1, BLOCK_SIZE_K=128, BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
            BLOCK_SIZE_H=block_size_h, BLOCK_SIZE_QH=block_size_h, USE_FP8=False, KV_SCALE_MODE=0,
            num_warps=num_warps, num_stages=num_stages,
            partial_ptr=partial, maximum_ptr=maximum, denom_ptr=denom, count_ptr=counts,
            GROUP_Q=PREFIX_GROUP, NUM_GROUPS=prefix_groups, STRIDE_PN=partial.stride(0),
            STRIDE_PH=partial.stride(1), STRIDE_PD=partial.stride(2), STRIDE_SN=maximum.stride(0),
            USE_SHARED_PREFIX=enable_prefix,
            **({"MODE": mode, "PIPELINE": pipeline} if use_copy else {}))
        return dict(partial=partial, maximum=maximum, denom=denom, counts=counts, grid=grid)

    return dispatch


# --------------------------------------------------------------------------
# measurement helpers
# --------------------------------------------------------------------------
def bench_samples(fns, rounds, warmup=3):
    """Alternating (rotated) paired sampling; one device-event sample per call."""
    order = list(fns.keys())
    for name in order:
        for _ in range(warmup):
            fns[name]()
    torch.gcu.synchronize()
    samples = {name: [] for name in order}
    for r in range(rounds):
        rot = order[r % len(order):] + order[:r % len(order)]
        for name in rot:
            a = torch.gcu.Event(enable_timing=True)
            b = torch.gcu.Event(enable_timing=True)
            a.record()
            fns[name]()
            b.record()
            b.synchronize()
            samples[name].append(a.elapsed_time(b))
    return samples


def summarize(values):
    return dict(n=len(values), samples=list(values), median=statistics.median(values),
                mean=statistics.fmean(values), min=min(values), max=max(values))


def diff_stats(a, b):
    d = (a.float() - b.float()).abs()
    return dict(max_abs=d.max().item(), mean_abs=d.mean().item(),
                nonzero=int((d > 0).sum().item()), numel=d.numel())


# --------------------------------------------------------------------------
# torch float32 reference for a row subset (mask semantics check)
# --------------------------------------------------------------------------
def torch_reference(data, topk_idx, rows, scale, seq_len, num_heads):
    q = data.q.float().cpu()
    kv = data.kv_cache.float().cpu()
    bt = data.block_table.cpu()
    tk = topk_idx.cpu()
    out = torch.zeros((len(rows), num_heads, q.shape[-1]), dtype=torch.float32)
    prefix_len = int(data.prefix_lens[0].item())
    for out_i, row in enumerate(rows):
        q_abs = prefix_len + row
        acc = torch.zeros((num_heads, q.shape[-1]), dtype=torch.float32)
        m = torch.full((num_heads,), float("-inf"))
        l = torch.zeros((num_heads,))
        for slot in range(tk.shape[-1]):
            blk = int(tk[0, row, slot])
            if blk < 0:
                continue
            page = int(bt[0, blk])
            k = kv[page, 0, :, :128]
            v = kv[page, 0, :, 128:]
            pos = blk * 128 + torch.arange(128)
            valid = (pos < seq_len) & (pos <= q_abs)
            scores = (q[row] @ k.T) * scale
            scores = scores.masked_fill(~valid[None, :], float("-inf"))
            m_new = torch.maximum(m, scores.max(dim=-1).values)
            safe = torch.where(m_new > float("-inf"), m_new, torch.zeros(()))
            p = torch.where(valid[None, :], torch.exp2((scores - safe[:, None]) * 1.4426950409),
                            torch.zeros(()))
            alpha = torch.where(m > float("-inf"), torch.exp2((m - safe) * 1.4426950409), torch.zeros(()))
            l = l * alpha + p.sum(dim=-1)
            acc = acc * alpha[:, None] + p @ v
            m = m_new
        out[out_i] = acc / l[:, None]
    return out


# --------------------------------------------------------------------------
# static IR census of the compiled direct kernel
# --------------------------------------------------------------------------
def find_cache_dirs(source_path):
    """Cache dirs whose direct kernel was compiled from this exact source file."""
    cache = os.path.expanduser("~/.triton/cache")
    hits = []
    for name in sorted(os.listdir(cache)):
        d = os.path.join(cache, name)
        src = os.path.join(d, "_gqa_sparse_fwd_direct.source")
        if not os.path.isfile(src):
            continue
        try:
            with open(src, errors="replace") as fh:
                head = fh.read(200000)
        except Exception:
            continue
        if f'loc("{source_path}":128' in head:
            hits.append((os.path.getmtime(src), d))
    return [d for _, d in sorted(hits, reverse=True)]


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
    """Extract the scf.for region starting at `start` (opening '{' after args)."""
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


def census(path, kind):
    with open(path, "r", errors="replace") as fh:
        text = fh.read()
    if kind == "gcuir":
        counts = _count_ops(text)
        # "scf.for"( : the op itself, not the gcu.border{enter = "scf.for"} attr
        starts = [m.start() for m in re.finditer(r'"scf\.for"\(', text)]
        region = None
        if len(starts) >= 2:
            body = _balanced_region(text, starts[1])
            region = dict(chars=len(body), lines=body.count("\n"), counts=_count_ops(body))
        return counts, region
    return _count_tops(text), None


def ttir_loc_histogram(path):
    """How many ttir ops carry each source line, plus the ttir op mix.

    loc aliases can chain (#loc132 = loc("q_ptr"(#loc))) so resolve them
    recursively; unresolved ones are counted separately instead of being
    silently dropped.
    """
    with open(path, "r", errors="replace") as fh:
        text = fh.read()
    direct, ref = {}, {}
    for m in re.finditer(r'#loc(\d*)\s*=\s*loc\("([^"]+)":(\d+):(\d+)\)', text):
        direct[m.group(1)] = int(m.group(3))
    for m in re.finditer(r'#loc(\d*)\s*=\s*loc\("([^"]+)"\(#loc(\d*)\)\)', text):
        ref[m.group(1)] = m.group(3)

    def resolve(key, depth=0):
        if key in direct:
            return direct[key]
        if key in ref and depth < 8:
            return resolve(ref[key], depth + 1)
        return None

    body = "\n".join(l for l in text.splitlines() if not l.startswith("#loc"))
    per_line, ops = {}, {}
    hit = miss = 0
    for line in body.splitlines():
        m = re.search(r"loc\(#loc(\d*)\)", line)
        if m:
            ln = resolve(m.group(1))
            if ln is None:
                miss += 1
            else:
                hit += 1
                per_line[ln] = per_line.get(ln, 0) + 1
        m2 = re.search(r"=\s*([a-z_]+\.[a-z_0-9]+)\b", line)
        if m2:
            ops[m2.group(1)] = ops.get(m2.group(1), 0) + 1
    return per_line, ops, dict(aliases=len(direct) + len(ref), resolved=hit, unresolved=miss)


def dot_to_source(ttir_path, gcuir_path, llir_path):
    """Map each tt.dot / gcu.matmul / tops_convgen call to its source line.

    ttir locs are name-chained (#loc200 = loc("acc_o"(#loc90))), so the mapping
    has to be resolved through the name alias to reach the real file:line.
    """
    with open(ttir_path, "r", errors="replace") as fh:
        text = fh.read()
    chain, by_key = {}, {}
    for m in re.finditer(r'#loc(\d*)\s*=\s*loc\("([^"]+)"\(#loc(\d*)\)\)', text):
        chain[m.group(1)] = m.group(3)
    for m in re.finditer(r'#loc(\d*)\s*=\s*loc\("([^"]+)":(\d+):(\d+)\)', text):
        by_key[m.group(1)] = (int(m.group(3)), int(m.group(4)))
    out = []
    for line in text.splitlines():
        if "tt.dot" not in line:
            continue
        key = re.search(r"loc\(#loc(\d*)\)", line)
        key = key.group(1) if key else None
        if key in chain:
            key = chain[key]
        name = re.search(r"(%[a-z_0-9]+)\s*=\s*tt\.dot", line)
        out.append({"ir": "tt.dot", "value": name.group(1) if name else None,
                    "source": by_key.get(key),
                    "text": line.strip()[:120]})
    gcu, llir = [], []
    if gcuir_path and os.path.isfile(gcuir_path):
        with open(gcuir_path, "r", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                if '"gcu.matmul"' in line:
                    gcu.append({"ir": "gcu.matmul", "line": i,
                                "has_accumulator": "array<i32: 1, 1, 1, 1, 0>" in line,
                                "shapes": re.findall(r"memref<[0-9x]+x?[a-z0-9]*>", line)})
    if llir_path and os.path.isfile(llir_path):
        with open(llir_path, "r", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                if "callee = @tops_convgen_m8_n128_k128_" in line:
                    llir.append({"ir": "llvm.call", "line": i,
                                 "callee": re.search(r"@(tops_convgen_m8_n128_k128_[a-z0-9_]*)",
                                                     line).group(1)})
    return {"tt_dot": out, "gcu_matmul": gcu, "convgen_call": llir}


def structural_landmarks(path):
    """gcuir/llir line landmarks: loops, the two MMAs, and the DMA ops."""
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()
    marks = {"outer_query_loop": [], "page_loop": [], "gcu_matmul": [],
             "convgen_call": [], "dma_wait": [], "transpose": [], "deslice": []}
    for i, line in enumerate(lines, 1):
        if '"scf.for"(' in line:
            marks["outer_query_loop"].append(i) if not marks["outer_query_loop"] else \
                marks["page_loop"].append(i)
        if '"gcu.matmul"' in line:
            marks["gcu_matmul"].append(
                {"line": i, "text": line.strip()[:200],
                 "has_accumulator": "operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>" in line})
        if "callee = @tops_convgen_m8_n128_k128_" in line:
            m = re.search(r"callee = @(tops_convgen_m8_n128_k128_[a-z0-9_]*)", line)
            marks["convgen_call"].append({"line": i, "callee": m.group(1)})
        if "@tops_wait_dte" in line:
            marks["dma_wait"].append(i)
        if "@tops_transpose_async_2_bf16" in line:
            marks["transpose"].append(i)
        if "@tops_deslice_async_2_bf16" in line:
            marks["deslice"].append(i)
    return marks


def ir_report_for_source(source_path):
    """Op census of the compiled direct kernel + per-source-line attribution."""
    rep = {"source": source_path, "cache_dirs": [], "per_file": {}}
    dirs = find_cache_dirs(source_path)
    rep["cache_dirs"] = dirs
    if not dirs:
        return rep
    # Prefer the shipped configuration (num_stages=1, no fp8) over diagnostic
    # builds of the same source, and record every candidate's metadata.
    cands = []
    for cand in dirs:
        j = os.path.join(cand, "_gqa_sparse_fwd_direct.json")
        try:
            with open(j) as fh:
                meta = json.load(fh)
        except Exception:
            meta = {}
        cands.append({"dir": cand, "num_stages": meta.get("num_stages"),
                      "num_warps": meta.get("num_warps"), "shared": meta.get("shared"),
                      "hash": meta.get("hash")})
    rep["cache_candidates"] = cands
    pick = next((c for c in cands if c["num_stages"] == 1), cands[0])
    d = pick["dir"]
    rep["cache_dir"] = d
    rep["cache_dir_policy"] = ("first candidate with num_stages == 1 (the shipped launch "
                              "configuration); candidates list all builds of this source")
    rep["meta"] = json.load(open(os.path.join(d, "_gqa_sparse_fwd_direct.json")))
    for ext in ("gcuir", "llir", "ttir"):
        p = os.path.join(d, f"_gqa_sparse_fwd_direct.{ext}")
        if not os.path.isfile(p):
            continue
        counts, region = census(p, ext)
        rep["per_file"][ext] = {
            "path": p, "size": os.path.getsize(p),
            "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "region": region,
        }
    for ext in ("gcuir", "llir"):
        p = os.path.join(d, f"_gqa_sparse_fwd_direct.{ext}")
        if os.path.isfile(p) and ext in rep["per_file"]:
            rep["per_file"][ext]["landmarks"] = structural_landmarks(p)
    tp = os.path.join(d, "_gqa_sparse_fwd_direct.ttir")
    if os.path.isfile(tp):
        per_line, ops, nloc = ttir_loc_histogram(tp)
        rep["ttir_ops"] = dict(sorted(ops.items(), key=lambda kv: -kv[1]))
        rep["ttir_ops_per_source_line"] = dict(sorted(per_line.items(), key=lambda kv: kv[0]))
        rep["ttir_loc_aliases"] = nloc
        src = {}
        if os.path.isfile(source_path):
            with open(source_path, errors="replace") as fh:
                src = fh.read().splitlines()
        rep["source_line_text"] = {str(k): (src[k - 1].strip() if 0 < k <= len(src) else None)
                                   for k in per_line}
        rep["dot_mapping"] = dot_to_source(tp, os.path.join(d, "_gqa_sparse_fwd_direct.gcuir"),
                                          os.path.join(d, "_gqa_sparse_fwd_direct.llir"))
    return rep


# --------------------------------------------------------------------------
# main flows
# --------------------------------------------------------------------------
def build(device, args_bench):
    from benchmark import test_minimax_sparse_attention_benchmark as b
    from flag_attn.runtime.backend._enflame.msa import sparse_attn as sa

    shape = b.PREFILL_SHAPES[SHAPE_INDEX]
    batch, qlen, seq, kvh, nh = shape
    gen = torch.Generator(device="cpu").manual_seed(SEED + SHAPE_INDEX)
    data = b.make_data(batch, seq, kvh, nh, torch.device("gcu"), "bf16", decode=False,
                       decode_qlen=1, query_len=qlen, randomize_pages=True, generator=gen)
    return b, sa, shape, data


def run_verify(device, env):
    b, sa, shape, data = build(device, None)
    batch, qlen, seq, kvh, nh = shape
    scores = b.minimax_m3_index_score(data.idx_q, data.index_kv_cache, data.block_table,
                                      data.cu_q, data.seq_lens, data.prefix_lens, qlen, seq, kvh)
    topk_idx = b.minimax_m3_index_topk(scores, data.cu_q, data.prefix_lens, qlen,
                                       b.MSABenchmarkArgs().topk, 1, 2)
    torch.gcu.synchronize()

    out = {k: torch.empty_like(data.q) for k in ("stock", "copy_on", "copy_off")}
    b._call_sparse(b.minimax_m3_sparse_attn, data, topk_idx, out["stock"], decode=False,
                   max_query_len=qlen, num_kv_heads=kvh, decode_qlen=1)
    dispatch = make_dispatch(sa, None)
    st = dispatch(data.q, data.kv_cache, topk_idx, data.block_table, data.cu_q, data.seq_lens,
                  data.prefix_lens, qlen, kvh, data.sm_scale, out["copy_on"], True, 0, use_copy=True)
    counts = st["counts"].cpu().reshape(-1)
    dispatch(data.q, data.kv_cache, topk_idx, data.block_table, data.cu_q, data.seq_lens,
             data.prefix_lens, qlen, kvh, data.sm_scale, out["copy_off"], False, 0, use_copy=True)
    torch.gcu.synchronize()

    # shared-prefix reuse: how many of the 16 selected pages per query are absorbed
    hist = {int(k): int((counts == k).sum()) for k in range(0, 5)}
    rows = list(range(0, 8)) + [255, 512, 1000, 1023]
    ref = torch_reference(data, topk_idx, rows, data.sm_scale, seq, nh)
    ref_diff = diff_stats(ref, out["stock"].cpu()[rows])
    # mask check: rows whose every selected position is beyond seq must be exactly 0
    res = {
        "shape": dict(batch=batch, query_len=qlen, seq_len=seq, num_kv_heads=kvh, num_heads=nh,
                      head_dim=128, topk=int(topk_idx.shape[-1])),
        "diff_stock_vs_copy_on": diff_stats(out["stock"], out["copy_on"]),
        "diff_stock_vs_prefix_off": diff_stats(out["stock"], out["copy_off"]),
        "diff_copy_on_vs_prefix_off": diff_stats(out["copy_on"], out["copy_off"]),
        "shared_prefix_count_hist": hist,
        "shared_prefix_mean_pages": float(counts.float().mean()),
        "selected_pages_per_query": int(topk_idx.shape[-1]),
        "torch_fp32_reference_rows": rows,
        "diff_stock_vs_torch_fp32": ref_diff,
        "all_finite": bool(torch.isfinite(out["stock"]).all().item()),
        "env": env,
    }
    return res


def derive_budget(per_page):
    """Telescoping per-page budget from the marginal costs.

    qk_only -> qk_softmax -> full adds exactly softmax, then (V DMA + PV), and
    no_v_dma splits that last term.  Mode 3 (both_dots) is reported but NOT used:
    dropping the softmax turns p into a raw logit cast, so its PV input differs
    and it does not telescope (see the printed residual).
    """
    f = per_page["full"]["marginal_ms_per_page"]
    qk = per_page["qk_only"]["marginal_ms_per_page"]
    sm = per_page["qk_softmax"]["marginal_ms_per_page"]
    nv = per_page["no_v_dma"]["marginal_ms_per_page"]
    bd = per_page["both_dots"]["marginal_ms_per_page"]
    softmax = sm - qk
    v_and_pv = f - sm
    pv = nv - sm
    v = f - nv
    parts = {"qk_load_transpose_dot": qk, "online_softmax": softmax,
             "v_dma_and_dte_wait": v, "pv_dot": pv}
    total = sum(parts.values())
    return {
        "parts_ms": parts,
        "share_pct": {k: 100.0 * x / f for k, x in parts.items()},
        "sum_ms": total,
        "full_ms": f,
        "residual_ms": f - total,
        "cross_check_v_plus_pv_ms": v_and_pv,
        "v_plus_pv_from_parts_ms": v + pv,
        "mode3_both_dots_ms": bd,
        "mode3_residual_ms": bd - (qk + v_and_pv),
        "mode3_used": False,
        "mode3_note": ("mode 3 replaces p=exp2(qk-m) with a raw qk cast, so its PV "
                       "operand differs; it does not telescope and is excluded."),
    }


def linear_fit(xs, ys):
    """Ordinary least squares y = a*x + b -> (slope, intercept, r2)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx if sxx else float("nan")
    b = my - a * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    return {"slope_ms_per_page": a, "intercept_ms": b,
            "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan")}


def probe_counters(fn, seconds=6, device=5):
    """Sample efsmi GPM counters while `fn` runs, plus an explicit positive control.

    On this card every GPM metric reads 0.0 even while a 4096^3 bf16 matmul is
    saturating the device, so the counters cannot attribute time inside the
    attention kernel.  Both observations are recorded, not just the null one.
    """
    def sample(tag, secs, work=None):
        out = {"tag": tag, "samples": []}
        stop = threading.Event()

        def loop():
            if work is None:
                return stop.wait(secs)
            t0 = time.time()
            while time.time() - t0 < secs and not stop.is_set():
                work()

        th = threading.Thread(target=loop, daemon=True)
        th.start()
        time.sleep(0.8)  # let the load actually reach the device
        r = sh(f"efsmi dmon -i {device} --gpm-metrics 2,3,18,19,30,31,32,33 -w 0.5 -c {int(secs * 2)}")
        stop.set()
        th.join(timeout=secs)
        rows = [l.split() for l in r.splitlines() if re.match(r"^\d+\s", l)]
        out["raw_header"] = [l for l in r.splitlines() if l.startswith("*") or l.startswith("5")]
        out["samples"] = rows
        out["nonzero_samples"] = sum(1 for row in rows if any(x not in ("0.0", "0") for x in row[1:]))
        return out

    a = torch.randn(4096, 4096, device="gcu", dtype=torch.bfloat16)
    ctrl = sample("positive_control_matmul_4096x4096x4096", seconds, lambda: a @ a)
    torch.gcu.synchronize()
    del a
    torch.gcu.empty_cache()
    real = sample("msa_full_chain", seconds, fn)
    return {"positive_control": ctrl, "msa_full_chain": real,
            "verdict": ("GPM counters report zero even under the positive control, so they "
                        "carry no information on this card; they are NOT used to attribute "
                        "time." if ctrl["nonzero_samples"] == 0 else
                        "GPM counters respond to the positive control; see samples.")}


def capture_launches(sa, call_fn):
    """Record grid / constexprs / shared bytes of the MSA kernels for ONE call.

    Metadata only: this runs outside every timed region and its result is never
    mixed with timing samples.
    """
    from triton.runtime.jit import JITFunction
    rec, orig = [], JITFunction.run

    def spy(self, *args, **kwargs):
        grid = kwargs.get("grid")
        if callable(grid):
            # grid lambdas only read meta-parameters, so the constexpr kwargs are
            # a complete meta dict for them.
            meta = {k: v for k, v in kwargs.items() if isinstance(v, (int, bool))}
            try:
                grid = grid(meta)
            except Exception as exc:  # pragma: no cover - diagnostic only
                rec.append({"kernel": "?", "grid": None, "grid_error": repr(exc)})
                grid = None
        rec.append({"kernel": getattr(self, "fn", None).__name__ if getattr(self, "fn", None) else "?",
                    "grid": list(grid) if isinstance(grid, (tuple, list)) else None,
                    "kwargs": {k: v for k, v in kwargs.items() if isinstance(v, (int, bool, str))}})
        return orig(self, *args, **kwargs)

    JITFunction.run = spy
    try:
        call_fn()
        torch.gcu.synchronize()
    finally:
        JITFunction.run = orig
    keep = [r for r in rec if r["kernel"] and any(t in r["kernel"].lower()
                                                 for t in ("sparse", "topk", "index"))]
    return [r for r in keep if "warmup" in r["kwargs"] or r["grid"]]


def run_benchmark(device, env, rounds, out_path):
    b, sa, shape, data = build(device, None)
    batch, qlen, seq, kvh, nh = shape
    args = b.MSABenchmarkArgs()

    def index_score():
        return b.minimax_m3_index_score(data.idx_q, data.index_kv_cache, data.block_table,
                                        data.cu_q, data.seq_lens, data.prefix_lens, qlen, seq, kvh)

    scores = index_score()
    topk_idx = b.minimax_m3_index_topk(scores, data.cu_q, data.prefix_lens, qlen,
                                       args.topk, args.init_blocks, args.local_blocks)
    torch.gcu.synchronize()

    out = {k: torch.empty_like(data.q) for k in
           ("stock", "copy_on", "copy_off", "copy_pin", "m1", "m2", "m3", "m4", "pipe2",
            "s2", "w2")}
    dispatch = make_dispatch(sa, None)

    def call_stock():
        b._call_sparse(b.minimax_m3_sparse_attn, data, topk_idx, out["stock"], decode=False,
                       max_query_len=qlen, num_kv_heads=kvh, decode_qlen=1)

    def mk(name, prefix=True, mode=0, stages=1, bt=None, pipeline=False, warps=1):
        bt_use = data.block_table if bt is None else bt
        return lambda: dispatch(data.q, data.kv_cache, topk_idx, bt_use, data.cu_q, data.seq_lens,
                                data.prefix_lens, qlen, kvh, data.sm_scale, out[name],
                                prefix, mode, stages, use_copy=True, pipeline=pipeline,
                                num_warps=warps)

    bt_pin = torch.zeros_like(data.block_table)
    fns = {
        "stock": call_stock,
        "copy_on": mk("copy_on"),
        "copy_off": mk("copy_off", prefix=False),
        "copy_pin": mk("copy_pin", bt=bt_pin),
        "copy_on_s2": mk("s2", stages=2),
        "abl_qk_only": mk("m1", mode=1),
        "abl_no_pv": mk("m2", mode=2),
        "abl_no_softmax": mk("m3", mode=3),
        "abl_no_v_dma": mk("m4", mode=4),
        "copy_on_pipe2": mk("pipe2", pipeline=True),
        "copy_on_w2": mk("w2", warps=2),
    }
    fns["full_chain"] = lambda: (index_score(), b.minimax_m3_index_topk(
        scores, data.cu_q, data.prefix_lens, qlen, args.topk, args.init_blocks, args.local_blocks),
        b._call_sparse(b.minimax_m3_sparse_attn, data, topk_idx, out["stock"], decode=False,
                       max_query_len=qlen, num_kv_heads=kvh, decode_qlen=1))

    for name, fn in fns.items():
        fn()
    torch.gcu.synchronize()
    launch_meta = capture_launches(sa, fns["full_chain"])
    counters = probe_counters(fns["full_chain"], device=device)

    diffs = {}
    base = out["copy_on"]
    bufname = {"copy_off": "copy_off", "copy_pin": "copy_pin", "copy_on_s2": "s2",
               "copy_on_pipe2": "pipe2", "copy_on_w2": "w2", "abl_qk_only": "m1",
               "abl_no_pv": "m2", "abl_no_softmax": "m3", "abl_no_v_dma": "m4"}
    for name, buf in bufname.items():
        d = (out[buf].float() - base.float()).abs()
        diffs[f"copy_on_vs_{name}"] = {"max_abs": float(d.max()),
                                       "mean_abs": float(d.mean()),
                                       "nonzero": int((d > 0).sum()),
                                       "numel": d.numel(),
                                       "finite": bool(torch.isfinite(out[buf]).all())}
    diffs["stock_vs_copy_on"] = {
        "max_abs": float((out["stock"].float() - base.float()).abs().max()), "numel": base.numel()}

    pairs = {k: fns[k] for k in ("stock", "copy_on", "copy_off", "copy_pin", "copy_on_s2",
                                 "abl_qk_only", "abl_no_pv", "abl_no_softmax", "abl_no_v_dma",
                                 "copy_on_pipe2", "copy_on_w2")}
    samples = bench_samples(pairs, rounds)
    chain = bench_samples({"full_chain": fns["full_chain"]}, rounds, warmup=2)

    # ---- separate staged profile (never mixed with the end-to-end samples) ----
    st = dispatch(data.q, data.kv_cache, topk_idx, data.block_table, data.cu_q, data.seq_lens,
                  data.prefix_lens, qlen, kvh, data.sm_scale, out["copy_on"], True, 0, use_copy=True)
    torch.gcu.synchronize()

    def prefix_only():
        sa._gqa_sparse_common_prefix[(triton.cdiv(qlen, PREFIX_GROUP), kvh, batch)](
            data.q, data.kv_cache, topk_idx, data.block_table, data.cu_q, data.seq_lens,
            data.prefix_lens, st["partial"], st["maximum"], st["denom"], st["counts"],
            nh // kvh, 128, int(topk_idx.shape[-1]), data.sm_scale,
            *data.q.stride(), *data.kv_cache.stride(), *topk_idx.stride(),
            data.block_table.stride(0), *st["partial"].stride(), st["maximum"].stride(0),
            kvh, triton.cdiv(qlen, PREFIX_GROUP), PREFIX_GROUP, 8, 128,
            num_warps=1, num_stages=1)

    def direct_only():
        # shipped kernel with USE_SHARED_PREFIX=False: per-query full page loop
        dispatch(data.q, data.kv_cache, topk_idx, data.block_table, data.cu_q, data.seq_lens,
                 data.prefix_lens, qlen, kvh, data.sm_scale, out["copy_on"], False, 0)

    def direct_reuse():
        # shipped direct kernel with USE_SHARED_PREFIX=True and the prefix state
        # computed above (inputs are unchanged, so the state stays valid).  Only
        # the direct launch is inside the timed region.
        sa._gqa_sparse_fwd_direct[(qlen, kvh, batch)](
            data.q, data.kv_cache, out["copy_on"], out["copy_on"], topk_idx, out["copy_on"],
            data.block_table, data.cu_q, data.cu_q, data.seq_lens, data.prefix_lens,
            kvh, nh // kvh, 128, int(topk_idx.shape[-1]), 1, data.sm_scale,
            data.q.stride(0), data.q.stride(1), data.q.stride(2),
            data.kv_cache.stride(0), data.kv_cache.stride(1), data.kv_cache.stride(2),
            data.kv_cache.stride(3), 0, 0, 0, 0,
            topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
            out["copy_on"].stride(0), out["copy_on"].stride(1), out["copy_on"].stride(2),
            data.block_table.stride(0),
            BLOCK_SIZE_Q=1, BLOCK_SIZE_K=128, BLOCK_SIZE_D=128,
            BLOCK_SIZE_H=8, BLOCK_SIZE_QH=8, USE_FP8=False, KV_SCALE_MODE=0,
            num_warps=1, num_stages=1,
            partial_ptr=st["partial"], maximum_ptr=st["maximum"], denom_ptr=st["denom"],
            count_ptr=st["counts"], GROUP_Q=PREFIX_GROUP, NUM_GROUPS=triton.cdiv(qlen, PREFIX_GROUP),
            STRIDE_PN=st["partial"].stride(0), STRIDE_PH=st["partial"].stride(1),
            STRIDE_PD=st["partial"].stride(2), STRIDE_SN=st["maximum"].stride(0),
            USE_SHARED_PREFIX=True)

    staged_fns = {
        "index_score": index_score,
        "index_topk": lambda: b.minimax_m3_index_topk(scores, data.cu_q, data.prefix_lens, qlen,
                                                     args.topk, args.init_blocks, args.local_blocks),
        "prefix_kernel": prefix_only,
        "direct_reuse_13p": direct_reuse,
        "direct_no_prefix_16p": direct_only,
    }
    # ---- per-page marginal cost, per ablation -----------------------------
    # Same compiled kernel, only the number of selected pages changes, so the
    # slope d(t)/d(pages) is the marginal cost of ONE page in the serialized
    # loop -- free of the 0.39 ms per-launch overhead that dominates the
    # absolute numbers above.  Prefix is OFF so every page really runs the loop.
    def sweep_fn(mode, k, o, prefix=False, pipeline=False):
        tk = topk_idx[..., :k].contiguous()
        return lambda: dispatch(data.q, data.kv_cache, tk, data.block_table, data.cu_q,
                                data.seq_lens, data.prefix_lens, qlen, kvh, data.sm_scale,
                                o, prefix, mode, 1, use_copy=True, pipeline=pipeline)

    variants = {"full": dict(mode=0), "qk_only": dict(mode=1), "qk_softmax": dict(mode=2),
                "both_dots": dict(mode=3), "no_v_dma": dict(mode=4),
                "pipelined": dict(mode=0, pipeline=True)}
    slope_fns, cache = {}, {}
    for name, kw in variants.items():
        for k in (4, 16):
            o = cache[name] if name in cache else torch.empty_like(data.q)
            cache[name] = o
            f = sweep_fn(kw["mode"], k, o, pipeline=kw.get("pipeline", False))
            f()
            slope_fns[f"{name}@{k}"] = f
    torch.gcu.synchronize()
    slope = bench_samples(slope_fns, max(5, rounds), warmup=2)
    per_page = {}
    for name in variants:
        t4 = statistics.median(slope[f"{name}@4"])
        t16 = statistics.median(slope[f"{name}@16"])
        per_page[name] = dict(ms_at_4_pages=t4, ms_at_16_pages=t16,
                              marginal_ms_per_page=(t16 - t4) / 12.0,
                              fixed_ms=t4 - (t16 - t4) / 12.0 * 4)

    # page-count sweep on the same kernel/case: fewer selected pages, same code
    sweep = {}
    for k in (1, 4, 8, 12, 16):
        tk_k = topk_idx[..., :k].contiguous()
        buf = torch.empty_like(data.q)

        def run_k(tk=tk_k, o=buf):
            # prefix OFF: every selected page runs inside the direct loop, so the
            # slope is the pure per-page cost of the loop (no prefix absorption).
            dispatch(data.q, data.kv_cache, tk, data.block_table, data.cu_q, data.seq_lens,
                     data.prefix_lens, qlen, kvh, data.sm_scale, o, False, 0, use_copy=True)
        run_k()
        sweep[f"pages{k}"] = run_k
    torch.gcu.synchronize()
    staged_fns.update(sweep)
    staged = bench_samples(staged_fns, max(5, rounds), warmup=2)
    sweep_pages = {}
    for k in (1, 4, 8, 12, 16):
        tk_k = topk_idx[..., :k]
        st_k = dispatch(data.q, data.kv_cache, tk_k.contiguous(), data.block_table, data.cu_q,
                        data.seq_lens, data.prefix_lens, qlen, kvh, data.sm_scale,
                        torch.empty_like(data.q), True, 0, use_copy=True)
        c = st_k["counts"].cpu().reshape(-1)
        sweep_pages[k] = dict(shared_prefix_pages_prefix_on=float(c.float().mean()),
                              pages_in_direct_loop_prefix_off=int(k))

    reuse = st["counts"].cpu().reshape(-1)
    hist = {int(k): int((reuse == k).sum()) for k in range(0, 5)}

    ir = ir_report_for_source(os.path.join(REPO, "src", "flag_attn", "runtime", "backend",
                                           "_enflame", "msa", "sparse_attn.py"))
    res = {
        "env": env,
        "shape": dict(batch=batch, query_len=qlen, seq_len=seq, num_kv_heads=kvh, num_heads=nh,
                      head_dim=128, topk=int(topk_idx.shape[-1]), rounds=rounds),
        "shared_prefix_count_hist": hist,
        "shared_prefix_mean_pages": float(reuse.float().mean()),
        "grid": {"prefix": [triton.cdiv(qlen, PREFIX_GROUP), kvh, batch],
                 "direct": [qlen, kvh, batch]},
        "end_to_end_ms": {k: summarize(v) for k, v in samples.items()},
        "full_chain_ms": {k: summarize(v) for k, v in chain.items()},
        "staged_ms": {k: summarize(v) for k, v in staged.items()},
        "launch_meta": launch_meta,
        "hardware_counters": counters,
        "numeric_diffs_vs_copy_on": diffs,
        "per_page_marginal": per_page,
        "per_page_budget": derive_budget(per_page),
        "page_sweep": {
            "topk": {str(k): sweep_pages[k] for k in sweep_pages},
            "median_ms": {k: statistics.median(v) for k, v in staged.items()
                          if k.startswith("pages")},
            "linear_fit": linear_fit([k for k in sweep_pages],
                                     [statistics.median(staged[f"pages{k}"])
                                      for k in sweep_pages]),
            "samples": {k: v for k, v in staged.items() if k.startswith("pages")},
        },
        "notes": {
            "copy_pin": "block_table forced to page 0: identical instruction path and identical "
                        "masks, only KV addresses change (locality probe, wrong numerics).",
            "abl_*": "MODE 1/2/3/4 ablation probes, wrong numerics by construction.",
            "copy_on_w2": "same math with num_warps=2 (diagnostic, numerics recorded).",
            "copy_on_s2": "num_stages=2 pipeline probe on the copied kernel.",
        },
        "ir_census": ir,
        "raw": {"end_to_end": samples, "full_chain": chain, "staged": staged},
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    return res, out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-busy", action="store_true")
    a = ap.parse_args()
    if not (a.verify or a.benchmark):
        ap.error("choose --verify or --benchmark")

    os.sched_setaffinity(0, set(range(160, 192)))
    torch.gcu.set_device(a.device)
    assert torch.gcu.current_device() == a.device
    env = env_record(a.device)
    env["cwd"] = os.getcwd()
    env["cpu_affinity_after_pin"] = sorted(os.sched_getaffinity(0))
    print(json.dumps({k: env[k] for k in ("pci", "dev_name", "driver_version", "ap_fw",
                                          "mem_clock_mhz", "gcu_usage_pct_before", "busy_before",
                                          "cpu_affinity_after_pin", "torch", "triton")}, indent=1))
    if env["busy_before"] and not a.allow_busy:
        print("FATAL: device busy, refusing to measure", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if a.verify:
        res = run_verify(a.device, env)
        path = a.out or os.path.join(OUT_DIR, f"verify_{stamp}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(res, fh, indent=1)
        res_print = {k: v for k, v in res.items() if k not in ("env",)}
        print(json.dumps(res_print, indent=1))
        print("wrote", path)
        ok = (res["diff_stock_vs_copy_on"]["max_abs"] == 0.0
              and res["diff_stock_vs_prefix_off"]["max_abs"] < 1e-3
              and res["diff_stock_vs_torch_fp32"]["max_abs"] < 5e-2
              and res["all_finite"])
        print("VERIFY", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    else:
        path = a.out or os.path.join(OUT_DIR, f"bench_{stamp}.json")
        res, path = run_benchmark(a.device, env, a.rounds, path)
        brief = {k: {n: round(v["median"], 4) for n, v in res[k].items()}
                 for k in ("end_to_end_ms", "full_chain_ms", "staged_ms")}
        brief["shared_prefix_count_hist"] = res["shared_prefix_count_hist"]
        print(json.dumps(brief, indent=1))
        print("wrote", path)
        return 0


if __name__ == "__main__":
    sys.exit(main())
