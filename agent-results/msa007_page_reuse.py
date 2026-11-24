#!/usr/bin/env python3
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
"""MSA-007: long-prefill KV page reuse analysis for MiniMax M3 sparse attention.

Two modes:

``--collect``
    Runs on GCU 5 only.  Builds the *stock* benchmark input via
    ``benchmark/test_minimax_sparse_attention_benchmark.py::make_data`` with the
    stock seed, runs the stock index score / top-k path, then runs the stock
    sparse-attention entry while recording the real ``counts`` tensor produced
    by ``_gqa_sparse_common_prefix``.  Only small tensors are copied back to the
    host (top-k ids, page table, lengths, counts).  Everything else is dropped.

``--verify``
    Pure offline re-computation from the committed JSON.  It re-derives the
    ordered common-prefix rule with the exact kernel semantics, checks the
    device ``counts`` bit-for-bit, recomputes every reported statistic and
    compares it with the committed numbers.  No torch / no device / no network.

Evidence boundary (see ``agent-results/MSA-007.md``): one BF16 shape, one seed,
one random page mapping, one device, synthetic random inputs, ideal *logical*
payload accounting -- not measured HBM traffic, and not a benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from collections import Counter
from pathlib import Path

TASK_ID = "MSA-007"

# Fixed case required by the task (benchmark PREFILL_SHAPES index 6).
BATCH, QUERY_LEN, SEQ_LEN, KV_HEADS, Q_HEADS = 1, 1024, 16384, 1, 8
SHAPE = (BATCH, QUERY_LEN, SEQ_LEN, KV_HEADS, Q_HEADS)
SEED = 6  # benchmark default --seed 0 + shape index 6
TOPK = 16
INIT_BLOCKS = 1
LOCAL_BLOCKS = 2
BLOCK = 128
DEVICE_INDEX = 5
DTYPE = "bf16"

# 128 tokens x 128 dim x 2 bytes x (K and V) x 1 KV head.
PAGE_BYTES = BLOCK * 128 * 2 * 2 * KV_HEADS

HERE = Path(__file__).resolve().parent
RAW_PATH = HERE / "msa007_raw.json"


# ---------------------------------------------------------------------------
# Pure-python statistics (shared by --collect and --verify)
# ---------------------------------------------------------------------------
def real_topk_of(prefix_len: int, q_local: int, topk: int = TOPK) -> int:
    """Valid block count, identical to the direct kernel's ``real_topk``."""
    valid_blocks = (prefix_len + q_local + BLOCK) // BLOCK
    return min(topk, valid_blocks)


def valid_blocks_of(prefix_len: int, q_local: int) -> int:
    return (prefix_len + q_local + BLOCK) // BLOCK


def prefix_counts(topk_rows, prefix_len: int, pos0: int, width: int, topk: int = TOPK) -> int:
    """Reimplementation of ``_gqa_sparse_common_prefix``'s ordered match rule.

    ``topk_rows[q][slot]`` holds the logical block id selected for query ``q``.
    Only the first ``min(4, topk)`` slots are inspected, exactly like the kernel.
    Slot ``i`` is consumed by the shared prefix only while every valid query row
    in the group selects the same id, that id exists for the group-start query,
    and the id is non-negative.  The scan stops at the first violation.
    """
    local = pos0
    count = 0
    qvalid = [local + i < len(topk_rows) for i in range(width)]
    for slot in range(min(4, topk)):
        first = topk_rows[local][slot]
        equal = 0
        for i in range(width):
            if qvalid[i]:
                if topk_rows[local + i][slot] == first:
                    equal += 1
            else:
                # The kernel treats masked rows as equal to ``first``.
                equal += 1
        exists = slot < min(topk, (prefix_len + local + BLOCK) // BLOCK)
        if not (exists and first >= 0 and equal == width):
            break
        count += 1
    return count


def _quantile(sorted_values, q):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def analyze_width(topk_rows, block_table, seq_len, prefix_len, width, shared_counts, topk=TOPK):
    """Per-group page-reuse statistics for a fixed query grouping."""
    total_q = len(topk_rows)
    groups = []

    per_page_hist = Counter()          # multiplicity -> pages (full selection)
    per_page_hist_rem = Counter()      # multiplicity -> pages (after prefix)
    multiplicity_all = []
    multiplicity_rem = []
    multiplicity_emu = []

    for g, start in enumerate(range(0, total_q, width)):
        qs = list(range(start, min(start + width, total_q)))
        full = Counter()
        rem = Counter()
        accesses = 0
        accesses_rem = 0
        active = 0
        active_rem = 0
        invalid_sentinel = 0
        invalid_beyond_valid = 0
        invalid_beyond_topk = 0
        fully_masked = 0
        c = shared_counts[g]

        for q in qs:
            rt = real_topk_of(prefix_len, q, topk)
            vb = valid_blocks_of(prefix_len, q)
            row = topk_rows[q]
            hit = 0
            hit_rem = 0
            for slot in range(topk):
                blk = row[slot]
                if slot >= rt:
                    invalid_beyond_topk += 1
                    continue
                if blk < 0:
                    invalid_sentinel += 1
                    continue
                if blk >= vb:
                    invalid_beyond_valid += 1
                    continue
                # Every selected page holds at least one causally visible token:
                # the page start is 128*blk <= prefix_len + q.
                if BLOCK * blk > prefix_len + q:
                    fully_masked += 1
                full[blk] += 1
                accesses += 1
                hit += 1
                if slot >= c:
                    rem[blk] += 1
                    accesses_rem += 1
                    hit_rem += 1
            if hit:
                active += 1
            if hit_rem:
                active_rem += 1

        union = len(full)
        union_rem = len(rem)
        for blk, mult in full.items():
            per_page_hist[mult] += 1
            multiplicity_all.append(mult)
        for blk, mult in rem.items():
            per_page_hist_rem[mult] += 1
            multiplicity_rem.append(mult)

        groups.append(
            {
                "group": g,
                "q_start": start,
                "q_end": qs[-1] + 1,
                "queries": len(qs),
                "active_queries": active,
                "accesses": accesses,
                "union_pages": union,
                "reused_accesses": accesses - union,
                "reuse_ratio": (accesses - union) / accesses if accesses else 0.0,
                "active_over_union": active / union if union else None,
                "page_multiplicity": dict(sorted(Counter(full.values()).items())),
                "prefix_shared_pages": c,
                "remaining_accesses": accesses_rem,
                "remaining_union_pages": union_rem,
                "remaining_reused_accesses": accesses_rem - union_rem,
                "remaining_reuse_ratio": (accesses_rem - union_rem) / accesses_rem if accesses_rem else 0.0,
                "remaining_active_queries": active_rem,
                "remaining_active_over_union": active_rem / union_rem if union_rem else None,
                "remaining_page_multiplicity": dict(sorted(Counter(rem.values()).items())),
                "invalid_sentinel": invalid_sentinel,
                "invalid_beyond_valid": invalid_beyond_valid,
                "invalid_beyond_topk": invalid_beyond_topk,
                "fully_masked": fully_masked,
            }
        )

    def agg(key):
        return sum(gr[key] for gr in groups)

    accesses = agg("accesses")
    union_saved = agg("reused_accesses")
    rem_accesses = agg("remaining_accesses")
    rem_saved = agg("remaining_reused_accesses")
    mult_sorted = sorted(multiplicity_all)
    mult_rem_sorted = sorted(multiplicity_rem)

    summary = {
        "group_width": width,
        "num_groups": len(groups),
        "queries": total_q,
        "accesses": accesses,
        "union_pages": agg("union_pages"),
        "reused_accesses": union_saved,
        "reuse_ratio": union_saved / accesses if accesses else 0.0,
        "accesses_per_query": accesses / total_q if total_q else 0.0,
        "active_queries": agg("active_queries"),
        "active_over_union": (agg("active_queries") / agg("union_pages")) if agg("union_pages") else None,
        "remaining_accesses": rem_accesses,
        "remaining_union_pages": agg("remaining_union_pages"),
        "remaining_reused_accesses": rem_saved,
        "remaining_reuse_ratio": rem_saved / rem_accesses if rem_accesses else 0.0,
        "remaining_active_over_union": (
            agg("remaining_active_queries") / agg("remaining_union_pages")
            if agg("remaining_union_pages")
            else None
        ),
        "prefix_shared_pages_total": sum(gr["prefix_shared_pages"] * gr["queries"] for gr in groups),
        "invalid_sentinel": agg("invalid_sentinel"),
        "invalid_beyond_valid": agg("invalid_beyond_valid"),
        "invalid_beyond_topk": agg("invalid_beyond_topk"),
        "fully_masked": agg("fully_masked"),
        "mask_slots_total": total_q * topk,
        "logical_full_access_bytes": accesses * PAGE_BYTES,
        "logical_full_union_bytes": agg("union_pages") * PAGE_BYTES,
        "logical_full_saving_bytes": union_saved * PAGE_BYTES,
        "logical_remaining_access_bytes": rem_accesses * PAGE_BYTES,
        "logical_remaining_union_bytes": agg("remaining_union_pages") * PAGE_BYTES,
        "logical_remaining_saving_bytes": rem_saved * PAGE_BYTES,
        "page_multiplicity_hist": {str(k): v for k, v in sorted(per_page_hist.items())},
        "page_multiplicity_hist_remaining": {str(k): v for k, v in sorted(per_page_hist_rem.items())},
        "page_multiplicity_quantiles": {
            "p50": _quantile(mult_sorted, 0.50),
            "p90": _quantile(mult_sorted, 0.90),
            "p99": _quantile(mult_sorted, 0.99),
            "max": float(mult_sorted[-1]) if mult_sorted else None,
        },
        "page_multiplicity_quantiles_remaining": {
            "p50": _quantile(mult_rem_sorted, 0.50),
            "p90": _quantile(mult_rem_sorted, 0.90),
            "p99": _quantile(mult_rem_sorted, 0.99),
            "max": float(mult_rem_sorted[-1]) if mult_rem_sorted else None,
        },
    }
    return summary, groups


def build_report(topk_rows, block_table, seq_len, prefix_len, counts16, topk=TOPK):
    """Full offline analysis: 8-query primary grouping plus 16-query check."""
    total_q = len(topk_rows)

    # Offline re-derivation of the real (16-query) common-prefix counts.
    counts16_recomputed = [
        prefix_counts(topk_rows, prefix_len, g * 16, 16, topk) for g in range((total_q + 15) // 16)
    ]

    # Offline emulation of the same rule with 8-query groups (diagnostic only:
    # the shipped kernel always uses 16, so this cannot be device-validated).
    counts8 = [prefix_counts(topk_rows, prefix_len, g * 8, 8, topk) for g in range((total_q + 7) // 8)]

    # "remaining" for an 8-query group uses the real pipeline's shared count of
    # the enclosing 16-query group, i.e. what the direct kernel actually skips.
    shared8_real = [counts16[g // 2] for g in range((total_q + 7) // 8)]

    s8, g8 = analyze_width(topk_rows, block_table, seq_len, prefix_len, 8, shared8_real, topk)
    s8e, _ = analyze_width(topk_rows, block_table, seq_len, prefix_len, 8, counts8, topk)
    s16, g16 = analyze_width(topk_rows, block_table, seq_len, prefix_len, 16, counts16, topk)

    # What the shipped prefix kernel already removes from the direct loop:
    # ``count`` pages loaded once per group instead of once per query.
    saved_pages_current = 0
    for gr in g16:
        saved_pages_current += gr["prefix_shared_pages"] * max(gr["active_queries"] - 1, 0)

    return {
        "page_bytes": PAGE_BYTES,
        "real_topk_values": sorted({real_topk_of(prefix_len, q, topk) for q in range(total_q)}),
        "valid_blocks_range": [
            min(valid_blocks_of(prefix_len, q) for q in range(total_q)),
            max(valid_blocks_of(prefix_len, q) for q in range(total_q)),
        ],
        "counts16_device": list(counts16),
        "counts16_recomputed": counts16_recomputed,
        "counts16_match": list(counts16) == counts16_recomputed,
        "counts16_hist": {str(k): v for k, v in sorted(Counter(counts16).items())},
        "counts16_max": max(counts16) if counts16 else 0,
        "counts8_emulated": counts8,
        "counts8_diagnostic_hist": {str(k): v for k, v in sorted(Counter(counts8).items())},
        "current_prefix": {
            "saved_page_loads": saved_pages_current,
            "saved_bytes": saved_pages_current * PAGE_BYTES,
            "saved_fraction_of_full_accesses": saved_pages_current / (total_q * topk),
        },
        "group8": s8,
        "group8_groups": g8,
        "group8_emulated_prefix": s8e,
        "group16": s16,
        "group16_groups": g16,
        "logical_total_access_bytes": total_q * topk * PAGE_BYTES,
    }


# ---------------------------------------------------------------------------
# Device collection
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_i32(rows) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update(struct.pack("<%di" % len(row), *row))
    return h.hexdigest()


def _run(cmd):
    import subprocess

    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return (out.stdout or "") + (out.stderr or "")
    except Exception as exc:  # pragma: no cover - diagnostics only
        return f"<failed: {exc}>"


def collect() -> int:
    import time

    import torch

    repo = HERE.parent
    bench_dir = repo / "benchmark"
    for p in (str(repo), str(bench_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    torch.gcu.set_device(DEVICE_INDEX)
    current = torch.gcu.current_device()
    if current != DEVICE_INDEX:
        raise RuntimeError(f"device guard failed: current={current} expected={DEVICE_INDEX}")

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "msa007_bench", bench_dir / "test_minimax_sparse_attention_benchmark.py"
    )
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench  # dataclasses needs the module registered
    spec.loader.exec_module(bench)

    from flag_attn.runtime.backend._enflame.msa import sparse_attn as sa

    device = torch.device("gcu")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    data = bench.make_data(
        BATCH,
        SEQ_LEN,
        KV_HEADS,
        Q_HEADS,
        device,
        DTYPE,
        decode=False,
        decode_qlen=1,
        query_len=QUERY_LEN,
        randomize_pages=True,
        generator=generator,
    )

    score = bench.minimax_m3_index_score(
        data.idx_q,
        data.index_kv_cache,
        data.block_table,
        data.cu_q,
        data.seq_lens,
        data.prefix_lens,
        QUERY_LEN,
        SEQ_LEN,
        KV_HEADS,
    )
    topk_idx = bench.minimax_m3_index_topk(
        score,
        data.cu_q,
        data.prefix_lens,
        QUERY_LEN,
        TOPK,
        INIT_BLOCKS,
        LOCAL_BLOCKS,
    )
    torch.gcu.synchronize()

    calls = []
    captured = {}

    class _Proxy:
        def __init__(self, kernel, name):
            self._kernel = kernel
            self._name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(self._name)
                if self._name == "_gqa_sparse_common_prefix":
                    captured["counts"] = args[10]
                    captured["partial"] = args[7]
                return self._kernel[grid](*args, **kwargs)

            return launch

        def __getattr__(self, item):
            return getattr(self._kernel, item)

    originals = {}
    for name in ("_gqa_sparse_common_prefix", "_gqa_sparse_fwd_direct", "_gqa_sparse_fwd_grouped8"):
        originals[name] = getattr(sa, name)
        setattr(sa, name, _Proxy(originals[name], name))
    try:
        output = torch.empty_like(data.q)

        def run_once():
            bench._call_sparse(
                bench.minimax_m3_sparse_attn,
                data,
                topk_idx,
                output,
                decode=False,
                max_query_len=QUERY_LEN,
                num_kv_heads=KV_HEADS,
                decode_qlen=1,
            )

        run_once()  # compiles + produces the recorded counts
        torch.gcu.synchronize()
        if "counts" not in captured:
            raise RuntimeError("shared-prefix kernel was not launched; counts unavailable")
        counts = captured["counts"].to("cpu").to(torch.int32).tolist()

        for _ in range(2):
            run_once()
        torch.gcu.synchronize()
        samples = []
        for _ in range(5):
            t0 = time.perf_counter()
            run_once()
            torch.gcu.synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)
        samples.sort()
    finally:
        for name, kernel in originals.items():
            setattr(sa, name, kernel)

    topk_rows = topk_idx.to("cpu").to(torch.int32).tolist()
    topk_rows = [[int(v) for v in row] for row in topk_rows[0]]
    table = [[int(v) for v in row] for row in data.block_table.to("cpu").tolist()]
    seq_lens = [int(v) for v in data.seq_lens.to("cpu").tolist()]
    prefix_lens = [int(v) for v in data.prefix_lens.to("cpu").tolist()]
    cu_q = [int(v) for v in data.cu_q.to("cpu").tolist()]

    counts16 = counts[0][0]
    report = build_report(topk_rows, table[0], seq_lens[0], prefix_lens[0], counts16)
    if not report["counts16_match"]:
        raise RuntimeError(
            "offline common-prefix rule does not reproduce device counts: "
            f"{report['counts16_device']} vs {report['counts16_recomputed']}"
        )

    import triton

    env_keys = [
        "PYTHONPATH",
        "TRITON_CACHE_DIR",
        "TRITON_GCU_ENABLE_I64",
        "PYTORCH_GCU_ALLOC_CONF",
        "FLAGTREE_HOME",
    ]
    meta = {
        "task": TASK_ID,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "git_branch": _run(f"cd {repo} && git rev-parse --abbrev-ref HEAD").strip(),
        "git_head": _run(f"cd {repo} && git rev-parse HEAD").strip(),
        "git_status": _run(f"cd {repo} && git status --porcelain").strip(),
        "device_index": DEVICE_INDEX,
        "device_current": int(torch.gcu.current_device()),
        "device_name": torch.gcu.get_device_name(DEVICE_INDEX),
        "device_properties": str(torch.gcu.get_device_properties(DEVICE_INDEX)),
        "efsmi": _run(f"efsmi -i {DEVICE_INDEX}"),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": sys.version.split()[0],
        "shape": list(SHAPE),
        "dtype": DTYPE,
        "seed": SEED,
        "topk": TOPK,
        "init_blocks": INIT_BLOCKS,
        "local_blocks": LOCAL_BLOCKS,
        "block": BLOCK,
        "kv_heads": KV_HEADS,
        "q_heads": Q_HEADS,
        "page_bytes": PAGE_BYTES,
        "env": {k: os.environ.get(k, "") for k in env_keys},
        "source_sha256": {
            "benchmark/test_minimax_sparse_attention_benchmark.py": _sha256_file(
                bench_dir / "test_minimax_sparse_attention_benchmark.py"
            ),
            "src/flag_attn/runtime/backend/_enflame/msa/sparse_attn.py": _sha256_file(
                repo / "src/flag_attn/runtime/backend/_enflame/msa/sparse_attn.py"
            ),
            "src/flag_attn/runtime/backend/_enflame/msa/index_topk.py": _sha256_file(
                repo / "src/flag_attn/runtime/backend/_enflame/msa/index_topk.py"
            ),
        },
        "kernel_calls_observed": sorted(set(calls)),
        "kernel_call_counts": dict(sorted(Counter(calls).items())),
        "grouped8_selected": "_gqa_sparse_fwd_grouped8" in calls,
        "shared_prefix_selected": "_gqa_sparse_common_prefix" in calls,
        "direct_selected": "_gqa_sparse_fwd_direct" in calls,
        "diagnostic_attention_ms": {
            "samples": samples,
            "median": samples[len(samples) // 2],
            "min": samples[0],
            "max": samples[-1],
            "note": "single-process wall clock around one sparse-attn call; diagnostic only, not a benchmark",
        },
        "raw_sha256": {"topk_idx": _sha256_i32(topk_rows)},
    }

    payload = {
        "meta": meta,
        "block_table": table,
        "seq_lens": seq_lens,
        "prefix_lens": prefix_lens,
        "cu_q": cu_q,
        "topk_idx": topk_rows,
        "counts": counts,
        "analysis": report,
    }
    RAW_PATH.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")

    print(f"[collect] wrote {RAW_PATH} ({RAW_PATH.stat().st_size} bytes)")
    print(f"[collect] kernels: {meta['kernel_call_counts']}")
    print(f"[collect] counts16 hist: {report['counts16_hist']}")
    print(f"[collect] counts16 == recomputed: {report['counts16_match']}")
    print(f"[collect] diagnostic attention ms: {samples}")
    print(
        "[collect] group8 accesses/union/reused: "
        f"{report['group8']['accesses']}/{report['group8']['union_pages']}/{report['group8']['reused_accesses']}"
    )
    print(
        "[collect] group8 remaining accesses/union/reused: "
        f"{report['group8']['remaining_accesses']}/{report['group8']['remaining_union_pages']}"
        f"/{report['group8']['remaining_reused_accesses']}"
    )
    return 0


# ---------------------------------------------------------------------------
# Offline verification
# ---------------------------------------------------------------------------
def verify() -> int:
    if not RAW_PATH.exists():
        print(f"[verify] FAIL missing {RAW_PATH}")
        return 2
    payload = json.loads(RAW_PATH.read_text())
    meta = payload["meta"]
    analysis = payload["analysis"]
    failures = []

    def check(name, ok, detail=""):
        print(f"[verify] {'PASS' if ok else 'FAIL'} {name}{(' :: ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    check("task id", meta.get("task") == TASK_ID, str(meta.get("task")))
    check("shape", tuple(meta.get("shape", ())) == SHAPE, str(meta.get("shape")))
    check("seed", meta.get("seed") == SEED, str(meta.get("seed")))
    check("topk/init/local", (meta.get("topk"), meta.get("init_blocks"), meta.get("local_blocks")) == (TOPK, INIT_BLOCKS, LOCAL_BLOCKS))
    check("device index", meta.get("device_index") == DEVICE_INDEX, str(meta.get("device_index")))
    check("device current", meta.get("device_current") == DEVICE_INDEX, str(meta.get("device_current")))
    check("page bytes", meta.get("page_bytes") == PAGE_BYTES, str(meta.get("page_bytes")))

    topk_rows = [[int(v) for v in row] for row in payload["topk_idx"]]
    table = [int(v) for v in payload["block_table"][0]]
    seq_len = int(payload["seq_lens"][0])
    prefix_len = int(payload["prefix_lens"][0])
    cu_q = [int(v) for v in payload["cu_q"]]
    counts = [int(v) for row in payload["counts"][0] for v in row]

    check("topk rows", len(topk_rows) == QUERY_LEN, str(len(topk_rows)))
    check("topk width", all(len(r) == TOPK for r in topk_rows))
    check("topk sha256", _sha256_i32(topk_rows) == meta["raw_sha256"]["topk_idx"])
    check("page table size", len(table) == SEQ_LEN // BLOCK and sorted(table) == list(range(len(table))))
    check("cu_q", cu_q == [0, QUERY_LEN], str(cu_q))
    check("seq/prefix", (seq_len, prefix_len) == (SEQ_LEN, SEQ_LEN - QUERY_LEN), f"{seq_len}/{prefix_len}")
    check("dtype", meta.get("dtype") == DTYPE, str(meta.get("dtype")))
    check("shared prefix selected", meta.get("shared_prefix_selected") is True)
    check("grouped8 not selected", meta.get("grouped8_selected") is False)
    check("counts shape", len(counts) == QUERY_LEN // 16, str(len(counts)))

    # Re-derive the ordered matching rule and require an exact match.
    recomputed = [prefix_counts(topk_rows, prefix_len, g * 16, 16) for g in range(QUERY_LEN // 16)]
    check("counts16 exact match", recomputed == counts, f"{recomputed[:8]} vs {counts[:8]}")
    check("counts<=4 (kernel cap)", max(counts) <= 4, str(max(counts)))
    check("counts utf8 ordering", all(c <= real_topk_of(prefix_len, g * 16) for g, c in enumerate(counts)))

    report = build_report(topk_rows, table, seq_len, prefix_len, counts)
    for key, want in analysis.items():
        if key in ("group8_groups", "group16_groups"):
            continue
        got = report[key]
        check(f"analysis.{key}", got == want, "" if got == want else f"{got!r} != {want!r}")

    # Structural invariants that must hold for any valid input.
    for name, summary in (("group8", report["group8"]), ("group16", report["group16"])):
        check(f"{name} accesses<=slots", summary["accesses"] <= summary["mask_slots_total"])
        check(f"{name} union<=accesses", summary["union_pages"] <= summary["accesses"])
        check(
            f"{name} remaining<=accesses",
            summary["remaining_accesses"] <= summary["accesses"],
        )
        check(f"{name} no fully masked page", summary["fully_masked"] == 0)
        check(f"{name} no sentinel", summary["invalid_sentinel"] == 0)
        check(f"{name} active>=1", summary["active_queries"] >= 1)

    # Cross-check the two groupings agree on totals that cannot depend on width.
    g8, g16 = report["group8"], report["group16"]
    check("accesses width-invariant", g8["accesses"] == g16["accesses"])
    check("active queries width-invariant", g8["active_queries"] == g16["active_queries"])
    check("prefix saving width-invariant", g8["prefix_shared_pages_total"] == g16["prefix_shared_pages_total"])
    check(
        "group16 remaining == direct loop",
        g16["remaining_accesses"] == g16["accesses"] - g16["prefix_shared_pages_total"],
    )
    check(
        "payload identity",
        report["logical_total_access_bytes"] == QUERY_LEN * TOPK * PAGE_BYTES,
        str(report["logical_total_access_bytes"]),
    )

    # Diagnostic artifacts, if committed alongside the raw data.  Their summary
    # numbers are re-derived from the per-sample timings rather than trusted.
    diag_path = HERE / "msa007_diag_grouped8.json"
    if diag_path.exists():
        d = json.loads(diag_path.read_text())
        check("diag device", d["meta"]["device_index"] == DEVICE_INDEX, str(d["meta"]["device_index"]))
        check("diag shape", tuple(d["meta"]["shape"]) == SHAPE)
        check("diag seed", d["meta"]["seed"] == SEED)
        check("diag gate is 2*topk", d["meta"]["page_table_gate"] == 2 * TOPK)
        check(
            "diag gate excludes table",
            d["meta"]["block_table_pages"] > d["meta"]["page_table_gate"],
            f"{d['meta']['block_table_pages']} > {d['meta']['page_table_gate']}",
        )
        for arm in ("direct", "grouped8", "grouped16"):
            samples = d[arm]["samples_ms"]
            check(f"diag {arm} median", abs(sorted(samples)[len(samples) // 2] - d[arm]["median_ms"]) < 1e-12)
            check(f"diag {arm} min", abs(min(samples) - d[arm]["min_ms"]) < 1e-12)
            check(f"diag {arm} sorted", samples == sorted(samples))
        for arm in ("grouped8", "grouped16"):
            want = d["direct"]["median_ms"] / d[arm]["median_ms"]
            check(f"diag {arm} speedup", abs(want - d[arm]["speedup_vs_direct"]) < 1e-9, f"{want:.6f}")
            check(
                f"diag {arm} slower than direct",
                d[arm]["median_ms"] > d["direct"]["median_ms"],
                f"{d[arm]['median_ms']:.3f} vs {d['direct']['median_ms']:.3f} ms",
            )
            check(f"diag {arm} agrees with direct", d[arm]["cosine_vs_direct"] > 0.999)
        check(
            "diag grouped16 faster than grouped8",
            d["grouped16"]["median_ms"] < d["grouped8"]["median_ms"],
        )

    prefix_path = HERE / "msa007_diag_prefix.json"
    if prefix_path.exists():
        d = json.loads(prefix_path.read_text())
        check("prefix diag device", d["meta"]["device_index"] == DEVICE_INDEX)
        check("prefix diag seed", d["meta"]["seed"] == SEED)
        dev_counts = [int(v) for row in d["meta"]["counts"][0] for v in row]
        check("prefix diag counts == raw counts", dev_counts == counts, str(set(dev_counts)))
        for arm in ("prefix_kernel", "direct_real_count", "direct_no_prefix"):
            samples = d[arm]["samples_ms"]
            check(f"prefix diag {arm} median", abs(sorted(samples)[len(samples) // 2] - d[arm]["median_ms"]) < 1e-12)
        saving = d["direct_no_prefix"]["median_ms"] - d["direct_real_count"]["median_ms"]
        check("prefix diag marginal saving", abs(saving - d["marginal_saving_ms"]) < 1e-9, f"{saving:+.4f} ms")
        net = saving - d["prefix_kernel"]["median_ms"]
        check("prefix diag net", abs(net - d["net_ms"]) < 1e-9, f"{net:+.4f} ms")
        check("prefix diag verdict", (net > 0) == d["prefix_pays_for_itself"], str(net > 0))
        # A skip only ever removes work, so the real-count arm must not be slower
        # by more than run-to-run noise.
        check(
            "prefix diag real<=no-prefix",
            d["direct_real_count"]["median_ms"] <= d["direct_no_prefix"]["median_ms"],
        )
        # The two arms must compute the same attention: if neutralising the
        # prefix state changed the result, the timing delta would be meaningless.
        check(
            "prefix diag arms agree",
            d["output_cosine"] > 0.9999 and d["output_max_rel_diff_vs_rms"] < 0.05,
            f"cos={d['output_cosine']:.8f} maxdiff/rms={d['output_max_rel_diff_vs_rms']:.4f}",
        )
        # Cross-artifact identity: the grouped8 diagnostic times the *whole*
        # sparse-attn pipeline (prefix + direct) as its "direct" arm, so it must
        # match prefix-kernel + direct-kernel measured separately here.
        gp = HERE / "msa007_diag_grouped8.json"
        if gp.exists():
            g = json.loads(gp.read_text())
            pipeline = d["prefix_kernel"]["median_ms"] + d["direct_real_count"]["median_ms"]
            other = g["direct"]["median_ms"]
            rel = abs(pipeline - other) / other
            check(
                "prefix+direct == pipeline direct",
                rel < 0.05,
                f"{pipeline:.3f} vs {other:.3f} ms ({rel:.2%})",
            )
            # grouped8 is a single kernel that replaces prefix+direct, so its
            # verdict must be judged against the pipeline, not the bare kernel.
            check(
                "grouped8 verdict vs pipeline",
                g["grouped8"]["median_ms"] > g["direct"]["median_ms"],
                f"{g['grouped8']['median_ms']:.3f} vs {g['direct']['median_ms']:.3f} ms",
            )

        saved_loads = report["current_prefix"]["saved_page_loads"]
        if saved_loads:
            per_page = d["marginal_saving_ms"] / saved_loads
            print(f"[verify] marginal saving per skipped page load: {per_page * 1e3:.3f} us")

    print()
    print(f"[verify] group8 : {json.dumps({k: g8[k] for k in ('accesses','union_pages','reused_accesses','reuse_ratio','remaining_accesses','remaining_union_pages','remaining_reused_accesses','remaining_reuse_ratio')})}")
    print(f"[verify] group16: {json.dumps({k: g16[k] for k in ('accesses','union_pages','reused_accesses','reuse_ratio','remaining_accesses','remaining_union_pages','remaining_reused_accesses','remaining_reuse_ratio')})}")
    print(f"[verify] counts16 hist: {report['counts16_hist']}")
    print(f"[verify] current prefix saved page loads: {report['current_prefix']['saved_page_loads']} "
          f"({report['current_prefix']['saved_fraction_of_full_accesses']:.4f} of accesses)")
    if failures:
        print(f"[verify] FAILED {len(failures)} check(s): {failures}")
        return 1
    print("[verify] OK: all checks passed offline (no torch, no device)")
    return 0


# ---------------------------------------------------------------------------
# Diagnostic: what a page-table-gate relaxation would actually run
# ---------------------------------------------------------------------------
DIAG_PATH = HERE / "msa007_diag_grouped8.json"


def diag_grouped8() -> int:
    """Time the existing grouped-union kernel at 8- and 16-query groups.

    The shipped dispatch never selects ``_gqa_sparse_fwd_grouped8`` for this
    shape because ``block_table.shape[1] (128) > 2 * topk (32)``.  Relaxing only
    that gate would run it with ``grouped_query_size = 8`` (the 16 branch has
    the same gate), so this calls the unmodified kernel directly with
    GROUP_SIZE_Q=8 and, for reference, 16.  Baseline source and default
    dispatch are untouched; this is a diagnostic, not a benchmark.
    """
    import time

    import torch
    import triton

    repo = HERE.parent
    bench_dir = repo / "benchmark"
    for p in (str(repo), str(bench_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    torch.gcu.set_device(DEVICE_INDEX)
    if torch.gcu.current_device() != DEVICE_INDEX:
        raise RuntimeError("device guard failed")

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "msa007_bench_diag", bench_dir / "test_minimax_sparse_attention_benchmark.py"
    )
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)

    from flag_attn.runtime.backend._enflame.msa import sparse_attn as sa

    device = torch.device("gcu")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    data = bench.make_data(
        BATCH, SEQ_LEN, KV_HEADS, Q_HEADS, device, DTYPE,
        decode=False, decode_qlen=1, query_len=QUERY_LEN,
        randomize_pages=True, generator=generator,
    )
    score = bench.minimax_m3_index_score(
        data.idx_q, data.index_kv_cache, data.block_table, data.cu_q,
        data.seq_lens, data.prefix_lens, QUERY_LEN, SEQ_LEN, KV_HEADS,
    )
    topk_idx = bench.minimax_m3_index_topk(
        score, data.cu_q, data.prefix_lens, QUERY_LEN, TOPK, INIT_BLOCKS, LOCAL_BLOCKS,
    )
    torch.gcu.synchronize()

    q, kv = data.q, data.kv_cache
    head_dim = q.shape[-1]
    gqa = Q_HEADS // KV_HEADS
    block_h = triton.next_power_of_2(gqa)
    output = torch.empty_like(q)

    def run_direct():
        bench._call_sparse(
            bench.minimax_m3_sparse_attn, data, topk_idx, output,
            decode=False, max_query_len=QUERY_LEN, num_kv_heads=KV_HEADS, decode_qlen=1,
        )

    def make_grouped(group_q, out):
        def run():
            sa._gqa_sparse_fwd_grouped8[(triton.cdiv(QUERY_LEN, group_q), KV_HEADS, BATCH)](
                q, kv, topk_idx, out, data.block_table,
                data.cu_q, data.seq_lens, data.prefix_lens,
                0, gqa, head_dim, data.sm_scale,
                q.stride(0), q.stride(1), q.stride(2),
                kv.stride(0), kv.stride(1), kv.stride(2), kv.stride(3),
                topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                data.block_table.stride(0),
                GROUP_SIZE_Q=group_q,
                BLOCK_SIZE_K=BLOCK,
                BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
                BLOCK_SIZE_H=block_h,
                MAX_TOPK=TOPK,
                BLOCK_TOPK=triton.next_power_of_2(TOPK),
                DECODE_MODE=False,
                num_warps=1,
                num_stages=1,
            )

        return run

    def timed(fn, warmup=3, reps=7):
        for _ in range(warmup):
            fn()
        torch.gcu.synchronize()
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            torch.gcu.synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)
        samples.sort()
        return {"samples_ms": samples, "median_ms": samples[len(samples) // 2], "min_ms": samples[0]}

    run_direct()
    torch.gcu.synchronize()
    ref = output.clone()

    results = {"meta": {"device_index": DEVICE_INDEX, "shape": list(SHAPE), "seed": SEED,
                        "topk": TOPK, "page_table_gate": 2 * TOPK,
                        "block_table_pages": data.block_table.shape[1],
                        "note": "diagnostic only; unmodified kernel called from a standalone script"},
               "direct": timed(run_direct)}

    outs = {}
    for group_q in (8, 16):
        out = torch.empty_like(q)
        fn = make_grouped(group_q, out)
        fn()
        torch.gcu.synchronize()
        outs[group_q] = out.clone()
        stats = timed(fn)
        a = outs[group_q].float().flatten()
        b = ref.float().flatten()
        diff = (a - b).abs()
        norm = max(torch.linalg.vector_norm(a).item() * torch.linalg.vector_norm(b).item(), 1e-30)
        stats.update(
            {
                "group_size_q": group_q,
                "grid": [triton.cdiv(QUERY_LEN, group_q), KV_HEADS, BATCH],
                "max_abs_diff_vs_direct": diff.max().item(),
                "mean_abs_diff_vs_direct": diff.mean().item(),
                "cosine_vs_direct": (a @ b).item() / norm,
                "speedup_vs_direct": results["direct"]["median_ms"] / stats["median_ms"],
            }
        )
        results[f"grouped{group_q}"] = stats
    results["grouped8_vs_grouped16_max_abs_diff"] = (
        (outs[8].float() - outs[16].float()).abs().max().item()
    )

    DIAG_PATH.write_text(json.dumps(results, indent=1) + "\n")
    print(f"[diag] wrote {DIAG_PATH}")
    print(f"[diag] direct     median {results['direct']['median_ms']:.3f} ms")
    for group_q in (8, 16):
        r = results[f"grouped{group_q}"]
        print(
            f"[diag] grouped{group_q:<2} median {r['median_ms']:.3f} ms  "
            f"speedup {r['speedup_vs_direct']:.2f}x  max|diff| {r['max_abs_diff_vs_direct']:.4g}  "
            f"cos {r['cosine_vs_direct']:.6f}"
        )
    return 0


# ---------------------------------------------------------------------------
# Diagnostic: marginal value of the shared prefix the kernel already computes
# ---------------------------------------------------------------------------
PREFIX_DIAG_PATH = HERE / "msa007_diag_prefix.json"


def diag_prefix() -> int:
    """Time the direct kernel with the real ``count`` versus ``count = 0``.

    Same kernel, same data, same launch: only ``count_ptr`` contents differ.  The
    delta is the marginal value of skipping the shared pages; the prefix kernel
    itself costs extra time that this delta must repay.
    """
    import time

    import torch
    import triton

    repo = HERE.parent
    bench_dir = repo / "benchmark"
    for p in (str(repo), str(bench_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    torch.gcu.set_device(DEVICE_INDEX)
    if torch.gcu.current_device() != DEVICE_INDEX:
        raise RuntimeError("device guard failed")

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "msa007_bench_prefix", bench_dir / "test_minimax_sparse_attention_benchmark.py"
    )
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)

    from flag_attn.runtime.backend._enflame.msa import sparse_attn as sa

    device = torch.device("gcu")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    data = bench.make_data(
        BATCH, SEQ_LEN, KV_HEADS, Q_HEADS, device, DTYPE,
        decode=False, decode_qlen=1, query_len=QUERY_LEN,
        randomize_pages=True, generator=generator,
    )
    score = bench.minimax_m3_index_score(
        data.idx_q, data.index_kv_cache, data.block_table, data.cu_q,
        data.seq_lens, data.prefix_lens, QUERY_LEN, SEQ_LEN, KV_HEADS,
    )
    topk_idx = bench.minimax_m3_index_topk(
        score, data.cu_q, data.prefix_lens, QUERY_LEN, TOPK, INIT_BLOCKS, LOCAL_BLOCKS,
    )
    torch.gcu.synchronize()

    q, kv, table = data.q, data.kv_cache, data.block_table
    head_dim = q.shape[-1]
    gqa = Q_HEADS // KV_HEADS
    block_h = triton.next_power_of_2(gqa)
    prefix_group = 16
    prefix_groups = triton.cdiv(QUERY_LEN, prefix_group)
    output = torch.empty_like(q)

    partial = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    maximum = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
    denom = torch.empty_like(maximum)
    counts = torch.empty((BATCH, KV_HEADS, prefix_groups), device=q.device, dtype=torch.int32)

    def run_prefix():
        sa._gqa_sparse_common_prefix[(prefix_groups, KV_HEADS, BATCH)](
            q, kv, topk_idx, table, data.cu_q, data.seq_lens, data.prefix_lens,
            partial, maximum, denom, counts,
            gqa, head_dim, TOPK, data.sm_scale,
            *q.stride(), *kv.stride(), *topk_idx.stride(), table.stride(0),
            *partial.stride(), maximum.stride(0),
            KV_HEADS, prefix_groups, prefix_group, block_h,
            triton.next_power_of_2(head_dim), num_warps=1, num_stages=1,
        )

    def direct_fn(count_buf, out):
        def run():
            sa._gqa_sparse_fwd_direct[(QUERY_LEN, KV_HEADS, BATCH)](
                q, kv, out, out, topk_idx, out, table,
                data.cu_q, data.cu_q, data.seq_lens, data.prefix_lens,
                KV_HEADS, gqa, head_dim, TOPK, 1, data.sm_scale,
                q.stride(0), q.stride(1), q.stride(2),
                kv.stride(0), kv.stride(1), kv.stride(2), kv.stride(3),
                0, 0, 0, 0,
                topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
                out.stride(0), out.stride(1), out.stride(2), table.stride(0),
                BLOCK_SIZE_Q=1, BLOCK_SIZE_K=BLOCK,
                BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
                BLOCK_SIZE_H=block_h, BLOCK_SIZE_QH=block_h,
                USE_FP8=False, KV_SCALE_MODE=0, num_warps=1, num_stages=1,
                partial_ptr=partial, maximum_ptr=maximum, denom_ptr=denom,
                count_ptr=count_buf, GROUP_Q=prefix_group, NUM_GROUPS=prefix_groups,
                STRIDE_PN=partial.stride(0), STRIDE_PH=partial.stride(1),
                STRIDE_PD=partial.stride(2), STRIDE_SN=maximum.stride(0),
                USE_SHARED_PREFIX=True,
            )

        return run

    def timed(fn, warmup=3, reps=7):
        for _ in range(warmup):
            fn()
        torch.gcu.synchronize()
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            torch.gcu.synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)
        samples.sort()
        return {"samples_ms": samples, "median_ms": samples[len(samples) // 2], "min_ms": samples[0]}

    run_prefix()
    torch.gcu.synchronize()
    real_counts = counts.clone().to("cpu").to(torch.int32).tolist()

    # Clean "no shared prefix" arm: same kernel, same launch, but the prefix
    # state is neutralised (count=0, m=-inf, l=0, partial=0) so the loop covers
    # all ``topk`` pages from scratch.  This is the semantics of the code path
    # when no page is shareable, so the two arms must agree numerically and the
    # timing delta is the marginal value of skipping ``count`` pages.
    zero_counts = torch.zeros_like(counts)
    neg_inf_max = torch.full_like(maximum, float("-inf"))
    zero_denom = torch.zeros_like(denom)
    zero_partial = torch.zeros_like(partial)
    saved = (partial, maximum, denom, counts)
    out_real = torch.empty_like(q)
    out_full = torch.empty_like(q)

    direct_fn(saved[3], out_real)()
    partial, maximum, denom, counts = zero_partial, neg_inf_max, zero_denom, zero_counts
    direct_fn(counts, out_full)()
    partial, maximum, denom, counts = saved
    torch.gcu.synchronize()

    a = out_real.float().flatten()
    b = out_full.float().flatten()
    diff = (a - b).abs()
    # NOTE: must not shadow ``denom``/``maximum``/``partial``/``counts`` -- those
    # names are the live device tensors captured by the kernel closures above.
    norm = max(torch.linalg.vector_norm(a).item() * torch.linalg.vector_norm(b).item(), 1e-30)
    cosine = (a @ b).item() / norm
    rms = torch.sqrt(torch.mean(a * a)).item()

    results = {
        "meta": {
            "device_index": DEVICE_INDEX,
            "shape": list(SHAPE),
            "seed": SEED,
            "counts": real_counts,
            "counts_hist": {str(k): v for k, v in sorted(Counter(real_counts[0][0]).items())},
            "note": (
                "diagnostic only; prefix cost and direct-kernel delta measured separately. "
                "The no-prefix arm neutralises the prefix state (count=0, m=-inf, l=0, "
                "partial=0) so both arms compute the same attention and differ only in "
                "whether the shared pages are re-loaded."
            ),
        },
        "prefix_kernel": timed(run_prefix),
        "direct_real_count": timed(direct_fn(saved[3], out_real)),
        "direct_no_prefix": timed(direct_fn(zero_counts, out_full)),
        "output_max_abs_diff_real_vs_no_prefix": diff.max().item(),
        "output_mean_abs_diff_real_vs_no_prefix": diff.mean().item(),
        "output_rms": rms,
        "output_cosine": cosine,
        "output_max_rel_diff_vs_rms": diff.max().item() / rms if rms else None,
    }
    results["prefix_ms"] = results["prefix_kernel"]["median_ms"]
    results["marginal_saving_ms"] = (
        results["direct_no_prefix"]["median_ms"] - results["direct_real_count"]["median_ms"]
    )
    results["net_ms"] = results["marginal_saving_ms"] - results["prefix_ms"]
    results["prefix_pays_for_itself"] = results["net_ms"] > 0

    PREFIX_DIAG_PATH.write_text(json.dumps(results, indent=1) + "\n")
    print(f"[prefix-diag] wrote {PREFIX_DIAG_PATH}")
    print(f"[prefix-diag] counts {results['meta']['counts_hist']}")
    print(f"[prefix-diag] prefix kernel      {results['prefix_ms']:.4f} ms")
    print(f"[prefix-diag] direct, count=real {results['direct_real_count']['median_ms']:.4f} ms")
    print(f"[prefix-diag] direct, no prefix  {results['direct_no_prefix']['median_ms']:.4f} ms")
    print(f"[prefix-diag] marginal saving    {results['marginal_saving_ms']:+.4f} ms")
    print(f"[prefix-diag] net of prefix cost {results['net_ms']:+.4f} ms  "
          f"(pays for itself: {results['prefix_pays_for_itself']})")
    print(
        f"[prefix-diag] |real-full| max    {diff.max().item():.6g} "
        f"(output rms {rms:.4g}, cos {cosine:.8f})"
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--collect", action="store_true", help="collect raw data on GCU 5")
    mode.add_argument("--verify", action="store_true", help="offline verification only")
    mode.add_argument(
        "--diag-prefix",
        action="store_true",
        help="diagnostic: prefix cost vs direct-kernel marginal saving (device 5)",
    )
    mode.add_argument(
        "--diag-grouped8",
        action="store_true",
        help="diagnostic timing of the unused grouped-union kernel (device 5)",
    )
    args = parser.parse_args(argv)
    if args.collect:
        return collect()
    if args.diag_grouped8:
        return diag_grouped8()
    if args.diag_prefix:
        return diag_prefix()
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
