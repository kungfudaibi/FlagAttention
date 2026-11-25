#!/usr/bin/env python
"""MSA-011: pre-resolved physical page ids for C200 long-prefill direct MSA.

One experiment, one variable: instead of resolving ``physical_page =
block_table[logical_page]`` once per page *inside* the direct attention kernel,
resolve the whole TopK row once (in the existing shared-prefix launch, in the
same slot order as ``topk_idx``) and let the kernel read that id directly.
Logical page ids stay authoritative for the causal/boundary masks.

Fixed case (upstream benchmark, unmodified):
  benchmark/test_minimax_sparse_attention_benchmark.py
  PREFILL_SHAPES[6] = (B, Q, KV, Hkv, Hq) = (1, 1024, 16384, 1, 8)
  D = 128, page = 128, TopK = 16, seed + shape_index = 6,
  upstream ``make_data`` random physical page permutation.

Both arms keep the verified upstream entry point:
  baseline   FLAG_ATTN_MSA_PACKED_K=1                        (packed K + early V)
  candidate  FLAG_ATTN_MSA_PACKED_K=1 FLAG_ATTN_MSA_PHYS_PAGE=1

  --device 5 --verify      mapping/bitwise/FP32/mask/bypass/IR checks
  --device 5 --benchmark    >=3 interleaved A/B rounds, full chain plus the
                            direct and Score->TopK stages, every raw sample
                            printed to stdout (nothing is written into the repo)

GCU 5 only (PCI 0000:83:00.0).  The gate fails closed on a wrong device, a
wrong PCI address or a foreign process on the card.
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
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

import torch  # noqa: E402
import triton  # noqa: E402

SHAPE_INDEX = 6                  # PREFILL_SHAPES[6] == (1, 1024, 16384, 1, 8)
BYPASS_INDEX = 1                 # PREFILL_SHAPES[1] == (1, 512, 4096, 1, 8)
PACKED_ENV = "FLAG_ATTN_MSA_PACKED_K"
PHYS_ENV = "FLAG_ATTN_MSA_PHYS_PAGE"
EXPECTED_PCI = "0000:83:00.0"
DEVICE = 5

# Pristine fb664f8, recorded before the candidate was applied (see MSA-011.md):
PRISTINE = {
    "chain_out_sha256": "4559539690ae255a79bef16fbdfdcba89d1ebaeb70642a02681e56528fd4a9cd",
    "topk_sha256": "361843aa1b536115b7ed4429d0c8605c9a16d9fbf8a6707074945a9db012f611",
    "block_table_sha256": "38947b677568a98393c38d66de472e2dced8c238056dace7933bae5b57013f2e",
    # Compiled-artifact hashes are salted per process/cache on this backend,
    # so the stable identity used here is the GCUIR op census plus shared size.
    "direct_ir_default": {
        "counts": {"memref.dma_start": 9, "memref.dma_wait": 26, "gcu.matmul": 4,
                   "memref_ext.transpose_start": 1, "scf.for": 3, "memref.load": 8,
                   "ttir_tt_trans": 0, "ttir_tt_dot": 0},
        "convgen": ["tops_convgen_m8_n128_k128_bf16_fp32",
                    "tops_convgen_m8_n128_k128_bias_bf16_fp32"],
        "shared": 65536,
    },
    "direct_ir_packed": {
        "counts": {"memref.dma_start": 9, "memref.dma_wait": 23, "gcu.matmul": 4,
                   "memref_ext.transpose_start": 0, "scf.for": 3, "memref.load": 8,
                   "ttir_tt_trans": 0, "ttir_tt_dot": 0},
        "convgen": ["tops_convgen_m8_n128_k128_bf16_fp32",
                    "tops_convgen_m8_n128_k128_bias_bf16_fp32"],
        "shared": 0,
    },
    "grouped8_ir": {
        "counts": {"memref.dma_start": 6, "memref.dma_wait": 39, "gcu.matmul": 4,
                   "memref_ext.transpose_start": 2, "scf.for": 14, "memref.load": 11,
                   "ttir_tt_trans": 0, "ttir_tt_dot": 0},
        "convgen": ["tops_convgen_m64_n128_k128_bf16_fp32",
                    "tops_convgen_m64_n128_k128_bias_bf16_fp32"],
        "shared": 99328,
    },
    "short_out_sha256": "f684383fc6941ea7eada64ff7f64e7d8bc2af3c7ac9288a243b4e540f86797bb",
}


# --------------------------------------------------------------------------
# gate / environment
# --------------------------------------------------------------------------
def sh(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except Exception as exc:
        return f"<error {type(exc).__name__}: {exc}>"


def process_tree_pids():
    pids, cur = set(), os.getpid()
    while cur and cur not in pids:
        pids.add(cur)
        try:
            cur = int(open(f"/proc/{cur}/stat").read().rsplit(")", 1)[1].split()[1])
        except Exception:
            break
    return pids


def foreign_pids(pmon_text):
    mine, out = process_tree_pids(), []
    for line in pmon_text.splitlines():
        m = re.match(r"\s*\d+\s+(\d+)\s", line)
        if m and int(m.group(1)) not in mine:
            out.append([int(m.group(1)), line.strip()])
    return out


def device_pci(index):
    q = sh(f"efsmi -i {index} -q")
    dom = re.search(r"Domain\s*:\s*([0-9a-fA-F]+)", q)
    bus = re.search(r"Bus\s*:\s*(\d+)", q)
    dev = re.search(r"Dev\s*:\s*(\d+)", q)
    fn = re.search(r"Func\s*:\s*(\d+)", q)
    if not (dom and bus and dev and fn):
        return None
    return f"{int(dom.group(1), 16):04x}:{bus.group(1)}:{dev.group(1)}.{fn.group(1)}"


def gate(device):
    """Fail closed unless this is GCU 5 at 0000:83:00.0 and the card is idle."""
    if device != DEVICE:
        print(f"FATAL: MSA-011 is restricted to GCU {DEVICE} (got --device {device})")
        sys.exit(3)
    pmon = sh(f"efsmi pmon -i {device} -c 1")
    usage = sh(f"efsmi -i {device} -q -d USAGE")
    m = re.search(r"GCU Usage\s*:\s*([\d.]+)", usage)
    pct = float(m.group(1)) if m else None
    foreign = foreign_pids(pmon)
    pci = device_pci(device)
    print(f"[gate] pci={pci} usage={pct}% foreign_pids={foreign}")
    if foreign:
        print("[gate] FATAL: another process is on GCU 5; nothing was measured.")
        sys.exit(4)
    if pci != EXPECTED_PCI:
        print(f"[gate] FATAL: GCU 5 reports PCI {pci}, expected {EXPECTED_PCI}")
        sys.exit(4)
    if pct is None or pct > 1.0:
        print("[gate] FATAL: GCU 5 is busy; nothing was measured.")
        sys.exit(4)
    return dict(pci=pci, usage_pct=pct, foreign_pids=foreign, pmon_raw=pmon,
                usage_raw=usage)


def gate_soft(device):
    """Post-run check: only foreign processes matter (this process is on the card)."""
    pmon = sh(f"efsmi pmon -i {device} -c 1")
    usage = sh(f"efsmi -i {device} -q -d USAGE")
    m = re.search(r"GCU Usage\s*:\s*([\d.]+)", usage)
    return {"pci": device_pci(device),
            "usage_pct": float(m.group(1)) if m else None,
            "foreign_pids": foreign_pids(pmon)}


def env_record(device, gate_rec):
    props = torch.gcu.get_device_properties(device)
    rec = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": sh(f"git -C {REPO} rev-parse HEAD").strip(),
        "git_branch": sh(f"git -C {REPO} rev-parse --abbrev-ref HEAD").strip(),
        "git_dirty": bool(sh(f"git -C {REPO} status --porcelain").strip()),
        "device_index": device,
        "pci": gate_rec["pci"],
        "device_name": torch.gcu.get_device_name(device),
        "torch_gcu_current_device": torch.gcu.current_device(),
        "stream": str(torch.gcu.current_stream()),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": sys.version.split()[0],
        "props": {
            "total_memory_mb": round(props.total_memory / (1 << 20)),
            "multi_processor_count": props.multi_processor_count,
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "env": {k: os.environ.get(k) for k in
                ("PYTHONPATH", "TRITON_CACHE_DIR", PACKED_ENV, PHYS_ENV)},
        "gate": {k: gate_rec[k] for k in ("usage_pct", "foreign_pids")},
        "disk": sh("df -h /workspace | tail -1").strip(),
    }
    return rec


class arms:
    """Explicit baseline / candidate arm context (never leaks between arms)."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in (PACKED_ENV, PHYS_ENV)}
        for k in self.old:
            os.environ.pop(k, None)
        os.environ[PACKED_ENV] = "1"
        if self.name == "candidate":
            os.environ[PHYS_ENV] = "1"
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def sha256(t):
    return hashlib.sha256(
        t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def diff_stats(a, b):
    d = (a.float().cpu() - b.float().cpu()).abs()
    af, bf = a.float().cpu().reshape(-1), b.float().cpu().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(af, bf, dim=0).item()
    return dict(max_abs=d.max().item(), mean_abs=d.mean().item(),
                nonzero=int((d > 0).sum()), numel=d.numel(), cosine=cos)


# --------------------------------------------------------------------------
# upstream case
# --------------------------------------------------------------------------
def load_bench():
    from benchmark import test_minimax_sparse_attention_benchmark as bench
    return bench


def build_case(bench, shape_index):
    batch, qlen, seq, kvh, nh = bench.PREFILL_SHAPES[shape_index]
    gen = torch.Generator(device="cpu").manual_seed(bench.MSABenchmarkArgs().seed + shape_index)
    data = bench.make_data(batch, seq, kvh, nh, torch.device(f"gcu:{DEVICE}"), "bf16",
                           decode=False, decode_qlen=1, query_len=qlen,
                           randomize_pages=True, generator=gen)
    return data, (batch, qlen, seq, kvh, nh)


def score_topk(bench, data, qlen, seq, kvh, topk=16, init_blocks=1, local_blocks=2):
    scores = bench.minimax_m3_index_score(
        data.idx_q, data.index_kv_cache, data.block_table, data.cu_q,
        data.seq_lens, data.prefix_lens, qlen, seq, kvh)
    return bench.minimax_m3_index_topk(
        scores, data.cu_q, data.prefix_lens, qlen, topk, init_blocks, local_blocks)


def call_sparse(bench, data, tk, out, qlen, kvh):
    bench._call_sparse(bench.minimax_m3_sparse_attn, data, tk, out, decode=False,
                       max_query_len=qlen, num_kv_heads=kvh, decode_qlen=1)


def run_chain(bench, data, out, qlen, seq, kvh):
    tk = score_topk(bench, data, qlen, seq, kvh)
    call_sparse(bench, data, tk, out, qlen, kvh)
    return tk


# --------------------------------------------------------------------------
# torch reference (independent of the kernels)
# --------------------------------------------------------------------------
def torch_reference(data, tk, rows, scale, seq_len, num_heads):
    q = data.q.float().cpu()
    kv = data.kv_cache.float().cpu()
    bt = data.block_table.cpu()
    tkc = tk.cpu()
    prefix_len = int(data.prefix_lens[0].item())
    out = torch.zeros((len(rows), num_heads, q.shape[-1]), dtype=torch.float32)
    for i, row in enumerate(rows):
        q_abs = prefix_len + row
        acc = torch.zeros((num_heads, q.shape[-1]), dtype=torch.float32)
        m = torch.full((num_heads,), float("-inf"))
        low = torch.zeros((num_heads,))
        for slot in range(tkc.shape[-1]):
            blk = int(tkc[0, row, slot])
            if blk < 0:
                continue
            page = int(bt[0, blk])
            k = kv[page, 0, :, :128]
            v = kv[page, 0, :, 128:]
            pos = blk * 128 + torch.arange(128)
            valid = (pos < seq_len) & (pos <= q_abs)
            s = (q[row] @ k.T) * scale
            s = s.masked_fill(~valid[None, :], -float("inf"))
            m_new = torch.maximum(m, s.max(dim=-1).values)
            safe = torch.where(m_new > float("-inf"), m_new, torch.zeros(()))
            p = torch.where(valid[None, :],
                            torch.exp2((s - safe[:, None]) * 1.4426950409),
                            torch.zeros(()))
            alpha = torch.where(m > float("-inf"),
                                torch.exp2((m - safe) * 1.4426950409), torch.zeros(()))
            low = low * alpha + p.sum(dim=-1)
            acc = acc * alpha[:, None] + p @ v
            m = m_new
        out[i] = acc / low[:, None]
    return out


def mask_check(bench, data, tk, clean, qlen, kvh, rows, poison=30000.0):
    """Falsifiable causal-mask check on *logical* positions.

    For row r: future positions (>q_abs) must not change it (changed == 0);
    visible positions must change it (positive control).  Poison is written via
    the physical page that the random block table maps the logical block to.
    """
    bt = data.block_table.cpu()
    blocks = sorted({int(x) for x in tk[0].reshape(-1).tolist() if int(x) >= 0})
    page_of = {b: int(bt[0, b]) for b in blocks}
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
            o = torch.empty_like(data.q)
            call_sparse(bench, data, tk, o, qlen, kvh)
        finally:
            data.kv_cache = saved
        return o, n

    res = {}
    for r in rows:
        q_abs = prefix_len + r
        fut, vis = {}, {}
        for b in blocks:
            page, base = page_of[b], b * 128
            f = [o for o in range(128) if base + o > q_abs]
            v = [o for o in range(128) if base + o <= q_abs]
            if f:
                fut[page] = f
            if v:
                vis[page] = v
        dirty, n_f = run(fut)
        ch_f = int((dirty[r].cpu() != clean[r]).sum())
        ctrl, n_v = run(vis)
        ch_v = int((ctrl[r].cpu() != clean[r]).sum())
        res[f"row{r}"] = dict(q_abs=q_abs, future_positions=n_f, future_changed=ch_f,
                              control_positions=n_v, control_changed=ch_v,
                              enforced=(ch_f == 0 and ch_v > 0))
        print(f"[mask r={r}] q_abs={q_abs} future n={n_f} changed={ch_f} | "
              f"control n={n_v} changed={ch_v} -> "
              f"{'ENFORCED' if ch_f == 0 and ch_v > 0 else 'LEAK/INCONCLUSIVE'}")
    return res


# --------------------------------------------------------------------------
# launch capture / IR summary
# --------------------------------------------------------------------------
def capture_launch(bench, data, out, qlen, seq, kvh):
    """One chain call, recording launch metadata and compiled artifacts."""
    log, bins, phys = [], {}, {}
    orig = triton.runtime.jit.JITFunction.run

    def hook(inst, *a, **kw):
        name = getattr(inst, "__name__", "")
        warm = kw.get("warmup", False)
        ret = orig(inst, *a, **kw)
        if warm or name not in ("_gqa_sparse_common_prefix", "_gqa_sparse_fwd_direct",
                                "_gqa_sparse_fwd_grouped8"):
            return ret
        log.append({"kernel": name, "grid": list(kw["grid"]) if "grid" in kw else None,
                    "USE_PACKED_K": kw.get("USE_PACKED_K"),
                    "EARLY_V_DMA": kw.get("EARLY_V_DMA"),
                    "USE_PHYS_PAGE": kw.get("USE_PHYS_PAGE")})
        bins[name] = ret
        if name == "_gqa_sparse_fwd_direct":
            phys["ptr"] = kw.get("phys_page_ptr")
            phys["use"] = kw.get("USE_PHYS_PAGE")
        return ret

    triton.runtime.jit.JITFunction.run = hook
    try:
        run_chain(bench, data, out, qlen, seq, kvh)
        torch.gcu.synchronize()
    finally:
        triton.runtime.jit.JITFunction.run = orig
    return log, bins, phys


def _text(x):
    return x.decode(errors="replace") if isinstance(x, (bytes, bytearray)) else (x or "")


def ir_summary(binary):
    if binary is None or not hasattr(binary, "asm"):
        return None
    asm = binary.asm
    ir = _text(asm.get("gcuir", ""))
    llvm = _text(asm.get("llir", ""))
    ttir = _text(asm.get("ttir", ""))
    counts = {k: ir.count(k) for k in (
        "memref.dma_start", "memref.dma_wait", "gcu.matmul",
        "memref_ext.transpose_start", "scf.for", "memref.load")}
    counts["ttir_tt_trans"] = ttir.count('"tt.trans"')
    counts["ttir_tt_dot"] = ttir.count('"tt.dot"')
    return {
        "hash": getattr(binary, "hash", None),
        "shared": getattr(binary.metadata, "shared", None),
        "counts": counts,
        "convgen": sorted(set(re.findall(r"callee = @(tops_convgen[^, >]+)", llvm))),
        "llir_dte_calls": len(re.findall(r"callee = @[^\n]*dte", llvm)),
    }


def struct_ir(rec):
    """Lowering identity that is stable across process/cache salts."""
    if rec is None:
        return None
    return {"counts": rec["counts"], "convgen": rec["convgen"], "shared": rec["shared"]}


def page_loop_signature(binary):
    """Ordered op-name sequence of the innermost page loop (lowering identity)."""
    if binary is None:
        return None
    lines = _text(binary.asm.get("gcuir", "")).splitlines()
    starts = [i for i, l in enumerate(lines) if '"scf.for"' in l]
    if not starts:
        return None
    begin = starts[-1]
    body, depth, indent0 = [], 0, None
    for line in lines[begin:]:
        s = line.strip()
        if indent0 is None:
            indent0 = len(line) - len(line.lstrip())
        if s.startswith("^bb0"):
            body.append("bb0")
            continue
        if s.startswith('"') and s.count('"') >= 2:
            body.append(s.split('"')[1])
    return body


# --------------------------------------------------------------------------
# --verify
# --------------------------------------------------------------------------
def run_verify(device):
    if device != DEVICE:
        print(f"FATAL: --device must be {DEVICE}")
        return 3
    g = gate(device)
    torch.gcu.set_device(device)
    env = env_record(device, g)
    print("[env]", json.dumps(env, ensure_ascii=False))
    bench = load_bench()
    data, shape = build_case(bench, SHAPE_INDEX)
    batch, qlen, seq, kvh, nh = shape
    print(f"[case] PREFILL_SHAPES[{SHAPE_INDEX}] = {shape} D=128 page=128 TopK=16")

    ok = True
    tk = score_topk(bench, data, qlen, seq, kvh)
    torch.gcu.synchronize()

    # ---- 1. mapping actually used (never identity) -----------------------
    bt = data.block_table.cpu()
    tkc = tk.cpu()
    sel = tkc[0][tkc[0] >= 0]
    phys = bt[0][sel]
    n_slots = int(tkc.shape[-1])
    n_logical = int(sel.numel())
    n_diff = int((phys != sel).sum())
    table_identity = bool(torch.equal(bt[0], torch.arange(bt.shape[1], dtype=bt.dtype)))
    mapping = {
        "topk_shape": list(tkc.shape),
        "topk_sha256": sha256(tk),
        "block_table_sha256": sha256(data.block_table),
        "slots": n_slots,
        "logical_ids_row0": tkc[0, 0].tolist(),
        "physical_ids_row0": bt[0][tkc[0, 0]].tolist(),
        "logical_ne_physical": f"{n_diff}/{n_logical}",
        "block_table_is_identity": table_identity,
        "distinct_logical_row0": len(set(tkc[0, 0].tolist())),
        "distinct_physical_row0": len(set(bt[0][tkc[0, 0]].tolist())),
        "topk_sha_matches_pristine": sha256(tk) == PRISTINE["topk_sha256"],
        "block_table_matches_pristine": sha256(data.block_table) == PRISTINE["block_table_sha256"],
    }
    print("[mapping]", json.dumps(mapping, ensure_ascii=False))
    ok &= bool(n_diff == n_logical)
    ok &= not table_identity
    ok &= mapping["topk_sha_matches_pristine"] and mapping["block_table_matches_pristine"]

    # ---- 2. bitwise baseline vs candidate, on both entry points ----------
    outs, routes, irs, physbuf = {}, {}, {}, {}
    for arm in ("baseline", "candidate"):
        with arms(arm):
            o = torch.empty_like(data.q)
            log, bins, ph = capture_launch(bench, data, o, qlen, seq, kvh)
            outs[arm] = o.cpu()
            routes[arm] = log
            irs[arm] = {k: ir_summary(v) for k, v in bins.items()}
            physbuf[arm] = (ph.get("ptr"), ph.get("use"))
            print(f"[launch/{arm}]", json.dumps(log))
            print(f"[ir/{arm}]", json.dumps(irs[arm], ensure_ascii=False))
    bit = diff_stats(outs["baseline"], outs["candidate"])
    all_finite = bool(torch.isfinite(outs["candidate"]).all())
    nonzero = int(torch.count_nonzero(outs["candidate"]))
    print("[bitwise]", json.dumps(dict(bit, all_finite=all_finite, nonzero=nonzero)))
    ok &= bit["max_abs"] == 0.0 and bit["nonzero"] == 0
    ok &= all_finite and nonzero > 0

    # route shape: same two kernels, only USE_PHYS_PAGE differs
    rb = routes["baseline"][-1]
    rc = routes["candidate"][-1]
    route_ok = (rb["kernel"] == rc["kernel"] == "_gqa_sparse_fwd_direct"
                and rb["USE_PACKED_K"] is True and rc["USE_PACKED_K"] is True
                and rb["EARLY_V_DMA"] is True and rc["EARLY_V_DMA"] is True
                and rb["USE_PHYS_PAGE"] is False and rc["USE_PHYS_PAGE"] is True)
    print("[route]", json.dumps(dict(route_ok=route_ok, baseline=rb, candidate=rc)))
    ok &= route_ok

    # ---- 3. resolver output equals block_table[topk] in TopK slot order --
    ptr, used = physbuf["candidate"]
    resolved_ok = False
    if used and isinstance(ptr, torch.Tensor):
        exp = bt[0][tkc[0].clamp(min=0)].to(torch.int32)
        got = ptr.cpu()
        same = bool(torch.equal(got[0], exp))
        resolved_ok = same
        print("[resolver]", json.dumps({
            "shape": list(got.shape), "dtype": str(got.dtype),
            "equals_torch_gather": same,
            "row0_first8": got[0, 0, :8].tolist(),
            "expected_first8": exp[0, :8].tolist(),
            "slots_checked": int((tkc[0] >= 0).sum()),
        }))
        ok &= same
    else:
        print("[resolver] FATAL: candidate did not expose the resolved buffer")
        ok = False

    # ---- 4. independent FP32 reference ----------------------------------
    rows = sorted({0, 1, qlen // 2, qlen - 1})
    ref = torch_reference(data, tk, rows, data.sm_scale, seq, nh)
    refs = {}
    for arm in ("baseline", "candidate"):
        got = outs[arm][rows]
        refs[arm] = diff_stats(ref, got)
    print("[reference]", json.dumps(refs, ensure_ascii=False))
    ok &= refs["baseline"]["max_abs"] < 2e-3 and refs["candidate"]["max_abs"] < 2e-3
    ok &= refs["baseline"]["cosine"] > 0.999 and refs["candidate"]["cosine"] > 0.999
    ok &= refs["baseline"] == refs["candidate"]

    # ---- 5. falsifiable causal mask, candidate arm ----------------------
    masks = {}
    for arm in ("baseline", "candidate"):
        with arms(arm):
            masks[arm] = mask_check(bench, data, tk, outs[arm], qlen, kvh,
                                    [0, 1, qlen // 2, qlen - 1])
    print("[mask]", json.dumps(masks, ensure_ascii=False))
    ok &= all(v["enforced"] for v in masks["candidate"].values())
    ok &= masks["baseline"] == masks["candidate"]

    # ---- 6. existing short-shape bypass (grouped8) ----------------------
    sdata, sshape = build_case(bench, BYPASS_INDEX)
    souts, sroutes = {}, {}
    for arm in ("baseline", "candidate"):
        with arms(arm):
            o = torch.empty_like(sdata.q)
            log, bins, _ = capture_launch(bench, sdata, o, sshape[1], sshape[2], sshape[3])
            souts[arm] = o.cpu()
            sroutes[arm] = (log, {k: ir_summary(v) for k, v in bins.items()})
    sbit = diff_stats(souts["baseline"], souts["candidate"])
    sroute = [r["kernel"] for r in sroutes["candidate"][0]]
    bypass = {
        "shape": sshape,
        "route": sroute,
        "grouped8": sroute == ["_gqa_sparse_fwd_grouped8"],
        "bitwise": sbit,
        "sha256": sha256(souts["candidate"]),
        "sha_matches_pristine": sha256(souts["candidate"]) == PRISTINE["short_out_sha256"],
        "ir_matches_pristine":
            struct_ir(sroutes["baseline"][1]["_gqa_sparse_fwd_grouped8"])
            == struct_ir(sroutes["candidate"][1]["_gqa_sparse_fwd_grouped8"])
            == PRISTINE["grouped8_ir"],
    }
    print("[bypass]", json.dumps(bypass, ensure_ascii=False))
    ok &= bypass["grouped8"] and sbit["max_abs"] == 0.0 and bypass["sha_matches_pristine"]
    ok &= bypass["ir_matches_pristine"]

    # ---- 7. default production route untouched --------------------------
    old = {k: os.environ.pop(k, None) for k in (PACKED_ENV, PHYS_ENV)}
    try:
        o = torch.empty_like(data.q)
        log, bins, _ = capture_launch(bench, data, o, qlen, seq, kvh)
    finally:
        for k, v in old.items():
            if v is not None:
                os.environ[k] = v
    dflt = {
        "route": log,
        "sha256": sha256(o.cpu()),
        "sha_matches_pristine": sha256(o.cpu()) == PRISTINE["chain_out_sha256"],
        "ir": ir_summary(bins["_gqa_sparse_fwd_direct"]),
        "ir_matches_pristine":
            struct_ir(ir_summary(bins["_gqa_sparse_fwd_direct"])) == PRISTINE["direct_ir_default"],
    }
    print("[default-route]", json.dumps(dflt, ensure_ascii=False))
    ok &= dflt["sha_matches_pristine"] and dflt["ir_matches_pristine"]

    # ---- 8. lowering unchanged outside the page-address load -----------
    pb = page_loop_signature(bins.get("_gqa_sparse_fwd_direct")) if False else None
    packed_ir_ok = {
        a: struct_ir(irs[a]["_gqa_sparse_fwd_direct"]) == PRISTINE["direct_ir_packed"]
        for a in ("baseline", "candidate")
    }
    print("[packed-ir-vs-pristine]", json.dumps(packed_ir_ok))
    ok &= all(packed_ir_ok.values())

    loop = {
        "baseline": irs["baseline"]["_gqa_sparse_fwd_direct"]["counts"],
        "candidate": irs["candidate"]["_gqa_sparse_fwd_direct"]["counts"],
        "convgen_baseline": irs["baseline"]["_gqa_sparse_fwd_direct"]["convgen"],
        "convgen_candidate": irs["candidate"]["_gqa_sparse_fwd_direct"]["convgen"],
        "shared_baseline": irs["baseline"]["_gqa_sparse_fwd_direct"]["shared"],
        "shared_candidate": irs["candidate"]["_gqa_sparse_fwd_direct"]["shared"],
    }
    same_lowering = (loop["baseline"] == loop["candidate"]
                     and loop["convgen_baseline"] == loop["convgen_candidate"]
                     and loop["shared_baseline"] == loop["shared_candidate"] == 0)
    print("[lowering]", json.dumps(dict(loop, unchanged=same_lowering), ensure_ascii=False))
    ok &= same_lowering

    print("[verify]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# --benchmark
# --------------------------------------------------------------------------
KERNELS = ("_gqa_sparse_common_prefix", "_gqa_sparse_fwd_direct")


def kernel_spans(bench, data, tk, qlen, kvh, reps=12):
    """Per-kernel time inside the operator, so the launch that absorbed the
    gather is visible separately from the kernel that stopped doing it."""
    raw = {}
    for arm in ("baseline", "candidate"):
        with arms(arm):
            call_sparse(bench, data, tk, torch.empty_like(data.q), qlen, kvh)
            torch.gcu.synchronize()
            events, cur = {}, [None]
            orig = triton.runtime.jit.JITFunction.run

            def hook(inst, *a, **kw):
                name = getattr(inst, "__name__", "")
                if name not in KERNELS or kw.get("warmup", False):
                    return orig(inst, *a, **kw)
                beg = torch.gcu.Event(enable_timing=True)
                end = torch.gcu.Event(enable_timing=True)
                beg.record()
                ret = orig(inst, *a, **kw)
                end.record()
                events.setdefault(name, []).append((beg, end))
                return ret

            triton.runtime.jit.JITFunction.run = hook
            try:
                for _ in range(reps):
                    call_sparse(bench, data, tk, torch.empty_like(data.q), qlen, kvh)
                torch.gcu.synchronize()
            finally:
                triton.runtime.jit.JITFunction.run = orig
            raw[arm] = {k: [b.elapsed_time(e) for b, e in v] for k, v in events.items()}
    out = {}
    for name in KERNELS:
        if name not in raw["baseline"] or name not in raw["candidate"]:
            continue
        b, c = raw["baseline"][name], raw["candidate"][name]
        out[name] = {"median_baseline": statistics.median(b),
                     "median_candidate": statistics.median(c),
                     "delta_ms": statistics.median(c) - statistics.median(b),
                     "samples_baseline": b, "samples_candidate": c}
    return out


def timed(fn, sync=True):
    start = torch.gcu.Event(enable_timing=True)
    end = torch.gcu.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    if sync:
        end.synchronize()
    return start.elapsed_time(end)


def run_benchmark(device, rounds):
    if device != DEVICE:
        print(f"FATAL: --device must be {DEVICE}")
        return 3
    g = gate(device)
    torch.gcu.set_device(device)
    env = env_record(device, g)
    print("[env]", json.dumps(env, ensure_ascii=False))
    bench = load_bench()
    data, shape = build_case(bench, SHAPE_INDEX)
    batch, qlen, seq, kvh, nh = shape
    print(f"[case] PREFILL_SHAPES[{SHAPE_INDEX}] = {shape} D=128 page=128 TopK=16 seed=6")

    tk = score_topk(bench, data, qlen, seq, kvh)
    torch.gcu.synchronize()
    print("[case-hashes]", json.dumps({
        "topk_sha256": sha256(tk),
        "block_table_sha256": sha256(data.block_table),
        "q_sha256": sha256(data.q),
        "kv_sha256": sha256(data.kv_cache),
    }))

    outs = {a: torch.empty_like(data.q) for a in ("baseline", "candidate")}

    def chain(arm):
        with arms(arm):
            run_chain(bench, data, outs[arm], qlen, seq, kvh)

    def attn(arm):
        with arms(arm):
            call_sparse(bench, data, tk, outs[arm], qlen, kvh)

    def score_topk_stage(arm):
        with arms(arm):
            score_topk(bench, data, qlen, seq, kvh)

    # "attn" is the whole attention operator (prefix launch + direct launch);
    # per-kernel attribution is measured separately by kernel_spans().
    stages = {"chain": chain, "attn": attn, "score_topk": score_topk_stage}

    # cold compile + warmup, one arm at a time (never inside a sample)
    for arm in ("baseline", "candidate"):
        for _ in range(3):
            chain(arm)
    torch.gcu.synchronize()

    # equivalence gate before any number is reported (uses its own buffers,
    # so the timed device buffers above are never replaced by CPU copies)
    chk = {}
    for arm in ("baseline", "candidate"):
        with arms(arm):
            o = torch.empty_like(data.q)
            run_chain(bench, data, o, qlen, seq, kvh)
            torch.gcu.synchronize()
            chk[arm] = o.cpu()
    pre = diff_stats(chk["baseline"], chk["candidate"])
    print("[pre-check]", json.dumps(dict(pre, finite=bool(torch.isfinite(chk["candidate"]).all()))))
    if pre["max_abs"] != 0.0:
        print("[benchmark] FATAL: arms are not bit-identical; refusing to report timings")
        return 1

    samples = {s: {a: [] for a in ("baseline", "candidate")} for s in stages}
    for r in range(rounds):
        order = ("baseline", "candidate") if r % 2 == 0 else ("candidate", "baseline")
        for stage, fn in stages.items():
            for arm in order:
                ms = timed(lambda: fn(arm))
                samples[stage][arm].append(ms)
                print(f"[sample] {json.dumps({'round': r, 'stage': stage, 'arm': arm, 'ms': ms})}",
                      flush=True)

    summary = {}
    for stage in stages:
        b = samples[stage]["baseline"]
        c = samples[stage]["candidate"]
        mb, mc = statistics.median(b), statistics.median(c)
        summary[stage] = {
            "baseline": b, "candidate": c,
            "median_baseline": mb, "median_candidate": mc,
            "delta_ms": mc - mb, "speedup": mb / mc,
            "candidate_faster_rounds": sum(1 for x, y in zip(c, b) if x < y),
            "rounds": rounds,
        }
    print("[summary]", json.dumps(summary))

    spans = kernel_spans(bench, data, tk, qlen, kvh)
    print("[kernel-spans]", json.dumps(spans))
    for name, v in spans.items():
        print(f"[span] {name}: baseline {v['median_baseline']:.4f} ms | "
              f"candidate {v['median_candidate']:.4f} ms | delta {v['delta_ms']:+.4f} ms")
    for stage, s in summary.items():
        print(f"[result] {stage}: baseline {s['median_baseline']:.4f} ms | "
              f"candidate {s['median_candidate']:.4f} ms | "
              f"delta {s['delta_ms']:+.4f} ms | {s['speedup']:.4f}x | "
              f"candidate faster in {s['candidate_faster_rounds']}/{rounds} rounds")
    verdict = ("candidate wins" if summary["chain"]["delta_ms"] < 0
               else "no end-to-end gain")
    for name, v in spans.items():
        print(f"[span-result] {name}: {v['delta_ms']:+.4f} ms")
    print(f"[verdict] chain {verdict}")
    post = gate_soft(device)
    print("[post-gate]", json.dumps(post))
    if post["foreign_pids"]:
        print("[benchmark] WARNING: a foreign process appeared during the run")
    print("[benchmark] PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--rounds", type=int, default=7)
    a = ap.parse_args()
    if a.verify == a.benchmark:
        print("FATAL: pass exactly one of --verify / --benchmark")
        return 2
    if a.verify:
        return run_verify(a.device)
    if a.rounds < 3:
        print("FATAL: --benchmark needs --rounds >= 3")
        return 2
    if a.rounds < 7:
        print("[warn] fewer than 7 rounds is not enough here: one cold first round "
              "or one outlier moves the median by ~0.1-0.3 ms, far above the "
              "0.03 ms effect being measured")
    return run_benchmark(a.device, a.rounds)


if __name__ == "__main__":
    sys.exit(main())
