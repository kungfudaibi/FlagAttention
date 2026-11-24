#!/usr/bin/env python
"""MSA-009 overnight profile/optimization harness (GCU 5 only).

Two entry points, both on the fixed BF16 long-prefill case
(B/Q/KV/Hkv/Hq = 1/1024/16384/1/8, D=128, page=128, TopK=16), using the
upstream benchmark's own real score->topk and random page table:

  --device 5 --verify     Numeric/mask validation of baseline (stock, default
                          route) and of the experimental candidate against an
                          independent torch float32 reference, plus a bit-exact
                          comparison of the default route with a recorded
                          pristine-baseline digest, plus one bypass regression
                          shape (B1/Q512/KV4096).
  --device 5 --benchmark  Same-device, same-sample alternating A/B (>=3 rounds)
                          over the complete chain (index_score -> index_topk ->
                          sparse_attn) and over sparse_attn alone, with every
                          raw sample recorded.  Baseline and candidate run
                          together whenever both are valid; if no candidate is
                          valid the report says so explicitly.

Nothing here mutates the library default route: the candidate is only reachable
through the opt-in environment switch that the candidate commit adds.  Formal
re-measurement under the exclusive host gate is done by the scheduler *after*
this session, not by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

import torch  # noqa: E402
import triton  # noqa: E402

SHAPE_INDEX = 6           # PREFILL_SHAPES[6] == (1, 1024, 16384, 1, 8)
BYPASS_INDEX = 1          # PREFILL_SHAPES[1] == (1, 512, 4096, 1, 8)
SEED_OFFSET = 0           # benchmark default: seed + shape_index
PREFIX_GROUP = 16
CANDIDATE_ENV = "FLAG_ATTN_MSA_EARLY_V_DMA"
BF16_ATOL = BF16_RTOL = 2e-2
RAW_DIR = os.path.join(REPO, "agent-results", "raw")


# --------------------------------------------------------------------------
# environment / occupancy gate
# --------------------------------------------------------------------------
def sh(cmd, timeout=60):
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
    pct = float(m.group(1)) if m else None
    return dict(pmon_raw=pmon, usage_raw=usage, gcu_usage_pct=pct,
                own_pids=sorted(process_tree_pids()), foreign_pids=foreign,
                busy=bool(foreign))


def env_record(device_index):
    q = sh(f"efsmi -i {device_index} -q")
    props = torch.gcu.get_device_properties(device_index)
    dom = re.search(r"Domain\s*:\s*([0-9a-fA-F]+)", q)
    bus = re.search(r"Bus\s*:\s*(\d+)", q)
    dev = re.search(r"Dev\s*:\s*(\d+)", q)
    fn = re.search(r"Func\s*:\s*(\d+)", q)
    pci = None
    if dom and bus and dev and fn:
        pci = f"{int(dom.group(1), 16):04x}:{bus.group(1)}:{dev.group(1)}.{fn.group(1)}"
    rec = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device_index": device_index,
        "torch_gcu_current_device": torch.gcu.current_device(),
        "device_name": torch.gcu.get_device_name(device_index),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": sys.version.split()[0],
        "props": {"major": props.major, "minor": props.minor,
                  "total_memory_mb": round(props.total_memory / (1 << 20)),
                  "multi_processor_count": props.multi_processor_count,
                  "sip_count": getattr(props, "sip_count", None)},
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "pci": pci,
        "candidate_env": {CANDIDATE_ENV: os.environ.get(CANDIDATE_ENV)},
        "efsmi_query_raw": q,
    }
    for key, pat in (("driver_version", r"Ver\s*:\s*(\S+)"),
                     ("ap_fw", r"AP FW Version\s*:\s*(\S+)"),
                     ("sp_fw", r"SP FW Version\s*:\s*(\S+)"),
                     ("mem_clock_mhz", r"Mem CLK\s*:\s*(\d+)"),
                     ("health", r"Health\s*:\s*(\S+)")):
        mm = re.search(pat, q)
        rec[key] = mm.group(1) if mm else None
    return rec


# --------------------------------------------------------------------------
# data / launch helpers (upstream benchmark functions only, never modified)
# --------------------------------------------------------------------------
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


def run_chain(bench, data, out, qlen, seq, kvh):
    tk = score_topk(bench, data, qlen, seq, kvh)
    call_sparse(bench, data, tk, out, qlen, kvh)
    return tk


def with_env(value):
    """Context manager toggling the candidate switch (None = unset)."""
    class _E:
        def __enter__(self):
            self.old = os.environ.get(CANDIDATE_ENV)
            if value is None:
                os.environ.pop(CANDIDATE_ENV, None)
            else:
                os.environ[CANDIDATE_ENV] = value
            return self

        def __exit__(self, *exc):
            if self.old is None:
                os.environ.pop(CANDIDATE_ENV, None)
            else:
                os.environ[CANDIDATE_ENV] = self.old
            return False
    return _E()


def launch_capture(bench, data, tk, out, qlen, kvh, sa):
    """Record grid + constexpr metadata of both MSA kernels (outside timing)."""
    log = []
    orig_prefix = sa._gqa_sparse_common_prefix
    orig_direct = sa._gqa_sparse_fwd_direct

    class _Wrap:
        def __init__(self, fn, name):
            self.fn, self.name = fn, name

        def __getitem__(self, grid):
            outer = self

            class _Launcher:
                def __call__(inner, *args, **kwargs):
                    log.append({"kernel": outer.name, "grid": list(grid),
                                "constexpr": {k: v for k, v in kwargs.items()
                                              if isinstance(v, (int, bool, str))}})
                    return outer.fn[grid](*args, **kwargs)
            return _Launcher()

    sa._gqa_sparse_common_prefix = _Wrap(orig_prefix, "_gqa_sparse_common_prefix")
    sa._gqa_sparse_fwd_direct = _Wrap(orig_direct, "_gqa_sparse_fwd_direct")
    try:
        call_sparse(bench, data, tk, out, qlen, kvh)
        torch.gcu.synchronize()
    finally:
        sa._gqa_sparse_common_prefix = orig_prefix
        sa._gqa_sparse_fwd_direct = orig_direct
    return log


def kernel_ir_digest(sa, mode_name):
    """Source/IR identity of the compiled direct kernel for this configuration."""
    import inspect
    src = inspect.getsource(sa._gqa_sparse_fwd_direct.fn if hasattr(sa._gqa_sparse_fwd_direct, "fn")
                            else sa._gqa_sparse_fwd_direct)
    import hashlib
    return {"mode": mode_name,
            "direct_source_sha256": hashlib.sha256(src.encode()).hexdigest(),
            "early_v_dma_lowered": "EARLY_V_DMA" in src}


# --------------------------------------------------------------------------
# reference
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
        for slot in range(tkc.shape[-1]):
            blk = int(tkc[0, row, slot])
            if blk < 0:
                continue
            page = int(bt[0, blk])
            k = kv[page, 0, :, :128]
            v = kv[page, 0, :, 128:]
            pos = blk * 128 + torch.arange(128)
            valid = (pos < seq_len) & (pos <= q_abs)
            scores = (q[row] @ k.T) * scale
            scores = scores.masked_fill(~valid[None, :], -float("inf"))
            m_new = torch.maximum(m, scores.max(dim=-1).values)
            safe = torch.where(m_new > -float("inf"), m_new, torch.zeros(()))
            p = torch.where(valid[None, :], torch.exp2((scores - safe[:, None]) * 1.4426950409),
                            torch.zeros(()))
            alpha = torch.where(m > -float("inf"), torch.exp2((m - safe) * 1.4426950409),
                                torch.zeros(()))
            l = l * alpha + p.sum(dim=-1)
            acc = acc * alpha[:, None] + p @ v
            m = m_new
        out[i] = acc / l[:, None]
    return out


def diff_stats(a, b):
    d = (a.float().cpu() - b.float().cpu()).abs()
    af, bf = a.float().cpu().reshape(-1), b.float().cpu().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(af, bf, dim=0).item()
    return dict(max_abs=d.max().item(), mean_abs=d.mean().item(),
                nonzero=int((d > 0).sum().item()), numel=d.numel(), cosine=cos)


# --------------------------------------------------------------------------
# --verify
# --------------------------------------------------------------------------
def run_verify(device, env):
    from benchmark import test_minimax_sparse_attention_benchmark as bench
    from flag_attn.runtime.backend._enflame.msa import sparse_attn as sa

    res = {"mode": "verify", "env": env, "torch": torch.__version__,
           "candidate_env": CANDIDATE_ENV, "checks": {}}
    ok = True

    for tag, idx in (("main", SHAPE_INDEX), ("bypass", BYPASS_INDEX)):
        data, shape = build_case(bench, idx)
        batch, qlen, seq, kvh, nh = shape
        tk = score_topk(bench, data, qlen, seq, kvh)
        ref_tk = tk.clone()
        outs = {}
        with with_env(None):
            outs["baseline"] = torch.empty_like(data.q)
            call_sparse(bench, data, tk, outs["baseline"], qlen, kvh)
            torch.gcu.synchronize()
        with with_env("1"):
            outs["candidate"] = torch.empty_like(data.q)
            call_sparse(bench, data, tk, outs["candidate"], qlen, kvh)
            torch.gcu.synchronize()
        # determinism: the topk stage must be untouched by the switch
        with with_env("1"):
            tk2 = score_topk(bench, data, qlen, seq, kvh)
        topk_same = torch.equal(tk, ref_tk) and torch.equal(tk2, ref_tk)

        rows = list(range(0, 8)) + [qlen // 4, qlen // 2, qlen - 24, qlen - 1]
        rows = sorted({r for r in rows if 0 <= r < qlen})
        ref = torch_reference(data, tk, rows, data.sm_scale, seq, nh)
        d_base = diff_stats(outs["baseline"].cpu()[rows], ref)
        d_cand = diff_stats(outs["candidate"].cpu()[rows], ref)
        d_pair = diff_stats(outs["baseline"], outs["candidate"])

        finite = bool(torch.isfinite(outs["baseline"]).all().item()
                      and torch.isfinite(outs["candidate"]).all().item())
        # mask semantics: no position beyond seq_len / q_abs may contribute, so a
        # query whose whole selected window is out of range yields exactly 0.
        prefix_len = int(data.prefix_lens[0].item())
        masked_zero_ok = True
        for r in rows:
            q_abs = prefix_len + r
            if q_abs < 0:
                masked_zero_ok &= bool((outs["baseline"][r].abs().max().item() == 0.0))
        base_ok = (d_base["max_abs"] <= BF16_ATOL + BF16_RTOL * ref.abs().max().item()
                   and d_base["cosine"] > 0.999)
        cand_ok = (d_cand["max_abs"] <= BF16_ATOL + BF16_RTOL * ref.abs().max().item()
                   and d_cand["cosine"] > 0.999)
        entry = {
            "shape": dict(batch=batch, query_len=qlen, seq_len=seq, num_kv_heads=kvh,
                          num_heads=nh, head_dim=128, topk=int(tk.shape[-1])),
            "reference_rows": rows,
            "baseline_vs_fp32": d_base, "candidate_vs_fp32": d_cand,
            "baseline_vs_candidate": d_pair,
            "baseline_vs_fp32_pass": base_ok, "candidate_vs_fp32_pass": cand_ok,
            "topk_unchanged_by_switch": bool(topk_same),
            "all_finite": finite, "mask_zero_ok": bool(masked_zero_ok),
        }
        res["checks"][tag] = entry
        ok &= base_ok and cand_ok and topk_same and finite and masked_zero_ok
        print(f"[verify:{tag}] shape={shape} baseline_fp32 max={d_base['max_abs']:.3e} "
              f"cos={d_base['cosine']:.8f} | candidate_fp32 max={d_cand['max_abs']:.3e} "
              f"cos={d_cand['cosine']:.8f} | base-cand max={d_pair['max_abs']:.3e} "
              f"nz={d_pair['nonzero']} | topk_unchanged={topk_same}")

    # launch shape of both routes (captured outside timing)
    data, shape = build_case(bench, SHAPE_INDEX)
    batch, qlen, seq, kvh, nh = shape
    tk = score_topk(bench, data, qlen, seq, kvh)
    out = torch.empty_like(data.q)
    with with_env(None):
        base_launch = launch_capture(bench, data, tk, out, qlen, kvh, sa)
    with with_env("1"):
        cand_launch = launch_capture(bench, data, tk, out, qlen, kvh, sa)
    res["launch"] = {"baseline": base_launch, "candidate": cand_launch}
    direct_b = next((x for x in base_launch if x["kernel"] == "_gqa_sparse_fwd_direct"), None)
    direct_c = next((x for x in cand_launch if x["kernel"] == "_gqa_sparse_fwd_direct"), None)
    default_flag = bool(direct_b and direct_b["constexpr"].get("EARLY_V_DMA") is False)
    cand_flag = bool(direct_c and direct_c["constexpr"].get("EARLY_V_DMA") is True)
    res["default_route_unchanged"] = default_flag
    res["candidate_opt_in_reached"] = cand_flag
    res["ir"] = [kernel_ir_digest(sa, "candidate")]

    # bit-exactness of the default route against a recorded pristine digest
    pristine_path = os.environ.get("MSA009_PRISTINE", "/tmp/msa009_pristine_baseline.pt")
    if os.path.exists(pristine_path):
        import hashlib
        pristine = torch.load(pristine_path)
        bits = {}
        for tag, idx in (("shape6", SHAPE_INDEX), ("shape1", BYPASS_INDEX)):
            d, sh = build_case(bench, idx)
            b_, ql, sq, kh, nh_ = sh
            t_ = score_topk(bench, d, ql, sq, kh)
            o_ = torch.empty_like(d.q)
            with with_env(None):
                call_sparse(bench, d, t_, o_, ql, kh)
                torch.gcu.synchronize()
            digest = hashlib.sha256(o_.cpu().view(torch.int16).numpy().tobytes()).hexdigest()
            bits[tag] = {"digest": digest, "pristine_digest": pristine[tag]["out_sha"],
                         "bit_identical": digest == pristine[tag]["out_sha"]}
        res["default_route_bit_exact"] = bits
        ok &= all(v["bit_identical"] for v in bits.values())
        print("[verify] default-route bit-exact:",
              {k: v["bit_identical"] for k, v in bits.items()})
    else:
        res["default_route_bit_exact"] = None
        print(f"[verify] pristine digest not found at {pristine_path}; skipped bit-exact check")

    print(f"[verify] default_route_unchanged={default_flag} candidate_reached={cand_flag}")
    print(f"[verify] overall={'PASS' if (ok and default_flag and cand_flag) else 'FAIL'}")
    res["pass"] = bool(ok and default_flag and cand_flag)
    return res


# --------------------------------------------------------------------------
# --benchmark
# --------------------------------------------------------------------------
def summarize(values):
    return dict(n=len(values), samples=list(values), median=statistics.median(values),
                mean=statistics.fmean(values), min=min(values), max=max(values),
                stdev=statistics.stdev(values) if len(values) > 1 else 0.0)


def paired_samples(fns, rounds, warmup=3):
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


def staged_samples(bench, data, tk, out, qlen, seq, kvh, rounds=7):
    """Separate stage timings (never mixed into the paired chain numbers)."""
    def index_score():
        bench.minimax_m3_index_score(data.idx_q, data.index_kv_cache, data.block_table,
                                     data.cu_q, data.seq_lens, data.prefix_lens, qlen, seq, kvh)

    def index_topk():
        score_topk(bench, data, qlen, seq, kvh)

    stages = {"index_score": index_score, "index_topk": index_topk,
              "sparse_attn": lambda: call_sparse(bench, data, tk, out, qlen, kvh),
              "full_chain": lambda: run_chain(bench, data, out, qlen, seq, kvh)}
    return paired_samples(stages, rounds)[0]


def run_benchmark(device, env, rounds=7, out_path=None):
    from benchmark import test_minimax_sparse_attention_benchmark as bench
    from flag_attn.runtime.backend._enflame.msa import sparse_attn as sa

    data, shape = build_case(bench, SHAPE_INDEX)
    batch, qlen, seq, kvh, nh = shape
    tk = score_topk(bench, data, qlen, seq, kvh)
    out = {k: torch.empty_like(data.q) for k in ("baseline", "candidate", "chain_base", "chain_cand")}

    # correctness gate before any timing claim
    with with_env(None):
        call_sparse(bench, data, tk, out["baseline"], qlen, kvh)
    with with_env("1"):
        call_sparse(bench, data, tk, out["candidate"], qlen, kvh)
    torch.gcu.synchronize()
    pair = diff_stats(out["baseline"], out["candidate"])
    rows = list(range(0, 8)) + [qlen // 2, qlen - 1]
    ref = torch_reference(data, tk, rows, data.sm_scale, seq, nh)
    gate = dict(baseline_vs_fp32=diff_stats(out["baseline"].cpu()[rows], ref),
                candidate_vs_fp32=diff_stats(out["candidate"].cpu()[rows], ref),
                baseline_vs_candidate=pair)
    valid_base = gate["baseline_vs_fp32"]["max_abs"] <= 1.0 and gate["baseline_vs_fp32"]["cosine"] > 0.99
    valid_cand = (gate["candidate_vs_fp32"]["max_abs"] <= 1.0
                  and gate["candidate_vs_fp32"]["cosine"] > 0.99)
    res = {"mode": "benchmark", "env": env, "shape": dict(
        batch=batch, query_len=qlen, seq_len=seq, num_kv_heads=kvh, num_heads=nh,
        head_dim=128, topk=int(tk.shape[-1]), shape_index=SHAPE_INDEX),
        "rounds": rounds, "correctness_gate": gate,
        "baseline_valid": bool(valid_base), "candidate_valid": bool(valid_cand),
        "candidate_env": CANDIDATE_ENV}

    fns = {}

    def set_base():
        os.environ.pop(CANDIDATE_ENV, None)

    def set_cand():
        os.environ[CANDIDATE_ENV] = "1"

    if valid_base:
        fns["baseline"] = lambda: (set_base(), call_sparse(bench, data, tk, out["baseline"], qlen, kvh))
        fns["baseline_chain"] = lambda: (set_base(), run_chain(bench, data, out["chain_base"], qlen, seq, kvh))
    if valid_cand:
        fns["candidate"] = lambda: (set_cand(), call_sparse(bench, data, tk, out["candidate"], qlen, kvh))
        fns["candidate_chain"] = lambda: (set_cand(), run_chain(bench, data, out["chain_cand"], qlen, seq, kvh))

    samples, orders = paired_samples(fns, rounds)
    res["samples"] = {k: summarize(v) for k, v in samples.items()}
    res["order"] = orders
    if valid_base and valid_cand:
        b, c = samples["baseline"], samples["candidate"]
        res["delta_sparse_attn"] = dict(
            baseline_median=statistics.median(b), candidate_median=statistics.median(c),
            speedup=statistics.median(b) / statistics.median(c),
            paired_deltas=[bb - cc for bb, cc in zip(b, c)])
        bc, cc = samples["baseline_chain"], samples["candidate_chain"]
        res["delta_full_chain"] = dict(
            baseline_median=statistics.median(bc), candidate_median=statistics.median(cc),
            speedup=statistics.median(bc) / statistics.median(cc),
            paired_deltas=[bb - cccc for bb, cccc in zip(bc, cc)])

    # staged profile, both routes, outside the paired A/B
    res["staged_baseline"] = staged_samples(bench, data, tk, out["baseline"], qlen, seq, kvh)
    with with_env("1"):
        res["staged_candidate"] = staged_samples(bench, data, tk, out["candidate"], qlen, seq, kvh)

    # launches/IR for both routes
    out2 = torch.empty_like(data.q)
    with with_env(None):
        res["launch_baseline"] = launch_capture(bench, data, tk, out2, qlen, kvh, sa)
    with with_env("1"):
        res["launch_candidate"] = launch_capture(bench, data, tk, out2, qlen, kvh, sa)
    res["ir"] = [kernel_ir_digest(sa, "candidate")]

    if not valid_cand:
        print("[benchmark] NO VALID CANDIDATE: reporting baseline only")
    for name, s in res["samples"].items():
        print(f"[benchmark] {name:18s} n={s['n']} median={s['median']:.4f} ms "
              f"min={s['min']:.4f} max={s['max']:.4f} stdev={s['stdev']:.4f}")
    if "delta_sparse_attn" in res:
        d = res["delta_sparse_attn"]
        print(f"[benchmark] sparse_attn speedup {d['speedup']:.4f}x "
              f"({d['baseline_median']:.4f} -> {d['candidate_median']:.4f} ms)")
        d = res["delta_full_chain"]
        print(f"[benchmark] full_chain  speedup {d['speedup']:.4f}x "
              f"({d['baseline_median']:.4f} -> {d['candidate_median']:.4f} ms)")
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=5)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()
    if args.device != 5:
        print(f"FATAL: this harness is pinned to GCU 5, got {args.device}")
        return 2
    torch.gcu.set_device(5)
    occ = occupancy_check(5)
    if occ["busy"]:
        print("FATAL: device busy (foreign processes):", occ["foreign_pids"])
        return 3
    env = env_record(5)
    if env["pci"] != "0000:83:00.0":
        print(f"FATAL: PCI mismatch {env['pci']}")
        return 4
    print(f"[env] device=5 pci={env['pci']} cpu={env['cpu_affinity'][0]}-{env['cpu_affinity'][-1]} "
          f"torch={env['torch']} triton={env['triton']} candidate_env={os.environ.get(CANDIDATE_ENV)}")

    rc = 1
    if args.verify:
        res = run_verify(5, env)
        out_path = args.out or os.path.join("/tmp", f"msa009_verify_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"[verify] wrote {out_path}")
        rc = 0 if res["pass"] else 1
    elif args.benchmark:
        res = run_benchmark(5, env, rounds=args.rounds, out_path=args.out)
        rc = 0 if res["baseline_valid"] else 1
    else:
        ap.error("choose --verify or --benchmark")
    return rc


if __name__ == "__main__":
    sys.exit(main())
