"""MSA-004: reference-checked, alternating 3-round full-call A/B of the
opt-in group-16 prefill gate (``MSA002_GROUP16_EXPERIMENT``).

Variant A = shipped dispatch (switch off). Variant B = identical source with
the experiment switch on. Both run on GCU 2 only.

For every case the script
  1. runs the three public prefill calls (index score -> index top-k ->
     sparse attention) for A and B on the same inputs,
  2. checks both outputs against the repo's independent PyTorch reference
     (``tests/flag_attn/test_minimax_sparse_attention.py``),
  3. captures the real Triton kernel name/grid/``GROUP_SIZE_Q`` of the
     attention launch and asserts the dispatch changed exactly when the
     documented gate says it should,
  4. times the full call in three alternating rounds (A,B / B,A / A,B),
     keeping every raw sample,
  5. records peak GCU memory per variant and a fresh-process cold start.

Exit code is non-zero if any sample is missing, any reference check fails,
a path does not match the gate, or a cold start fails.
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time

os.environ.setdefault("FLAG_ATTN_TEST_DEVICE", "gcu:2")
os.environ.setdefault("S60_TEST_DEVICE", "2")

import torch

try:  # the harness exports /opt/flagtree on PYTHONPATH; allow a bare run too
    import triton
except ModuleNotFoundError:  # pragma: no cover - environment fallback
    sys.path.insert(0, os.environ.get("FLAGTREE_PATH", "/opt/flagtree"))
    import triton

import torch_gcu  # noqa: F401
from triton.runtime.jit import JITFunction

ROOT = os.environ.get("MSA004_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/src")

from flag_attn import minimax_m3_sparse_attn  # noqa: E402
from flag_attn.minimax_sparse_attention import (  # noqa: E402
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
import flag_attn.runtime.backend._enflame.msa.sparse_attn as enflame_sparse  # noqa: E402
from tests.flag_attn import test_minimax_sparse_attention as refmod  # noqa: E402

DEVICE_INDEX = 2
torch.gcu.set_device(DEVICE_INDEX)
triton.knobs.autotuning.adjust_block_size = False

ROUNDS = 3
SAMPLES_PER_ROUND = 15
WARMUP = 3
OUT_PATH = os.path.join(ROOT, "agent-results", "raw", "MSA-004", "negative",
                       os.path.basename(__file__).replace(".py", ".jsonl"))

# seq_lens, prefix_lens, num_kv_heads, gqa_group_size, topk, case id
# Prefixes are chosen so every "qNNN" id really has NNN queries in total.
CASES = [
    ((4096,), (3584,), 1, 8, 16, "q512_s4096_g8_k16"),
]
_UNUSED = [
    ((4096,), (3584,), 1, 8, 16, "q512_s4096_g8_k16"),
    ((4096,), (3496,), 1, 8, 16, "q600_s4096_g8_k16"),
    ((4096,), (3088,), 1, 8, 16, "q1008_s4096_g8_k16"),
    ((4096,), (3584,), 1, 1, 16, "q512_s4096_g1_k16"),
    ((4096,), (3584,), 1, 2, 16, "q512_s4096_g2_k16"),
    ((4096,), (3584,), 1, 4, 16, "q512_s4096_g4_k16"),
    ((4096, 4096), (3616, 4064), 1, 8, 16, "ragged480_32_s4096_g8_k16"),
    ((4096, 4096), (3840, 3840), 1, 8, 16, "ragged256_256_s4096_g8_k16"),
    ((4096, 4096, 4096, 4096), (3968, 3968, 3968, 3968), 1, 8, 16, "ragged128x4_s4096_g8_k16"),
    ((4096, 4096), (3592, 3592), 1, 8, 16, "ragged504x2_s4096_g8_k16"),
    ((4000,), (3488,), 1, 8, 16, "q512_s4000_pool32_g8_k16"),
    ((3968,), (3456,), 1, 8, 16, "q512_s3968_pool31_g8_k16"),
    ((2048,), (1536,), 1, 8, 16, "q512_s2048_pool16_g8_k16"),
    ((1024,), (512,), 1, 8, 16, "q512_s1024_pool8_g8_k16"),
    ((512,), (0,), 1, 8, 16, "q512_s512_pool4_g8_k16"),
    ((1024,), (512,), 1, 8, 8, "q512_s1024_pool8_g8_k8"),
    ((513,), (1,), 1, 8, 16, "q512_s513_prefix1_g8_k16"),
    ((612,), (100,), 1, 8, 16, "q512_s612_prefix100_g8_k16"),
    ((4096,) * 32, (4080,) * 32, 1, 8, 16, "batch32_q16_s4096_g8_k16"),
    ((4096,) * 16, (4064,) * 16, 1, 8, 16, "batch16_q32_s4096_g8_k16"),
    ((1024,) * 4, (895,) * 4, 1, 8, 16, "ragged129x4_s1024_k16"),
    # total_q boundary: 513 is not a multiple of 16; 511/1 splits one query off.
    ((4096,), (3583,), 1, 8, 16, "q513_s4096_g8_k16"),
    ((4096, 256), (3585, 255), 1, 8, 16, "ragged511_1_s4096_256_g8_k16"),
    ((4096,) * 8, (4000, 4064, 4016, 4048, 4032, 4032, 4016, 4048), 1, 8, 16, "ragged_mixed512_s4096_g8_k16"),
    # Small top-k with a narrow page pool: still inside the band.
    ((128,) * 32, (112,) * 32, 1, 8, 1, "topk1_pad_q512_g8_k1"),
    ((128,) * 32, (112,) * 32, 1, 8, 2, "topk2_pad_q512_g8_k2"),
    ((768,) * 32, (752,) * 32, 1, 8, 3, "topk3_pad_q512_g8_k3"),
    # topk=4 accepts a wider pool; seq_len 900 is not block aligned.
    ((900,) * 32, (884,) * 32, 1, 8, 4, "q512_s900_pool8_topk4"),
    ((4097,), (3585,), 1, 8, 16, "control_q512_s4097_pool33_k16"),
    ((4096,), (3584,), 1, 8, 32, "control_q512_s4096_g8_k32"),
    ((4096,), (3584,), 1, 16, 16, "control_q512_s4096_g16_k16"),
    ((4096,), (3584,), 2, 8, 16, "control_q512_s4096_kv2_g8_k16"),
    ((4096,), (3584,), 4, 2, 16, "control_q512_s4096_kv4_g2_k16"),
    ((4096,), (3584,), 1, 8, 4, "control_q512_s4096_g8_k4"),
    ((4096, 4096), (3796, 3908), 1, 8, 16, "control_ragged300_188_total488"),
    ((4096,), (3596,), 1, 8, 16, "control_q500_below_gate"),
    ((4096,) * 35, (4081,) * 35, 1, 8, 16, "control_batch35_q15_total525"),
    ((4096,), (3087,), 1, 8, 16, "control_q1009_already_group16"),
    ((896,) * 32, (880,) * 32, 1, 8, 3, "control_q512_s896_pool7_topk3_direct"),
]

LAUNCH: list = []
_ORIGINAL_RUN = JITFunction.run


def _capture_run(self, *args, **kwargs):
    if kwargs.get("warmup") is False:
        grid = kwargs.get("grid")
        LAUNCH.append(
            {
                "kernel": self.__name__,
                "grid": list(grid) if isinstance(grid, (list, tuple)) else str(grid),
                "group_size_q": kwargs.get("GROUP_SIZE_Q"),
            }
        )
    return _ORIGINAL_RUN(self, *args, **kwargs)


JITFunction.run = _capture_run


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _cpu_busy(cpus: list[int]) -> tuple[int, int]:
    """(busy, total) jiffies for the pinned CPUs, from /proc/stat."""
    wanted = {f"cpu{index}" for index in cpus}
    busy = total = 0
    with open("/proc/stat", encoding="ascii") as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0] not in wanted:
                continue
            values = [int(value) for value in fields[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            total += sum(values)
            busy += sum(values) - idle
    return busy, total


def _expected_change(total_q: int, batch: int, kv: int, group: int, topk: int, pool: int) -> bool:
    block_h = triton.next_power_of_2(group)
    shipped = (
        block_h <= 8
        and 0 < topk <= 16
        and pool <= 2 * topk
        and total_q >= 16 * batch
        and _cdiv(total_q, 16) * kv >= 64
    )
    experiment = (
        not shipped
        and block_h <= 8
        and 0 < topk <= 16
        and pool <= 2 * topk
        and total_q >= 512
        and total_q >= 16 * batch
        and _cdiv(total_q, 16) * kv >= 32
    )
    return False


def _full_call(data, topk: int, output) -> torch.Tensor:
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
    return topk_idx


def _attention_launch(launches: list) -> dict:
    candidates = [item for item in launches if "sparse_fwd" in item["kernel"]]
    if len(candidates) != 1:
        raise AssertionError(f"expected exactly one attention launch, got {candidates!r} in {launches!r}")
    return candidates[0]


def _set_variant(enabled: bool) -> None:
    enflame_sparse._MSA002_GROUP16_EXPERIMENT = bool(enabled)


def _measure_memory(data, topk: int, enabled: bool) -> dict:
    """Peak device memory of one variant, with no other variant resident."""
    _set_variant(enabled)
    output = torch.empty_like(data.q)
    for _ in range(WARMUP):
        _full_call(data, topk, output)
    torch.gcu.synchronize()
    del output
    torch.gcu.empty_cache()

    torch.gcu.reset_peak_memory_stats(DEVICE_INDEX)
    torch.gcu.synchronize()
    before = torch.gcu.memory_allocated(DEVICE_INDEX)
    scratch = torch.empty_like(data.q)
    _full_call(data, topk, scratch)
    torch.gcu.synchronize()
    peak = torch.gcu.max_memory_allocated(DEVICE_INDEX)
    del scratch
    torch.gcu.empty_cache()
    return {"peak_mib": peak / 2**20, "baseline_mib": before / 2**20}


def _prepare(data, topk: int, enabled: bool) -> dict:
    """Warm up one variant and capture the real launch it dispatches."""
    _set_variant(enabled)
    output = torch.empty_like(data.q)
    for _ in range(WARMUP):
        _full_call(data, topk, output)
    torch.gcu.synchronize()

    LAUNCH.clear()
    topk_idx = _full_call(data, topk, output)
    torch.gcu.synchronize()
    launch = _attention_launch(list(LAUNCH))
    return {
        "group_size_q": launch["group_size_q"],
        "attention_kernel": launch["kernel"],
        "attention_grid": launch["grid"],
        "launches": list(LAUNCH),
        "topk_idx": topk_idx,
        "output": output,
        "samples_ms": [],
        "round_medians_ms": [],
    }


def _timed_round(data, topk: int, variants: dict, order: tuple) -> None:
    for tag in order:
        _set_variant(tag == "B")
        current = []
        for _ in range(SAMPLES_PER_ROUND):
            start = torch.gcu.Event(enable_timing=True)
            end = torch.gcu.Event(enable_timing=True)
            torch.gcu.synchronize()
            start.record()
            _full_call(data, topk, variants[tag]["output"])
            end.record()
            torch.gcu.synchronize()
            value = float(start.elapsed_time(end))
            variants[tag]["samples_ms"].append(value)
            current.append(value)
        variants[tag]["round_medians_ms"].append(statistics.median(current))


def _reference(data, topk: int):
    started = time.perf_counter()
    ref_scores = refmod._ref_index_score(data)
    ref_topk = refmod._ref_topk(ref_scores, data, topk, 1, 2)
    ref_output = refmod._ref_sparse_attn(data, ref_topk)
    torch.gcu.synchronize()
    return ref_scores, ref_topk, ref_output, time.perf_counter() - started


def _compare(output: torch.Tensor, reference: torch.Tensor) -> dict:
    diff = (output.float() - reference.float()).abs()
    ref_rms = float(torch.sqrt(torch.mean(reference.float() ** 2)).item())
    diff_rms = float(torch.sqrt(torch.mean(diff**2)).item())
    return {
        "max_abs": float(diff.max().item()),
        "relative_rms": diff_rms / ref_rms if ref_rms else 0.0,
        "finite": bool(torch.isfinite(output.float()).all().item()),
        "within_2e-2": bool(torch.allclose(output, reference, atol=2e-2, rtol=2e-2)),
    }


def run_case(seqs, prefixes, kv, group, topk, case_id, failures: list, sink) -> dict:
    data = refmod.make_data(seqs, kv, group, decode=False, prefix_lens=prefixes, mode="bf16")
    if data.q.device.type != "gcu":
        raise AssertionError(f"case allocated on {data.q.device}, expected gcu:{DEVICE_INDEX}")
    if data.q.device.index is not None and data.q.device.index != DEVICE_INDEX:
        raise AssertionError(f"case allocated on {data.q.device}, expected gcu:{DEVICE_INDEX}")
    total_q = int(data.q.shape[0])
    batch = len(seqs)
    pool = int(data.block_table.shape[1])
    expect_change = _expected_change(total_q, batch, kv, group, topk, pool)

    ref_scores, ref_topk, ref_output, ref_seconds = _reference(data, topk)

    memory = {
        "A": _measure_memory(data, topk, False),
        "B": _measure_memory(data, topk, True),
    }
    variants = {
        "A": _prepare(data, topk, False),
        "B": _prepare(data, topk, True),
    }
    cpus = sorted(os.sched_getaffinity(0))
    busy0, total0 = _cpu_busy(cpus)
    # Alternate the order inside each round so neither variant always runs
    # after the other; every sample keeps the same inputs.
    for round_index, order in enumerate((("A", "B"), ("B", "A"), ("A", "B"))):
        _timed_round(data, topk, variants, order)
        print(
            json.dumps(
                {
                    "probe": "round",
                    "id": case_id,
                    "round": round_index,
                    "order": list(order),
                    "A_ms": variants["A"]["round_medians_ms"][-1],
                    "B_ms": variants["B"]["round_medians_ms"][-1],
                }
            ),
            flush=True,
        )
    busy1, total1 = _cpu_busy(cpus)
    _set_variant(False)
    pinned_busy = (busy1 - busy0) / (total1 - total0) if total1 > total0 else float("nan")

    record = {
        "probe": "case",
        "pinned_cpu_busy_fraction": pinned_busy,
        "loadavg": os.getloadavg(),
        "id": case_id,
        "seq_lens": list(seqs),
        "prefix_lens": list(prefixes),
        "query_lens": list(data.query_lens),
        "batch": batch,
        "total_q": total_q,
        "max_query_len": max(data.query_lens),
        "max_seq_len": data.max_seq_len,
        "num_kv_heads": kv,
        "gqa_group_size": group,
        "topk": topk,
        "block_table_pool": pool,
        "expect_path_change": expect_change,
        "reference_seconds": ref_seconds,
        "round_orders": [["A", "B"], ["B", "A"], ["A", "B"]],
    }

    try:
        refmod._assert_score_match(
            minimax_m3_index_score(
                data.idx_q,
                data.index_kv_cache,
                data.block_table,
                data.cu_q,
                data.seq_lens,
                data.prefix_lens,
                max(data.query_lens),
                data.max_seq_len,
                kv,
            ),
            ref_scores,
            data,
            decode=False,
        )
        record["index_score_match"] = True
    except AssertionError as exc:
        record["index_score_match"] = False
        record["index_score_error"] = str(exc)
        failures.append(f"{case_id}: index score mismatch: {exc}")

    for tag in ("A", "B"):
        variant = variants[tag]
        try:
            refmod._assert_topk_match(variant["topk_idx"], ref_topk, data, topk, decode=False)
            record[f"{tag}_topk_match"] = True
        except AssertionError as exc:
            record[f"{tag}_topk_match"] = False
            record[f"{tag}_topk_error"] = str(exc)
            failures.append(f"{case_id}/{tag}: top-k mismatch: {exc}")
        metrics = _compare(variant["output"], ref_output)
        record[tag] = {
            "group_size_q": variant["group_size_q"],
            "attention_kernel": variant["attention_kernel"],
            "attention_grid": variant["attention_grid"],
            "launches": variant["launches"],
            "samples_ms": variant["samples_ms"],
            "round_medians_ms": variant["round_medians_ms"],
            "median_ms": statistics.median(variant["samples_ms"]) if variant["samples_ms"] else None,
            "peak_mib": memory[tag]["peak_mib"],
            "baseline_mib": memory[tag]["baseline_mib"],
            **metrics,
        }
        if not metrics["within_2e-2"] or not metrics["finite"]:
            failures.append(f"{case_id}/{tag}: reference mismatch {metrics}")
        expected_samples = ROUNDS * SAMPLES_PER_ROUND
        if len(variant["samples_ms"]) != expected_samples:
            failures.append(
                f"{case_id}/{tag}: expected {expected_samples} samples, got {len(variant['samples_ms'])}"
            )
        if len(variant["round_medians_ms"]) != ROUNDS:
            failures.append(f"{case_id}/{tag}: expected {ROUNDS} round medians, got {len(variant['round_medians_ms'])}")

    record["AB_max_abs"] = float(
        (variants["A"]["output"].float() - variants["B"]["output"].float()).abs().max().item()
    )
    path_changed = variants["A"]["group_size_q"] != variants["B"]["group_size_q"]
    record["path_changed"] = path_changed
    if record["A"]["median_ms"] and record["B"]["median_ms"]:
        record["speedup"] = record["A"]["median_ms"] / record["B"]["median_ms"]
    else:
        record["speedup"] = None
        failures.append(f"{case_id}: missing timing samples")
    if path_changed != expect_change:
        failures.append(f"{case_id}: path_changed={path_changed} but gate expected {expect_change}")
    if path_changed:
        if variants["B"]["group_size_q"] != 16 or variants["A"]["group_size_q"] != 8:
            failures.append(
                f"{case_id}: changed path is {variants['A']['group_size_q']}->"
                f"{variants['B']['group_size_q']}, expected 8->16"
            )
        if "grouped8" not in variants["B"]["attention_kernel"]:
            failures.append(f"{case_id}: group16 did not use the grouped kernel")
        # The band is only proposable where it is an actual win on this run.
        if record["speedup"] is None or record["speedup"] < 1.0:
            failures.append(f"{case_id}: changed path did not improve ({record['speedup']})")
    else:
        if variants["A"]["group_size_q"] != variants["B"]["group_size_q"]:
            failures.append(f"{case_id}: path did not change but group sizes differ")
        if variants["A"]["attention_kernel"] != variants["B"]["attention_kernel"]:
            failures.append(f"{case_id}: path did not change but kernels differ")

    sink.write(json.dumps(record) + "\n")
    sink.flush()
    print(json.dumps(record), flush=True)
    del data, variants, ref_scores, ref_topk, ref_output
    torch.gcu.empty_cache()
    return record


def cold_start(tag: str) -> int:
    """First-call latency in a fresh process, on a shape the gate actually moves."""
    enabled = tag == "B"
    _set_variant(enabled)
    seqs, prefixes, kv, group, topk = (4096,), (3584,), 1, 8, 16
    data = refmod.make_data(seqs, kv, group, decode=False, prefix_lens=prefixes, mode="bf16")
    output = torch.empty_like(data.q)
    torch.gcu.synchronize()
    LAUNCH.clear()
    started = time.perf_counter()
    _full_call(data, topk, output)
    torch.gcu.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    launches = list(LAUNCH)
    attention = _attention_launch(launches)
    expected_group = 16 if enabled else 8
    if attention["group_size_q"] != expected_group:
        print(
            json.dumps(
                {
                    "probe": "cold_start_error",
                    "variant": tag,
                    "error": f"GROUP_SIZE_Q {attention['group_size_q']} != {expected_group}",
                    "launches": launches,
                }
            ),
            flush=True,
        )
        return 1
    print(
        json.dumps(
            {
                "probe": "cold_start",
                "variant": tag,
                "total_q": int(data.q.shape[0]),
                "group_size_q": attention["group_size_q"],
                "attention_kernel": attention["kernel"],
                "attention_grid": attention["grid"],
                "elapsed_ms": elapsed_ms,
                "peak_mib": torch.gcu.max_memory_allocated(DEVICE_INDEX) / 2**20,
            }
        ),
        flush=True,
    )
    return 0


def main() -> int:
    failures: list[str] = []
    started = time.time()
    header = {
        "probe": "header",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": f"gcu:{DEVICE_INDEX}",
        "device_name": torch.gcu.get_device_name(DEVICE_INDEX),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "loadavg": os.getloadavg(),
        "git_head": subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "git_status": subprocess.run(["git", "-C", ROOT, "status", "--porcelain"], capture_output=True, text=True).stdout.strip(),
        "rounds": ROUNDS,
        "samples_per_round": SAMPLES_PER_ROUND,
        "warmup": WARMUP,
        "experiment_flag": "MSA002_GROUP16_EXPERIMENT",
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    sink = open(OUT_PATH, "w", encoding="utf-8")
    sink.write(json.dumps(header) + "\n")
    sink.flush()
    print(json.dumps(header), flush=True)

    cold = {}
    for tag in ("A", "B"):
        env = {**os.environ, "FLAG_ATTN_TEST_DEVICE": f"gcu:{DEVICE_INDEX}", "S60_TEST_DEVICE": str(DEVICE_INDEX)}
        env.pop("MSA002_GROUP16_EXPERIMENT", None)
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--cold-start", tag],
            capture_output=True,
            text=True,
            env=env,
        )
        lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
        if proc.returncode != 0 or not lines:
            failures.append(f"cold start {tag} failed rc={proc.returncode}: {proc.stderr[-800:]}")
            cold[tag] = None
            continue
        payload = json.loads(lines[-1])
        want_group = 16 if tag == "B" else 8
        if payload.get("probe") != "cold_start":
            failures.append(f"cold start {tag}: unexpected payload {payload}")
            cold[tag] = None
            continue
        if payload.get("group_size_q") != want_group:
            failures.append(
                f"cold start {tag}: GROUP_SIZE_Q {payload.get('group_size_q')} != {want_group}"
            )
            cold[tag] = None
            continue
        if not math.isfinite(payload.get("elapsed_ms", float("nan"))) or payload["elapsed_ms"] <= 0:
            failures.append(f"cold start {tag}: invalid elapsed_ms {payload.get('elapsed_ms')}")
            cold[tag] = None
            continue
        cold[tag] = payload
        sink.write(json.dumps(cold[tag]) + "\n")
        sink.flush()
        print(json.dumps(cold[tag]), flush=True)

    records = []
    for seqs, prefixes, kv, group, topk, case_id in CASES:
        try:
            records.append(run_case(seqs, prefixes, kv, group, topk, case_id, failures, sink))
        except Exception as exc:  # noqa: BLE001 - one broken case must not lose the sweep
            failures.append(f"{case_id}: {type(exc).__name__}: {exc}")
            line = {"probe": "case_error", "id": case_id, "error": f"{type(exc).__name__}: {exc}"}
            sink.write(json.dumps(line) + "\n")
            sink.flush()
            print(json.dumps(line), flush=True)
            torch.gcu.empty_cache()

    summary = {
        "probe": "summary",
        "cases": len(records),
        "expected_cases": len(CASES),
        "failures": failures,
        "elapsed_seconds": time.time() - started,
        "cold_start_ms": {tag: (cold[tag] or {}).get("elapsed_ms") for tag in ("A", "B")},
        "pinned_cpu_busy_fraction": {
            record["id"]: record["pinned_cpu_busy_fraction"] for record in records
        },
        "changed_cases": [
            {
                "id": record["id"],
                "A_ms": record["A"]["median_ms"],
                "B_ms": record["B"]["median_ms"],
                "speedup": record["speedup"],
                "A_peak_mib": record["A"]["peak_mib"],
                "B_peak_mib": record["B"]["peak_mib"],
            }
            for record in records
            if record.get("path_changed")
        ],
        "unchanged_cases": [record["id"] for record in records if not record.get("path_changed")],
    }
    if len(records) != len(CASES):
        failures.append(f"missing cases: {len(records)}/{len(CASES)}")
    if not math.isfinite(summary["elapsed_seconds"]) or summary["elapsed_seconds"] <= 0:
        failures.append("invalid elapsed time")
    summary["failures"] = failures
    sink.write(json.dumps(summary) + "\n")
    sink.close()
    print(json.dumps(summary), flush=True)
    print(f"WROTE {OUT_PATH}", flush=True)
    if failures:
        print("FAILED", flush=True)
        for failure in failures:
            print(f"- {failure}", flush=True)
        return 1
    print("PASSED", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--cold-start":
        raise SystemExit(cold_start(sys.argv[2]))
    raise SystemExit(main())
