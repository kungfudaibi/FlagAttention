"""GLA-008: dtype / shape dispatch checks for the Enflame GLA parallel state.

Two layers:

* in-process numeric + route checks (fast, always run on GCU 4);
* a subprocess check of the formal benchmark entry point so a broken driver
  (missing samples, wrong device, silent fallback) fails here too.

The opt-in flags are read at import/launch time, so the in-process layer forks a
helper that sets them before importing FlagAttention.  Nothing here changes the
production default route: the default-route test asserts it explicitly.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PY = "/flagos/bin/python"
HELPER = ROOT / "agent-results/scripts/gla008_case.py"


def _run(script_args, env_extra=None, timeout=900):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}:/opt/flagtree"
    for k in ("FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP16_STATE",
              "FLAGGEMS_ENFLAME_GLA_EXPERIMENTAL_FP32_STATE",
              "FLAGGEMS_ENFLAME_GLA_PARALLEL_STATE",
              "FLAGGEMS_ENFLAME_GLA_PARALLEL_BWD_STATE"):
        env.pop(k, None)
    env["TRITON_F32_DEFAULT"] = "ieee"
    env.update(env_extra or {})
    run = subprocess.run([PY] + script_args, cwd=ROOT, env=env, text=True,
                         capture_output=True, timeout=timeout)
    rec = None
    for ln in run.stdout.splitlines():
        if ln.startswith("GLA008_JSON "):
            rec = json.loads(ln[len("GLA008_JSON "):])
    return run, rec


def _case(shape, dtype, gln, route, extra=None, tag="pytest"):
    K, V = shape
    args = [str(HELPER), "--B", "2", "--T", "256", "--H", "2", "--K", str(K),
            "--V", str(V), "--dtype", dtype, "--gln", str(gln),
            "--samples", "1", "--warmup", "0", "--no-timing", "--tag", tag]
    if route == "default":
        args += ["--exp-fp16", "0", "--exp-fp32", "0"]
    else:
        args += ["--exp-fp16", "1", "--exp-fp32", "1"]
    if route == "fallback":
        args.append("--force-fallback")
    return args + (extra or [])


@pytest.fixture(scope="module")
def preflight():
    run = subprocess.run(
        [PY, "-c",
         "import torch, torch_gcu; torch.gcu.set_device(4);"
         "print(torch.gcu.current_device());"
         "print(torch.gcu.current_stream(4).device)"],
        env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}:/opt/flagtree"},
        text=True, capture_output=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    lines = run.stdout.strip().splitlines()
    assert lines[0].strip() == "4", lines
    assert lines[-1].strip() == "gcu:4", lines
    return True


@pytest.mark.parametrize("shape", [(32, 64), (96, 128)])
@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
def test_fallback_matches_parallel(preflight, tmp_path, shape, dtype):
    """Both routes on the same commit must agree, on the real kernels."""
    ref = tmp_path / "ref.pt"
    fb_run, fb = _run(_case(shape, dtype, 1.0, "fallback",
                            ["--save", str(ref)]))
    assert fb_run.returncode == 0, fb_run.stderr[-3000:]
    assert fb and fb["exit"] == 0, (fb or {}).get("error")
    assert fb["device_ok"] and fb["gcu_current_device"] == 4
    assert fb["path_fwd_h"]["done"].get("chunk_fwd_kernel_h", 0) >= 1
    assert not fb["parallel_fwd_h"], "forced fallback still launched parallel"

    # ``--oracle`` must be requested explicitly: without it the case probe skips
    # the FP64 recurrence and ``oracle_quant_grade`` is None, which is not the
    # same thing as a pass.
    par_run, par = _run(_case(shape, dtype, 1.0, "parallel",
                              ["--compare", str(ref), "--oracle"]))
    assert par_run.returncode == 0, par_run.stderr[-3000:]
    assert par and par["exit"] == 0, (par or {}).get("error")
    assert par["parallel_fwd_h"] and par["parallel_bwd_dh"]
    assert par["path_fwd_h"]["done"].get("_chunk_fwd_state_scan_kernel", 0) >= 1
    assert par["grade"] == "pass", [(r["name"], r["max_abs"], r["rel_rms"])
                                    for r in par["compare"] if not r["ok"]]
    assert not par["nan_mismatch_names"]
    assert par["all_grads_finite"]
    # The FP64 recurrence on the quantised inputs bounds the path error too, so
    # a route that is merely self-consistent cannot pass.
    assert par["oracle_quant_grade"] == "pass", [
        (r["name"], r["rel_rms"]) for r in par["oracle_quant"] if not r["ok"]]


@pytest.mark.parametrize("shape", [(32, 64), (64, 64), (96, 128), (256, 64)])
@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
def test_local_state_is_always_fp32(preflight, tmp_path, shape, dtype):
    """The local-state buffer is FP32 whatever the K/V dtype is.

    This is the fact the earlier report got wrong: FP32 inputs do not double the
    buffer, because it is already FP32 on every route.
    """
    ref = tmp_path / "ref.pt"
    fb_run, fb = _run(_case(shape, dtype, 1.0, "fallback",
                            ["--save", str(ref)]))
    assert fb_run.returncode == 0 and fb and fb["exit"] == 0
    par_run, par = _run(_case(shape, dtype, 1.0, "parallel",
                              ["--compare", str(ref)]))
    assert par_run.returncode == 0 and par and par["exit"] == 0
    assert par["local_state_dtypes"] == ["torch.float32"]
    assert par["h0_is_fp32"] and par["dht_is_fp32"]
    theory = par["local_state_fp32_bytes_theory"]
    observed = {x["bytes"] for x in par["local_state_observed"]}
    assert theory["fwd"] in observed, (theory, observed)
    assert theory["bwd"] in observed, (theory, observed)
    # h (the per-chunk state fed to the scan) follows the input dtype; dh is
    # FP32 because the backward always requests states_in_fp32=True.
    assert par["state_stats"]["h"]["dtype"] != "torch.float32" or \
        dtype == "fp32"
    assert par["state_stats"]["dh"]["dtype"] == "torch.float32"


def test_production_default_route_is_unchanged(preflight, tmp_path):
    """No flags: the default route must be the pre-existing one, per shape."""
    # The pre-existing route: BF16 is admitted for every shape, FP16 only for
    # the legacy whitelist, FP32 never.  FP16 64x64 is in that whitelist and is
    # therefore parallel by default too -- expecting only BF16 here would have
    # asserted the opposite of the code under test.
    expect_parallel = {(64, 64), (128, 64), (128, 128), (32, 32)}
    seen = {}
    for shape in [(32, 64), (64, 64), (96, 128), (256, 64)]:
        for dtype in ["fp16", "bf16"]:
            fb_run, fb = _run(_case(shape, dtype, 1.0, "default"))
            assert fb_run.returncode == 0 and fb and fb["exit"] == 0, \
                (fb or {}).get("error")
            used_parallel = bool(fb["parallel_fwd_h"])
            # BF16 is admitted for every guarded shape; FP16 only for the
            # legacy whitelist.  (The earlier version of this test applied the
            # whitelist to BF16 as well, which the code has never done.)
            want = (dtype == "bf16") or (dtype == "fp16"
                                         and shape in expect_parallel)
            assert used_parallel == want, (shape, dtype, used_parallel, want)
            seen[(shape, dtype)] = used_parallel
    # FP32 must still refuse the parallel route with both flags absent.
    for shape in [(32, 64), (96, 128)]:
        fp32_run, fp32 = _run(_case(shape, "fp32", 1.0, "default"))
        assert fp32_run.returncode == 0 and fp32 and fp32["exit"] == 0
        assert not fp32["parallel_fwd_h"] and not fp32["parallel_bwd_dh"], \
            (shape, fp32["parallel_fwd_h"], fp32["parallel_bwd_dh"])
        seen[(shape, "fp32")] = False
    assert len(seen) == 10


def test_semantic_guards_still_fall_back(preflight):
    """cu_seqlens / split_size / state_v_first / HQ!=H / other gate forms."""
    run = subprocess.run(
        [PY, str(ROOT / "agent-results/scripts/gla008_drive.py"),
         "--mode", "guards", "--out", "/tmp/gla008_pytest_guards.jsonl"],
        cwd=ROOT, text=True, capture_output=True, timeout=1800,
        env={**os.environ,
             "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}:/opt/flagtree"})
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    rows = [json.loads(ln) for ln in
            Path("/tmp/gla008_pytest_guards.jsonl").read_text().splitlines()]
    guards = [r for r in rows if r.get("kind") == "guard"]
    assert guards, "guard probe produced no record"
    cases = {r["case"]: r for r in guards[0]["rows"]}
    for name in ("fwd_varlen_cu_seqlens", "fwd_state_v_first",
                 "fwd_given_gate_tensor", "fwd_split_size",
                 "bwd_varlen_cu_seqlens", "bwd_given_gate_tensor"):
        assert cases[name]["done"].get("chunk_fwd_kernel_h") or \
            cases[name]["done"].get("chunk_bwd_kernel_dh"), (name, cases[name])
        assert not cases[name]["done"].get("_chunk_fwd_local_state_kernel"), name
        assert not cases[name]["done"].get("_chunk_bwd_local_state_kernel"), name
    assert cases["bwd_hq_ne_h"]["done"].get("chunk_bwd_kernel_dh")
    # states_in_fp32 is a forward *semantic* switch, not a dtype allow-list.
    assert cases["fwd_states_in_fp32_bwd_on"]["done"].get(
        "_chunk_fwd_state_scan_fp32_2d_group_kernel")
    assert cases["fwd_states_in_fp32_bwd_off"]["done"].get(
        "chunk_fwd_kernel_h")


def test_benchmark_entry_point_ab_smoke(preflight, tmp_path):
    """The formal benchmark must fail closed on missing/invalid samples."""
    run = subprocess.run(
        [PY, str(ROOT / "agent-results/scripts/gla008_drive.py"),
         "--mode", "ab", "--smoke", "--rounds", "1", "--samples", "2",
         "--out", str(tmp_path / "ab.jsonl")],
        cwd=ROOT, text=True, capture_output=True, timeout=1800,
        env={**os.environ,
             "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}:/opt/flagtree"})
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-4000:]
    rows = [json.loads(ln) for ln in
            (tmp_path / "ab.jsonl").read_text().splitlines()]
    steady = [r for r in rows if r.get("phase") == "steady"]
    assert steady, "no steady-state A/B samples"
    for row in steady:
        ms = row["timing"]["full_fwd_bwd"]["ms"]
        assert len(ms) == 2, row["tag"]
        assert all(isinstance(x, float) and x > 0 for x in ms), row["tag"]
        assert row["device_ok"]
    sides = {r["side"] for r in steady}
    assert sides == {"fallback", "parallel"}
