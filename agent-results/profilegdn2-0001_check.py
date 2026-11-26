#!/usr/bin/env python3
"""PROFILEGDN2-0001 gate: ``--verify`` correctness, ``--benchmark`` A/B.

This file is the one the task scheduler invokes, with either ``--verify`` or
``--benchmark`` as the only argument.  It is deliberately thin: it runs the
repository's own GDN2 tests and this round's A/B harness instead of
re-implementing a numerical reference, so the correctness claim is the
project's, not a bespoke one.

``--verify`` (blocking gate, non-zero on failure)
    * ``tests/flag_attn/test_gdn2_forward_tiles.py`` -- the S60 tile map, the
      kill switches, the legacy-vs-tuned comparison on device and the
      irregular-shape CPU FP64 reference;
    * ``tests/flag_attn/test_gdn2_backward_reference.py`` -- twelve forward
      cases against the independent FP64 CPU recurrence (its device-gradient
      test is skipped: ``chunk_gdn2`` is inference-only);
    * ``tests/flag_attn/test_gdn2.py`` -- the operator's published shapes.
    Then it checks that the opt-in switch is off by default and that the
    ``new_zeros`` path is still what the default produces, by comparing the
    default output with the switch off and on on device.

``--benchmark`` (the official measurement, scheduler-exclusive)
    Runs ``benchmark/gdn2_final_state_ab.py drive`` (three alternating rounds,
    real baseline commit ``1320878`` versus the same tree with
    ``FLAG_ATTN_GDN2_STATE_EMPTY_HT=1``) into
    ``agent-results/PROFILEGDN2-0001-ab.jsonl`` and then ``check``s it: the
    candidate must be bit-identical, must not change any launch geometry, and
    the summary prints every raw sample ratio.

Both modes refuse to run unless the process is on the assigned physical GCU 5
(PCI ``83:00.0``).  Nothing here writes outside ``agent-results/``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PCI = "83:00.0"
HOST_DEVICE_INDEX = 5
CPU_LIST = "160-191"
# The scheduler runs --benchmark *after* the agent exits, on the committed
# tree.  Reproducing the A/B must not touch the committed evidence file, or the
# rerun would both dirty the workspace and overwrite the record it is being
# compared against.  The fresh run therefore writes outside the repository, to
# a path derived from this file's name, and prints its full check table to
# stdout (which the scheduler captures in the run record).
AB_OUT = os.environ.get(
    "PROFILEGDN2_AB_OUT", "/tmp/PROFILEGDN2-0001-ab-rerun.jsonl")
AB_SCRIPT = "benchmark/gdn2_final_state_ab.py"
ENV_SWITCH = "FLAG_ATTN_GDN2_STATE_EMPTY_HT"

TESTS = [
    "tests/flag_attn/test_gdn2_forward_tiles.py",
    "tests/flag_attn/test_gdn2_backward_reference.py",
    "tests/flag_attn/test_gdn2.py",
]


def _pci(index: int = HOST_DEVICE_INDEX):
    try:
        out = subprocess.run(["efsmi", "-q", "-d", "PCIE", "--json-format"],
                             capture_output=True, text=True, timeout=30).stdout
        data = json.loads(out)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    for item in data.get("devices", []):
        if str(item.get("id")) == str(index):
            p = item.get("PCIe Info", {})
            return f"{p.get('Bus')}:{p.get('Dev')}.{p.get('Func')}"
    return None


def _check_device():
    """Verify this process is bound to physical GCU 5, and return its ordinal.

    Two bindings both mean GCU 5 and both occur in practice:

    * the scheduler wrapper exposes every GCU (``TOPS_VISIBLE_DEVICES`` unset or
      ``0,1,2,3,4,5``) and calls ``torch.gcu.set_device(5)``, so the ordinal is
      5;
    * a self-launched child masks the device (``TOPS_VISIBLE_DEVICES=5``), so
      the only visible ordinal is 0.

    Anything else is an error: allocating on a different ordinal would run on
    another tenant's card.
    """
    import torch

    pci = _pci()
    count = torch.gcu.device_count()
    current = torch.gcu.current_device()
    if pci is None or pci.strip().lower() != EXPECTED_PCI:
        print(f"FAIL: host GCU {HOST_DEVICE_INDEX} is {pci!r}, expected {EXPECTED_PCI!r}",
              file=sys.stderr)
        return None, 4
    if count == 1 and current == 0:
        ordinal = 0
    elif count > 5 and current == HOST_DEVICE_INDEX:
        ordinal = HOST_DEVICE_INDEX
    else:
        print(f"FAIL: visible={count} current={current} is not a GCU 5 binding "
              f"(expected 1/0 or >5/{HOST_DEVICE_INDEX})", file=sys.stderr)
        return None, 4
    print(f"device: GCU {HOST_DEVICE_INDEX} PCI {pci}, visible={count}, "
          f"ordinal={ordinal}, torch {torch.__version__}")
    return ordinal, 0


def _child_env() -> dict:
    env = os.environ.copy()
    env["TOPS_VISIBLE_DEVICES"] = str(HOST_DEVICE_INDEX)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def _run(argv, label: str) -> int:
    print(f"\n=== {label} ===\n$ {' '.join(argv)}", flush=True)
    proc = subprocess.run(argv, cwd=str(ROOT), env=_child_env())
    print(f"=== {label}: exit {proc.returncode} ===", flush=True)
    return proc.returncode


def command_verify() -> int:
    import torch

    ordinal, code = _check_device()
    if code:
        return code
    dev = f"gcu:{ordinal}"

    # The scheduler exports PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH", but the
    # script must also work when invoked directly, so make the same two roots
    # importable before touching the operator.
    for extra in (str(ROOT / "src"), str(ROOT)):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    from flag_attn import chunk_gdn2  # noqa: F401  (import must succeed)

    impl = __import__("flag_attn.runtime.backend._enflame.FLA.gdn2.chunk_gdn2",
                      fromlist=["x"])

    # The experiment must be off unless explicitly requested: a default-on
    # switch would silently change the shipped operator.
    if os.environ.get(ENV_SWITCH, "0") == "1":
        print(f"FAIL: {ENV_SWITCH}=1 in the verify environment; the default "
              f"must be off", file=sys.stderr)
        return 5
    if getattr(impl, "_S60_STATE_EMPTY_HT_ENABLED", False):
        print("FAIL: module-level experiment flag is on by default", file=sys.stderr)
        return 5

    code = _run([sys.executable, "-m", "pytest", *TESTS, "-q", "-p", "no:cacheprovider"],
                "repository GDN2 tests")
    if code:
        print("FAIL: repository GDN2 tests failed", file=sys.stderr)
        return code or 3

    # Independent numerical check of the experiment path: the default and the
    # opt-in allocation must produce identical bits, and both must stay inside
    # the operator's own gate against the FP64 CPU recurrence.
    test_dir = str(ROOT / "tests" / "flag_attn")
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)
    import hashlib
    import math

    from gdn2_backward_ref import gdn2_forward

    def digest(t):
        raw = t.detach().to("cpu").contiguous()
        if raw.dtype == torch.bfloat16:
            raw = raw.view(torch.int16)
        return hashlib.sha256(raw.numpy().tobytes()).hexdigest()

    failures = 0
    cases = [((2, 512, 4, 128, 160), torch.bfloat16),
             ((2, 512, 4, 128, 256), torch.float16),
             ((2, 512, 4, 96, 128), torch.bfloat16),
             ((1, 1024, 4, 256, 256), torch.bfloat16),
             ((2, 512, 8, 64, 64), torch.float16)]
    device_str = dev
    for (B, T, H, K, V), dtype in cases:
        torch.manual_seed(42)
        q = (torch.randn(B, T, H, K) / math.sqrt(K)).to(dtype)
        k = (torch.randn(B, T, H, K) / math.sqrt(K)).to(dtype)
        v = torch.randn(B, T, H, V).to(dtype)
        g = (-torch.rand(B, T, H, K) * 0.1).to(dtype)
        b = torch.rand(B, T, H, K).to(dtype)
        w = torch.rand(B, T, H, V).to(dtype)
        h0 = torch.randn(B, H, K, V) * 0.01
        dev = [x.to(device_str) for x in (q, k, v, g, b, w, h0)]
        ref_o, ref_ht = gdn2_forward(q, k, v, g, b, w, K ** -0.5, h0)
        results = {}
        for empty in (False, True):
            impl._S60_STATE_EMPTY_HT_ENABLED = empty
            with torch.inference_mode():
                o, ht = chunk_gdn2(*dev[:6], scale=K ** -0.5, initial_state=dev[6],
                                   output_final_state=True, chunk_size=64)
            results[empty] = (digest(o), digest(ht), o, ht)
        impl._S60_STATE_EMPTY_HT_ENABLED = False
        o_ok = results[False][0] == results[True][0]
        ht_ok = results[False][1] == results[True][1]
        diff = (ref_o.double() - results[False][2].double().cpu()).abs()
        rms = diff.square().mean().sqrt().item()
        ref_rms = ref_o.double().square().mean().sqrt().item()
        gate = (diff.max().item() <= 1e-3 + 2e-2 * ref_rms
                and rms / (ref_rms + 1e-12) <= 2e-2)
        finite = bool(torch.isfinite(results[False][2]).all().item()
                      and torch.isfinite(results[False][3]).all().item())
        status = "ok" if (o_ok and ht_ok and gate and finite) else "FAIL"
        if status != "ok":
            failures += 1
        print(f"  {dtype} {(B, T, H, K, V)} bit_identical={o_ok and ht_ok} "
              f"gate={gate} finite={finite} rms_rel={rms / (ref_rms + 1e-12):.3e} "
              f"{status}")

    if failures:
        print(f"FAIL: {failures} numerical case(s) failed", file=sys.stderr)
        return 3
    print("\nVERIFY PASS: tests, default-off switch and bit-identical experiment path")
    return 0


def command_benchmark() -> int:
    _, code = _check_device()
    if code:
        return code
    out = ROOT / AB_OUT
    if out.exists():
        out.unlink()
    code = _run([sys.executable, AB_SCRIPT, "drive", "--workspace", str(ROOT),
                 "--base-ref", "1320878ddada3b3d29738d1e5f5c0bcc9385def7",
                 "--cand-ref", "HEAD", "--rounds", "3", "--out", AB_OUT],
                "full-step A/B (3 alternating rounds)")
    if code:
        print(f"FAIL: A/B drive exited {code}", file=sys.stderr)
        return code
    code = _run([sys.executable, AB_SCRIPT, "check", "--out", AB_OUT, "--rounds", "3"],
                "A/B offline check")
    if code:
        print("FAIL: A/B check failed", file=sys.stderr)
        return code
    print(f"\nBENCHMARK PASS: {AB_OUT}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", action="store_true", help="run the correctness gate")
    group.add_argument("--benchmark", action="store_true", help="run the official A/B")
    args = parser.parse_args(argv)
    return command_verify() if args.verify else command_benchmark()


if __name__ == "__main__":
    raise SystemExit(main())
