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

"""Regression tests for the MSA A/B noise and bimodality rules.

These tests need neither torch nor a GCU: the statistics module is pure
Python, and the integration test re-reads committed JSONL through the
``msa_ab_reanalyze.py`` CLI.  The historical false speedup is reproduced from
the formal MSA-005 samples so the regression cannot silently return.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "agent-results" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import msa_ab_stats as stats  # noqa: E402


def _alternating(fast: float, slow: float, slow_indexes) -> list[float]:
    """45 samples, a constant fast mode plus slow-mode values at indexes."""
    samples = [fast] * 45
    for index in slow_indexes:
        samples[index] = slow
    return samples


def _record(samples_a, samples_b, rounds_a, rounds_b, *, changed, case_id="case"):
    return {
        "probe": "case",
        "id": case_id,
        "path_changed": changed,
        "A": {
            "samples_ms": samples_a,
            "round_medians_ms": rounds_a,
            "median_ms": float(sorted(samples_a)[len(samples_a) // 2]),
        },
        "B": {
            "samples_ms": samples_b,
            "round_medians_ms": rounds_b,
            "median_ms": float(sorted(samples_b)[len(samples_b) // 2]),
        },
    }


def test_bimodal_control_with_median_artifact_is_not_treated_as_noise():
    # Shape of the formal control_q512_s896_pool7_topk3_direct record: both
    # variants run the same direct kernel, but A's all-sample median landed in
    # the slow mode while B's landed in the fast mode.
    samples_a = _alternating(1.105, 1.615, range(0, 45, 2))
    samples_b = _alternating(1.105, 1.615, range(0, 45, 3))
    rounds_a = [1.1054, 1.6125, 1.6116]
    rounds_b = [1.1272, 1.6105, 1.1069]
    verdict = stats.classify_case(
        _record(samples_a, samples_b, rounds_a, rounds_b, changed=False)
    )
    assert verdict["legacy_speedup"] > 1.3
    assert verdict["timing_status"] == "bimodal"
    assert verdict["ratio_usable"] is False
    assert verdict["verdict"] == "control_noise_unstable"


def test_round_median_ratio_recovers_the_unstable_control():
    samples_a = _alternating(1.105, 1.615, range(0, 45, 2))
    samples_b = _alternating(1.105, 1.615, range(0, 45, 3))
    verdict = stats.classify_case(
        _record(
            samples_a,
            samples_b,
            [1.1054, 1.6125, 1.6116],
            [1.1272, 1.6105, 1.1069],
            changed=False,
        )
    )
    # The round-median ratio is ~1.0; the all-sample ratio is not.  The module
    # must publish both and refuse the all-sample one.
    assert verdict["round_median_speedup"] == pytest.approx(1.0, abs=0.02)
    assert abs(verdict["legacy_speedup"] - verdict["round_median_speedup"]) > 0.05


def test_stable_control_stays_usable():
    samples = [3.45 + 0.001 * (index % 5) for index in range(45)]
    verdict = stats.classify_case(
        _record(samples, list(samples), [3.451, 3.450, 3.449], [3.450, 3.450, 3.451], changed=False)
    )
    assert verdict["timing_status"] == "stable"
    assert verdict["ratio_usable"] is True
    assert verdict["verdict"] == "control_noise_ok"


def test_isolated_spike_is_reported_but_not_a_second_mode():
    samples = [1.10] * 43 + [1.70, 1.71]
    verdict = stats.describe_variant(samples, [1.10, 1.10, 1.10])
    assert verdict["kind"] == "spikes"
    assert verdict["largest_gap"]["relative"] > stats.SPIKE_GAP_REL


def test_round_flip_marks_ratio_unusable():
    # Both variants unimodal individually, but one round's A/B ratio is 1.0
    # and another is 1.4: the ratio is not reproducible across rounds.
    samples_a = [1.10] * 45
    samples_b = [1.10] * 45
    verdict = stats.classify_case(
        _record(samples_a, samples_b, [1.10, 1.54, 1.10], [1.10, 1.10, 1.10], changed=True)
    )
    assert verdict["timing_status"] == "round_flip"
    assert verdict["ratio_usable"] is False
    assert verdict["verdict"] == "inconclusive_unstable"


def test_large_stable_gain_and_small_case_are_classified_differently():
    large = [2.40 + 0.002 * (index % 3) for index in range(45)]
    large_b = [1.90 + 0.002 * (index % 3) for index in range(45)]
    verdict_large = stats.classify_case(
        _record(large, large_b, [2.40, 2.40, 2.40], [1.90, 1.90, 1.90], changed=True, case_id="large")
    )
    assert verdict_large["verdict"] == "stable_gain"
    assert verdict_large["size_class"] == "large"

    small = [0.48 + 0.001 * (index % 3) for index in range(45)]
    small_b = [0.45 + 0.001 * (index % 3) for index in range(45)]
    verdict_small = stats.classify_case(
        _record(small, small_b, [0.48, 0.48, 0.48], [0.45, 0.45, 0.45], changed=True, case_id="small")
    )
    assert verdict_small["verdict"] == "inconclusive_small"


def test_summary_keeps_unstable_controls_as_negative_evidence():
    stable = stats.classify_case(
        _record(
            [3.45] * 45,
            [3.45] * 45,
            [3.451, 3.450, 3.449],
            [3.450, 3.450, 3.451],
            changed=False,
            case_id="stable_control",
        )
    )
    unstable = stats.classify_case(
        _record(
            _alternating(1.105, 1.615, range(0, 45, 2)),
            _alternating(1.105, 1.615, range(0, 45, 3)),
            [1.1054, 1.6125, 1.6116],
            [1.1272, 1.6105, 1.1069],
            changed=False,
            case_id="unstable_control",
        )
    )
    summary = stats.summarize([stable, unstable])
    assert summary["noise_floor"] == "unresolved"
    assert [item["id"] for item in summary["unstable_controls"]] == ["unstable_control"]
    assert summary["stable_control_range"] == [
        stable["round_median_speedup"],
        stable["round_median_speedup"],
    ]
    # The legacy all-sample ratio is still published alongside it.
    assert summary["stable_control_legacy_range"] == [
        stable["legacy_speedup"],
        stable["legacy_speedup"],
    ]


def test_reanalysis_skips_trailer_and_flags_formal_control(tmp_path):
    formal = (
        REPO
        / "agent-results"
        / "raw"
        / "MSA-005"
        / "formal"
        / "20261116T044252Z-99672caa"
        / "stdout.log"
    )
    if not formal.is_file():
        pytest.skip(f"formal evidence not present: {formal}")
    out = tmp_path / "reanalysis.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "msa_ab_reanalyze.py"),
            "--run",
            f"MSA-005-formal={formal}",
            "--json-out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    run = payload["runs"]["MSA-005-formal"]
    assert run["non_json_lines_skipped"] == 2
    assert run["case_verdicts"]["control_q512_s896_pool7_topk3_direct"]["verdict"] == (
        "control_noise_unstable"
    )
    assert run["case_verdicts"]["control_q512_s896_pool7_topk3_direct"]["legacy_speedup"] > 1.4
    assert run["case_verdicts"]["control_q512_s896_pool7_topk3_direct"]["round_median_speedup"] == (
        pytest.approx(1.0013, abs=0.001)
    )
