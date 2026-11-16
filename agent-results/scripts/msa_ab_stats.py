"""Noise and bimodality classification for the MSA group-16 A/B records.

The historical ``msa003_ab.py`` used the median of all 45 samples as the
per-variant statistic.  On sub-2 ms shapes this is not robust when the timing
distribution has two modes: the median can land in the fast mode for one
variant and in the slow mode for the other, producing a large apparent
speedup while both variants execute exactly the same dispatch.

This module is pure Python (no torch/GCU import) so the classification can be
tested and re-run on committed evidence on any host.  It deliberately does not
discard samples or rewrite a run; it labels a ratio as usable or not.

Definitions
-----------
* a *bimodal* variant has two adjacent clusters, each with at least
  ``max(4, 20% of samples)`` members, separated by a relative gap of at least
  8%;
* a *spike* variant has a large gap but the minority side is too small to be a
  second mode (one or a few context-switch outliers);
* a *round flip* has per-round A/B ratios spanning more than 5%;
* a ratio is *usable* only for unimodal/spike timing whose round-median ratio
  agrees with the all-sample median ratio to within 5%.
"""

from __future__ import annotations

import math
import statistics
from typing import Iterable, Sequence

BIMODAL_GAP_REL = 0.08
BIMODAL_MIN_FRACTION = 0.20
BIMODAL_MIN_COUNT = 4
SPIKE_GAP_REL = 0.10
ROUND_FLIP_REL = 0.05
RATIO_AGREE_REL = 0.05
SMALL_CASE_MS = 0.95
LARGE_CASE_MS = 1.3


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _gaps(values: Sequence[float]) -> list[tuple[float, float, float, int, int]]:
    """Adjacent gaps as (relative_gap, low, high, low_count, high_count)."""
    ordered = sorted(float(value) for value in values)
    gaps = []
    total = len(ordered)
    for index in range(1, total):
        low = ordered[index - 1]
        high = ordered[index]
        if low <= 0 or not math.isfinite(low) or not math.isfinite(high):
            continue
        gaps.append(((high - low) / low, low, high, index, total - index))
    return gaps


def describe_variant(samples: Sequence[float], round_medians: Sequence[float]) -> dict:
    """Classify one variant's timing distribution without dropping samples."""
    ordered = sorted(float(value) for value in samples)
    total = len(ordered)
    gaps = _gaps(ordered)
    largest = max(gaps, default=None, key=lambda item: item[0])
    threshold = max(BIMODAL_MIN_COUNT, math.ceil(BIMODAL_MIN_FRACTION * total))

    kind = "unimodal"
    gap = None
    if largest is not None and largest[0] >= BIMODAL_GAP_REL:
        rel, low, high, low_n, high_n = largest
        if low_n >= threshold and high_n >= threshold:
            kind = "bimodal"
            gap = {
                "relative": rel,
                "low_ms": low,
                "high_ms": high,
                "low_count": low_n,
                "high_count": high_n,
                "low_fraction": low_n / total,
                "high_fraction": high_n / total,
                "low_median_ms": _median(ordered[:low_n]),
                "high_median_ms": _median(ordered[low_n:]),
            }
        elif rel >= SPIKE_GAP_REL and min(low_n, high_n) >= 1:
            kind = "spikes"
            gap = {
                "relative": rel,
                "low_ms": low,
                "high_ms": high,
                "low_count": low_n,
                "high_count": high_n,
                "low_fraction": low_n / total,
                "high_fraction": high_n / total,
                "low_median_ms": _median(ordered[:low_n]),
                "high_median_ms": _median(ordered[low_n:]),
            }

    rounds = [float(value) for value in round_medians if math.isfinite(float(value))]
    spread = (max(rounds) / min(rounds) - 1.0) if len(rounds) >= 2 and min(rounds) > 0 else 0.0
    return {
        "kind": kind,
        "largest_gap": gap,
        "samples": total,
        "round_medians_ms": rounds,
        "round_median_spread_rel": spread,
        "median_ms": _median(ordered),
        "min_ms": min(ordered) if ordered else float("nan"),
        "max_ms": max(ordered) if ordered else float("nan"),
    }


def _coerce_variant(value) -> dict:
    if isinstance(value, dict):
        samples = value.get("samples_ms") or []
        rounds = value.get("round_medians_ms") or []
        median = value.get("median_ms")
    else:  # pragma: no cover - defensive
        raise TypeError(f"variant must be a dict, got {type(value)!r}")
    described = describe_variant(samples, rounds)
    if median is not None:
        described["record_median_ms"] = float(median)
    return described


def classify_case(record: dict) -> dict:
    """Return the timing verdict for one ``probe=case`` record."""
    a = _coerce_variant(record["A"])
    b = _coerce_variant(record["B"])
    a_median = a.get("record_median_ms", a["median_ms"])
    b_median = b.get("record_median_ms", b["median_ms"])
    legacy = (a_median / b_median) if b_median else float("nan")

    a_rounds = a["round_medians_ms"]
    b_rounds = b["round_medians_ms"]
    round_ratios = [x / y for x, y in zip(a_rounds, b_rounds) if y]
    round_ratio_median = _median(round_ratios)
    round_ratio_range = (max(round_ratios) - min(round_ratios)) if len(round_ratios) >= 2 else 0.0
    delta = abs(round_ratio_median - legacy) if math.isfinite(legacy) else float("inf")

    reasons = []
    if a["kind"] == "bimodal" or b["kind"] == "bimodal":
        status = "bimodal"
        reasons.append(
            "bimodal timing: "
            + ", ".join(
                tag
                for tag, variant in (("A", a), ("B", b))
                if variant["kind"] == "bimodal"
            )
        )
    elif round_ratio_range > ROUND_FLIP_REL:
        status = "round_flip"
        reasons.append(f"round A/B ratios span {round_ratio_range:.3f}")
    elif a["kind"] == "spikes" or b["kind"] == "spikes":
        status = "spikes"
        reasons.append("isolated timing spikes (minority too small to be a mode)")
    else:
        status = "stable"

    if delta > RATIO_AGREE_REL and status in ("stable", "spikes"):
        status = "ratio_disagreement"
        reasons.append(
            f"all-sample ratio {legacy:.4f} differs from round-median ratio "
            f"{round_ratio_median:.4f} by {delta:.3f}"
        )

    usable = status in ("stable", "spikes") and delta <= RATIO_AGREE_REL
    if a_median >= LARGE_CASE_MS:
        size = "large"
    elif a_median >= SMALL_CASE_MS:
        size = "boundary"
    else:
        size = "small"
    changed = bool(record.get("path_changed"))
    if not changed:
        verdict = "control_noise_ok" if usable else "control_noise_unstable"
    elif not usable:
        verdict = "inconclusive_unstable"
    elif size == "small":
        verdict = "inconclusive_small"
    elif size == "boundary":
        verdict = "stable_gain_boundary"
    else:
        verdict = "stable_gain"

    return {
        "id": record.get("id"),
        "path_changed": changed,
        "size_class": size,
        "a_median_ms": a_median,
        "b_median_ms": b_median,
        "legacy_speedup": legacy,
        "round_median_speedup": round_ratio_median,
        "round_speedups": round_ratios,
        "round_ratio_range": round_ratio_range,
        "ratio_delta": delta,
        "timing_status": status,
        "ratio_usable": usable,
        "verdict": verdict,
        "reasons": reasons,
        "A": a,
        "B": b,
    }


def geomean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if value and math.isfinite(float(value))]
    if not values:
        return float("nan")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize(cases: Sequence[dict]) -> dict:
    """Aggregate verdicts, keeping unstable controls as negative evidence."""
    stable_controls = [c for c in cases if c["verdict"] == "control_noise_ok"]
    unstable_controls = [c for c in cases if c["verdict"] == "control_noise_unstable"]
    stable_gains = [c for c in cases if c["verdict"] == "stable_gain"]
    boundary_gains = [c for c in cases if c["verdict"] == "stable_gain_boundary"]
    small_inconclusive = [c for c in cases if c["verdict"] == "inconclusive_small"]
    unstable = [c for c in cases if c["verdict"] == "inconclusive_unstable"]

    # Controls are judged on the round-median ratio, the same metric used for
    # changed cases; the legacy ratio is kept per case but is bimodal-prone.
    control_ratios = [c["round_median_speedup"] for c in stable_controls]
    stable_gain_speedups = [c["round_median_speedup"] for c in stable_gains]
    boundary_gain_speedups = [c["round_median_speedup"] for c in boundary_gains]
    return {
        "cases": len(cases),
        "changed_cases": sum(1 for c in cases if c["path_changed"]),
        "control_cases": sum(1 for c in cases if not c["path_changed"]),
        "stable_gain_cases": [c["id"] for c in stable_gains],
        "stable_gain_boundary_cases": [c["id"] for c in boundary_gains],
        "inconclusive_small_cases": [c["id"] for c in small_inconclusive],
        "inconclusive_unstable_cases": [c["id"] for c in unstable],
        "stable_gain_geomean": geomean(stable_gain_speedups),
        "stable_gain_min": min(stable_gain_speedups) if stable_gain_speedups else None,
        "stable_gain_max": max(stable_gain_speedups) if stable_gain_speedups else None,
        "boundary_gain_geomean": geomean(boundary_gain_speedups),
        "boundary_gain_min": min(boundary_gain_speedups) if boundary_gain_speedups else None,
        "boundary_gain_max": max(boundary_gain_speedups) if boundary_gain_speedups else None,
        "stable_control_range": [min(control_ratios), max(control_ratios)] if control_ratios else None,
        "stable_control_legacy_range": (
            [
                min(c["legacy_speedup"] for c in stable_controls),
                max(c["legacy_speedup"] for c in stable_controls),
            ]
            if stable_controls
            else None
        ),
        "stable_control_max_abs_deviation": (
            max(abs(value - 1.0) for value in control_ratios) if control_ratios else None
        ),
        "unstable_controls": [
            {
                "id": c["id"],
                "legacy_speedup": c["legacy_speedup"],
                "round_speedups": c["round_speedups"],
                "timing_status": c["timing_status"],
                "reasons": c["reasons"],
            }
            for c in unstable_controls
        ],
        "noise_floor": "unresolved" if unstable_controls else "stable",
    }
