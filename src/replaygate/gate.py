"""The gate: a run of cases becomes one exit code.

Two kinds of threshold, because they answer different questions.

**Absolute-zero checks** are for defects with no acceptable rate. One invented
catalog entry is a failure at any volume — "only 2% of replies hallucinate" is
not a passing grade, it is an incident waiting for the right customer.

**Rate thresholds** are for everything else. Some cases will always sit at the
edge of the rubric, and a gate that demands 100 gets disabled within a month.

Pick both from a baseline, not from intuition: run the suite against the current
production revision, take what it scores, and set the floor just under it. A gate
calibrated against an aspiration fails on day one and teaches the team to pass
`--no-verify`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .rubric import CaseResult, Rubric


@dataclass(frozen=True, slots=True)
class Thresholds:
    min_mean_score: float = 85.0
    max_critical_rate: float = 0.0
    """Share of cases allowed in the worst bucket."""
    max_bucket_rates: dict[str, float] = field(default_factory=dict)
    max_signal_rates: dict[str, float] = field(default_factory=dict)
    """e.g. {"model_fallback": 0.10} — up to 10% of cases may fall back."""


@dataclass
class GateOutcome:
    passed: bool
    failures: list[str]
    mean_score: float
    bucket_counts: dict[str, int]
    signal_counts: dict[str, int]
    absolute_hits: list[str]
    total: int

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "exit_code": self.exit_code,
            "total_cases": self.total,
            "mean_score": round(self.mean_score, 2),
            "buckets": self.bucket_counts,
            "signals": self.signal_counts,
            "absolute_violations": self.absolute_hits,
            "failures": self.failures,
        }


def evaluate(
    results: list[CaseResult], rubric: Rubric, thresholds: Thresholds
) -> GateOutcome:
    total = len(results)
    if total == 0:
        return GateOutcome(
            passed=False,
            failures=["no cases ran — an empty suite is not a passing suite"],
            mean_score=0.0,
            bucket_counts={},
            signal_counts={},
            absolute_hits=[],
            total=0,
        )

    bucket_counts: dict[str, int] = {}
    signal_counts: dict[str, int] = {}
    absolute_hits: list[str] = []

    for result in results:
        bucket_counts[result.bucket] = bucket_counts.get(result.bucket, 0) + 1
        for signal in result.defects:
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
            if rubric.is_absolute(signal):
                absolute_hits.append(f"{result.case_id}: {signal}")

    mean_score = sum(r.score for r in results) / total
    failures: list[str] = []

    if absolute_hits:
        failures.append(
            f"{len(absolute_hits)} absolute-zero violation(s): "
            + ", ".join(absolute_hits[:5])
            + ("…" if len(absolute_hits) > 5 else "")
        )

    if mean_score < thresholds.min_mean_score:
        failures.append(
            f"mean score {mean_score:.1f} below floor {thresholds.min_mean_score}"
        )

    worst_bucket = rubric.buckets[-1][0]
    critical_rate = bucket_counts.get(worst_bucket, 0) / total
    if critical_rate > thresholds.max_critical_rate:
        failures.append(
            f"{worst_bucket} rate {critical_rate:.1%} above "
            f"{thresholds.max_critical_rate:.1%}"
        )

    for bucket, limit in thresholds.max_bucket_rates.items():
        rate = bucket_counts.get(bucket, 0) / total
        if rate > limit:
            failures.append(f"bucket {bucket} rate {rate:.1%} above {limit:.1%}")

    for signal, limit in thresholds.max_signal_rates.items():
        rate = signal_counts.get(signal, 0) / total
        if rate > limit:
            failures.append(f"signal {signal} rate {rate:.1%} above {limit:.1%}")

    return GateOutcome(
        passed=not failures,
        failures=failures,
        mean_score=mean_score,
        bucket_counts=bucket_counts,
        signal_counts=signal_counts,
        absolute_hits=absolute_hits,
        total=total,
    )
