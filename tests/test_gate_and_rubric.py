"""Scoring and gating: the arithmetic that turns a run into an exit code."""

from __future__ import annotations

import pytest

from replaygate import CaseResult, Rubric, Signal, Thresholds, evaluate
from replaygate.rubric import DEFAULT_RUBRIC

RUBRIC = Rubric(
    signals={
        "minor": Signal("minor", 10),
        "major": Signal("major", 40),
        "never": Signal("never", 50, absolute=True),
    }
)


def result(case_id: str, *defects: str, kind: str = "general") -> CaseResult:
    r = CaseResult(case_id=case_id, kind=kind)
    for d in defects:
        r.add(d)
    return r.finalise(RUBRIC)


def test_score_is_one_hundred_minus_the_weights():
    assert result("a").score == 100
    assert result("b", "minor").score == 90
    assert result("c", "minor", "major").score == 50


def test_score_floors_at_zero_rather_than_going_negative():
    assert result("d", "never", "major", "major").score == 0


def test_an_unknown_signal_is_an_error_not_a_zero():
    """Silently scoring an undeclared signal as free is how a rubric drifts out
    of sync with the code that emits it."""
    r = CaseResult(case_id="x", kind="general")
    r.add("typo_in_signal_name")
    with pytest.raises(KeyError, match="typo_in_signal_name"):
        r.finalise(RUBRIC)


def test_buckets_are_assigned_from_the_score():
    assert result("a").bucket == "ok"           # 100
    assert result("b", "minor").bucket == "ok"  # 90
    assert result("c", "minor", "minor").bucket == "needs-review"  # 80
    assert RUBRIC.bucket(85) == "ok"
    assert RUBRIC.bucket(84) == "needs-review"
    assert RUBRIC.bucket(59) == "critical"


def test_one_absolute_violation_fails_a_high_scoring_run():
    """Rate thresholds must not be able to average away a defect that has no
    acceptable rate."""
    results = [result(f"ok-{i}") for i in range(19)]
    results.append(result("bad", "never"))

    outcome = evaluate(results, RUBRIC, Thresholds(min_mean_score=50.0))

    assert outcome.passed is False
    assert outcome.mean_score > 90
    assert outcome.absolute_hits == ["bad: never"]
    assert "absolute-zero" in outcome.failures[0]


def test_signal_rate_threshold():
    results = [result("a", "minor"), result("b", "minor"), result("c")]
    tight = evaluate(
        results, RUBRIC, Thresholds(min_mean_score=0, max_signal_rates={"minor": 0.5})
    )
    assert tight.passed is False
    assert "minor" in tight.failures[0]

    loose = evaluate(
        results, RUBRIC, Thresholds(min_mean_score=0, max_signal_rates={"minor": 0.9})
    )
    assert loose.passed is True


def test_an_empty_run_fails():
    outcome = evaluate([], RUBRIC, Thresholds())
    assert outcome.passed is False
    assert outcome.exit_code == 1
    assert "empty suite" in outcome.failures[0]


def test_exit_code_is_the_whole_interface():
    passing = evaluate([result("a")], RUBRIC, Thresholds(min_mean_score=50))
    failing = evaluate([result("b", "major")], RUBRIC, Thresholds(min_mean_score=90))
    assert passing.exit_code == 0
    assert failing.exit_code == 1


def test_the_default_rubric_prices_a_model_fallback():
    """The opinionated default, asserted so a future edit is deliberate."""
    assert DEFAULT_RUBRIC.weight("model_fallback") == 20
    assert DEFAULT_RUBRIC.is_absolute("hallucinated_entity") is True
    assert DEFAULT_RUBRIC.is_absolute("model_fallback") is False
