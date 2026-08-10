"""The judge's failure behaviour, and an end-to-end replay of the demo agent."""

from __future__ import annotations

import demo.deps as deps
from demo import fixtures
from demo.agent import BookingAgent
from replaygate import (
    Count,
    Fake,
    Forbid,
    Judge,
    ReplayConfig,
    ScriptedJudge,
    Thresholds,
    evaluate,
    load_cases,
    replay_all,
    seam,
)
from replaygate.rubric import DEFAULT_RUBRIC

CASES = "demo/cases.yaml"


def _config(**overrides):
    base = dict(
        rubric=DEFAULT_RUBRIC,
        seams=[
            seam("demo.deps.llm.complete", Count(_stub_model, label="model")),
            seam("demo.deps.store", Fake(deps.store)),
            seam("demo.deps.counter", Fake(deps.counter)),
        ],
        count_signal_map={"model": "model_fallback"},
        count_exempt_kinds={"model_fallback": ("escalation",)},
    )
    base.update(overrides)
    return ReplayConfig(**base)


def _stub_model(prompt: str, **_: object) -> str:  # noqa: ARG001
    return "Could you rephrase that?"


def _prepare() -> None:
    deps.store = fixtures.build_store(read_only=True)
    deps.counter = fixtures.build_counter()
    deps.llm = type("M", (), {"complete": staticmethod(_stub_model)})()


# -- judge -----------------------------------------------------------------


def test_judge_fails_open_when_the_backend_raises():
    judge = Judge(backend=ScriptedJudge(raises=RuntimeError("502")), budget=1.0)
    verdict = judge.grade("q", "a", {})

    assert verdict.failed_open is True
    assert verdict.acceptable is True, "a grader outage must not read as a regression"
    assert judge.summary()["fail_opens"] == 1


def test_judge_fails_open_on_non_json():
    backend = ScriptedJudge()
    backend.verdicts = []
    judge = Judge(backend=_Garbage(), budget=1.0)
    verdict = judge.grade("q", "a", {})
    assert verdict.failed_open is True
    assert "non-JSON" in judge.summary()["fail_open_reasons"][0]


def test_judge_stops_at_its_budget_instead_of_spending_on():
    judge = Judge(backend=ScriptedJudge(cost=0.4), budget=1.0)
    for _ in range(5):
        judge.grade("q", "a", {})

    assert judge.exhausted is True
    assert judge.spent <= judge.budget
    assert judge.summary()["budget_exhausted"] is True


def test_a_fail_open_is_visible_in_the_summary():
    """A green run built on judge outages must not look like a clean one."""
    judge = Judge(backend=ScriptedJudge(raises=TimeoutError()), budget=1.0)
    for _ in range(3):
        judge.grade("q", "a", {})
    assert judge.summary()["fail_opens"] == 3


class _Garbage:
    cost_per_call = 0.001

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002
        return "not json at all"


# -- end to end ------------------------------------------------------------


def test_the_demo_suite_passes_its_own_gate():
    _prepare()
    results = replay_all(load_cases(CASES), BookingAgent, _config())
    outcome = evaluate(
        results,
        DEFAULT_RUBRIC,
        Thresholds(min_mean_score=85.0, max_signal_rates={"model_fallback": 0.35}),
    )

    assert outcome.passed is True, outcome.failures
    assert outcome.total == 9


def test_forbidding_the_model_turns_a_fallback_into_a_failure():
    """The flag has to move the gate, not only the seam."""
    _prepare()
    config = _config(
        seams=[
            seam("demo.deps.llm.complete", Forbid("deterministic path")),
            seam("demo.deps.store", Fake(deps.store)),
            seam("demo.deps.counter", Fake(deps.counter)),
        ]
    )
    results = replay_all(load_cases(CASES), BookingAgent, config)
    outcome = evaluate(
        results,
        DEFAULT_RUBRIC,
        Thresholds(min_mean_score=85.0, max_signal_rates={"model_fallback": 0.0}),
    )

    assert outcome.passed is False
    assert outcome.signal_counts.get("model_fallback", 0) > 0


class _Hallucinating(BookingAgent):
    """Offers a room that is not in the catalog."""

    def _search(self, state):  # noqa: ANN001
        reply = super()._search(state)
        reply.text += " We also have the Terrace."
        return reply


def test_a_denylist_only_catches_what_someone_thought_to_name():
    """The honest limit of not_contains_any, asserted so nobody oversells it.

    Only the cases that happen to list 'Terrace' notice. A denylist cannot
    enumerate everything an agent might invent, which is the entire reason the
    judged tier exists.
    """
    _prepare()
    results = replay_all(load_cases(CASES), _Hallucinating, _config())
    caught = [r for r in results if r.defects]

    assert len(caught) == 1, "expected exactly the one case that names the room"
    assert "Terrace" in caught[0].notes[0]

    outcome = evaluate(results, DEFAULT_RUBRIC, Thresholds(min_mean_score=85.0))
    assert outcome.passed is True, (
        "a single structural catch is averaged away — this is the gap the judge fills"
    )


def test_the_judge_catches_the_invention_the_denylist_missed():
    """Same broken agent, judged tier on: grounded=False is absolute, so one
    occurrence fails the gate no matter how high the mean is."""
    _prepare()
    ungrounded = ScriptedJudge(
        verdicts=[{"grounded": False, "useful": 3, "why": "Terrace is not in the facts"}]
        * 40
    )
    config = _config(
        judge=Judge(backend=ungrounded, budget=10.0),
        judge_kinds=("booking",),
    )

    results = replay_all(load_cases(CASES), _Hallucinating, config)
    outcome = evaluate(results, DEFAULT_RUBRIC, Thresholds(min_mean_score=85.0))

    assert outcome.passed is False
    assert outcome.absolute_hits, "hallucinated_entity should be an absolute failure"
    assert any("hallucinated_entity" in hit for hit in outcome.absolute_hits)
