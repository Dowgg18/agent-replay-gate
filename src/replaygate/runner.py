"""Driving cases through the agent, inside the sandbox, into results.

The order of operations matters and is easy to get subtly wrong:

1. Seal the seams **before** the agent is touched, so a module-level side effect
   at import time cannot escape.
2. Replay the turns in order, carrying the state the agent returns. A harness
   that resets state between turns is testing single-turn behaviour and calling
   it multi-turn.
3. Score seam crossings **and** expectation failures. A case can satisfy every
   expectation and still be defective, because it got there by calling the model
   on a path that was supposed to be deterministic.
4. Judge last, and only what survived — there is no point paying a model to grade
   a reply that already failed a structural expectation.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .agent import AgentReply, AgentUnderTest
from .cases import Case
from .judge import Judge
from .rubric import CaseResult, Rubric
from .sandbox import Seam, SeamViolation, sealed


@dataclass
class ReplayConfig:
    rubric: Rubric
    seams: list[Seam] = field(default_factory=list)
    judge: Judge | None = None
    judge_kinds: tuple[str, ...] = ()
    """Only these case kinds are judged. Empty means judge nothing."""
    count_signal_map: dict[str, str] = field(default_factory=dict)
    """Maps a COUNT seam label to the rubric signal it should charge, e.g.
    {"model": "model_fallback"}."""

    count_exempt_kinds: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Case kinds that must not be charged for a given signal.

    Some cases exist precisely to exercise the fallback: a suite that penalises
    an escalation case for escalating is measuring the wrong thing, and the team
    learns to distrust the score rather than fix the agent."""


def replay_case(
    case: Case,
    build_agent: Callable[[], AgentUnderTest],
    config: ReplayConfig,
) -> CaseResult:
    result = CaseResult(case_id=case.id, kind=case.kind)
    state: dict[str, Any] = dict(case.seed_state)

    try:
        with sealed(*config.seams) as sandbox:
            agent = build_agent()

            for index, turn in enumerate(case.turns):
                try:
                    reply = agent.handle(case.id, turn.user, state)
                except SeamViolation as exc:
                    result.add("model_fallback", f"turn {index + 1}: {exc}")
                    result.error = str(exc)
                    break
                except Exception as exc:  # noqa: BLE001
                    result.add("crashed", f"turn {index + 1}: {type(exc).__name__}: {exc}")
                    result.error = traceback.format_exc(limit=4)
                    break

                if not isinstance(reply, AgentReply):
                    result.add(
                        "state_corrupted",
                        f"turn {index + 1}: handle() returned {type(reply).__name__}",
                    )
                    break

                state = reply.state or {}
                result.transcript.append(
                    {
                        "user": turn.user,
                        "reply": reply.text,
                        "node": reply.node,
                        "intent": reply.intent,
                    }
                )

                if turn.expect is not None:
                    failures = turn.expect.check(reply.text, reply.node, state)
                    for failure in failures:
                        result.add("expectation_failed", f"turn {index + 1}: {failure}")

                _judge_turn(result, config, turn.user, reply, case)

            _charge_crossings(result, sandbox.crossings, config, case)

    except Exception as exc:  # noqa: BLE001 - harness failure, not agent failure
        result.add("crashed", f"harness error: {type(exc).__name__}: {exc}")
        result.error = traceback.format_exc(limit=4)

    return result.finalise(config.rubric)


def _charge_crossings(
    result: CaseResult, crossings: dict[str, int], config: ReplayConfig, case: Case
) -> None:
    for label, count in crossings.items():
        signal = config.count_signal_map.get(label)
        if not signal or count == 0:
            continue
        if case.kind in config.count_exempt_kinds.get(signal, ()):
            result.notes.append(f"{label} crossed {count}x (exempt for kind {case.kind!r})")
            continue
        # Charged once per case, not once per call: the defect is "this path
        # needed the model at all", and billing it per turn would make one
        # chatty case dominate the whole report.
        result.add(signal, f"{label} crossed {count}x")


def _judge_turn(
    result: CaseResult, config: ReplayConfig, user: str, reply: AgentReply, case: Case
) -> None:
    if config.judge is None or case.kind not in config.judge_kinds:
        return
    if "expectation_failed" in result.defects:
        return  # already broken structurally; do not pay to grade it

    verdict = config.judge.grade(user, reply.text, reply.facts)
    if verdict.failed_open:
        result.notes.append(f"judge failed open: {verdict.why}")
        return
    if not verdict.grounded:
        result.add("hallucinated_entity", f"judge: {verdict.why}")
    elif verdict.useful < 3:
        result.add("expectation_failed", f"judge: unhelpful ({verdict.useful}/4) {verdict.why}")


def replay_all(
    cases: list[Case],
    build_agent: Callable[[], AgentUnderTest],
    config: ReplayConfig,
) -> list[CaseResult]:
    return [replay_case(case, build_agent, config) for case in cases]
