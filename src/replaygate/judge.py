"""The judged tier: a model grading replies the deterministic tier cannot.

Some qualities have no assertion. "Did the agent actually answer the question?"
and "was this reply useful or evasive?" are real regressions and no substring
catches them.

Four decisions worth defending:

**Tool results are the ground truth, not the prompt.** The judge is asked whether
the reply is supported by what the agent actually retrieved. Otherwise it grades
plausibility, and a confident invention scores well.

**Temperature 0 and JSON out.** A judge that phrases its verdict differently on
reruns turns the gate into a coin flip.

**Fail open, loudly.** If the judge API is down, a run must not turn red — an
outage at the grader is not a regression in the agent. Every fail-open is
recorded in the report so a green run built on ten judge timeouts is visible
rather than reassuring.

**A budget.** Judged runs cost money. The tier stops when the budget is spent and
says so, rather than quietly invoicing whoever pushed the branch.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger("replaygate.judge")

SYSTEM_PROMPT = """\
You grade a single reply produced by a customer-service assistant.

You are given the user's message, the assistant's reply, and the RETRIEVED FACTS
the assistant had available. The retrieved facts are the only source of truth.

Grade on two axes and return JSON only:

  "grounded": true only if every concrete claim in the reply (names, numbers,
  availability, prices) is supported by the retrieved facts. A claim that is
  plausible but absent from the facts is NOT grounded.

  "useful": 0-4. 0 = ignores the question. 2 = partially answers or answers with
  an unnecessary question back. 4 = answers the question directly and moves the
  conversation forward.

Return exactly: {"grounded": bool, "useful": int, "why": "one short sentence"}
"""


@dataclass(frozen=True, slots=True)
class Verdict:
    grounded: bool
    useful: int
    why: str = ""
    failed_open: bool = False

    @property
    def acceptable(self) -> bool:
        return self.grounded and self.useful >= 3


class JudgeBackend(Protocol):
    def complete(self, system: str, user: str) -> str: ...

    @property
    def cost_per_call(self) -> float: ...


@dataclass
class Judge:
    backend: JudgeBackend | None = None
    budget: float = 1.0
    """Currency units for a whole run. Exceeded means the tier stops, not that
    the gate fails."""
    spent: float = 0.0
    fail_opens: list[str] = field(default_factory=list)
    exhausted: bool = False

    def grade(self, user_message: str, reply: str, facts: Any) -> Verdict:
        if self.backend is None:
            return self._open("no judge backend configured")
        if self.exhausted:
            return self._open("judge budget exhausted")
        if self.spent + self.backend.cost_per_call > self.budget:
            self.exhausted = True
            return self._open(f"judge budget of {self.budget} would be exceeded")

        payload = json.dumps(
            {
                "user_message": user_message,
                "assistant_reply": reply,
                "retrieved_facts": facts,
            },
            ensure_ascii=False,
            default=str,
        )

        try:
            raw = self.backend.complete(SYSTEM_PROMPT, payload)
            self.spent += self.backend.cost_per_call
        except Exception as exc:  # noqa: BLE001
            return self._open(f"judge call failed: {type(exc).__name__}: {exc}")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return self._open(f"judge returned non-JSON: {raw[:120]!r}")

        try:
            return Verdict(
                grounded=bool(data["grounded"]),
                useful=int(data["useful"]),
                why=str(data.get("why", ""))[:200],
            )
        except (KeyError, TypeError, ValueError) as exc:
            return self._open(f"judge JSON missing fields: {exc}")

    def _open(self, reason: str) -> Verdict:
        """Fail open: assume the reply was fine, and record that we assumed it."""
        self.fail_opens.append(reason)
        logger.warning("judge failed open: %s", reason)
        return Verdict(grounded=True, useful=4, why=reason, failed_open=True)

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.backend is not None,
            "spent": round(self.spent, 4),
            "budget": self.budget,
            "budget_exhausted": self.exhausted,
            "fail_opens": len(self.fail_opens),
            "fail_open_reasons": self.fail_opens[:10],
        }


@dataclass
class ScriptedJudge:
    """A deterministic backend for testing the harness itself."""

    verdicts: list[dict[str, Any]] = field(default_factory=list)
    calls: int = 0
    raises: Exception | None = None
    cost: float = 0.001

    @property
    def cost_per_call(self) -> float:
        return self.cost

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002
        if self.raises is not None:
            raise self.raises
        index = self.calls
        self.calls += 1
        if index < len(self.verdicts):
            return json.dumps(self.verdicts[index])
        return json.dumps({"grounded": True, "useful": 4, "why": "default"})
