"""Scoring: named defects with weights, published alongside the score.

Two things this deliberately does not do.

It does not reduce a run to pass/fail. "17 of 20 cases passed" tells you nothing
about whether the three failures were cosmetic or the agent quoting a price that
does not exist.

It does not hide the rubric. A score that cannot be audited is a number someone
will learn to game. The weights ship with the report, so "84" can always be
expanded into the list of defects that produced it.

The one opinionated default: **falling back to the model is itself a defect with
a price.** A deterministic path that needs the LLM to answer "what are your
opening hours" is not wrong, but it is slower, costlier and non-reproducible, and
a rubric that scores it as free will watch it spread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Signal:
    """One named way a case can be defective."""

    name: str
    weight: int
    description: str = ""
    absolute: bool = False
    """If True, a single occurrence fails the gate regardless of the score.
    Reserved for defects with no acceptable rate: inventing a catalog entry,
    leaking another user's data."""


@dataclass(frozen=True, slots=True)
class Rubric:
    signals: dict[str, Signal]
    buckets: tuple[tuple[str, int], ...] = (
        ("ok", 85),
        ("needs-review", 60),
        ("critical", 0),
    )

    def weight(self, name: str) -> int:
        signal = self.signals.get(name)
        if signal is None:
            raise KeyError(
                f"unknown signal {name!r}; declare it in the rubric so the score "
                "stays auditable"
            )
        return signal.weight

    def is_absolute(self, name: str) -> bool:
        signal = self.signals.get(name)
        return bool(signal and signal.absolute)

    def bucket(self, score: int) -> str:
        for name, floor in self.buckets:
            if score >= floor:
                return name
        return self.buckets[-1][0]

    def score(self, defects: list[str]) -> int:
        total = 100 - sum(self.weight(d) for d in defects)
        return max(0, min(100, total))

    def as_dict(self) -> dict[str, Any]:
        return {
            "signals": {
                name: {
                    "weight": s.weight,
                    "absolute": s.absolute,
                    "description": s.description,
                }
                for name, s in self.signals.items()
            },
            "buckets": [{"name": n, "min_score": f} for n, f in self.buckets],
        }


@dataclass
class CaseResult:
    case_id: str
    kind: str
    defects: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    score: int = 100
    bucket: str = "ok"
    error: str | None = None

    def add(self, signal: str, note: str = "") -> None:
        self.defects.append(signal)
        if note:
            self.notes.append(f"{signal}: {note}")

    def finalise(self, rubric: Rubric) -> CaseResult:
        self.score = rubric.score(self.defects)
        self.bucket = rubric.bucket(self.score)
        return self


# The defaults. A project overrides them in YAML; these are what the demo uses
# and what the README explains.
DEFAULT_SIGNALS = {
    "expectation_failed": Signal(
        "expectation_failed", 20, "A declared expectation did not hold."
    ),
    "model_fallback": Signal(
        "model_fallback",
        20,
        "The deterministic path resolved the turn by calling the model.",
    ),
    "classifier_fallback": Signal(
        "classifier_fallback", 8, "An intent needed the classifier tier."
    ),
    "hallucinated_entity": Signal(
        "hallucinated_entity",
        60,
        "The reply named something absent from the source of truth.",
        absolute=True,
    ),
    "escalation_missed": Signal(
        "escalation_missed", 25, "A hand-off rule that should have fired did not."
    ),
    "state_corrupted": Signal(
        "state_corrupted", 30, "Conversation state left in an unusable shape."
    ),
    "crashed": Signal("crashed", 100, "The agent raised.", absolute=True),
}

DEFAULT_RUBRIC = Rubric(signals=DEFAULT_SIGNALS)


def load_yaml(path: str) -> Rubric:
    import yaml

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    signals = {
        name: Signal(
            name=name,
            weight=int(body["weight"]),
            description=body.get("description", ""),
            absolute=bool(body.get("absolute", False)),
        )
        for name, body in (data.get("signals") or {}).items()
    }
    buckets = tuple(
        (b["name"], int(b["min_score"])) for b in (data.get("buckets") or [])
    ) or Rubric.buckets

    return Rubric(signals=signals or DEFAULT_SIGNALS, buckets=buckets)
