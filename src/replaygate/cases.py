"""Cases and expectations, declared rather than asserted.

Substring assertions are how agent suites rot. ``assert "available" in reply``
passes until someone improves the wording, then fails for a reason that has
nothing to do with correctness, and after the third false alarm the team starts
ignoring the suite.

An expectation here is a claim about *observable structure* — which node the
conversation reached, what state it left behind, which intent was resolved —
with text matching available but never the only tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Turn:
    """One user message and, optionally, what must be true right after it."""

    user: str
    expect: Expectation | None = None


@dataclass(frozen=True, slots=True)
class Expectation:
    node: str | None = None
    """The conversation node the agent should have reached."""

    intent: str | None = None

    state: dict[str, Any] = field(default_factory=dict)
    """Subset match against the conversation state the agent persisted."""

    contains_all: tuple[str, ...] = ()
    """Case-insensitive substrings that must appear. Use sparingly."""

    not_contains_any: tuple[str, ...] = ()
    """The stronger half of text matching: things that must never be said —
    a price that does not exist, a room that is not in the catalog."""

    matches: str | None = None
    """Regex, when the shape matters but the wording does not."""

    def check(self, reply: str, node: str | None, state: dict[str, Any]) -> list[str]:
        """Return a list of human-readable failures. Empty means satisfied."""
        failures: list[str] = []
        low = (reply or "").lower()

        if self.node is not None and node != self.node:
            failures.append(f"node: expected {self.node!r}, reached {node!r}")

        for key, want in self.state.items():
            got = state.get(key)
            if got != want:
                failures.append(f"state.{key}: expected {want!r}, got {got!r}")

        for needle in self.contains_all:
            if needle.lower() not in low:
                failures.append(f"missing text: {needle!r}")

        for needle in self.not_contains_any:
            if needle.lower() in low:
                failures.append(f"forbidden text present: {needle!r}")

        if self.matches is not None and not re.search(self.matches, reply or "", re.I):
            failures.append(f"regex did not match: {self.matches!r}")

        return failures


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    turns: tuple[Turn, ...]
    kind: str = "general"
    """Groups cases for rubric weighting and report sections."""
    tags: tuple[str, ...] = ()
    seed_state: dict[str, Any] = field(default_factory=dict)
    """Conversation state to start from — how a multi-turn case skips ahead
    without replaying twenty turns to reach the interesting one."""

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError(f"case {self.id!r} has no turns")


def from_dict(raw: dict[str, Any]) -> Case:
    turns = []
    for t in raw.get("turns") or []:
        expect_raw = t.get("expect")
        expect = None
        if expect_raw:
            expect = Expectation(
                node=expect_raw.get("node"),
                intent=expect_raw.get("intent"),
                state=expect_raw.get("state") or {},
                contains_all=tuple(expect_raw.get("contains_all") or ()),
                not_contains_any=tuple(expect_raw.get("not_contains_any") or ()),
                matches=expect_raw.get("matches"),
            )
        turns.append(Turn(user=t["user"], expect=expect))

    return Case(
        id=raw["id"],
        kind=raw.get("kind", "general"),
        tags=tuple(raw.get("tags") or ()),
        seed_state=raw.get("seed_state") or {},
        turns=tuple(turns),
    )


def load_yaml(path: str) -> list[Case]:
    import yaml

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return [from_dict(item) for item in (data.get("cases") or [])]
