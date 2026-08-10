"""The contract the code under test has to satisfy — deliberately small.

One method, three inputs, one structured reply. Anything larger and the harness
starts dictating the architecture of the thing it is supposed to observe.

``facts`` is the one field people skip and then regret. It is whatever the agent
retrieved to build this reply, and it is what the judge grades against. Without
it the judge can only assess plausibility, and a confident invention reads as
plausible every time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class AgentReply:
    text: str
    node: str | None = None
    state: dict[str, Any] = field(default_factory=dict)
    facts: Any = None
    """What the agent retrieved for this turn: rows, tool results, nothing."""
    intent: str | None = None


@runtime_checkable
class AgentUnderTest(Protocol):
    def handle(
        self, conversation_id: str, message: str, state: dict[str, Any]
    ) -> AgentReply: ...
