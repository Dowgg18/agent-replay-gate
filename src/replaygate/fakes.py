"""Fakes with declared fidelity.

A fake that returns canned answers tests the canned answers. The two places that
bites hardest in conversational agents:

**Counters.** Escalation rules are almost always "after N invalid attempts, hand
to a human". A fake counter that returns 1 forever means the escalation branch is
never reached in replay — the exact branch most likely to be broken, because it
is the one nobody exercises by hand. So ``FakeCounter`` implements real
increment/expire semantics.

**Queries.** If the fake store ignores filters and returns everything, a case
that checks "the agent must not offer a room that does not fit" passes for the
wrong reason. ``FakeStore`` implements filtering and ordering, and records every
write so a test can assert that a read-only replay stayed read-only.

Each fake states what it is faithful to and what it is not, because the honest
answer is never "everything".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


class UnexpectedWrite(AssertionError):
    """A replay that was supposed to be read-only tried to write."""


@dataclass
class FakeCounter:
    """Faithful to: increment, read, expiry, reset. Not faithful to: atomicity
    across processes (there is only one), or millisecond-accurate TTL."""

    _values: dict[str, int] = field(default_factory=dict)
    _expiry: dict[str, float] = field(default_factory=dict)

    def incr(self, key: str, ttl: float | None = None) -> int:
        self._expire_if_due(key)
        self._values[key] = self._values.get(key, 0) + 1
        if ttl is not None:
            self._expiry[key] = time.monotonic() + ttl
        return self._values[key]

    def get(self, key: str) -> int:
        self._expire_if_due(key)
        return self._values.get(key, 0)

    def reset(self, key: str) -> None:
        self._values.pop(key, None)
        self._expiry.pop(key, None)

    def _expire_if_due(self, key: str) -> None:
        due = self._expiry.get(key)
        if due is not None and time.monotonic() >= due:
            self._values.pop(key, None)
            self._expiry.pop(key, None)


@dataclass
class FakeStore:
    """An in-memory table with real filtering.

    Faithful to: equality and comparison filters, ordering, limit, and the
    distinction between "no rows" and "no such table". Not faithful to:
    transactions, joins, or any query planner behaviour.
    """

    rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    read_only: bool = False
    writes: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def select(
        self,
        table: str,
        where: dict[str, Any] | None = None,
        order_by: str | None = None,
        descending: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if table not in self.rows:
            raise KeyError(f"no such table in fixture: {table!r}")

        out = list(self.rows[table])
        for key, want in (where or {}).items():
            out = [r for r in out if _matches(r.get(key), want)]
        if order_by:
            out.sort(key=lambda r: r.get(order_by), reverse=descending)
        if limit is not None:
            out = out[:limit]
        return [dict(r) for r in out]

    def insert(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        if self.read_only:
            raise UnexpectedWrite(f"insert into {table!r} during a read-only replay")
        self.writes.append((table, dict(row)))
        self.rows.setdefault(table, []).append(dict(row))
        return dict(row)

    def update(self, table: str, where: dict[str, Any], patch: dict[str, Any]) -> int:
        if self.read_only:
            raise UnexpectedWrite(f"update on {table!r} during a read-only replay")
        touched = 0
        for row in self.rows.get(table, []):
            if all(_matches(row.get(k), v) for k, v in where.items()):
                row.update(patch)
                touched += 1
        if touched:
            self.writes.append((table, {"where": where, "patch": patch}))
        return touched


def _matches(value: Any, want: Any) -> bool:
    """Supports ``{"gte": x}`` / ``{"lte": x}`` / ``{"in": [...]}`` alongside
    plain equality, which is enough vocabulary for slot-filling agents."""
    if isinstance(want, dict):
        if "gte" in want and not (value is not None and value >= want["gte"]):
            return False
        if "lte" in want and not (value is not None and value <= want["lte"]):
            return False
        if "in" in want and value not in want["in"]:
            return False
        return True
    return value == want


@dataclass
class RecordedModel:
    """A stand-in for an LLM that replays fixed answers in order.

    Used for the *judged* tier only. On the deterministic tier the model seam is
    ``Forbid``, so this never gets a chance to run — which is the point.
    """

    answers: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    default: str = ""

    def complete(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        index = len(self.calls) - 1
        if index < len(self.answers):
            return self.answers[index]
        return self.default
