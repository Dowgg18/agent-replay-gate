"""Module-level dependencies, which is what makes them seamable.

The agent reaches for ``deps.store`` rather than importing a client directly, so
a seam can be pointed at ``demo.deps.store``. This is the one design constraint
the harness imposes, and it is the same one that makes the agent testable at all.
"""

from __future__ import annotations

from typing import Any

store: Any = None
"""Read side: the room catalog and existing bookings."""

counter: Any = None
"""Attempt counters, for the escalation rule."""

llm: Any = None
"""The model. On the deterministic tier this seam is FORBID, so any call raises."""
