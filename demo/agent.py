"""A room-booking assistant for a coworking space — the agent under test.

Non-trivial on purpose. A harness demonstrated against a toy proves nothing, so
this has the three properties that make conversational agents hard to test:

* **Multi-turn state.** Date, time and headcount are collected across turns and
  any of them can arrive out of order, or be corrected later.
* **A source of truth.** Rooms come from the store. Offering a room that is not
  in the catalog is the failure the rubric treats as absolute.
* **A threshold rule.** Three unparseable answers in a row hands off to a human.
  That branch only fires if the counter has real semantics, which is why the
  fake implements increment and expiry rather than returning 1.

The deterministic path handles menu numbers, dates, times and headcounts. Free
text falls through to ``deps.llm`` — legitimate, and scored, because a path that
needs the model to understand "tomorrow at 2" is slower and non-reproducible.
"""

from __future__ import annotations

import re
from typing import Any

from replaygate.agent import AgentReply

from . import deps

MAX_INVALID = 3

MENU = (
    "How can I help?\n"
    "1 - Book a room\n"
    "2 - Check an existing booking\n"
    "3 - Talk to a person"
)

_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_SIZE = re.compile(r"\b(\d{1,2})\s*(?:people|persons?|pax|seats?)\b", re.I)

# Time is only recognised in shapes that cannot be confused with a date fragment
# or a headcount: an explicit clock time, a meridiem, or an "at N".
_TIME = re.compile(
    r"(?:\b([01]?\d|2[0-3]):([0-5]\d)\b)"
    r"|(?:\b([01]?\d|2[0-3])\s*(am|pm)\b)"
    r"|(?:\bat\s+([01]?\d|2[0-3])\b(?!\s*(?:people|persons?|pax|seats?)))",
    re.I,
)


class BookingAgent:
    """Stateless between turns; all continuity lives in the state dict."""

    def handle(
        self, conversation_id: str, message: str, state: dict[str, Any]
    ) -> AgentReply:
        state = dict(state)
        text = (message or "").strip()

        if not state:
            state["node"] = "greeting"
            return self._reply("Hi! " + MENU, "menu", state)

        node = state.get("node", "menu")

        if node == "menu":
            return self._menu(conversation_id, text, state)
        if node in {"ask_date", "ask_time", "ask_size"}:
            return self._collect(conversation_id, text, state)
        if node == "offer":
            return self._confirm(conversation_id, text, state)
        if node in {"handoff", "confirmed"}:
            return self._reply(
                "A colleague will follow up here shortly.", node, state
            )

        return self._reply(MENU, "menu", state)

    # -- steps --------------------------------------------------------------

    def _menu(self, conversation_id: str, text: str, state: dict) -> AgentReply:
        choice = text.strip()
        if choice == "1":
            deps.counter.reset(f"invalid:{conversation_id}")
            state["node"] = "ask_date"
            return self._reply("Sure. What date? (YYYY-MM-DD)", "ask_date", state)
        if choice == "2":
            state["node"] = "handoff"
            return self._reply(
                "Let me get someone to look that up for you.", "handoff", state
            )
        if choice == "3":
            state["node"] = "handoff"
            return self._reply("Putting you through to a colleague.", "handoff", state)

        return self._invalid(conversation_id, text, state, "menu")

    def _collect(self, conversation_id: str, text: str, state: dict) -> AgentReply:
        """Slots can arrive in any order and out of turn, which is how people
        actually answer. One message can fill several."""
        filled = False
        remaining = text

        # Consume in order of specificity, removing each match so a later, looser
        # pattern cannot claim digits that already belong to something else. The
        # date has to go first: "2026-09-02" contains two perfectly good hours.
        date_match = _DATE.search(remaining)
        if date_match:
            state["date"] = date_match.group(1)
            remaining = remaining.replace(date_match.group(0), " ", 1)
            filled = True

        size_match = _SIZE.search(remaining)
        if size_match:
            state["size"] = int(size_match.group(1))
            remaining = remaining.replace(size_match.group(0), " ", 1)
            filled = True

        time_value = _parse_time(remaining)
        if time_value is not None:
            state["time"] = time_value
            filled = True

        if not filled:
            return self._invalid(conversation_id, text, state, state.get("node", "ask_date"))

        deps.counter.reset(f"invalid:{conversation_id}")

        if "date" not in state:
            state["node"] = "ask_date"
            return self._reply("What date? (YYYY-MM-DD)", "ask_date", state)
        if "time" not in state:
            state["node"] = "ask_time"
            return self._reply("What time?", "ask_time", state)
        if "size" not in state:
            state["node"] = "ask_size"
            return self._reply("How many people?", "ask_size", state)

        return self._search(state)

    def _search(self, state: dict) -> AgentReply:
        rooms = deps.store.select(
            "rooms",
            where={"capacity": {"gte": state["size"]}, "active": True},
            order_by="capacity",
            limit=3,
        )
        taken = {
            b["room"]
            for b in deps.store.select(
                "bookings", where={"date": state["date"], "time": state["time"]}
            )
        }
        free = [r for r in rooms if r["name"] not in taken]

        if not free:
            state["node"] = "handoff"
            return self._reply(
                "I could not find a room that fits. Let me get a colleague to help.",
                "handoff",
                state,
                facts={"rooms": rooms, "taken": sorted(taken)},
            )

        state["node"] = "offer"
        state["offered"] = [r["name"] for r in free]
        names = ", ".join(r["name"] for r in free)
        return self._reply(
            f"These fit {state['size']} on {state['date']} at {state['time']}: {names}. "
            "Reply with the name to book it.",
            "offer",
            state,
            facts={"rooms": free},
        )

    def _confirm(self, conversation_id: str, text: str, state: dict) -> AgentReply:
        offered = state.get("offered") or []
        chosen = next((r for r in offered if r.lower() in text.lower()), None)
        if chosen is None:
            return self._invalid(conversation_id, text, state, "offer")

        deps.counter.reset(f"invalid:{conversation_id}")
        state["node"] = "confirmed"
        state["room"] = chosen
        return self._reply(
            f"Booked: {chosen} on {state['date']} at {state['time']}.",
            "confirmed",
            state,
            facts={"room": chosen},
        )

    # -- fallback and escalation -------------------------------------------

    def _invalid(
        self, conversation_id: str, text: str, state: dict, node: str
    ) -> AgentReply:
        attempts = deps.counter.incr(f"invalid:{conversation_id}", ttl=600)
        if attempts >= MAX_INVALID:
            state["node"] = "handoff"
            return self._reply(
                "I am not following — let me bring in a colleague.", "handoff", state
            )

        # Legitimate, and scored: the deterministic path could not parse this and
        # is asking the model to. See `model_fallback` in the rubric.
        guess = deps.llm.complete(
            f"Extract a date, time or headcount from: {text!r}. Reply with one word."
        )
        state["node"] = node
        return self._reply(
            f"Sorry, I did not catch that. {guess}".strip(), node, state
        )

    def _reply(
        self, text: str, node: str, state: dict, facts: Any = None
    ) -> AgentReply:
        state["node"] = node
        return AgentReply(text=text, node=node, state=state, facts=facts)


def _parse_time(text: str) -> str | None:
    match = _TIME.search(text)
    if not match:
        return None

    clock_h, clock_m, mer_h, meridiem, at_h = match.groups()
    if clock_h is not None:
        hour, minute = int(clock_h), int(clock_m)
    elif mer_h is not None:
        hour, minute = int(mer_h), 0
        if meridiem.lower() == "pm" and hour < 12:
            hour += 12
        if meridiem.lower() == "am" and hour == 12:
            hour = 0
    else:
        hour, minute = int(at_h), 0

    if not (0 <= hour <= 23):
        return None
    return f"{hour:02d}:{minute:02d}"
