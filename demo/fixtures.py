"""The synthetic world the demo agent lives in.

Small enough to reason about, structured enough that filtering matters: three
rooms with different capacities, one already booked at a specific slot, so a
case can prove the agent does not offer a room that is taken.
"""

from __future__ import annotations

from replaygate.fakes import FakeCounter, FakeStore

ROOMS = [
    {"name": "Alcove", "capacity": 2, "active": True},
    {"name": "Studio", "capacity": 6, "active": True},
    {"name": "Atrium", "capacity": 14, "active": True},
    {"name": "Cellar", "capacity": 8, "active": False},  # decommissioned
]

BOOKINGS = [
    {"room": "Studio", "date": "2026-09-01", "time": "14:00"},
]


def build_store(read_only: bool = True) -> FakeStore:
    return FakeStore(
        rows={"rooms": [dict(r) for r in ROOMS], "bookings": [dict(b) for b in BOOKINGS]},
        read_only=read_only,
    )


def build_counter() -> FakeCounter:
    return FakeCounter()


ROOM_NAMES = {r["name"] for r in ROOMS}
ACTIVE_ROOM_NAMES = {r["name"] for r in ROOMS if r["active"]}
