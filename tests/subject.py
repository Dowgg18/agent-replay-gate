"""A deliberately boring module for the sandbox tests to patch."""

from __future__ import annotations

from typing import Any

store: Any = {"rows": []}


def fetch(key: str) -> str:
    return f"real:{key}"


def call_model(prompt: str) -> str:
    return f"real-model:{prompt}"


async def fetch_async(value: int) -> int:
    return value


class _Client:
    def send(self, message: str) -> str:
        return f"real-send:{message}"


client = _Client()
