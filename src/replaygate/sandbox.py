"""The sealed sandbox: every I/O boundary gets a policy, and one of them can say no.

A replay harness is only worth its runtime if the code under test cannot quietly
escape it. The usual approach — patch the database, patch the HTTP client, hope
you got them all — fails silently in the direction that costs the most: a
deterministic path that was supposed to be pure starts calling the model, the
suite gets slower and non-reproducible, and nobody notices for months because
the assertions still pass.

So a seam is not "replaced or not". It carries a policy:

    FAKE    substitute a test double. The ordinary case.
    FORBID  raise on call. For boundaries the path under test must never cross.
    COUNT   let it through, but record the crossing as a scored defect.

FORBID is the one that changes behaviour in review. A deterministic conversation
step that reaches for an LLM does not produce a slightly worse score — it produces
a red run with the stack trace of the exact call.

COUNT is for boundaries that are *allowed* but *expensive*: falling back to the
model is legitimate, and it should still cost points (see ``rubric.py``).
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any


class SeamViolation(AssertionError):
    """A forbidden boundary was crossed. Deliberately an AssertionError so it
    surfaces as a test failure rather than something a broad except swallows."""


@dataclass(frozen=True, slots=True)
class Fake:
    replacement: Any


@dataclass(frozen=True, slots=True)
class Forbid:
    reason: str


@dataclass(frozen=True, slots=True)
class Count:
    """Allowed, but every crossing is recorded and can be scored."""

    replacement: Any
    label: str | None = None


Policy = Fake | Forbid | Count


@dataclass(frozen=True, slots=True)
class Seam:
    target: str
    """Dotted path to the attribute to patch, e.g. ``demo.agent.llm.complete``."""
    policy: Policy


def seam(target: str, policy: Policy) -> Seam:
    return Seam(target=target, policy=policy)


@dataclass
class SandboxReport:
    crossings: dict[str, int] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    def crossed(self, target: str) -> int:
        return self.crossings.get(target, 0)

    @property
    def clean(self) -> bool:
        return not self.violations


def _split(target: str) -> tuple[Any, str]:
    module_path, _, attribute = target.rpartition(".")
    if not module_path:
        raise ValueError(f"seam target must be dotted, got {target!r}")

    # Walk from the longest importable prefix so both `pkg.mod.attr` and
    # `pkg.mod.Class.method` work.
    parts = module_path.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for extra in parts[cut:]:
            obj = getattr(obj, extra)
        return obj, attribute
    raise ImportError(f"cannot import any prefix of {target!r}")


@contextlib.contextmanager
def sealed(*seams: Seam) -> Iterator[SandboxReport]:
    """Apply every seam, restore in reverse order, always.

    The restore loop is the boring part that matters: a harness that leaks a
    patch poisons every test that runs after it, and the failure surfaces
    somewhere unrelated.
    """
    report = SandboxReport()
    applied: list[tuple[Any, str, Any, bool]] = []

    try:
        for s in seams:
            owner, attribute = _split(s.target)
            had = hasattr(owner, attribute)
            original = getattr(owner, attribute, None)
            applied.append((owner, attribute, original, had))
            setattr(owner, attribute, _build(s, report))
        yield report
    finally:
        for owner, attribute, original, had in reversed(applied):
            if had:
                setattr(owner, attribute, original)
            else:  # pragma: no cover - defensive
                with contextlib.suppress(AttributeError):
                    delattr(owner, attribute)


def _build(s: Seam, report: SandboxReport) -> Any:
    policy = s.policy

    if isinstance(policy, Fake):
        return _wrap_counting(policy.replacement, s.target, report)

    if isinstance(policy, Forbid):
        def _forbidden(*_args: Any, **_kwargs: Any) -> Any:
            message = f"{s.target} was called: {policy.reason}"
            report.violations.append(message)
            raise SeamViolation(message)

        return _forbidden

    if isinstance(policy, Count):
        return _wrap_counting(policy.replacement, policy.label or s.target, report)

    raise TypeError(f"unknown policy {policy!r}")


def _wrap_counting(replacement: Any, key: str, report: SandboxReport) -> Any:
    """Count calls without changing behaviour.

    Non-callables (a fake store object, a constant) are returned untouched:
    wrapping them would break attribute access for no benefit.
    """
    if not callable(replacement):
        return replacement

    import functools
    import inspect

    if inspect.iscoroutinefunction(replacement):

        @functools.wraps(replacement)
        async def _async(*args: Any, **kwargs: Any) -> Any:
            report.crossings[key] = report.crossings.get(key, 0) + 1
            return await replacement(*args, **kwargs)

        return _async

    @functools.wraps(replacement)
    def _sync(*args: Any, **kwargs: Any) -> Any:
        report.crossings[key] = report.crossings.get(key, 0) + 1
        return replacement(*args, **kwargs)

    return _sync


def forbid_model(target: str) -> Seam:
    """Shorthand for the seam this library exists to make loud."""
    return seam(
        target,
        Forbid("the deterministic path must not call the model during replay"),
    )


__all__ = [
    "Count",
    "Fake",
    "Forbid",
    "Policy",
    "SandboxReport",
    "Seam",
    "SeamViolation",
    "forbid_model",
    "sealed",
    "seam",
]


# Re-exported for type checkers that want the callable alias spelled out.
CallableAny = Callable[..., Any]
