"""The sandbox is the part everything else trusts, so it gets tested hardest.

A leaked patch is the worst failure mode here: the suite goes green, and
something unrelated three files away starts failing next week.
"""

from __future__ import annotations

import pytest

from replaygate import Count, Fake, Forbid, SeamViolation, sealed, seam

from . import subject


def test_fake_replaces_and_restores():
    original = subject.fetch

    with sealed(seam("tests.subject.fetch", Fake(lambda *_: "faked"))):
        assert subject.fetch("x") == "faked"

    assert subject.fetch is original
    assert subject.fetch("x") == "real:x"


def test_forbid_raises_and_records():
    with sealed(seam("tests.subject.call_model", Forbid("no model here"))) as report:
        with pytest.raises(SeamViolation, match="no model here"):
            subject.call_model("hello")

    assert report.violations == ["tests.subject.call_model was called: no model here"]
    assert report.clean is False


def test_count_lets_the_call_through_and_records_it():
    with sealed(
        seam("tests.subject.call_model", Count(lambda p: f"counted:{p}", label="model"))
    ) as report:
        assert subject.call_model("a") == "counted:a"
        assert subject.call_model("b") == "counted:b"

    assert report.crossed("model") == 2


def test_patches_are_restored_even_when_the_body_raises():
    original = subject.fetch

    with pytest.raises(RuntimeError):
        with sealed(seam("tests.subject.fetch", Fake(lambda *_: "faked"))):
            raise RuntimeError("boom")

    assert subject.fetch is original


def test_patches_are_restored_in_reverse_order():
    """Two seams on the same attribute must unwind to the original, not to the
    first replacement."""
    original = subject.fetch

    with sealed(seam("tests.subject.fetch", Fake(lambda *_: "one"))):
        with sealed(seam("tests.subject.fetch", Fake(lambda *_: "two"))):
            assert subject.fetch("x") == "two"
        assert subject.fetch("x") == "one"

    assert subject.fetch is original


def test_a_seam_can_target_a_method_on_an_object():
    with sealed(
        seam("tests.subject.client.send", Count(lambda m: f"sent:{m}", label="send"))
    ) as report:
        assert subject.client.send("hi") == "sent:hi"

    assert report.crossed("send") == 1
    assert subject.client.send("hi") == "real-send:hi"


async def test_count_wraps_async_callables():
    with sealed(
        seam("tests.subject.fetch_async", Count(_async_double, label="async"))
    ) as report:
        assert await subject.fetch_async(21) == 42

    assert report.crossed("async") == 1


async def _async_double(n: int) -> int:
    return n * 2


def test_non_callables_pass_through_untouched():
    store = {"rows": [1, 2, 3]}
    with sealed(seam("tests.subject.store", Fake(store))):
        assert subject.store is store
        assert subject.store["rows"] == [1, 2, 3]


def test_unknown_target_fails_loudly():
    with pytest.raises(ImportError):
        with sealed(seam("nope.not.a.module", Fake(object()))):
            pass  # pragma: no cover

    with pytest.raises(ValueError, match="dotted"):
        with sealed(seam("bare", Fake(object()))):
            pass  # pragma: no cover
